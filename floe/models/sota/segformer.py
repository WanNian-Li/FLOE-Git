"""SegFormer (Xie et al., 2021) — no mmseg required.

Uses SMP's MiT encoder backbone (default mit_b2, configurable via
options['segformer_encoder_name'], mit_b0~mit_b5) for feature extraction
plus a pure-PyTorch all-MLP SegformerHead decoder.
Only requires segmentation_models_pytorch (already a project dependency).

MiT-B1~B5 stage channels: [64, 128, 320, 512] at strides [4, 8, 16, 32].

E2（2026-07-19）：可选全分辨率纹理旁路 options['segformer_detail_branch']（默认
False，关闭时模块树/state_dict/前向与历史版本逐位一致）。动机与判定见
configs/FLOE/sota_segformer_b5_in1k_dlr_417_detail.py 顶部说明。

M1（2026-07-26）：可选对比度不变二阶纹理模块 options['segformer_vario_texture']
（默认 False，关闭时与历史版本逐位一致）。动机与判定见
configs/FLOE/sota_segformer_in1k_dlr_417_vario.py 顶部说明。

F3（2026-07-28）：可选跨景区域级 SupCon 辅助损失的投影头 options['segformer_supcon']
（默认 False）。只在 self.training 时往输出 dict 加 '_supcon' 键（stride-4 投影特征），
eval 前向与关闭时逐位一致 ⇒ evaluate/inference/按需读全部零改动。损失本体在
functions.supcon_region_loss，由 quickstart 训练循环组装。动机与判定见
configs/FLOE/sota_segformer_in1k_dlr_417_sgate2_supcon.py 顶部说明。

U1（2026-07-31）：可选 encoder **内部** 的逐像素滤波器组选择适配器
options['segformer_sk_adapter']（默认 False）。开启时 forward 走 `_encoder_forward_sk`
（显式复刻 smp 0.5.0 MixVisionTransformerEncoder 的分段前向），在每个 stage 的 NCHW
输出上插入一个零初始化残差适配器，**适配后的特征既进 head 也进下一 stage**。关闭时
encoder 调用路径与历史版本完全相同。动机与判定见
configs/FLOE/sota_segformer_in1k_dlr_417_sk.py 顶部说明。

U2（2026-07-31）：可选逐像素 log10-尺寸回归辅助头 options['segformer_scale_reg']
（默认 False）。走 '_supcon' 的同款模式——只在 self.training 时往输出 dict 加
'_scale_reg' 键（encoder stride-4 特征上的 1 通道回归图），eval 前向与关闭时逐位一致。
目标构造与损失在 quickstart.scale_reg_aux_loss。动机与判定见
configs/FLOE/sota_segformer_in1k_dlr_417_sreg.py 顶部说明。

V2（2026-08-05）：可选场景对抗（DANN）判别器 options['segformer_dann']（默认 False）。
梯度反转层 + 场景判别器，逼 encoder 抹掉特征里的"这是哪一景"信息。同样是 '_supcon'
契约（只在 self.training 时输出 '_dann' 键），eval/推理零改动。λ 由 quickstart 每
epoch 写进 `net.dann_lambda`。动机与判定见
configs/FLOE/sota_segformer_in1k_dlr_417_dann.py 顶部说明。

PGSG（2026-08-11）：可选证据条件化尺度门控 options['segformer_gate_evidence']
（默认 False，关闭时与历史版本逐位一致）。在 sgate2（softmax 尺度门控）基础上，把
两路物理接地、辐射稳健的"尺度证据"拼进门控的**决策输入**（不是主特征通路）：
① 对比度不变的 vario 剖面证据（_VarioTexture，zero_proj=False）；② 极化比先验分支
（_PolPrior，PA2-Swin 遗产）。与 E1（vario 残差与门控并联，campaign 最差）的本质
区别 = 证据为决策服务、不与决策争增益。门控末层零初始化不变 ⇒ 初始与无门控基线
逐位等价保持。故事与预注册判读见 docs/2026-08-11_PGSG_paper_story.md。

PGSG-v3（2026-08-11，用户拍板"方案一+方案二合体"）：两个新可选模块，默认全关、
关闭时与历史逐位一致：
① options['segformer_pba'] —— **物理偏置注意力 PBA**：monkey-patch MiT encoder 每个
   Attention 的 forward（逐行复刻 smp 0.5.0 mix_transformer.Attention.forward，从服务器
   环境原样抄写），在注意力 logits 上加极化相似度偏置 pol_proj(pol_q ⊗ pol_kv)。
   pol_proj 零初始化 ⇒ 初始逐位等价；新参数注册在顶层 self.pba_projs（不落 encoder.
   命名空间 ⇒ 拿满 LR）。这是 PA2_Swin_Improved.py 的 pol_bias 机制在 MiT 上的移植。
② options['segformer_scale_xattn'] —— **尺度轴 transformer**：head 里把每个像素的
   4 个尺度特征 + 1 个证据 token 当作长度 5 的序列做自注意力（跨尺度通信），输出经
   零初始化 out_proj 残差加回各尺度特征。sgate2 门控保留在其后（选择），本模块负责
   通信——对着首枪暴露的"大浮冰被碎判成中浮冰"失效模式（粗尺度告知细尺度背景语境）。
"""

import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


class _SegformerHead(nn.Module):
    """All-MLP decoder: project each scale to embed_dim, upsample to
    finest scale, concatenate, fuse, dropout, classify."""

    def __init__(self, in_channels: list, embed_dim: int, n_classes: int,
                 scale_gate: bool = False, gate_mode: str = 'tanh',
                 gate_groups: int = 1, gate_temp: float = 1.0,
                 evidence_ch: int = 0):
        super().__init__()
        # extra_pre 幅度监控用（普通 int，不进 state_dict）：见 forward 里的说明。
        self._pre_calls = 0
        # 门分布监控用（同上，普通 int）：见 forward 里门控分支的说明。
        self._gate_calls = 0
        self.n_scales = len(in_channels)
        self.embed_dim = embed_dim
        self.projections = nn.ModuleList([
            nn.Conv2d(c, embed_dim, 1, bias=False) for c in in_channels
        ])
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels), embed_dim, 1, bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.dropout = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(embed_dim, n_classes, 1)

        # N4（2026-07-27）逐像素尺度门控。现在 head 把 4 个 stage 投影后直接 concat 再 1×1
        # 融合 ⇒ 每个 stage 的权重是**全图固定**的。但"该信任哪个尺度"本就该随位置变：
        # 密集碎冰场的判别证据在细尺度、巨浮冰在粗尺度，而一景里两者常常共存。这里从
        # concat 自身预测 n_scales 个逐像素门 g_s(x)=1+tanh(a_s(x))∈(0,2)，乘回各 stage
        # 的那 embed_dim 个通道。**末层 weight+bias 全零 ⇒ tanh(0)=0 ⇒ g≡1 ⇒ 初始与基线
        # 逐位等价**（1.0 乘法在 IEEE 下精确），且 concat 维度不变、fuse 保持 1024→256 同构。
        # ⚠️ **必须在 projections/fuse/cls 之后【最后】构造**：Conv2d.reset_parameters 会
        # 消耗全局 RNG（BN 不会），建在前面会平移后续所有权重的初始化流 ⇒ 与基线的共有
        # 权重不再逐位相同 ⇒ N4 相当于同时换了 seed（07-20 实测纯 seed 噪声总分 0.033、
        # medium 0.078，足以淹没 <0.05 的效应，是假阴/假阳的直接来源）。
        # 这条纪律与 SegFormer.__init__ 里 detail_branch/vario_texture/context_module 一致。
        # N5（2026-07-28）两处强化，用来测"N4 的增益能否随门控表达力继续涨"：
        #  · gate_mode='softmax'：g = n_scales·softmax(a, dim=scale)，把"各尺度独立缩放"
        #    变成**真正的尺度竞争/预算分配**（Σ_s g_s ≡ n_scales）。零初始化时 softmax 给
        #    精确 0.25（exp(0)=1、和为 4、1/4 二进制精确），×4 = **精确 1.0** ⇒ 初始逐位
        #    等价照样成立。
        #  · gate_groups=G：每个 stage 的 embed_dim 通道拆 G 组各给一个门，提高自适应粒度。
        # H1（2026-07-29）gate_temp：softmax 门控的温度 T，g = S·softmax(a/T)。
        # T<1 = 竞争更锋利（同样的 logit 差产生更极端的选择）。动机：E2 赢在加约束
        # （softmax>tanh）、g4 输在放松约束（分组 −0.031），而 g4 日志显示选择熵训练
        # 后期仍停在 1.31/1.386 nats——门控只做温和重加权、从未真正"选"过；T<1 是
        # 约束方向的下一步。零初始化时 a≡0 ⇒ a/T≡0 ⇒ 任何 T 下 g 仍精确 1.0，初始
        # 逐位等价保持；且不增参数 ⇒ 与 E2 的 RNG 流逐位一致（E2-vs-N4 级干净对照）。
        assert gate_mode in ('tanh', 'softmax'), f'未知 gate_mode: {gate_mode}'
        assert embed_dim % gate_groups == 0, \
            f'embed_dim {embed_dim} 不能被 gate_groups {gate_groups} 整除'
        assert gate_temp > 0, f'gate_temp 必须 >0，得到 {gate_temp}'
        assert gate_mode == 'softmax' or gate_temp == 1.0, \
            'gate_temp 仅对 softmax 门控有意义（tanh 模式请保持 1.0）'
        # PGSG（2026-08-11）证据条件化门控：evidence_ch>0 时门控首层 conv 加宽，
        # 额外吃 SegFormer.forward 传入的证据张量（对比度不变 vario 剖面特征 + 极化比
        # 先验特征，见 _PolPrior 与 configs/FLOE/sota_segformer_in1k_dlr_cv5f0_pgsg.py）。
        # 与 E1（vario 走 extra_pre 残差 + 门控并联，0.5100 全 campaign 最差）的本质
        # 区别：证据**只**进门控的决策输入、不进主特征通路 ⇒ 两个机制不再争同一份
        # 增益，而是"证据为决策服务"。末层零初始化不变 ⇒ 初始 g≡1、与无门控基线
        # 逐位等价的性质保持（证据值是多少都无所谓）。
        assert evidence_ch == 0 or scale_gate, \
            'evidence_ch>0 需要 scale_gate=True（证据没有别的消费者）'
        self.evidence_ch = int(evidence_ch)
        self.gate_mode = gate_mode
        self.gate_groups = int(gate_groups)
        self.gate_temp = float(gate_temp)
        self.scale_gate = None
        if scale_gate:
            self.scale_gate = nn.Sequential(
                nn.Conv2d(embed_dim * self.n_scales + self.evidence_ch, 64, 1,
                          bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, self.n_scales * self.gate_groups, 1),
            )
            nn.init.zeros_(self.scale_gate[-1].weight)
            nn.init.zeros_(self.scale_gate[-1].bias)
        # PGSG-v3 方案二的挂载点：尺度轴 transformer（跨尺度通信，作用在门控之前）。
        # ⚠️ **只留占位，不在这里构造**（code-review 🔴1）：head 在 SegFormer.__init__
        # 中段就构造完了，若在此构造 _ScaleXAttn，其 Conv/MHA/Linear 消耗的大量 RNG
        # 会平移 head 之后所有模块（gate_ev_vario/gate_ev_pol）的初始化流 ⇒ v3-vs-v2
        # 的证据分支相当于换了 seed（07-20 实测纯 seed 噪声 macro 0.033）。构造挪到
        # SegFormer.__init__ 末尾（gate_ev 之后）再赋值到这里，state_dict 前缀不变
        # （仍是 heads.<chart>.scale_xattn.*）。
        self.scale_xattn = None

    def forward(self, features: list, extra: torch.Tensor = None,
                extra_pre: torch.Tensor = None,
                gate_evidence: torch.Tensor = None) -> torch.Tensor:
        target = features[0].shape[-2:]
        projected = []
        for feat, proj in zip(features, self.projections):
            x = proj(feat)
            if x.shape[-2:] != target:
                x = F.interpolate(x, size=target, mode='bilinear', align_corners=False)
            projected.append(x)
        # v3 尺度轴 transformer：先通信（零初始化残差，初始不改变 projected），
        # 再进下面的 concat + 门控选择。证据张量与门控共用同一份 gate_evidence。
        if self.scale_xattn is not None:
            projected = self.scale_xattn(projected, gate_evidence)
        x = torch.cat(projected, dim=1)
        # N4 逐像素尺度门控：g = 1 + tanh(·)∈(0,2)，按 stage 分块乘回去。末层零初始化时
        # g≡1 ⇒ 与基线逐位等价（1.0 乘法在 IEEE 下精确）。
        if self.scale_gate is not None:
            # 用 view 广播而不是 repeat_interleave：后者会实体化一份 [B,1024,H,W]
            # （训练 b16 约 134MB、验证 vbs32 约 268MB）。两者逐位等价——concat 与门控
            # 输出都是 **stage 主序、组次序** 的块状排列，view 后按 dim=1/2 广播即对齐。
            B, _, Hh, Ww = x.shape
            G, S, E = self.gate_groups, self.n_scales, self.embed_dim
            # PGSG：证据只拼进门控的**输入**（决策依据），不改被门控的 x 本身。
            # 显式互斥断言：evidence_ch>0 而证据缺失会在下面 AttributeError（报错不指
            # 向真因），evidence_ch==0 却传了证据则会被**静默丢弃**——两种都是接线
            # bug，宁可在这里干净地炸（code-review 🟢3）。
            assert (self.evidence_ch > 0) == (gate_evidence is not None), \
                f'evidence_ch={self.evidence_ch} 与 gate_evidence ' \
                f'{"缺失" if gate_evidence is None else "多余"} 不匹配（接线 bug）'
            if self.evidence_ch > 0:
                if gate_evidence.shape[-2:] != x.shape[-2:]:
                    gate_evidence = F.interpolate(
                        gate_evidence, size=x.shape[-2:],
                        mode='bilinear', align_corners=False)
                gate_in = torch.cat([x, gate_evidence.to(x.dtype)], dim=1)
            else:
                gate_in = x
            a = self.scale_gate(gate_in).view(B, S, G, 1, Hh, Ww)
            if self.gate_mode == 'softmax':
                # 尺度间竞争，Σ_s g_s ≡ S；gate_temp<1 锐化竞争（H1），T=1 与 E2 逐位一致
                g = S * torch.softmax(a.float() / self.gate_temp, dim=1)
            else:
                g = 1.0 + torch.tanh(a)                   # 各尺度独立缩放（N4 原版）
            # 门分布监控（只读、不进梯度/不耗 RNG，纪律同 extra_pre 监控）：
            # 平/负结果时用来区分「表达力没被用」和「用了但无益」——E3 的教训是读数
            # 必须跑前就位，否则负结果不可判读。三项读数：
            #   stage-mean g：各尺度平均门值（初始恒 1.0，偏离=在重分配预算）；
            #   选择熵（仅 softmax）：g/S 是尺度上的分布，熵≈ln S=没在选、≈0=硬选择；
            #   组间 std（仅 G>1）：4 组学出的门若几乎相同 ⇒ 分组容量未被用（g4 的判据）。
            # 间隔 = 几何前段 + 每 2 万次一打：纯几何间隔在 best epoch（e50~e120，
            # train+val 合计 ~4-8 万次前向）附近拿不到读数（code-review 🟡2）。
            self._gate_calls += 1
            if (self._gate_calls in (1, 100, 1000, 10000)
                    or self._gate_calls % 20000 == 0):
                with torch.no_grad():
                    gf = g.float()
                    stage_mean = gf.mean(dim=(0, 2, 3, 4, 5)).tolist()
                    msg = (f'[SegformerHead] gate #{self._gate_calls} '
                           f'({"train" if self.training else "val"}): '
                           'stage-mean g = ['
                           + ', '.join(f'{v:.3f}' for v in stage_mean) + ']')
                    if self.gate_mode == 'softmax':
                        p = (gf / S).clamp_min(1e-12)
                        ent = -(p * p.log()).sum(dim=1).mean().item()
                        # |a| 与 T 一起打（code-review 🟡3）：H1 温度枪判读必需——
                        # 「熵降 + |a| 与 E2 同量级」= 真·温度效应；「熵降但 |a| 同步
                        # 翻倍」= 门 logit 只是长得快，与 T 的锐化混淆、不可归因。
                        a_abs = a.float().abs().mean().item()
                        msg += (f' | 选择熵 {ent:.3f}/{math.log(S):.3f} nats'
                                '（≈上限=没在选尺度）'
                                f' | |a|={a_abs:.3f} T={self.gate_temp:g}')
                    if G > 1:
                        msg += (f' | 组间 std {gf.std(dim=2).mean().item():.3f}'
                                '（≈0=分组容量未被用）')
                    # PGSG 判读读数（预注册，docs/2026-08-11_PGSG_paper_story.md §二）：
                    # 定案口径是【激活级贡献比】（项目先例：M1/N2 的 ‖extra_pre‖/‖concat‖
                    # 范数占比才判得动，权重 RMS 只有相对自身初值的方向意义——列数
                    # 1024 vs 96 不对称、且证据/特征幅度可差数倍并随训练漂移）。
                    # 这里用 conv1 的两个权重切片分别卷"特征块"与"证据块"，比较两路
                    # 对 conv1 输出的实际贡献 RMS。conv 留在输入 dtype（fp16 亦可，
                    # 值域 O(1)~O(10)），统计在 fp32 里累加（历史坑：fp16 norm 溢出
                    # 会把"证据在说话"读成 0）。仅几何间隔触发，开销可忽略。
                    if self.evidence_ch > 0:
                        w1 = self.scale_gate[0].weight
                        n_main = S * E
                        y_f = F.conv2d(x, w1[:, :n_main].to(x.dtype))
                        y_e = F.conv2d(gate_evidence.to(x.dtype),
                                       w1[:, n_main:].to(x.dtype))
                        c_f = y_f.pow(2).mean(dtype=torch.float32).sqrt()
                        c_e = y_e.pow(2).mean(dtype=torch.float32).sqrt()
                        contrib = (c_e / c_f.clamp_min(1e-12)).item()
                        rms_f = w1[:, :n_main].pow(2).mean(
                            dtype=torch.float32).sqrt().item()
                        rms_e = w1[:, n_main:].pow(2).mean(
                            dtype=torch.float32).sqrt().item()
                        msg += (f' | 证据/特征激活贡献比 {contrib:.3f}'
                                '（趋 0=证据被闲置；升=门控在听证据）'
                                f' | conv1 权重 RMS f {rms_f:.4f} / e {rms_e:.4f}')
                print(msg, flush=True)
            x = (x.view(B, S, G, E // G, Hh, Ww)
                 * g.to(x.dtype)).view(B, -1, Hh, Ww)
        # M1 走 **fuse 之前**（加在 concat 上）：这样它的贡献要经过 fuse 的
        # conv+BN+ReLU 与 encoder 特征做非线性混合，而不是像 extra（E2）那样只能
        # 加性平移 logits（extra 之后只剩 1×1 cls，表达力被钉死成"一个独立线性
        # 分类器的偏置"）。零初始化下仍是加 0 ⇒ 与基线逐位等价、fuse 形状不变。
        if extra_pre is not None:
            if extra_pre.shape[-2:] != x.shape[-2:]:
                extra_pre = F.interpolate(extra_pre, size=x.shape[-2:],
                                          mode='bilinear', align_corners=False)
            # pre-fuse 注入自带一个风险：extra_pre 扰动的是 fuse 里 BN 的输入，若它长到
            # 与 concat 同量级，BN 的重归一化会等比压低 encoder 通路的有效幅度——模块就
            # 从"补充信息"变成"抢占带宽"。零初始化下它渐进长大，正常不会失控，但这是
            # "训练明显不如 baseline"时的首要怀疑机制，所以在几何间隔的若干次前向上打印
            # 一次幅度比（不加任何门控——加了就变双零不动点）。
            self._pre_calls += 1
            if self._pre_calls in (1, 100, 1000, 10000, 100000):
                with torch.no_grad():
                    # ⚠️ 必须在 fp32 里取 norm。autocast 下 x 是 fp16：norm 的 kernel 虽然
                    # 用 fp32 累加，但**最终 sqrt 结果被转回 fp16** ⇒ 判据是 ‖x‖<65504。
                    # 训练(b16) ‖x‖≈√6.7e7·RMS = 8192·RMS、验证(vbs32) 11585·RMS，初始
                    # RMS≈0.58 ⇒ 余量仅 ~10×，且随 √numel 缩小（换 p384 或调大 vbs 就吃掉
                    # 一半）。一旦溢出 ‖x‖=inf ⇒ ratio=0.0000，日志会读成"vario 完全没
                    # 启动"——恰好在这个诊断该起作用的场景下给出反向误导。另外
                    # clamp_min(1e-12) 在 fp16 下会下溢成 0、除零保护失效。
                    # 用 pow/sum(dtype) 而不是 .float().norm()：不实体化 fp32 副本。
                    num = extra_pre.pow(2).sum(dtype=torch.float32).sqrt()
                    den = x.pow(2).sum(dtype=torch.float32).sqrt().clamp_min(1e-12)
                    ratio = (num / den).item()
                print(f'[SegformerHead] forward #{self._pre_calls} '
                      f'({"train" if self.training else "val"}): '
                      f'‖extra_pre‖/‖concat‖ = {ratio:.4f}'
                      f'（远大于 ~1 = vario 在抢占 fuse 的带宽）', flush=True)
            x = x + extra_pre.to(x.dtype)
        x = self.fuse(x)
        # E2 纹理旁路走【残差相加】而非 concat 进 fuse：fuse 保持 1024→256 与基线
        # 完全同构，且零初始化的 extra 使初始函数与基线严格等价。
        # 注意：从无旁路 checkpoint 暖启动时 functions.load_model 是 strict=True，
        # 会因缺 detail_branch.* 键报错——届时需用 strict=False 加载（形状全兼容）。
        if extra is not None:
            if extra.shape[-2:] != x.shape[-2:]:
                extra = F.interpolate(extra, size=x.shape[-2:],
                                      mode='bilinear', align_corners=False)
            # .to(x.dtype) 是防御性的：这条 extra 路径（E2 detail branch）本身与 x 同为
            # autocast 下的 fp16，此处恒为 no-op、逐位不变；留着是为了将来若有旁路返回
            # fp32（如早期版本的 M1）不会把整条尾部提升成 fp32、改变显存/数值口径。
            x = x + extra.to(x.dtype)
        x = self.dropout(x)
        return self.cls(x)


class _DetailBranch(nn.Module):
    """E2 全分辨率纹理旁路：为亚特征格点尺度的纹理统计提供专用通道。

    small(0.25~1.25px)/medium(1.25~6px) 的判别信号是 1~6 px 尺度的冰-水交替
    粒度，低于 MiT stem（7×7/s4）之后的最细特征格点（stride 4 = 320 m/格）。
    本分支以比 stem 更缓和的方式下采样（s1→s2→s2，感受野从 3×3 起步），把
    高频纹理统计压进通道后在 stride-4 与 head 融合特征残差相加。

    零初始化【只零 proj 一处】：初始贡献为 0、函数与无旁路基线严格等价；
    梯度经 head 后续随机初始化权重回传，proj 第一步更新后全分支正常生长。
    ⚠️ 不要再叠加"标量门控零初始化"——双零级联时 ∂L/∂gate ∝ out=0、
    ∂L/∂W ∝ gate=0，是 AdamW 下的零梯度不动点，分支永远无法启动。
    """

    def __init__(self, in_ch: int, embed_dim: int):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(64, embed_dim, 1, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.convs(x))


class _VarioTexture(nn.Module):
    """M1：对比度不变的二阶纹理模块（可微变异函数 / 归一化结构函数剖面）。

    动机
    ----
    Stage 0 探针（`floe_separability_spatial.py`）里跨景可分性的赢因**不是信息量、
    是不变性**：variogram 相关长度 / 连通域尺寸是二阶量，且对场景亮度-对比度不变，
    所以能跨景（small-vs-medium AUC 0.729 vs marginal 0.651）。B1（多尺度局部标准差
    作输入通道，2026-07-25，hard-ice Δ−0.023）失败的最可能原因正是**缺这一步归一化**
    ——局部 std 同时编码"粒度尺度"和"后向散射对比度"，后者是景特异捷径（与 detail
    branch 的 speckle 捷径同族，也与 cake 的 train/val IoU 巨大 gap 自洽）。

    本模块把探针的赢因铸进网络：在 encoder 特征上算一组 lag 的结构函数（变异函数）

        γ_r(x) = mean_win[(f(x) − f(x±r))²]        （±r 两轴共 4 位移平均，固定窗口）

    然后**只把跨 lag 归一化后的剖面**送下去

        v_r(x) = log(γ_r + ε) − log(mean_r γ_r + ε)

    v 对 f 的乘性增益与加性偏置都（在 γ ≫ ε 时精确）不变：增益 g 使每个 log γ_r 平移
    同一个 2·log g、被去均值消掉；加性偏置对差分本就无影响。**幅度/对比度那一路
    （log mean_r γ）被刻意丢弃**——它正是要排除的捷径；encoder 主路仍携带幅度，
    全局并没有丢信息，只是这个模块的贡献被约束成"纯形状"。剖面的**形状**就是相关
    长度的稠密估计（γ 何时爬到 sill），也正是 small(20-100m)↔medium(100-500m) 的
    定义性差异。

    与 B1 / detail branch 的三点关键区别（前两个负结果不适用于此）：
      1. **归一化**：喂剖面比值而非绝对方差 → 对比度捷径在结构上被切断；
      2. **算在 encoder 特征上、不加输入通道**：不破坏与 in1k 首层的干净单变量对比
         （B1 的 8→12 通道破坏了），也不需要改 loaders / evaluate_scenes / 按需读；
      3. **不绕过 stem 的低通保护**：在 stride-4 特征上算，最小 lag = 4 px = 320 m，
         远离 1-px 斑点带（detail branch 正是在这里翻车）；B0 的滞后带消融也证明判别
         信号几乎全在 ≥320 m 的粗带（coarse AUC 0.606 ≈ all 0.625）。

    尺度换算（stride 4、80 m 像元 → 1 特征格 = 320 m）：
        radii (1,2,4,8,16) = 320 m / 640 m / 1.28 km / 2.56 km / 5.12 km
    覆盖 small↔medium↔big 的粒度带，最大 lag 用于估 sill（相关长度需要 plateau 参照）。
    注意 small/medium 的**浮冰粒度本身**（20-500 m）比最小 lag 还细，本模块测的是
    ≥320 m 的场结构——这不是设计疏漏：B0 的滞后带消融实测判别信号几乎全在 ≥320 m 的
    粗带（coarse 0.606 ≈ all 0.625），<320 m 的细带单独只有 0.568。
    若把 vario_stage 改成 1（stride 8）则每格 640 m，radii 语义整体 ×2。

    两个**估计器口径**（2026-07-26 code-review 后定案，跑前定死，改动即口径断代）
    ------------------------------------------------------------------------
    1. **4 方向对称位移**（±r 两轴），不是只取 +r 两轴。只取正方向时被平方的差分是
       (f(y)−f(y+r))、而窗口以 x 为中心 ⇒ γ_r 的实际测量支撑是 [x−pad, x+pad+r]、
       **中心漂到 x+r/2**（r=16 时偏 2.56 km），各 lag 的中心互不对齐，剖面就不是
       "同一位置的多尺度剖面"了。更糟的是这样对 h/v flip **不等变**（flip 把 γ⁺ 换成
       γ⁻），而增强里 `Random_h_flip/v_flip=0.5` 恒开 ⇒ 同一块物理纹理在翻转与否下
       给出不同特征，等于往正在被检验的这条通路里注入噪声、系统性偏向"没起效"。
       对称化后 γ_r 精确以 x 为中心、且对 flip/rot90 精确等变。
    2. **固定窗口**（默认 11 格 ≈ 3.5 km），不是随 r 放大。窗口随 r 放大会让
       log γ̂ 的估计噪声与 lag 强相关（std ≈ √(2/n)：3×3 窗只有 n=9 ⇒ 0.47，而
       scale 2→8 的真实剖面差异也才 ~0.5-1.5，最细的 lag 通道信噪比接近 1）。固定
       11×11 × 4 方向 ⇒ n≈484、噪声 std ≈0.06，各 lag 同口径。区域级支撑本来也正是
       想要的（标签是多边形级常数，polygon 尺度远大于 3.5 km）。

    数值/稳定性
    -----------
    - **只有 `profile()` 走 fp32**（γ 是平方量、又要取 log，fp16 下会下溢）；`mix`/`proj`
      是普通 conv+BN，留在外层 autocast 里走 fp16（v∈[−6,6] 精度够，且省掉 fp32 的
      [B,1024,H/4,W/4] 中间张量，见 `forward` 的注释）。
    - clamp(±6)：ratio 落在 [e⁻⁶, e⁶]；真实变异函数剖面不会超出该范围，越界只可能来自
      近常数区（黑边/掩膜区 γ→0）。全 lag 同时为 0 时 v 恰好为 0（无需特判）。
      **clamp 不会破坏不变性**：它是单调 1-Lipschitz 映射 ⇒ |clamp(v₁)−clamp(v₀)| ≤
      |v₁−v₀|，只会**缩小**仿射变换前后的差异，永远不是嫌疑人（不变性唯一的破功来源是
      ε 地板，它作用在 clamp 之前）。命中率要监控的是**另一件事**：命中意味着该 lag 通道
      的剖面被截断、失去对 γ 比值的分辨力（信息损失）⇒ 首次前向打印一次 γ 分位数与命中率
      （`_log_diag`），**训练日志里必须核对命中率极低**。
    - **单零初始化**（只零 proj 一处）：初始贡献为 0、函数与基线严格等价，梯度经 head
      后续随机权重回传，proj 第一步更新后整个模块正常生长。⚠️ 绝不再叠加标量门控
      零初始化——双零级联是 AdamW 下的零梯度不动点（E2 已踩实）。注意 `mix` 里 BN 的
      bias 天然零初始化，但那是**加性项、不阻断梯度**，不构成双零。
    """

    def __init__(self, in_ch: int, out_ch: int,
                 radii=(1, 2, 4, 8, 16), n_probe: int = 16,
                 mid: int = 64, win: int = 11, eps: float = 1e-6,
                 clamp: float = 6.0, out_pool: int = 1,
                 zero_proj: bool = True):
        # zero_proj=False 是 PGSG（2026-08-11）证据通路用的：模块输出只作为门控的
        # 条件输入、不是主通路残差 ⇒ 单零纪律不适用（门控自己的末层已零初始化，保证
        # 初始 g≡1；证据再零初始化只会让"经由门控零权重"的梯度多迟滞一层）。
        # nn.init.zeros_ 不消耗 RNG ⇒ 两种取值下模块的 RNG 消耗逐位相同。
        super().__init__()
        # out_pool>1：在【输入分辨率】上算剖面、再平均池化到 head 的 stride-4 网格。
        # 供 N6 细尺度模块用（在原始输入上测 lag 2~8 px = 160~640 m），这样 mix/proj
        # 仍然只在 stride-4 上跑、开销可控；剖面本身已是窗口统计量，再 4×4 平均无害。
        self.out_pool = int(out_pool)
        self.radii = tuple(int(r) for r in radii)
        assert all(r >= 1 for r in self.radii), 'radii 必须为正整数格数'
        self.win = int(win)
        assert self.win >= 1 and self.win % 2 == 1, 'win 必须是正奇数'
        self.eps = float(eps)
        self.clamp = float(clamp)
        self._diag_done = False
        self._warned_lag = False

        # 探针：把 C 维 encoder 特征线性压到 G 个"测纹理用"的标量场。逐通道增益由
        # 后面的跨 lag 归一化自动消掉，所以这里不需要（也不应该）做任何标准化。
        self.probe = nn.Conv2d(in_ch, n_probe, 1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(n_probe * len(self.radii), mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        if zero_proj:
            nn.init.zeros_(self.proj.weight)

    def _gamma(self, f: torch.Tensor, r: int) -> torch.Tensor:
        """局部结构函数 γ_r：±r 两轴共 4 个位移、固定窗口内平均 → [B,G,H,W]。

        边界用 **reflect**：镜像近似保持二阶统计量，而 replicate 会把边缘拉成常数、
        使大 lag 的 γ 在边界系统性偏小。位移按 min(r, dim−1) 截断：一是 reflect 要求
        pad < 该维长度，二是 lag 超过特征图本身时 γ 只能饱和到最大可测 lag（= 变异
        函数的 sill），这正是正确的退化行为。生产路径上永不触发——验证/训练的 tile 都
        被补齐到 patch_size=256 ⇒ stride-4 特征恒为 64×64 > max(radii)。
        """
        h, w = f.shape[-2:]
        # reflect 的隐性代价（2026-07-27 code-review）：越界索引 x+r 被映射到 2(h−1)−(x+r)，
        # 于是这些位置的**有效 lag** 变成 |2(h−1)−2x−r|，在 x = h−1−r/2 处**恰为 0**
        # （同一个像素、差分恒等于 0）。受影响的位置比例 ≈ r/h，而且它是与 tile 网格绑定的
        # **固定空间图案**（每个 patch 的同一相对位置），网络能直接学这个伪影。
        # M1（stage-0, h=64, r=16）是 25%；N1 若在 stage-1（h=32）也用 r=16 就是 50%，
        # 恰好落在"新增的那条粗 lag 带"上 = 最不可信的通道。r/h > 1/4 时打印一次告警。
        if (not self._warned_lag) and r * 4 > min(h, w):
            self._warned_lag = True
            print(f'[VarioTexture] ⚠️ 特征图 {h}×{w} 上 r={r}：约 {100 * r / min(h, w):.0f}% 的'
                  f'位置因 reflect 而有效 lag 失真（x=h−1−r/2 处退化为 0 lag）。该 lag 通道'
                  f'不可信 —— 给这个 stage 单独配更小的 radii（per-stage radii 已支持）。',
                  flush=True)
        rh, rw = min(r, h - 1), min(r, w - 1)
        pad = self.win // 2
        # 一次对称 pad，之后 4 个位移都只是切片；P 的中心块恰等于 f。
        p = F.pad(f, (rw, rw, rh, rh), mode='reflect')
        # ⚠️ 先把 4 个位移的 d² 累加、**只池化一次**：avg_pool2d 是线性算子，且 4 个 d² 形状
        # 完全相同 ⇒ count_include_pad=False 在每个位置的除数也相同 ⇒ Σᵢpool(d²ᵢ) ≡ pool(Σᵢd²ᵢ)。
        # 原写法把最贵的那个 op 做了 4 遍：M1 在 64×64 上无所谓，但细尺度模块在 256×256 上
        # 这就是主开销（20 次 stride-1 fp32 池化 → 5 次）。数学恒等，仅求和顺序不同（ULP 级）。
        acc = None
        for dh, dw in ((rh, 0), (-rh, 0), (0, rw), (0, -rw)):
            fs = p[..., rh + dh:rh + dh + h, rw + dw:rw + dw + w]
            d2 = (f - fs).pow(2)
            acc = d2 if acc is None else acc + d2
        return F.avg_pool2d(acc / 4.0, self.win, stride=1, padding=pad,
                            count_include_pad=False)

    def _log_diag(self, g: torch.Tensor, v: torch.Tensor) -> None:
        """首次前向打印一次 γ 量级与 clamp 命中率（唯一能在真实数据上核对
        "ε 地板 / clamp 没有破坏不变性"的观测点，见类 docstring）。"""
        self._diag_done = True
        with torch.no_grad():
            # torch.quantile 对输入元素数有上限（2²⁴），验证期 g 可达上千万 → 先抽样。
            flat = g.reshape(-1).float()
            if flat.numel() > 1_000_000:
                flat = flat[::flat.numel() // 1_000_000]
            q = torch.quantile(flat, torch.tensor([0.01, 0.5, 0.99], device=g.device))
            # ⚠️ clamp 命中率只能在【ε 有效域内】的位置上统计：黑边/陆地/掩膜区的 γ≡0
            # ⇒ v≡0 永不命中，而真实景 mask 占全图 ~70%，按全体取均值会把命中率系统性
            # 稀释 2~3 倍（方向保守、不会漏报大问题，但读数含义会说错）。
            live = (g.amax(dim=2, keepdim=True) > 100 * self.eps).expand_as(v)
            n_live = live.sum()
            frac_live = (n_live.float() / live.numel()).item()
            hit = (((v.abs() >= self.clamp - 1e-6) & live).sum().float()
                   / n_live.clamp_min(1)).item()
            # ★ 剖面是否"被压平"的读数（2026-07-28 加）。加性 nugget（斑点残余）对**每个
            # lag** 贡献相同的 2σ²：γ_r = γ_t(r) + 2σ²。当 2σ² ≳ γ_t 时整条剖面一起趋于
            # 常数 ⇒ v→0 ⇒ 模块输出近似常数+噪声，**归一化挡不住这种加性稀释**（它只挡
            # 乘性增益）。没有这个读数，一个 Δ≈0 的结果分不清"这个 lag 带没信号"还是
            # "剖面被压平、模块根本没说话"——前者会被写进路线图关掉一条线。
            per_lag = v.mean(dim=(0, 1, 3, 4))            # 每个 lag 的空间均值
            xstd = v.std(dim=2)                            # 跨 lag 标准差（逐像素）
            xf = xstd.reshape(-1).float()
            if xf.numel() > 1_000_000:
                xf = xf[::xf.numel() // 1_000_000]
            xq = torch.quantile(xf, torch.tensor([0.5, 0.9], device=g.device))
        print(f'[VarioTexture] γ 分位数 p1/p50/p99 = {q[0]:.3e}/{q[1]:.3e}/{q[2]:.3e} '
              f'(ε={self.eps:.0e}，p1 应远大于 ε) | ε 有效域占比 {100 * frac_live:.1f}% '
              f'| clamp(±{self.clamp:g}) 命中率(仅有效域) {100 * hit:.3f}% '
              f'(应 ≪1%；命中=该 lag 通道剖面被截断、失去对 γ 比值的分辨力。'
              f'clamp 是 1-Lipschitz、不破坏不变性)', flush=True)
        print(f'[VarioTexture] 剖面 radii={self.radii} 逐 lag 均值 = '
              f'{[round(float(x), 3) for x in per_lag]} | 跨 lag std p50/p90 = '
              f'{xq[0]:.3f}/{xq[1]:.3f} '
              f'(★ p50 ≲0.05 = 剖面被压平、模块没在说话，此时空结果【不能】读成'
              f'"该 lag 带无信号")', flush=True)

    def profile(self, f: torch.Tensor) -> torch.Tensor:
        """归一化结构函数剖面 v，[B, G*K, H, W]（probe 主序、lag 次序）。

        这是本模块的**核心不变量**：对 f → a·f + b 不变（a≠0），且随纹理相关长度改变。
        单独暴露出来供 smoke_m1_vario.py 直接断言这两条性质。
        """
        with torch.cuda.amp.autocast(enabled=False):
            p = self.probe(f.float())                                  # [B,G,H,W]
            g = torch.stack([self._gamma(p, r) for r in self.radii],
                            dim=2)                                     # [B,G,K,H,W]
            v = (torch.log(g + self.eps)
                 - torch.log(g.mean(dim=2, keepdim=True) + self.eps))
            v = v.clamp(-self.clamp, self.clamp)
            if not self._diag_done:
                self._log_diag(g, v)
            return v.flatten(1, 2)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        v = self.profile(f)
        if self.out_pool > 1:
            v = F.avg_pool2d(v, self.out_pool)
        # 只有 profile() 需要 fp32（平方 + log），它内部自带 autocast(enabled=False)。
        # mix/proj 是普通 conv+BN，让它们留在外层 autocast 里走 fp16：v∈[−6,6] 的
        # 精度需求 fp16 完全够，且省掉 fp32 的 [B,1024,H/4,W/4] 中间张量——proj 宽度
        # 是 embed_dim×4=1024，若整个 forward 强制 fp32，训练期瞬时多 ~0.4GB、
        # 验证期（vbs=32）多 ~1.1GB，挪到 b5 上就不是白送的。
        return self.proj(self.mix(v))


class _ScaleXAttn(nn.Module):
    """PGSG-v3 方案二：尺度轴 transformer —— 把"选尺度"升级成"跨尺度通信"。

    动机（首枪读数直接给出）：PGSG 首枪的门控把预算强压向最细尺度（末期
    g=[1.76,0.99,0.58,0.67]），中浮冰 +0.023 的代价是大浮冰内部被碎判成中浮冰
    （big .650→.587、medium 过预测 1.7 倍）。纯"选择"没有信息交换——细尺度不知道
    "这一片在粗尺度看是完整大浮冰"。本模块在门控**之前**让尺度之间先通信：每个像素
    位置的 S 个尺度特征（+1 个证据 token）作为长度 S+1 的序列过一层标准 transformer
    （LN→MHA→残差→LN→MLP→残差），输出只取 S 个尺度 token、经**零初始化 out_proj**
    残差加回各尺度特征 ⇒ 初始与不开模块逐位等价。门控（sgate2）保留在其后做选择。

    结构/成本（2026-08-11 code-review 🟡2 后加瓶颈维）：token 先经 `down`（E→d=64）
    压进瓶颈维再做 transformer，out_proj（d→E，零初始化）回投——直接在 E=256 上做
    的第一版是 ~12 GMAC/图（≈1.5 个 b2 encoder，docstring 还错写成 +15%），瓶颈版
    ~1.3 GMAC/图（全模型 +~15%），通信语义不变。参数 ~0.1M。
    **尺度嵌入**（code-review 🟡3）：MHA 对 token 轴置换等变，5 个 token 的身份只能
    靠内容区分 ⇒ 加一个零初始化的可学习嵌入 [1, S+1, d]，让"我是细尺度/它是粗尺度"
    可显式条件化——没有它，"粗尺度告知细尺度背景语境"这句话在结构上无载体。
    fp16：LN/MHA/Linear 全部 autocast 原生支持，无溢出面（特征 O(1)）。

    监控（几何间隔，纪律同 gate monitor）：① ‖resid‖/‖feat‖ ——模块有没有在说话；
    ② 尺度 token 对证据 token 的平均注意力份额——通信有没有用上证据。
    """

    def __init__(self, embed_dim: int, n_scales: int, evidence_ch: int = 0,
                 dim: int = 64, heads: int = 4, mlp_ratio: int = 2):
        super().__init__()
        self.n_scales = int(n_scales)
        self.evidence_ch = int(evidence_ch)
        self._calls = 0
        self.down = nn.Linear(embed_dim, dim)
        self.ev_proj = (nn.Conv2d(evidence_ch, dim, 1)
                        if evidence_ch > 0 else None)
        n_tok = self.n_scales + (1 if evidence_ch > 0 else 0)
        self.tok_embed = nn.Parameter(torch.zeros(1, n_tok, dim))
        self.ln1 = nn.LayerNorm(dim)
        # ⚠️ 不用 nn.MultiheadAttention：它走 scaled_dot_product_attention 融合核，
        # 而本模块的"批量"是 B·H·W（训练 65536、验证 131072）条长度 5 的序列，
        # batch×heads 超过 CUDA 网格维上限 65535 ⇒ GPU 上直接
        # "invalid configuration argument"（2026-08-11 链内 GPU 冒烟门禁实拦；
        # CPU 冒烟测不出）。手写 Q/K/V + 批量 matmul（cuBLAS batched GEMM 无此限制，
        # 且 L=5 的小注意力手写反而省掉融合核的调度开销）。
        assert dim % heads == 0
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )
        self.out_proj = nn.Linear(dim, embed_dim, bias=False)
        nn.init.zeros_(self.out_proj.weight)      # 单零：只零最后的投影

    def forward(self, projected: list, evidence: torch.Tensor = None) -> list:
        B, E, H, W = projected[0].shape
        S = self.n_scales
        toks = torch.stack([p.flatten(2).transpose(1, 2) for p in projected],
                           dim=2)                              # [B, HW, S, E]
        toks = self.down(toks)                                 # [B, HW, S, d]
        if self.ev_proj is not None:
            assert evidence is not None, \
                '_ScaleXAttn 构造时带 evidence_ch>0，forward 却没收到 evidence（接线 bug）'
            if evidence.shape[-2:] != (H, W):
                evidence = F.interpolate(evidence, size=(H, W),
                                         mode='bilinear', align_corners=False)
            ev = self.ev_proj(evidence.to(projected[0].dtype))
            toks = torch.cat([toks, ev.flatten(2).transpose(1, 2).unsqueeze(2)],
                             dim=2)                            # [B, HW, S+1, d]
        L = toks.shape[2]
        t = toks.reshape(B * H * W, L, -1) + self.tok_embed.to(toks.dtype)
        h = self.ln1(t)
        Bn, d = t.shape[0], t.shape[2]
        dh = d // self.heads
        qkv = (self.qkv(h).reshape(Bn, L, 3, self.heads, dh)
               .permute(2, 0, 3, 1, 4))                        # [3, Bn, nH, L, dh]
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn_w = ((q @ k.transpose(-2, -1)) * dh ** -0.5).softmax(dim=-1)
        attn_out = self.attn_out((attn_w @ v).transpose(1, 2).reshape(Bn, L, d))
        t = t + attn_out
        t = t + self.mlp(self.ln2(t))
        resid = self.out_proj(t[:, :S])                        # [B*HW, S, E]
        resid = resid.reshape(B, H * W, S, E).permute(0, 2, 3, 1) \
                     .reshape(B, S, E, H, W)
        self._calls += 1
        if self._calls in (1, 100, 1000, 10000) or self._calls % 20000 == 0:
            with torch.no_grad():
                num = resid.pow(2).sum(dtype=torch.float32).sqrt()
                den = sum(p.pow(2).sum(dtype=torch.float32)
                          for p in projected).sqrt().clamp_min(1e-12)
                msg = (f'[ScaleXAttn] #{self._calls} '
                       f'({"train" if self.training else "val"}): '
                       f'‖resid‖/‖feat‖ = {(num / den).item():.4f}'
                       '（趋 0=通信结果被闲置）')
                if self.ev_proj is not None:
                    # 证据 token 注意力份额：手写注意力的权重矩阵本来就实体化在
                    # attn_w 里（[Bn,nH,L,L]，fp16 仅 ~13MB），直接读，零额外前向
                    ev_share = attn_w[:, :, :S, S].mean(
                        dtype=torch.float32).item()
                    msg += (f' | 尺度→证据注意力份额 {ev_share:.3f}'
                            f'（均匀基线 {1.0 / L:.3f}）')
            print(msg, flush=True)
        return [projected[s] + resid[:, s].to(projected[s].dtype)
                for s in range(S)]


class _PolPrior(nn.Module):
    """PGSG（2026-08-11）：极化比先验分支 —— PA2-Swin 遗产的 SegFormer 版。

    PA2-Swin（SOD 时代）把极化比（pol ratio，HH−HV；z-score/dB 域之差 = 线性域之比）
    作为**物理先验去引导机制**（注意力 logits 偏置），而不是仅作输入通道。本分支沿用
    这个哲学，但引导对象换成 FLOE 战役里被验证的机制——sgate2 逐像素尺度门控：

      · pol ratio 是**免定标**的散射机制指纹：绝对定标/入射角的共模分量在两通道之差里
        对消（radaug 的通道抖动设计"共模满 B、差分仅 B/3"正是为保护它，08-04 0a 探针
        实测 corr(HH,HV)=0.595 的部分共模结构）。
      · 散射机制（开阔水镜面 / 平整一年冰 / 形变冰体散射）决定了哪个空间尺度带携带
        浮冰粒度信息 ⇒ 它是"该信哪个尺度"这个决策的自然条件变量，喂给门控而不是
        喂给分类器。

    结构：全分辨率 pol 通道 → 4×4 平均池化到 stride-4（与 encoder 特征网格对齐，
    池化即抗斑点）→ 两层 3×3 conv+BN+ReLU → out_ch 通道证据图。
    **随机初始化**（不是残差、只进门控；门控末层的零初始化已保证初始等价，见
    _SegformerHead 的 evidence 说明）。BN 在这里顺带承担 pol 幅度的批归一化——
    分支输出的量纲与 vario 证据可比。
    """

    def __init__(self, out_ch: int = 32, pool: int = 4, mid: int = 16):
        super().__init__()
        self.pool = int(pool)
        self.net = nn.Sequential(
            nn.Conv2d(1, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, pol: torch.Tensor) -> torch.Tensor:
        # pol: [B,1,H,W]（输入张量的最后一个通道，SegFormer.forward 里断言过口径）
        p = F.avg_pool2d(pol, self.pool)
        return self.net(p)


def _pba_patch_attention(attn: nn.Module, proj: nn.Module, holder: dict,
                         tag: str = '', monitor: bool = False) -> None:
    """PGSG-v3 方案一：给一个 smp MiT `Attention` 实例打上极化偏置补丁。

    机制（PA2_Swin_Improved.py 的 pol_bias 移植 + 2026-08-11 code-review 🟡7 升级）：
    注意力 logits 上加**广义二次极化核**
        bias = pol_proj([p_i², p_j², p_i·p_j])，pol_proj: Linear(3, num_heads) **零初始化**
    ⇒ 初始 bias 恒为 0、`attn + 0` 逐位等于原实现 ⇒ 初始逐位等价（smoke 断言）。
    为什么是 3 项而不是 PA2 原版的单外积 p_i·p_j：外积核的最大化条件是"同号且都大"，
    不是"相近"（p_i=p_j=−2 得 +4、p_i=p_j=0 得 0），且 pol 通道的非零均值会混入
    行/列线性显著性项——它撑不起"同散射域互相多看"的叙事。3 项可学习组合**同时
    包含**点积核（w=[0,0,1]）与相似核 −(p_i−p_j)²（w=[−1,−1,2]），让数据自己选核形，
    叙事口径统一为"可学习的二次极化核偏置"。
    物理语义：极化比（免定标的散射机制指纹）刻画散射机制——注意力的聚合范围顺着
    "同散射域=同一块浮冰"走，在散射跳变（浮冰边界）处被抑制。

    实现要点：
    · **monkey-patch forward 而不是替换模块**：state_dict 键完全不变（pol_proj 注册在
      SegFormer 顶层 self.pba_projs），历史 checkpoint 兼容性与 [A] 冒烟模式照旧成立。
    · 前向主体**逐行复刻 smp 0.5.0 `mix_transformer.Attention.forward`**（2026-08-11 从
      服务器环境 inspect 原样抄写）——op 顺序一字不改，pol 缺席时与原实现逐位相同。
      ⚠️ smp 版本漂移的唯一哨兵是 smoke_pgsg3.py [E]（零偏置下与未打补丁 encoder 逐位
      对拍），升级 smp 前必须先过它。
    · pol 通过 `holder`（SegFormer 实例上的普通 dict）传入：SegFormer.forward 在调
      encoder 前放入全分辨率 pol 图、调完立刻 pop ⇒ 不会跨 batch 残留；encoder 被
      单独调用（无 pol）时自动退化为原始注意力。
    · pol_q 用 adaptive_avg_pool2d 池到该 stage 的 token 网格（height×width 参数由
      Block 传入，天然正确）；pol_kv 用 avg_pool2d(kernel=stride=sr_ratio) 池到 KV
      网格——与 sr conv（kernel=stride=sr_ratio）的几何精确一致。
    · fp16 安全：pol ~±3 z-unit、外积 O(10)、Linear(1,nH) 输出 O(1)，无溢出面。
    """
    _calls = [0]          # 闭包计数器（仅 monitor=True 的实例使用，不进 state_dict）

    def forward(self, x, height, width):
        batch_size, N, C = x.shape
        q = (
            self.q(x)
            .reshape(batch_size, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(batch_size, C, height, width)
            x_ = self.sr(x_).reshape(batch_size, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = (
                self.kv(x_)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        else:
            kv = (
                self.kv(x)
                .reshape(batch_size, -1, 2, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        pol = holder.get('pol')
        if pol is not None and pol.shape[0] != batch_size:
            # 静默降级改为 warn 一次（code-review 🟢12）：正常路径 holder 与 x 同批，
            # 走到这里说明有调用方在 SegFormer.forward 之外用了 encoder 且 holder 残留
            if not holder.get('_warned_bs', False):
                holder['_warned_bs'] = True
                print(f'[PBA] ⚠️ holder 里的 pol batch={pol.shape[0]} 与当前 '
                      f'batch={batch_size} 不符，本次前向跳过极化偏置（只警告一次）',
                      flush=True)
            pol = None
        if pol is not None:
            pol_q = F.adaptive_avg_pool2d(pol, (height, width)) \
                     .flatten(2).transpose(1, 2)                    # [B, N, 1]
            if self.sr_ratio > 1:
                pol_kv = F.avg_pool2d(
                    pol_q.transpose(1, 2).reshape(batch_size, 1, height, width),
                    self.sr_ratio).flatten(2).transpose(1, 2)       # [B, N_kv, 1]
            else:
                pol_kv = pol_q
            # 广义二次核的 3 个特征（见函数 docstring）：p_i²/p_j² 沿另一轴广播
            n_kv = pol_kv.shape[1]
            feats = torch.stack([
                (pol_q * pol_q).expand(-1, -1, n_kv),               # p_i²
                (pol_kv * pol_kv).transpose(-2, -1).expand(-1, N, -1),  # p_j²
                pol_q @ pol_kv.transpose(-2, -1),                   # p_i·p_j
            ], dim=-1)                                              # [B, N, N_kv, 3]
            bias = proj(feats).permute(0, 3, 1, 2)                  # [B, nH, N, N_kv]
            if monitor:
                _calls[0] += 1
                c = _calls[0]
                if c in (1, 1000, 10000) or c % 50000 == 0:
                    with torch.no_grad():
                        b_rms = bias.pow(2).mean(dtype=torch.float32).sqrt().item()
                        a_rms = attn.pow(2).mean(dtype=torch.float32).sqrt().item()
                    print(f'[PBA {tag}] #{c}: bias RMS {b_rms:.4f} vs attn-logit '
                          f'RMS {a_rms:.3f}（比值 {b_rms / max(a_rms, 1e-12):.4f}；'
                          '趋 0=物理偏置被闲置）', flush=True)
            attn = attn + bias.to(attn.dtype)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(batch_size, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    attn.forward = types.MethodType(forward, attn)


class _ContextAggregator(nn.Module):
    """N2（2026-07-27）：decoder 侧上下文聚合模块（ASPP 空洞金字塔 + 轻量 non-local）。

    为什么这条没被 G1 覆盖
    ----------------------
    G1（patch 256→384）改的是**模型能看到多大的输入**，Δ=−0.004。本模块改的是**已有特征
    怎么聚合**——这是两件事：SegFormer 的 all-MLP head 只做逐点 1×1 融合，四个 stage 的
    特征各自上采样后直接拼接，**decoder 里没有任何跨位置的聚合算子**。路线图 3b 列了它但
    一直没实现。

    机理留口（记忆里那条"关键留口"）：Stage 0 的 ~0.68 是 **context-free 的纯 tile 区域
    纹理天花板**——探针刻意剥离了上下文。浮冰尺寸在空间上是连贯的（冰图多边形级常数），
    所以"模糊区旁边若存在可自信分类的锚区"时，跨位置聚合能消歧。ASPP 给多个膨胀率的
    **局部-中程**证据，non-local 给**全局**证据（"这一景整体是什么粒度"是很强的先验）。

    ⚠️ **挂在 stride-8（32×32），不是最粗的 stride-32**（2026-07-27 code-review 后改）：
    patch_size=256 ⇒ stride-32 的特征图恒为 **8×8**（训练 tile、`fast_tiled_val_*` 的 tile、
    滑窗窗口全都是 patch_size，没有任何路径喂进更大的图）。3×3 空洞卷积在 8×8 上，
    **d=12/18 的 9 个抽头有 8 个永远落在零 padding 里、梯度恒为 0**，d=6 也只剩中心抽头
    ⇒ 整个 ASPP 退化成 4 个近似 1×1，"多尺度上下文"这一半**根本没实现**。那样跑出来的
    Δ≈0 会被误写成"上下文聚合无效"，是典型的假负结果。
    改挂 stride-8（32×32，1 格 = 640 m）+ dilations (1,2,4,8) ⇒ 物理覆盖
    640 m / 1.28 km / 2.56 km / 5.12 km，正好横跨 big(0.5-2km) 与 vast(2-10km) 的尺度，
    且每个膨胀率的有效位置比例为 (32−2d)/32 = 94%/88%/75%/50%，都不退化。
    ⚠️ 诚实边界：non-local 的 token 全部来自**当前这个 256 patch（20.5 km）**，模块看不到
    整景 ⇒ "这一景整体什么粒度"这种景级先验在当前推理口径下**拿不到**；可达的上下文扩张
    被 patch 卡死（G1 patch 256→384 实测 Δ−0.004）。别把结论写成"景级先验无效"。

    结构：
      - 4 路并行 3×3 空洞卷积 + 1 路全局池化分支 → concat → 1×1 压回；
      - 一个轻量 non-local（key/query 降维到 1/8，softmax 注意力；32×32=1024 token，
        vbs=32 下注意力矩阵 fp16 约 67MB，可接受）；
      - 1×1 投影到 embed_dim×n_scales，**单零初始化**，走 extra_pre 加进 head 的 concat。
    """

    def __init__(self, in_ch: int, out_ch: int, mid: int = 128,
                 dilations=(1, 2, 4, 8)):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        self._warned = False
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=d, dilation=d, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
            for d in dilations
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.merge = nn.Sequential(
            nn.Conv2d(mid * (len(dilations) + 1), mid, 1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        # 轻量 non-local（stride 32 上 token 很少，全局注意力几乎免费）
        self.q = nn.Conv2d(mid, mid // 8, 1, bias=False)
        self.k = nn.Conv2d(mid, mid // 8, 1, bias=False)
        self.v = nn.Conv2d(mid, mid, 1, bias=False)
        self.nl_out = nn.Sequential(nn.Conv2d(mid, mid, 1, bias=False),
                                    nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.proj = nn.Conv2d(mid, out_ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)          # 单零：只零最后的投影

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        B, _, H, W = f.shape
        # 护栏：3×3 空洞卷积要求 2d+1 ≤ 边长，否则该支的非中心抽头恒落在 padding 上、
        # 梯度恒为 0（等于静默退化成 1×1）。换 patch_size / 换 stage 时最容易踩，打印一次。
        if not self._warned:
            dead = [d for d in self.dilations if 2 * d + 1 > min(H, W)]
            self._warned = True
            if dead:
                print(f'[ContextAggregator] ⚠️ 特征图 {H}×{W} 上 dilation {dead} 的非中心抽头'
                      f'恒落在 padding 内（梯度为 0，该支退化成 1×1）—— ASPP 的多尺度部分'
                      f'未生效，换 stage 或改小 dilations！', flush=True)
            else:
                print(f'[ContextAggregator] 特征图 {H}×{W}，dilations={self.dilations}，'
                      f'各支有效位置比例 '
                      f'{[round(max(0, min(H, W) - 2 * d) / min(H, W), 2) for d in self.dilations]}',
                      flush=True)
        feats = [b(f) for b in self.branches]
        g = self.gap(f)
        feats.append(g.expand(-1, -1, H, W))
        x = self.merge(torch.cat(feats, dim=1))
        # non-local：残差加回，不替换（保住 ASPP 的局部证据）
        q = self.q(x).flatten(2).transpose(1, 2)              # [B, HW, c]
        k = self.k(x).flatten(2)                              # [B, c, HW]
        v = self.v(x).flatten(2).transpose(1, 2)              # [B, HW, mid]
        att = torch.softmax(q @ k / (q.shape[-1] ** 0.5), dim=-1)
        nl = (att @ v).transpose(1, 2).reshape(B, -1, H, W)
        x = x + self.nl_out(nl)
        return self.proj(x)


class _OrdinalHead(nn.Module):
    """N3（2026-07-27）：CORAL 式单调有序输出头。**不是软标签**——目标分布一点没改，
    改的是 6 个 logit 的**参数化方式**。

    动机：FLOE 的 5 个冰类在物理上是**同一条尺寸轴的量化**
    （cake<20m < small 20-100m < medium 100-500m < big 0.5-2km < vast 2-10km），
    但基线用 6 个互不相干的 logit 各自划决策面，完全没有利用这个序结构：整条尺寸轴上的
    数据无法互相支撑，small↔medium 的分界是一个自由曲面而不是一个阈值。

    参数化：head 只出 2 个通道
        z_w  —— 水/冰二分类分数
        t    —— 冰的**尺寸分数**（1 维、连续）
    外加 4 个**保序**阈值 b_1<b_2<b_3<b_4（由 b_1 + cumsum(softplus(d)) 构造，恒保序）：
        P(water) = σ(z_w)，  P(ice) = 1−σ(z_w)
        s_k = σ(t − b_k) = P(尺寸 ≥ 第 k 档 | ice)，k=1..4，因 b 递增故 s_1≥s_2≥s_3≥s_4
        P(cake)=P(ice)(1−s_1)  P(small)=P(ice)(s_1−s_2)  P(medium)=P(ice)(s_2−s_3)
        P(big)=P(ice)(s_3−s_4) P(vast)=P(ice)·s_4
    各项非负、和为 1。**输出 log(P) 当作 logits 返回** ⇒ 下游 CrossEntropyLoss（= NLL∘
    log_softmax，而 log_softmax(log P)=log P 因为 ΣP=1）、class_decider、混淆矩阵、
    evaluate_scenes 全部不用改。

    ⚠️ 与基线**不是**零初始化等价（输出结构本身变了），所以 smoke 里只断言形状/概率性质，
    不断言逐位等价；也不能从基线 checkpoint 暖启动 cls 权重。
    """

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        assert n_classes == 6, f'_OrdinalHead 假定 6 类（水+5 冰），收到 {n_classes}'
        self.n_classes = n_classes
        self.cls = nn.Conv2d(embed_dim, 2, 1)      # [z_w, t]
        # b_1 初始 -1.5，之后三个间隔 softplus(0.54)≈1.0 → 初始阈值 ≈ [-1.5,-0.5,0.5,1.5]
        self.b0 = nn.Parameter(torch.tensor(-1.5))
        self.dlt = nn.Parameter(torch.full((3,), 0.5413))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zw, t = self.cls(x).float().split(1, dim=1)          # 概率组合在 fp32 里算，避免 fp16 下溢
        # softplus 加 1e-3 地板：dlt 极负时 softplus 会 fp32 下溢成 0 ⇒ 两个阈值相等 ⇒
        # 夹在中间的那一类概率恒 0、梯度死。wd 会把 dlt 拉向 0，风险低，但地板是免费保险。
        b = torch.cat([self.b0.view(1),
                       self.b0 + torch.cumsum(F.softplus(self.dlt) + 1e-3, 0)])   # 恒保序
        # clamp(±12)：两个 sigmoid 同时饱和到 fp32 的 0/1 时 s_hi−s_lo 会灾难性抵消为精确 0，
        # 而 clamp_min(1e-8) 在被截断处**梯度为 0** ⇒ "自信预测巨浮冰、真值是 cake"这类最需要
        # 梯度的像素反而完全学不动。e⁻¹²≈6e-6 的表达力对 6 类足够，且把梯度留在活的区间。
        s = torch.sigmoid((t - b.view(1, -1, 1, 1)).clamp(-12.0, 12.0))   # [B,4,H,W]，沿 k 递减
        p_ice = torch.sigmoid(-zw)                            # = 1 − σ(zw)
        ones = torch.ones_like(s[:, :1])
        zeros = torch.zeros_like(s[:, :1])
        s_hi = torch.cat([ones, s], dim=1)                    # s_0=1
        s_lo = torch.cat([s, zeros], dim=1)                   # s_5=0
        p_size = (s_hi - s_lo).clamp_min(0)                   # [B,5,H,W]，5 个冰类，和为 1
        probs = torch.cat([torch.sigmoid(zw), p_ice * p_size], dim=1)
        return torch.log(probs.clamp_min(1e-8))


class _SKAdapter(nn.Module):
    """U1（2026-07-31）：encoder 内的**逐像素滤波器组选择**适配器（selective-kernel 式）。

    定位（与已关的两条线的区分）
    ----------------------------
    N4/E2 的门控是在 head 里选**已经算好的四个 stage**（尺度选择发生在解码端）；本模块把
    "选择"搬进 encoder：每个 stage 输出上并联一组 **depthwise** 3×3 空洞卷积（d 不同 ⇒
    支撑半径不同），由一个 1×1 的 `sel` 头逐像素 softmax 选择用哪一支，零初始化投影后残差
    加回。**适配后的特征继续喂给下一个 stage** ⇒ 选择结果在深度上复合，这才是"编码端选
    尺度"，而不是 v1 那种贴在 encoder 输出上、只影响 head 输入的写法（后者落回 E1"head 内
    机制互抢"的射程）。

    叙事口径（跑前定死）：这是"**滤波器组的逐像素选择**"，不是"不扩上下文半径"——d=4 在
    stride-8 上已是 2.5 km 量级。与 N2（跨位置聚合，−0.061）的区别在 **选择 vs 聚合**，
    且 depthwise ⇒ 无任何跨通道混合（跨通道混合留作 U1 赢后的加深，不混进本枪）。

    单零初始化（纪律同 _DetailBranch/_VarioTexture）
    ------------------------------------------------
    只零 `proj` 一处 ⇒ 初始 residual 恒为 0、`x + 0` 逐位等于 x；梯度先进 proj（∂L/∂proj ∝ y
    ≠ 0），proj 第一步更新后 branches/sel 正常生长。⚠️ 绝不再叠"标量门控零初始化"（双零是
    AdamW 下的零梯度不动点，E2 已踩实）。`sel` 刻意**不**零初始化：它是加权系数、经 softmax
    后恒 >0，零初始化只会让初始选择均匀，没有必要且会多一个不动点风险。

    风险（判读时必须一起看）
    ------------------------
    适配后的特征送进 **ImageNet 预训练** 的后续 stage ⇒ 存在分布偏移；零初始化只保证
    **初始**等价，训练中的偏移正是本枪要测的效应本身，但它也可能损伤预训练特征。平结果时
    结合 `_eval_radiometric.py` 判"表征有没有真的变"。

    监控（两层，纪律同 _SegformerHead 的 gate monitor）
    --------------------------------------------------
      · 几何间隔打印：各分支平均选择权重 / 选择熵 / ‖proj(y)‖/‖x‖ 范数比；
      · epoch 级累加器 `_acc`（detached GPU 标量，**不 .item()**，避免每 batch 一次 GPU
        同步），由 `SegFormer.pop_sk_stats()` 每 epoch 取一次落成 wandb 曲线——E3 的教训是
        读数必须跑前就位、且要能与 hard-ice 曲线对齐，几何间隔在 best epoch 附近拿不到。
      **计数与观测全部只在 self.training 时发生**：一次验证要前向几万个 tile，若把 val
      也算进来，① 几何间隔的 1/100/1000/10000 会被第一次验证吃光、训练期再无读数，
      ② 上一个 epoch 的验证会混进下一个 epoch 的 `_acc`。唯一例外是首次前向的特征图
      尺寸/dilation 护栏（下方 `_warned`），它在任何模式下都要打印一次。
    """

    def __init__(self, ch: int, dilations=(1, 2, 4), tag: str = ''):
        super().__init__()
        self.dilations = tuple(int(d) for d in dilations)
        assert len(self.dilations) >= 2, 'SK 适配器至少要有 2 个分支才谈得上"选择"'
        assert all(d >= 1 for d in self.dilations), f'dilations 必须为正整数：{dilations}'
        self.tag = str(tag)
        # 普通 int / None：不进 state_dict（与 _SegformerHead 的 _gate_calls 同纪律）。
        self._calls = 0
        self._warned = False
        self._acc = None
        self.branches = nn.ModuleList([
            nn.Conv2d(ch, ch, 3, padding=d, dilation=d, groups=ch, bias=False)
            for d in self.dilations
        ])
        self.sel = nn.Conv2d(ch, len(self.dilations), 1)
        self.proj = nn.Conv2d(ch, ch, 1, bias=False)
        nn.init.zeros_(self.proj.weight)          # 单零：只零最后的投影

    def _observe(self, x: torch.Tensor, resid: torch.Tensor,
                 sel: torch.Tensor, do_print: bool) -> None:
        """只读观测：累加 epoch 统计 +（按几何间隔）打印一行。不耗 RNG、不进梯度。"""
        with torch.no_grad():
            w_mean = sel.mean(dim=(0, 2, 3))                       # [K]，fp32
            ent = -(sel.clamp_min(1e-12) * sel.clamp_min(1e-12).log()).sum(dim=1).mean()
            # ⚠️ 范数必须用 pow/sum(dtype=fp32) 而不是 .float().norm()：autocast 下
            # resid/x 是 fp16，norm 的最终 sqrt 会被转回 fp16、大张量上溢出成 inf ⇒
            # ratio=0.0000 会被误读成"模块完全没启动"（恰在诊断最该起作用时反向误导）。
            num = resid.pow(2).sum(dtype=torch.float32).sqrt()
            den = x.pow(2).sum(dtype=torch.float32).sqrt().clamp_min(1e-12)
            ratio = num / den
            if self.training:                 # 防御性：唯一调用点已是 training-only
                if self._acc is None:
                    self._acc = {'w': torch.zeros_like(w_mean),
                                 'ent': torch.zeros_like(ent),
                                 'ratio': torch.zeros_like(ratio),
                                 'n': 0}
                self._acc['w'] += w_mean
                self._acc['ent'] += ent
                self._acc['ratio'] += ratio
                self._acc['n'] += 1
            if do_print:
                # 不打 train/val 标签：本函数只由训练前向调用（见 forward 的说明）。
                print(f'[SKAdapter {self.tag}] train #{self._calls}: '
                      'branch mean sel = ['
                      + ', '.join(f'{v:.3f}' for v in w_mean.tolist())
                      + f'] (d={self.dilations}) | 选择熵 {ent.item():.3f}/'
                      f'{math.log(len(self.dilations)):.3f} nats（≈上限=没在选）'
                      f' | ‖resid‖/‖x‖ = {ratio.item():.4f}', flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 护栏（打印一次）：3×3 空洞卷积要求 2d+1 ≤ 边长，否则该支的非中心抽头恒落在零
        # padding 上 ⇒ 静默退化成 1×1（N2 在 stride-32 上踩过的坑）。patch 256 时
        # stage2/3 的特征只有 16×16 / 8×8，所以它们默认用更小的 dilations。
        if not self._warned:
            self._warned = True
            h, w = x.shape[-2:]
            frac = [max(0, min(h, w) - 2 * d) / min(h, w) for d in self.dilations]
            dead = [d for d, f in zip(self.dilations, frac) if f <= 0]
            # 软告警档（<0.3）：patch 256 的 stage3 只有 8×8，d=3 的有效位置比例仅 0.25
            # ——没死但绝大多数位置的外圈抽头压在 padding 上，那一支的读数要打折看。
            weak = [d for d, f in zip(self.dilations, frac) if 0 < f < 0.3]
            msg = (f'[SKAdapter {self.tag}] 特征图 {h}×{w}，dilations={self.dilations}，'
                   f'各支有效位置比例 {[round(f, 2) for f in frac]}')
            if dead:
                msg += (f' | ⚠️ dilation {dead} 的非中心抽头恒落在 padding 内（梯度为 0、'
                        '该支退化成 1×1）—— 改小该 stage 的 segformer_sk_dilations！')
            elif weak:
                msg += (f' | ⚠️ dilation {weak} 的有效位置比例 <0.3（未死但大部分位置靠 '
                        'padding 支撑，该支的选择权重读数需打折）')
            print(msg, flush=True)
        # 选择权重在 fp32 里做 softmax（autocast 下 sel 的 logits 是 fp16，softmax 的
        # 归一化在小幅值上更敏感），再转回 x 的 dtype 与分支输出相乘。
        sel = torch.softmax(self.sel(x).float(), dim=1)
        y = None
        for i, branch in enumerate(self.branches):
            t = sel[:, i:i + 1].to(x.dtype) * branch(x)
            y = t if y is None else y + t
        resid = self.proj(y)
        # 计数与观测**只在训练前向**：验证一次要跑几万个 tile，若把 val 也计进来，几何
        # 间隔的 1/100/1000/10000 会全被第一次验证吃掉、训练期再也打不出读数（纪律与
        # 下方 _acc 只在 training 累加一致）。
        if self.training:
            self._calls += 1
            do_print = (self._calls in (1, 100, 1000, 10000)
                        or self._calls % 20000 == 0)
            self._observe(x, resid, sel, do_print)
        return x + resid


class _ScaleRegHead(nn.Module):
    """U2（2026-07-31）：逐像素 log10-尺寸回归的辅助头（1×1 → 64 → ReLU → 1×1 → 1）。

    "浮冰尺寸分割 = 逐像素的空间尺度推断"这条故事线里，本模块是最字面的一步：直接逼网络
    回答"这里的特征对应多大的浮冰"。目标 = 该 cell 内冰类的 log10(几何均值尺寸 m)，
    水与 255 不监督（构造见 quickstart.scale_reg_aux_loss）。

    - **搭在 encoder stride-4 特征上**（不是 head 的 concat）：更贴"塑造 encoder 表征"的
      叙事，且与 head 内的门控/vario 零耦合（E1 已证 head 内机制互抢）。
    - **不做零初始化**：它不在主通路的残差上，只喂辅助损失；主 logits 与基线逐位一致靠
      "仅当 option 开启才构造 + 构造顺序排在所有模块之后"（RNG 流不被平移）。
    - 只在 self.training 时被调用（见 SegFormer.forward），eval 前向与关闭时逐位一致 ⇒
      evaluate_scenes / inference / 按需读 + tile 门控全部零改动。
    """

    def __init__(self, in_ch: int, mid: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _GradReverse(torch.autograd.Function):
    """梯度反转层（GRL, Ganin & Lempitsky 2015）：前向恒等、反向把梯度取负并乘 λ。

    这是 DANN 的全部技巧所在：判别器照常**最小化**自己的分类损失（学着认出"这是哪一景"），
    但梯度回传到 encoder 时被翻转 ⇒ encoder 反而在**最大化**判别器的损失，也就是被逼着
    把特征里的场景身份（定标 / 风格指纹）抹掉。因此总损失里判别器项按 **1.0** 权重相加，
    强弱完全由 λ 控制（见 `SegFormer.dann_lambda`），这与 SupCon/ScaleReg 那种"λ 乘在
    loss 上"的写法**不同**，不要混。

    λ 存在 ctx 上而不是 buffer 里：它逐 epoch 变（warmup ramp），且不该进 state_dict。
    """

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)          # 恒等，但返回新 view 以免 autograd 把它当 inplace

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None      # 第二个输入 lambd 无梯度


class _DANNHead(nn.Module):
    """场景判别器：GAP → Linear(ch, 256) → ReLU → Linear(256, n_domains)。

    吃的是 encoder stride-4 特征（features[0]，b2 是 64ch），全局平均池化成一个 patch 级
    向量后做 n_domains 路分类（n_domains = train 景数 375，即"这个 patch 来自哪一景"）。

    · **GAP 是刻意的**：要抹掉的是场景级的全局性质（定标/亮度/风格），不是逐像素结构；
      逐像素判别器会把局部纹理也拖进对抗，风险大得多。
    · 只在训练态被调用，推理时整个头闲置（+~0.11M 参数躺在 checkpoint 里）。
    · **不做零初始化**：它不在主通路的残差上，只喂辅助损失；主 logits 与基线逐位一致靠
      "仅当 option 开启才构造 + 构造顺序排在所有模块之后"（RNG 流不被平移）。
    """

    def __init__(self, in_ch: int, n_domains: int, mid: int = 256):
        super().__init__()
        assert n_domains >= 2, f'n_domains 必须 ≥2，得到 {n_domains}'
        self.n_domains = int(n_domains)
        self.net = nn.Sequential(
            nn.Linear(in_ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, n_domains),
        )

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # GAP 用 fp32 累加（dtype= 而不是 .float()：不实体化整张 fp16→fp32 副本）。
        # autocast 下 f 是 fp16，64×64=4096 个元素在 fp16 里累加会掉精度；下游 Linear
        # 仍由 autocast 拉回 fp16，显存口径不变。
        return self.net(f.mean(dim=(2, 3), dtype=torch.float32))   # [B,C,h,w] → [B,n_domains]


class SegFormer(nn.Module):
    """SegFormer: MiT encoder (via SMP, default mit_b2) + SegformerHead.

    Requires: pip install segmentation-models-pytorch
    """

    def __init__(self, options: dict):
        super().__init__()
        try:
            import segmentation_models_pytorch as smp
        except ImportError as e:
            raise ImportError(
                "segmentation_models_pytorch not found. "
                "Run: pip install segmentation-models-pytorch"
            ) from e

        in_ch = len(options['train_variables'])
        if options.get('month_encoding', False):
            in_ch += 2
        if options.get('pol_ratio_channel', False):
            in_ch += 1
        if options.get('texture_channels', False):
            from floe.loaders import N_TEXTURE_CHANNELS  # 懒导入避免顶层循环依赖
            in_ch += N_TEXTURE_CHANNELS  # B1：3 多尺度局部std + 1 边界密度

        self.charts = options['charts']
        self.n_classes = options['n_classes']

        # Build a temporary SMP FPN model to extract the MiT encoder.
        # The encoder handles the full ViT backbone; we replace the FPN decoder
        # with our own SegformerHead.
        # segformer_encoder_name: mit_b0~mit_b5（默认 mit_b2，历史 run 全是 b2）。
        #   b1~b5 的 stage 通道数相同 [64,128,320,512]，只差深度/宽度内部结构，
        #   head 经 encoder.out_channels 自动适配；b0 是 [32,64,160,256]，同样自动适配。
        # segformer_encoder_weights: None(从零) 或 'imagenet'（SMP 自动把 3 通道
        # patch embedding 循环平铺到 in_ch 通道，机制同 DL3+·R50·in1k）。
        _tmp = smp.FPN(
            encoder_name=options.get('segformer_encoder_name', 'mit_b2'),
            encoder_weights=options.get('segformer_encoder_weights', None),
            in_channels=in_ch,
            classes=1,
        )
        self.encoder = _tmp.encoder
        del _tmp

        # MiT-B2 outputs 4 transformer stages with these channel sizes.
        # SMP encoder.out_channels typically looks like (in_ch, 64, 128, 320, 512).
        # We take the last 4 entries (all transformer stages).
        feat_channels = list(self.encoder.out_channels[-4:])

        embed_dim = 256
        # PGSG（2026-08-11）：证据条件化门控的开关与通道数。这里只**读取** options
        # （不消耗 RNG），证据模块本体在 __init__ 末尾、dann_head 之后才构造（RNG
        # 纪律同 detail/vario/context/supcon/sk/sreg/dann："新加的排更后"）。
        self._gate_ev_on = bool(options.get('segformer_gate_evidence', False))
        _EV_VARIO_CH, _EV_POL_CH = 64, 32
        ev_ch = (_EV_VARIO_CH + _EV_POL_CH) if self._gate_ev_on else 0
        if self._gate_ev_on:
            assert options.get('segformer_scale_gate', False), \
                'segformer_gate_evidence=True 需要 segformer_scale_gate=True（证据的' \
                '唯一消费者是门控）'
            assert options.get('pol_ratio_channel', False), \
                'segformer_gate_evidence=True 需要 pol_ratio_channel=True（极化比' \
                '先验分支从输入最后一个通道取 pol ratio）'
            assert not options.get('texture_channels', False), \
                'segformer_gate_evidence 与 texture_channels 不兼容：后者会追加在 ' \
                'pol_ratio 之后，"pol 是最后一个通道"的取法失效'
            # 正向自检（code-review 🟢4）：把预期输入通道数存下来，forward 里逐次
            # 断言——将来若有人加第四种派生通道，"pol=最后一个通道"的约定会静默
            # 错位（B1 时代 8→12 通道错位同族），这一行让它变成立即失败。
            self._gate_ev_in_ch = in_ch
        self.heads = nn.ModuleDict({
            chart: _SegformerHead(
                feat_channels, embed_dim, self.n_classes[chart],
                scale_gate=options.get('segformer_scale_gate', False),
                gate_mode=options.get('segformer_gate_mode', 'tanh'),
                gate_groups=options.get('segformer_gate_groups', 1),
                gate_temp=options.get('segformer_gate_temp', 1.0),
                evidence_ch=ev_ch)
            for chart in self.charts
        })
        # N3 有序头：替换掉 head 里的 cls（其余结构不动）。不是零初始化等价，见类 docstring。
        if options.get('segformer_ordinal_head', False):
            for chart in self.charts:
                self.heads[chart].cls = _OrdinalHead(embed_dim, self.n_classes[chart])

        # E2 纹理旁路（默认关；关闭时模块树与历史版本逐位一致）。
        # 必须在 encoder/heads 之后【最后】构造：这样开关不改变 encoder/heads
        # 的初始化 RNG 流，同 seed 下开/关两个模型的共有权重逐位相同，
        # smoke_e1e2.py 依赖该性质做"初始严格等价"断言。
        if options.get('segformer_detail_branch', False):
            self.detail_branch = _DetailBranch(in_ch, embed_dim)
        else:
            self.detail_branch = None

        # M1 对比度不变二阶纹理模块（默认关；同样必须在 encoder/heads 之后构造，
        # 使开/关两个模型的共有权重在同 seed 下逐位相同 → smoke_m1_vario.py 的
        # "初始严格等价"断言成立）。
        # N1（2026-07-27）：vario_stage 可以是 int 也可以是 list —— 多尺度时每个 stage 各建
        # 一个独立模块。stride 4 的 1 格 = 320 m、stride 8 的 1 格 = 640 m，同一组 radii 在
        # 两个 stage 上覆盖的物理 lag 带整体差 2 倍（320m~5.1km / 640m~10.2km），且两个
        # stage 的特征语义不同 ⇒ 是真·多尺度，不是同一件事做两遍。
        vs = options.get('segformer_vario_stage', 0)
        assert isinstance(vs, (int, list, tuple)) and not isinstance(vs, bool), \
            f'segformer_vario_stage 必须是 int 或 list/tuple，收到 {type(vs).__name__}: {vs!r}'
        self.vario_stages = ([int(vs)] if isinstance(vs, int)
                             else [int(v) for v in vs])
        # 每个 stage 可以有自己的 radii：stride 4 / 8 的物理格距不同（320m / 640m），且
        # 特征图边长不同（64 / 32），能安全使用的最大 lag 也不同。见 _gamma 的 reflect 说明。
        radii_cfg = options.get('segformer_vario_radii', (1, 2, 4, 8, 16))
        per_stage = (isinstance(radii_cfg, (list, tuple)) and radii_cfg
                     and isinstance(radii_cfg[0], (list, tuple)))
        if per_stage:
            assert len(radii_cfg) == len(self.vario_stages), \
                f'per-stage radii 有 {len(radii_cfg)} 组，但 vario_stage 有 ' \
                f'{len(self.vario_stages)} 个'
            radii_list = [tuple(r) for r in radii_cfg]
        else:
            radii_list = [tuple(radii_cfg)] * len(self.vario_stages)
        if options.get('segformer_vario_texture', False):
            for s in self.vario_stages:
                assert 0 <= s < len(feat_channels), \
                    f'segformer_vario_stage 越界：{s} 不在 [0,{len(feat_channels)})'
            assert len(set(self.vario_stages)) == len(self.vario_stages), \
                'segformer_vario_stage 有重复项'
            # out_ch = embed_dim × stage 数：残差加在 head 的 **concat 之上**（fuse 之前），
            # 使模块贡献经 fuse 的 conv+BN+ReLU 与 encoder 特征非线性混合，而不是只能
            # 加性平移 logits。零初始化下仍与基线逐位等价。
            self.vario_texture = nn.ModuleList([
                _VarioTexture(
                    feat_channels[s], embed_dim * len(feat_channels),
                    radii=r,
                    n_probe=options.get('segformer_vario_probes', 16),
                    win=options.get('segformer_vario_win', 11),
                ) for s, r in zip(self.vario_stages, radii_list)
            ])
        else:
            self.vario_texture = None

        # N2 上下文聚合模块：挂在最粗的 stage（stride 32，感受野最大、token 最少）。
        # 与 vario 共用 extra_pre 注入口（加在 head 的 concat 上、fuse 之前），同为单零
        # 初始化 ⇒ 初始与基线逐位等价。
        # ⚠️ 默认挂 stride-8（stage index 1，32×32）而不是最粗的 stride-32（8×8）——
        # 后者上 ASPP 会静默退化成 1×1，理由见 _ContextAggregator 的 docstring。
        if options.get('segformer_context_module', False):
            self.context_stage = int(options.get('segformer_context_stage', 1))
            assert 0 <= self.context_stage < len(feat_channels), \
                f'segformer_context_stage 越界：{self.context_stage}'
            self.context_module = _ContextAggregator(
                feat_channels[self.context_stage], embed_dim * len(feat_channels),
                dilations=options.get('segformer_context_dilations', (1, 2, 4, 8)))
        else:
            self.context_module = None
            self.context_stage = -1

        # N6（2026-07-28）细尺度归一化纹理：把 M1 的可微变异函数搬到**原始输入分辨率**，
        # lag 2~8 px = 160~640 m —— 这正是【中浮冰 100-500m】的粒度带，而至今没有任何一枪
        # 覆盖过它（M1 最小 lag 320m、N1 还往粗走；MiT 的 7×7/s4 stem 从第一层起就把单个
        # 中浮冰抹平了）。中浮冰是十一枪里唯一从未超过 .211 的类。
        # ⚠️ 它离两个已证伪的东西很近，但失败原因在这里不成立：
        #   · E2 detail branch 是从原始输入堆卷积、最强的 1-px 信号是斑点；这里喂的是
        #     **跨 lag 归一化的剖面**，而"归一化"正是 M1 与失败的 B1 之间唯一的实质差别。
        #   · 最小 lag 取 2 px 而非 1 px，避开纯斑点 nugget；且 γ 是窗口内统计量，斑点在
        #     窗口平均里本就被压掉。
        # 开销控制：n_probe=8、win=7（560m 支撑）、算完剖面 avg_pool 到 stride-4 再进
        # mix/proj，所以只有 γ 那几步在全分辨率上跑。
        if options.get('segformer_fine_vario', False):
            self.fine_vario = _VarioTexture(
                in_ch, embed_dim * len(feat_channels),
                radii=options.get('segformer_fine_vario_radii', (2, 3, 4, 6, 8)),
                n_probe=options.get('segformer_fine_vario_probes', 8),
                win=options.get('segformer_fine_vario_win', 7),
                out_pool=4)
        else:
            self.fine_vario = None

        # F3（2026-07-28）跨景区域级 SupCon 的投影头：吃 encoder stride-4 特征
        # （features[0]，b2 是 64ch），两层 1×1 conv 投到对比空间。**训练 only**——
        # forward 只在 self.training 时输出 '_supcon' 键，eval 前向与关闭时逐位一致。
        # 刻意不进 head（避开 E1"head 内机制不可叠加"的枪口）、不做零初始化（它不是
        # 残差、只喂辅助损失，随机初始化即可）。
        # 必须在所有其它模块之后【最后】构造（RNG 纪律同 detail/vario/gate）：开关
        # 不改变共有权重的初始化流，同 seed 下开/关两模型共有权重逐位相同。
        if options.get('segformer_supcon', False):
            _c0 = feat_channels[0]
            self.supcon_proj = nn.Sequential(
                nn.Conv2d(_c0, 128, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 64, 1),
            )
        else:
            self.supcon_proj = None

        # U1（2026-07-31）encoder 内 SK 适配器。**必须排在以上全部模块之后构造**（RNG
        # 纪律同 detail/vario/context/supcon）：Conv2d.reset_parameters 消耗全局 RNG，
        # 建在前面会平移后续所有权重的初始化流 ⇒ 与基线的共有权重不再逐位相同 ⇒ 相当于
        # 同时换了 seed（07-20 实测纯 seed 噪声总分 0.033，足以淹没 <0.05 的效应）。
        # ⚠️ 注册为**顶层** self.sk_adapters（不落 'encoder.' 命名空间）：
        # functions._optimizer_param_groups 按名字前缀分组，落进 encoder 组会拿 1e-5 的
        # LR、零初始化模块长不起来 = 假阴。
        if options.get('segformer_sk_adapter', False):
            sk_stages = [int(s) for s in options.get('segformer_sk_stages', [0, 1, 2, 3])]
            for s in sk_stages:
                assert 0 <= s < len(feat_channels), \
                    f'segformer_sk_stages 越界：{s} 不在 [0,{len(feat_channels)})'
            assert len(set(sk_stages)) == len(sk_stages), 'segformer_sk_stages 有重复项'
            # 默认 per-stage dilations：patch 256 下 stage2/3 的特征只有 16×16 / 8×8，
            # d=4 的 3×3 抽头（跨度 9）在 8×8 上恒落进 padding ⇒ 粗 stage 用 (1,2,3)。
            _dil_default = {0: (1, 2, 4), 1: (1, 2, 4), 2: (1, 2, 3), 3: (1, 2, 3)}
            dil_cfg = options.get('segformer_sk_dilations', None)
            assert dil_cfg is None or len(dil_cfg) > 0, 'segformer_sk_dilations 不能为空'
            if dil_cfg is None:
                dil_list = [_dil_default[s] for s in sk_stages]
            elif isinstance(dil_cfg[0], (list, tuple)):   # 嵌套 = 逐 stage
                assert len(dil_cfg) == len(sk_stages), \
                    f'per-stage dilations 有 {len(dil_cfg)} 组，但 sk_stages 有 {len(sk_stages)} 个'
                dil_list = [tuple(d) for d in dil_cfg]
            else:                                          # 单组 = 全 stage 同用
                dil_list = [tuple(dil_cfg)] * len(sk_stages)
            # `_encoder_forward_sk` 复刻的是 smp **0.4/0.5 系** 的 MixVisionTransformer
            # 分段前向。这里只做**存在性**校验（属性名 + 深度），它**证明不了调用约定
            # 一致**——smp ≤0.3.x 的 patch_embed 返回 (x,H,W) 三元组、block 是 ModuleList
            # 且要传 H/W，属性名却完全相同。⚠️ 不要把这几行读成"API 已验证"：
            # **新旧 API 的唯一哨兵是 smoke_u1_sk.py 的 [C] 对拍**（旁路适配器后与
            # self.encoder(x) 逐位相同），开卡前必须先过它。
            for i in range(1, 5):
                for attr in (f'patch_embed{i}', f'block{i}', f'norm{i}'):
                    if not hasattr(self.encoder, attr):
                        raise RuntimeError(
                            f'segformer_sk_adapter=True 需要显式分段前向，但 encoder 缺少 '
                            f'属性 {attr!r}（当前 encoder 类型 '
                            f'{type(self.encoder).__name__}）—— 不是 smp 的 MiT encoder，'
                            '或 smp 版本结构已变，需更新 _encoder_forward_sk。')
            _depth = getattr(self.encoder, '_depth', 5)
            if _depth != 5:
                raise RuntimeError(
                    f'segformer_sk_adapter=True 假定 encoder._depth==5（返回 '
                    f'[input, dummy] + 4 个 stage），实际 {_depth}。')
            self.sk_adapters = nn.ModuleDict({
                str(s): _SKAdapter(feat_channels[s], d, tag=f's{s}')
                for s, d in zip(sk_stages, dil_list)
            })
        else:
            self.sk_adapters = None

        # U2 逐像素 log10-尺寸回归辅助头（默认关）。排在 sk_adapters 之后 —— "新加的排
        # 更后"，使已有 config 的 RNG 流一律不受影响。训练态才被调用，见 forward。
        if options.get('segformer_scale_reg', False):
            self.scale_reg_head = _ScaleRegHead(feat_channels[0])
        else:
            self.scale_reg_head = None

        # V2（2026-08-05）DANN 场景判别器。**排在以上全部模块之后构造** —— "新加的排更
        # 后"这条 RNG 纪律同 detail/vario/context/supcon/sk/sreg：Linear/Conv 的
        # reset_parameters 消耗全局 RNG，建在前面会平移后续所有权重的初始化流 ⇒ 与基线的
        # 共有权重不再逐位相同 ⇒ 相当于同时换了 seed。
        # `segformer_dann_domains` **必填**（= train 景数）：给默认值等于允许"判别器类数
        # 与实际域数不一致"静默跑起来，那样域 acc 曲线没有任何可解释性。quickstart 里
        # 还会再与 len(train_list) 断言一次。
        if options.get('segformer_dann', False):
            if 'segformer_dann_domains' not in options:
                raise ValueError(
                    'segformer_dann=True 必须显式给出 segformer_dann_domains（= train '
                    '景数，417 数据集是 375）—— 它决定判别器输出维度与域 acc 的基线 1/N，'
                    '不设默认值。')
            self.dann_head = _DANNHead(feat_channels[0],
                                       int(options['segformer_dann_domains']))
            # 普通 float 属性（不是 buffer/parameter ⇒ 不进 state_dict）：quickstart 每
            # epoch 按 warmup ramp 覆写它。初值 0 ⇒ 即使调用方忘了设，encoder 也只是
            # 收不到对抗梯度（判别器自己照常学），不会静默施加一个错的压力。
            self.dann_lambda = 0.0
        else:
            self.dann_head = None

        # PGSG（2026-08-11）证据模块。**排在以上全部模块之后构造**（RNG 纪律："新加的
        # 排更后"，保证与基线/历史 config 的共有权重同 seed 下逐位相同）。
        #   gate_ev_vario：对比度不变的变异函数剖面证据（复用 _VarioTexture，radii
        #     1,2,4,8,16 @stride-4 = 320m~5.12km 覆盖 small→vast 的定义域），
        #     **zero_proj=False**——它不是残差，是门控的条件输入（见 _SegformerHead）。
        #   gate_ev_pol：极化比先验分支（PA2-Swin 遗产），从输入最后一个通道取
        #     pol ratio、池化到 stride-4 编码成散射机制指纹。
        # 两者输出在 forward 里 concat 成 [B,96,H/4,W/4] 传给 head 的 gate_evidence。
        # 优化器分组：名字不含 'encoder.' → rest 组拿满 LR 1e-4（随机初始化，正确）。
        if self._gate_ev_on:
            self.gate_ev_vario = _VarioTexture(
                feat_channels[0], _EV_VARIO_CH,
                radii=options.get('segformer_gate_ev_radii', (1, 2, 4, 8, 16)),
                n_probe=options.get('segformer_gate_ev_probes', 16),
                win=options.get('segformer_gate_ev_win', 11),
                zero_proj=False)
            self.gate_ev_pol = _PolPrior(out_ch=_EV_POL_CH, pool=4)
        else:
            self.gate_ev_vario = None
            self.gate_ev_pol = None

        # PGSG-v3 方案二：尺度轴 transformer。**在 gate_ev 之后、此处才构造**再赋值到
        # head 的占位符上（RNG 纪律，理由见 _SegformerHead.__init__ 的占位注释；
        # state_dict 前缀仍是 heads.<chart>.scale_xattn.*，checkpoint 口径不变）。
        if options.get('segformer_scale_xattn', False):
            for chart in self.charts:
                self.heads[chart].scale_xattn = _ScaleXAttn(
                    embed_dim, len(feat_channels), evidence_ch=ev_ch)

        # PGSG-v3 方案一：物理偏置注意力 PBA。**排在全部模块之后【最后】构造**
        # （RNG 纪律同上；nn.Linear 构造消耗 RNG，零初始化本身不消耗）。
        # pol_proj 注册在顶层 self.pba_projs：① 不落 'encoder.' 命名空间 ⇒
        # functions._optimizer_param_groups 归入 rest 组拿满 LR 1e-4（零初始化模块
        # 落进 encoder 组会拿 1e-5、长不起来=假阴，U1 的教训）；② monkey-patch
        # 不改 encoder 的 state_dict 键 ⇒ 历史 checkpoint 兼容。
        if options.get('segformer_pba', False):
            assert options.get('pol_ratio_channel', False), \
                'segformer_pba=True 需要 pol_ratio_channel=True（偏置从输入最后一个' \
                '通道取 pol ratio）'
            assert not options.get('texture_channels', False), \
                'segformer_pba 与 texture_channels 不兼容（pol 必须是最后一个通道）'
            self._pba_in_ch = in_ch
            pba_stages = [int(s) for s in
                          options.get('segformer_pba_stages', [0, 1, 2, 3])]
            assert len(set(pba_stages)) == len(pba_stages), 'pba_stages 有重复项'
            for s in pba_stages:
                assert 0 <= s <= 3, f'segformer_pba_stages 越界：{s}'
            # 结构存在性校验（口径同 SK adapter：只证明属性在，调用约定的唯一哨兵
            # 是 smoke_pgsg3.py [E] 的零偏置逐位对拍）
            for s in pba_stages:
                if not hasattr(self.encoder, f'block{s + 1}'):
                    raise RuntimeError(
                        f'segformer_pba 需要 encoder.block{s + 1}（smp MiT 结构），'
                        f'当前 encoder 类型 {type(self.encoder).__name__} 没有——'
                        'smp 版本结构已变，需更新 _pba_patch_attention。')
            self._pba_holder = {}
            _projs = []
            # EMA 与 monkey-patch 不兼容（code-review 🟡5）：copy.deepcopy 复制绑定方法
            # 时只深拷贝 __self__，闭包里的 proj/holder 仍指向原模型 ⇒ EMA 副本的 PBA
            # 会静默失效（读不到 pol 或用错权重）、验证/best 全走坏的副本、整个 run
            # 白跑且无报错。显式互斥，别靠"当前 config 没开 ema"。
            assert not options.get('ema', False), \
                'segformer_pba 与 ema 同开未适配：deepcopy 不会跟走 monkey-patch 的闭包'
            for s in pba_stages:
                blocks = getattr(self.encoder, f'block{s + 1}')
                for bi, blk in enumerate(blocks):
                    proj = nn.Linear(3, blk.attn.num_heads)
                    nn.init.zeros_(proj.weight)
                    nn.init.zeros_(proj.bias)
                    _projs.append(proj)
                    # 每个 stage 只让第一个 block 打监控（否则 b2 的 3+4+6+3=16 个
                    # attention 全打印，日志淹没）
                    _pba_patch_attention(blk.attn, proj, self._pba_holder,
                                         tag=f's{s}b{bi}', monitor=(bi == 0))
            self.pba_projs = nn.ModuleList(_projs)
        else:
            self.pba_projs = None
            self._pba_holder = None

    def _encoder_forward_sk(self, x: torch.Tensor) -> list:
        """显式复刻 smp **0.4/0.5 系** `MixVisionTransformerEncoder.forward`，在每个
        stage 的 NCHW 输出上插入 `_SKAdapter`。**适配后的特征既进 features 也进下一个
        stage** —— 这就是"真编码端改动"的定义（贴在 encoder 输出上只影响 head 输入）。

        与 smp 源的逐行对应（`encoders/mix_transformer.py`，v0.5.0）：
          `MixVisionTransformer.forward_features` 的每个 stage：
              x = self.patch_embed{i}(x)     # 返回**单个 NCHW 张量**（不是 (x,H,W)）
              x = self.block{i}(x)           # block{i} 是 nn.Sequential，整体调用
              x = self.norm{i}(x)            # 自定义 LayerNorm，自己处理 4D
              x = x.contiguous()
              outs.append(x)                 ← 适配器插在这一步之前
          `MixVisionTransformerEncoder.forward`：
              return [x, dummy] + self.forward_features(x)[: self._depth - 1]
              （dummy = torch.empty([B, 0, H//2, W//2])，_depth=5 ⇒ 取满 4 个 stage）
        ⚠️ smp **≤0.3.x 是另一套调用约定**（patch_embed 返回 (x,H,W)、block 是
        ModuleList 且 blk(x,H,W)、norm 后要 reshape/permute），而属性名完全相同 ⇒
        `__init__` 里的 hasattr 校验**分辨不出新旧**。**唯一的版本漂移哨兵是
        smoke_u1_sk.py 的 [C]**（把适配器旁路成 Identity 后与 `self.encoder(x)` 逐位
        对拍），开卡前必须先过。
        返回值与 `self.encoder(x)` 同构（SegFormer.forward 只用 `[-4:]`，但保持同构以防
        未来有别的消费者）。
        """
        enc = self.encoder
        feats = []
        h = x
        for i in range(1, 5):
            h = getattr(enc, f'patch_embed{i}')(h)
            h = getattr(enc, f'block{i}')(h)
            h = getattr(enc, f'norm{i}')(h).contiguous()
            key = str(i - 1)
            if key in self.sk_adapters:
                h = self.sk_adapters[key](h)
            feats.append(h)
        dummy = torch.empty([x.shape[0], 0, x.shape[2] // 2, x.shape[3] // 2],
                            dtype=x.dtype, device=x.device)
        return [x, dummy] + feats

    def pop_sk_stats(self):
        """取走并清零 U1 的 epoch 级读数（quickstart 每 epoch 调一次落 wandb 曲线）。
        关闭 SK 时返回 None；本 epoch 没有训练前向时返回空 dict。"""
        if self.sk_adapters is None:
            return None
        stats = {}
        for key, adapter in self.sk_adapters.items():
            acc = adapter._acc
            adapter._acc = None
            if acc is None or acc['n'] == 0:
                continue
            n = float(acc['n'])
            # 这里才 .item()/.tolist()（每 epoch 一次同步，而不是每 batch 一次）
            for i, v in enumerate((acc['w'] / n).tolist()):
                stats[f'SK s{key} branch{i} mean'] = float(v)
            stats[f'SK s{key} entropy'] = (acc['ent'] / n).item()
            stats[f'SK s{key} resid ratio'] = (acc['ratio'] / n).item()
        return stats

    def forward(self, x: torch.Tensor) -> dict:
        detail = self.detail_branch(x) if self.detail_branch is not None else None
        # v3 PBA：调 encoder 前把全分辨率 pol 图放进 holder（打过补丁的 Attention
        # 会按各自 stage 的网格池化取用），调完立刻 pop ⇒ 不跨 batch 残留；encoder
        # 被单独调用（无 pol）时补丁自动退化为原始注意力（零偏置路径逐位一致）。
        if self.pba_projs is not None:
            assert x.shape[1] == self._pba_in_ch, \
                f'输入通道数 {x.shape[1]} ≠ 构造时的 {self._pba_in_ch}——' \
                '"pol=最后一个通道"的前提失效（新加了派生通道？）'
            self._pba_holder['pol'] = x[:, -1:, :, :]
        try:
            # U1：开 SK 时走显式分段前向（适配器插在 stage 之间）；关闭时调用路径与
            # 历史版本完全相同（同一个 self.encoder(x)）。
            all_feats = (self._encoder_forward_sk(x) if self.sk_adapters is not None
                         else self.encoder(x))   # tuple/list of feature maps
        finally:
            if self.pba_projs is not None:
                self._pba_holder.pop('pol', None)
        # Use the last 4 features (MiT-B2 stages 1–4)
        features = list(all_feats[-4:])
        # M1/N1：在指定 stage(s) 的特征上算归一化变异函数剖面。注入点与 E2 不同——它走
        # extra_pre（fuse 之前、加在 concat 上），见 _SegformerHead.forward 的说明。
        # 多 stage 时先各自上采样到 features[0] 的分辨率再相加（各路都是零初始化投影，
        # 相加后整体初始仍为 0、与基线逐位等价）。
        vario = None
        if self.vario_texture is not None:
            tgt = features[0].shape[-2:]
            for mod, s in zip(self.vario_texture, self.vario_stages):
                v = mod(features[s])
                if v.shape[-2:] != tgt:
                    v = F.interpolate(v, size=tgt, mode='bilinear', align_corners=False)
                vario = v if vario is None else vario + v
        # N6：细尺度模块吃的是**原始输入**（不是 encoder 特征），走同一个 extra_pre 注入口
        if self.fine_vario is not None:
            fv = self.fine_vario(x)
            tgt = features[0].shape[-2:]
            if fv.shape[-2:] != tgt:
                fv = F.interpolate(fv, size=tgt, mode='bilinear', align_corners=False)
            vario = fv if vario is None else vario + fv.to(vario.dtype)
        # N2：同一个 extra_pre 注入口（两者都是零初始化投影，直接相加）
        if self.context_module is not None:
            c = self.context_module(features[self.context_stage])
            tgt = features[0].shape[-2:]
            if c.shape[-2:] != tgt:
                c = F.interpolate(c, size=tgt, mode='bilinear', align_corners=False)
            vario = c if vario is None else vario + c.to(vario.dtype)
        # PGSG：证据条件化门控的两路证据。vario 证据在 encoder stride-4 特征上算
        # （profile 内部自带 fp32，mix/proj 留在 autocast 里）；pol 先验从输入最后一个
        # 通道取（__init__ 已断言 pol_ratio_channel=True 且 texture_channels=False ⇒
        # pol ratio 恒为最后一个通道，与 loaders._augment_x / PA2 的取法一致）。
        gate_ev = None
        if self.gate_ev_vario is not None:
            assert x.shape[1] == self._gate_ev_in_ch, \
                f'输入通道数 {x.shape[1]} ≠ 构造时的 {self._gate_ev_in_ch}——' \
                '"pol=最后一个通道"的前提失效（新加了派生通道？）'
            ev_v = self.gate_ev_vario(features[0])
            ev_p = self.gate_ev_pol(x[:, -1:, :, :])
            if ev_p.shape[-2:] != ev_v.shape[-2:]:
                ev_p = F.interpolate(ev_p, size=ev_v.shape[-2:],
                                     mode='bilinear', align_corners=False)
            gate_ev = torch.cat([ev_v, ev_p.to(ev_v.dtype)], dim=1)
        out = {}
        for chart in self.charts:
            logits = self.heads[chart](features, extra=detail, extra_pre=vario,
                                       gate_evidence=gate_ev)
            if logits.shape[-2:] != x.shape[-2:]:
                logits = F.interpolate(logits, size=x.shape[-2:],
                                       mode='bilinear', align_corners=False)
            out[chart] = logits
        # F3：训练时附带 SupCon 投影特征（stride-4）。eval 模式不加键 ⇒ 验证/推理路径
        # （fast_tiled_val_predict / slide_inference / evaluate_scenes / inference.py）
        # 拿到的输出 dict 与基线完全一致，零改动。
        if self.supcon_proj is not None and self.training:
            out['_supcon'] = self.supcon_proj(features[0])
        # U2：同款训练 only 契约（'_scale_reg' 只在训练态出现）。features[0] 是 encoder
        # 的 stride-4 特征（与 supcon_proj 同一个 tap 点）；目标构造与损失见
        # quickstart.scale_reg_aux_loss。
        if self.scale_reg_head is not None and self.training:
            out['_scale_reg'] = self.scale_reg_head(features[0])
        # V2 DANN：同款训练 only 契约（'_dann' 只在训练态出现）。GRL 夹在 encoder 特征与
        # 判别器之间 ⇒ 判别器正常下降、encoder 收到反向的梯度。λ 由 quickstart 每 epoch
        # 写进 self.dann_lambda（warmup ramp）。
        if self.dann_head is not None and self.training:
            out['_dann'] = self.dann_head(
                _GradReverse.apply(features[0], self.dann_lambda))
        return out
