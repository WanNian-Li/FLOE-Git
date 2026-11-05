# PGSG：Physics-Guided Scale Gating —— 论文故事与组合模型设计（2026-08-11）

用户定向：以当前最佳模型为起点讲一个论文故事，围绕故事改模型；基线 = cv5 fold0；
从 PA2_Swin_Improved.py 的极化比思路取灵感；必须包含 sgate2；模块合体一次性训练。

## 一、论文故事（The Story）

**标题候选**：*Scale is a Decision, not a Feature: Physics-Guided Per-Pixel Scale
Gating for Sea-Ice Floe Size Segmentation from Sentinel-1 SAR*
（中文骨架：浮冰尺寸分割 = 逐像素的空间尺度推断；证据要物理接地，决策要逐像素做）

### 1. 问题的本质（§Intro）
浮冰尺寸类别由纯物理尺度定义（cake<20m / small 20-100m / medium 100-500m /
big 0.5-2km / vast 2-10km）。在 80m 分辨率的 EW SAR 里，small/medium 的单体浮冰只有
1~6 像素——**类别标签是关于"纹理空间尺度"的陈述，不是关于局部亮度的陈述**。
通用语义分割模型把尺度当作隐式特征学习，这在浮冰任务上有两个结构性错配：

- **尺度证据被辐射条件混淆**（本项目探针实测）：景间 HH 均值 std≈3dB；+2dB 使
  hard-ice 掉 0.06；暗景把 medium 系统性误判为 small（08-04 逐景分解：3 个 medium
  崩溃景全在暗侧 2dB 外）；±0.7 z-unit 的辐射平移可把单景 medium IoU 在 0.01↔0.99
  之间搬动（响应面逐景各异、非单调）。
- **跨景泛化失败而非容量不足**（U 批第 0 步诊断）：中浮冰 train IoU≈0.7 / val
  0.15-0.22（gap +0.5），20+ 枪结构魔改全平——"加容量/加结构"不解决问题。

### 2. 关键洞察（§Motivation，全部有本项目实验背书）
1. **对比度不变的二阶统计量跨景迁移**：变异函数（variogram，测"纹理隔多远开始
   不相似"的二阶统计量）剖面对仿射辐射变换不变，跨景 AUC 0.729 > 幅度口径 0.651
   （Stage 0 探针）；铸进网络的 M1 模块是 27 枪里平滑口径最稳的单件（+0.012）。
2. **极化比是免定标的散射机制指纹**（PA2-Swin 遗产）：HH−HV（dB 域之差=线性域之比）
   把绝对定标/入射角的共模分量对消，区分平整冰/形变冰/开阔水的散射机制。PA2 的
   设计哲学：**物理先验应该引导"机制"（注意力偏置），而不是仅作为输入通道**。
   radaug 的通道抖动设计（共模满 B、差分仅 B/3）也从反面确认 pol ratio 是要保护的
   真信号。
3. **逐像素尺度选择有效但"决策者吃的是脏证据"**：sgate2（softmax 竞争式尺度门控，
   Σg≡4 预算守恒）是门控家族稳定 +0.03 hard-ice 的机制。但它现在从 head 的 concat
   特征里做决策——正是被亮度/对比度污染的表征（辐射探针：模型显著依赖绝对辐射）。
4. **温和增益不叠加的教训**：E1（vario 残差 + 门控并联）0.5100 反而最差——两个
   head 内机制**争同一份增益**。C1/F2/H2 三次组合负结果同构。**正确的组合方式不是
   并联相加，而是让一个模块为另一个模块服务**。

### 3. 方法（§Method）：证据-决策解耦的门控
**PGSG = 把两路物理接地、辐射稳健的"尺度证据"作为门控的条件输入，
让尺度决策基于干净证据做出；证据不再与门控并联竞争。**

```
                       ┌────────────────────────────────────────┐
   x (8ch SAR) ──► MiT-B2 encoder ──► 4-stage features ──► proj ──► concat(1024)
        │               │ stride-4 feat                              │
        │               ├─► VarioEvidence(64ch)  ──┐                 ▼
        │               │   （对比度不变剖面+mix）  ├─► evidence ─► ScaleGate
        └─ pol ratio ──►┴─► PolPrior(32ch)       ──┘   (96ch)    softmax, Σg≡4
           (最后一个输入通道, avg_pool→s4)                            │
                                                        g(4尺度) ◄───┘
                        concat × g ─► fuse ─► cls ─► logits
   训练期另有：encoder stride-4 ─► GRL ─► 场景判别器（DANN，推理零开销）
```

四个模块（**一次性合体训练**，遵用户指令不拆单变量）：
- **M-A sgate2**（保留，用户指定）：决策机制本体。
- **M-B VarioEvidence**：_VarioTexture 复用（radii 1,2,4,8,16 @stride-4 =
  320m~5.12km，正好横跨 small→vast 的定义域），但 **proj 不再零初始化、
  输出只进门控**——它是证据不是残差（与 E1 的本质区别）。
- **M-C PolPrior**（PA2 灵感）：pol ratio 通道 4×4 池化到 stride-4，两层 3×3 conv
  编码成 32ch 散射机制指纹，进门控。物理语义：门控在决定"信哪个尺度"时知道自己
  站在什么散射面上（开阔水/平整冰/形变冰的可分辨尺度带不同）。
- **M-D DANN**（保留 V2）：encoder 层面的场景去身份对抗（训练 only、推理零改动），
  与 head 内门控不同层位；此前单跑平滑口径 0.5316=与 M1 并列 campaign 最高档，
  且是 conf-sel-gm 推理增益可叠加的 checkpoint（+0.022 hard-ice）。

**部署行（论文 "ours + test-time adaptation"）**：conf-sel-gm 多操作点推理
（V0 探针已证：在 DANN checkpoint 上近饱和吃掉 oracle 差距的 99%）。

### 4. 为什么这个组合有先验（§的自洽性检查）
- 每个组件单独为正：sgate2 +0.006~0.03、vario +0.012、DANN +0.012（平滑口径）。
- 已知的组合失败模式被结构性避开：证据不走 extra_pre 残差（E1 的坑）、不加输入
  通道（B1 的坑）、不绕过 stem（E2-detail 的坑）、门控保留 softmax 约束（F1-g4 的坑）。
- 机制互补而非重复：证据（表征什么）/ 决策（信哪个尺度）/ 正则（抹掉哪些景特异
  信息）作用在三个不同的自由度上。
- 诚实风险：本项目"温和增益不复合"是 4 次验证的稳定规律，PGSG 的"服务式组合"
  是对这条规律的一次结构性挑战——若失败，其判读读数（见 §6）也能区分
  "门控没用证据"vs"用了证据但无益"，不会是不可判读的空枪。

## 二、实验设计

- **基线锚**：floe_b2_dlr_cv5_f0_s42（cv5 fold0，b2+in1k+dlr+smsel s42；
  CM /root/autodl-tmp/cm/cv5/b2dlr_f0.npz）。fold0 是 5 折中最难的折
  （hard-ice 0.224 ↔ f2 0.333）。
- **本枪**：floe_b2_pgsg_cv5f0_s42，config
  `configs/FLOE/sota_segformer_in1k_dlr_cv5f0_pgsg.py`，唯一 datalist=fold0。
- **口径**：双报 raw@平滑best 与 smoothed@best，跨 run 以 smoothed 为准；
  主指标 macro mIoU + hard-ice=mean(small,medium)。
- **判读门**（预注册）：fold0 上 Δhard-ice ≥ +0.03 且中浮冰不劣于锚 → 提拔、
  补其余 4 折做配对判定（cv5 配对 se≈0.003，可判 0.01 级效应）；
  0 ~ +0.03 → 看读数决定微调方向；负 → 按读数拆解（先撤 DANN，再看门控读数）。
- **跑前就位的读数**（E3 教训：读数不齐的负结果不可判读）：
  1. gate conv1 的 特征列 vs 证据列 RMS 权重比（几何间隔打印）——
     "门控有没有在听证据"的直接观测；
  2. 既有 gate 监控（stage-mean g / 选择熵 / |a|）——与 sgate2 历史值对比；
  3. VarioTexture 首次前向诊断行（γ 分位数 / clamp 命中率 / 跨 lag std）；
  4. DANN Domain Acc / Clip Hit Frac（既有 wandb 曲线）；
  5. 跑后 `_eval_radiometric.py`（表征是否真变了，run 间噪声比 hard-ice 低一个
     量级）+ `_eval_multib.py` conf-sel-gm 可叠加性（预测：若 PGSG 的证据通路
     真的接管了辐射自由度，gm 增益应像 vario 那样变小——这本身就是机制验证）。

## 三、实现要点（详见 models/sota/segformer.py 注释）
- `segformer_gate_evidence=True`：head 的 scale_gate 首层 conv 加宽
  1024→1024+96；末层仍零初始化 ⇒ g≡1 ⇒ **初始与 fold0 基线逐位等价**照旧成立。
- 证据模块（gate_ev_vario / gate_ev_pol）**随机初始化**（不是残差、不受单零纪律
  约束；零初始化反而制造"经由门控末层零权重的梯度迟滞"）。
- RNG 纪律：证据模块在 dann_head 之后【最后】构造；共有权重与基线同 seed 逐位相同。
- 断言：gate_evidence 需要 scale_gate=True、pol_ratio_channel=True、
  texture_channels=False（pol 必须是最后一个通道）。
- DANN domains=334（fold0 train 景数，quickstart 运行时再断言）。
