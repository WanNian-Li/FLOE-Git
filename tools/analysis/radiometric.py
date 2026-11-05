#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""辐射域捷径探针：val 上做 SAR 辐射扰动，看 hard-ice 退化多少（2026-07-26，零训练成本）。

动机（docs/2026-07-26_M1_vario_texture.md §3）
--------------------------------------------
M1 的假设 H1 是"模型的纹理表征被对比度/亮度混淆 ⇒ 跨景不迁移"。但 M1 只是**新增**一条
不变通路、主路仍自由携带幅度 ⇒ **M1 平掉并不能否证 H1**（"模型压根没用这条新通路"同样
解释得通）。本探针直接测 H1 本身，且**不需要训练任何东西**：

  在已有 checkpoint 上给 val 的 SAR 通道做辐射扰动 x → a·x + b，看 hard-ice 退化多少。
    · 退化明显 ⇒ 模型确实吃了绝对辐射水平这个捷径（H1 成立）→ M1/M2/M3 这条线有靶子；
    · 几乎不退化 ⇒ 捷径不存在 ⇒ H1 否掉、整条不变性线可关（省下后续所有 run）。

口径与物理含义
--------------
nersc SAR 通道是 dB 后再全局 z-score 的，所以在这个空间里：
  · **加性偏置 b = 辐射定标/增益漂移**（线性功率乘 g ⇒ dB 加 10log10(g)）；
  · **乘性增益 a = 对比度变化**（dB 动态范围被拉伸/压缩）。
两者正是景与景之间真实存在的差异，也正是 `_VarioTexture` 的归一化剖面在特征域刻意
不变的那两个自由度。b 的单位是 z-score，1 z-unit = 全数据集 dB 标准差（HH 典型 5~8 dB）
⇒ **b=0.15 ≈ 1 dB、b=0.30 ≈ 2 dB**，都在真实景间差异的量级内。给 ±0.15/±0.30 两档是为了
拿到**剂量-响应**曲线：单点退化可以是任何东西，单调的剂量-响应才是证据。

扰动只作用在 ch0(HH)/ch1(HV) 与派生的 pol_ratio 上，且**只在 global_valid_mask==1 的
像元内**（定标影响真实信号，不影响 nodata 填充；否则会在黑边造出人工结构）。
⚠️ **入射角(ch2)、GLCM HH 纹理(ch3-5)、mask(ch6) 不动** ⇒ 本探针给出的是捷径依赖的
**下界**（GLCM 是二阶量、几乎不携带绝对水平，但严格来说模型仍可从它读到一点残余信息）。

推理路径与 quickstart 验证完全一致：`fast_tiled_val_predict`（原生分辨率、与 patch 网格
对齐的不重叠 tile、tile 门控）+ **按需读**。本探针**不改 tile 网格**（与 offset-TTA 那类
探针不同，没有网格错位），所以是 `val_ondemand_ok` 明确允许的用法。所有变体共用同一次
场景解码，7 个变体只多 6 次前向。
⚠️ 第一版用的是 `fast_tiled_val_inference`，实测 **2.5 分钟/景**（42 景 × 2 seed = 3.5 小时）
—— 它要实体化 n_classes×H×W 的整景 logits（最大景 3.7GB）、把整景输入搬上 GPU 再搬回、
且不门控，7 个变体把这些开销乘了 7 倍。闸门比它要守的 3 小时训练还贵，逻辑就颠倒了。
换 `fast_tiled_val_predict` 后逐 batch 就地 argmax、整景留 CPU、跳过 66% 的 tile。
指标从全局 float64 混淆矩阵导出，逐景 CM 按场景名排序 dump（与 evaluate_scenes --dump-cm
同序，可喂 analyze_scene_cm.py 做**同 checkpoint 配对 bootstrap**——这个场景下配对
bootstrap 有效，见 docs/2026-07-20 §5）。

用法（服务器，GPU 空闲时）：
  python -m tools.analysis.radiometric --self-test          # 先过扰动逻辑自检，不碰数据/GPU
  python -m tools.analysis.radiometric \
      --config configs/FLOE/sota_segformer_in1k_dlr_417.py \
      --checkpoint work_dir/floe_b2_dlr_smsel_s42/best_model_floe_b2_dlr_smsel_s42.pth \
      --val-list datalists/val_417.json \
      --dump-cm /root/autodl-tmp/cm/radio/b2_smsel_s42.npz
"""
import argparse
import copy
import json
import os
import os.path as osp
import sys

import numpy as np
import torch
from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeTestDataset,
                     limit_torch_threads, val_ondemand_ok)
from floe.functions import get_model, fast_tiled_val_predict

# (标签, a 乘性增益=对比度, b 加性偏置=定标漂移[z-score 单位])
VARIANTS = [
    ('id',      1.00,  0.00),   # 恒等参照（也是确定性地板）
    ('b+0.15',  1.00,  0.15),   # ≈ +1 dB 定标漂移
    ('b-0.15',  1.00, -0.15),
    ('b+0.30',  1.00,  0.30),   # ≈ +2 dB
    ('b-0.30',  1.00, -0.30),
    ('a0.90',   0.90,  0.00),   # 对比度 −10%
    ('a1.10',   1.10,  0.00),   # 对比度 +10%
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config')
    p.add_argument('--checkpoint')
    p.add_argument('--val-list')
    p.add_argument('--dump-cm', default=None, help='逐景 CM 输出 .npz 前缀（每变体一个文件）')
    p.add_argument('--self-test', action='store_true', help='只跑扰动逻辑自检后退出')
    # 2026-08-04（暗景因果验证）：允许覆盖扰动清单，格式 "tag:a:b,tag:a:b,..."，
    # 如 "id:1:0,b-0.70:1:-0.70"。省略时用模块级 VARIANTS，历史行为逐位不变。
    # 必须包含 id（Δ 的参照）。自检（--self-test）始终用完整 VARIANTS，不受本参数影响。
    p.add_argument('--variants', default=None,
                   help='覆盖扰动清单："tag:a:b" 逗号分隔；须含 id:1:0')
    return p.parse_args()


def parse_variants(spec):
    """除 "tag:a:b" 固定仿射外，认两个逐景自适应特殊 tag（a/b 占位填 0，运行时按
    该景 valid 区统计现算）：
      center —— 逐景居中：b_ch = −mean_valid(ch)，a=1（每景每通道一个常数）
      zscene —— 逐景标准化：a_ch = 1/std_valid(ch)，b_ch = −mean/std
    这是 0b「辐射对齐家族上界」探针（docs/2026-08-04_M2_radaug.md §1）的载体。"""
    out = []
    for item in spec.split(','):
        tag, a, b = item.split(':')
        out.append((tag, float(a), float(b)))
    assert any(t == 'id' and a == 1.0 and b == 0.0 for t, a, b in out), \
        '--variants 必须包含参照项 id:1:0'
    return out


def channel_layout(opts):
    """返回 (hh_idx, hv_idx, mask_idx, pol_idx, n_ch)。pol_idx 为 None 表示无该通道。

    通道顺序由 loaders._augment_x 定死：train_variables → [month×2] → [pol_ratio] → [texture]。
    """
    tv = list(opts['train_variables'])
    assert tv[0] == 'nersc_sar_primary' and tv[1] == 'nersc_sar_secondary', \
        f'前两个通道应为 nersc SAR primary/secondary，实际 {tv[:2]}'
    assert 'global_valid_mask' in tv, '需要 global_valid_mask 来把扰动限制在有效像元内'
    assert not opts.get('texture_channels', False), \
        'texture_channels=True 时纹理通道是从 HH 派生的，本探针未实现其同步重算 —— 拒绝运行'
    n_ch = len(tv) + (2 if opts.get('month_encoding', False) else 0) \
                   + (1 if opts.get('pol_ratio_channel', False) else 0)
    pol_idx = (len(tv) + (2 if opts.get('month_encoding', False) else 0)) \
        if opts.get('pol_ratio_channel', False) else None
    return 0, 1, tv.index('global_valid_mask'), pol_idx, n_ch


def apply_radiometric(x, hh0, hv0, valid, a, b, pol_idx, a2=None, b2=None):
    """就地把 x 的 SAR 通道置为 a·orig + b（仅 valid 处），并同步重算 pol_ratio。

    hh0/hv0 是**原始**通道的备份 ⇒ 每个变体都从原始出发，不会跨变体累积浮点漂移。
    pol_ratio = HH − HV，扰动后为 a·(HH−HV)（偏置 b 在差分里精确消掉），从扰动后的
    两个通道重算即自动一致；无效区两个通道都没动 ⇒ pol_ratio 也保持原值。

    a2/b2（2026-08-04 加，0b 逐景对齐探针用）：HV 通道的独立系数；None 时与 HH
    共用——所有历史调用不传 ⇒ 行为逐位不变。逐通道时 pol_ratio 不再精确消 b，
    仍从扰动后的两通道重算，保持自洽。
    """
    a2 = a if a2 is None else a2
    b2 = b if b2 is None else b2
    x[:, 0:1] = torch.where(valid, hh0 * a + b, hh0)
    x[:, 1:2] = torch.where(valid, hv0 * a2 + b2, hv0)
    if pol_idx is not None:
        x[:, pol_idx:pol_idx + 1] = x[:, 0:1] - x[:, 1:2]


def self_test():
    """扰动逻辑自检（不碰数据/GPU）：无效区不动、有效区精确仿射、pol_ratio 自洽、
    每个变体都从原始出发（无跨变体漂移）。"""
    torch.manual_seed(0)
    C = 8
    x = torch.randn(1, C, 16, 16)
    x[:, 6] = (torch.rand(1, 16, 16) > 0.3).float()      # ch6 = 0/1 mask
    x[:, 7] = x[:, 0] - x[:, 1]                          # ch7 = pol_ratio
    x_orig = x.clone()
    valid = x[:, 6:7] > 0.5
    hh0, hv0 = x[:, 0:1].clone(), x[:, 1:2].clone()

    for tag, a, b in VARIANTS + VARIANTS:   # 跑两轮，验证无跨变体漂移
        apply_radiometric(x, hh0, hv0, valid, a, b, 7)
        v = valid.squeeze(1)
        # 有效区：精确仿射
        assert torch.allclose(x[:, 0][v], x_orig[:, 0][v] * a + b, atol=1e-6), tag
        assert torch.allclose(x[:, 1][v], x_orig[:, 1][v] * a + b, atol=1e-6), tag
        # 无效区：逐位不动
        assert torch.equal(x[:, 0][~v], x_orig[:, 0][~v]), tag
        assert torch.equal(x[:, 1][~v], x_orig[:, 1][~v]), tag
        # pol_ratio 自洽：有效区 = a·原值（b 消掉），无效区 = 原值
        assert torch.allclose(x[:, 7][v], x_orig[:, 7][v] * a, atol=1e-5), tag
        assert torch.allclose(x[:, 7][~v], x_orig[:, 7][~v], atol=1e-6), tag
        # 未涉及的通道逐位不动
        for c in (2, 3, 4, 5, 6):
            assert torch.equal(x[:, c], x_orig[:, c]), f'{tag} ch{c} 被改动了'
    # id 变体必须把 x 完全还原
    apply_radiometric(x, hh0, hv0, valid, 1.0, 0.0, 7)
    assert torch.allclose(x, x_orig, atol=1e-6), 'id 变体未还原原始输入'
    print('[self-test] PASS 扰动逻辑：有效区精确仿射、无效区与其余通道逐位不动、'
          'pol_ratio 自洽、变体间无浮点漂移、id 可还原', flush=True)


def infer_pred(x, net, opts, labels, valid_maps):
    """一次完整场景推理，返回 (H,W) uint8 CPU 预测图。

    走 quickstart 验证同一条路径 `fast_tiled_val_predict`：整景留在 CPU（只有每批 tile
    上 GPU）、逐 batch 就地 argmax（不开整景 logits 缓冲）、tile 门控跳过标签全 255 的
    tile。被跳过的 tile 预测为 0，但它们的像素在打分时一律被 valid_maps 排除 ⇒ 不影响
    任何指标（本探针只算混淆矩阵，不做可视化，满足这个前提）。"""
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds, _ = fast_tiled_val_predict(x, net, opts, labels=labels,
                                          valid_maps=valid_maps, gate=True)
    return preds['FLOE']


def cm_iou(cm):
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def main():
    args = parse_args()
    limit_torch_threads()
    if args.self_test:
        self_test()
        return
    for req in ('config', 'checkpoint', 'val_list'):
        assert getattr(args, req), f'--{req.replace("_", "-")} 必填（或用 --self-test）'
    self_test()          # 真跑之前也过一遍，几乎零成本
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    opts = get_variable_options(copy.deepcopy(Config.fromfile(args.config).train_options))
    assert opts['down_sample_scale'] == 1, '需原生分辨率验证口径'
    assert opts['model_selection'] not in ('swin', 'PA2'), \
        'swin/PA2 走重叠滑窗（batched_slide_inference），与本探针的对齐 tile 路径 + 按需读' \
        '不兼容 —— 拒绝运行，免得静默读到 0'
    n_cls = opts['n_classes']['FLOE']
    variants = parse_variants(args.variants) if args.variants else VARIANTS
    if args.variants:
        print(f'[variants] 使用覆盖清单：{variants}', flush=True)
    hh_i, hv_i, mask_i, pol_i, n_ch = channel_layout(opts)
    print(f'[layout] n_ch={n_ch} | HH=ch{hh_i} HV=ch{hv_i} mask=ch{mask_i} '
          f'pol_ratio={"ch%d" % pol_i if pol_i is not None else "无"}；'
          f'扰动只作用于 HH/HV/pol_ratio 且仅在 mask==1 内', flush=True)

    net = get_model(opts, device)
    sd = torch.load(args.checkpoint, map_location=device, weights_only=False)['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    # 模型已按【完整 charts】建好并 strict load（多任务 checkpoint 含辅头）；此后的
    # 数据读取 / 前向 / 指标一律只跑 target chart（辅任务 train-only）——否则 val 景可能
    # 根本没有辅任务变量（KeyError），且 tile 门控会因"辅任务处处有效"而全失效。
    # 无 aux_charts 时返回的就是 opts 本身，行为不变。
    opts = val_scoped_options(opts)

    files = json.load(open(args.val_list))
    # 按需读：本探针不改 tile 网格、且只在门控后的对齐 tile 上前向 ⇒ 满足 val_ondemand_ok
    # 的前提。不满足时（或 config 关了 kill switch）自动退回整景读并打印原因。
    ok, why = val_ondemand_ok(opts, 1)
    ondemand = bool(opts.get('val_ondemand_read', True)) and ok
    print(f'[ondemand] {"启用" if ondemand else "退回整景读"}：{why}', flush=True)
    ds = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train',
                                       ondemand_read=ondemand)
    dl = torch.utils.data.DataLoader(ds, batch_size=None,
                                     num_workers=opts.get('num_workers_val', 0), shuffle=False)

    tags = [v[0] for v in variants]
    global_cm = {t: np.zeros((n_cls, n_cls), dtype=np.float64) for t in tags}
    scene_cms = {t: [] for t in tags}
    scene_ids = []

    for inf_x, inf_y, cfv_masks, tfv_mask, name, original_size in dl:
        sid = osp.splitext(str(name))[0]
        x = inf_x if inf_x.dim() == 4 else inf_x.unsqueeze(0)     # (1,C,H,W) CPU
        assert x.shape[1] == n_ch, f'{sid}: 通道数 {x.shape[1]} != 预期 {n_ch}'
        H, W = x.shape[-2:]
        gt = np.squeeze(np.asarray(inf_y['FLOE'])).astype(np.int64)
        m = gt != 255

        valid = x[:, mask_i:mask_i + 1] > 0.5
        hh0, hv0 = x[:, hh_i:hh_i + 1].clone(), x[:, hv_i:hv_i + 1].clone()
        # 门控/打分掩膜与 quickstart 一致：valid = ~cfv_masks（loader 已算好，别重算）
        valid_maps = {c: ~cfv_masks[c] for c in opts['charts']}
        line = []
        for tag, a, b in variants:
            if tag in ('center', 'zscene'):
                # 逐景自适应：从该景 valid 区的原始统计现算（hh0/hv0 是原始备份）
                m_hh = float(hh0[valid].mean()); m_hv = float(hv0[valid].mean())
                if tag == 'center':
                    apply_radiometric(x, hh0, hv0, valid, 1.0, -m_hh, pol_i,
                                      a2=1.0, b2=-m_hv)
                else:
                    s_hh = max(float(hh0[valid].std()), 1e-6)
                    s_hv = max(float(hv0[valid].std()), 1e-6)
                    apply_radiometric(x, hh0, hv0, valid,
                                      1.0 / s_hh, -m_hh / s_hh, pol_i,
                                      a2=1.0 / s_hv, b2=-m_hv / s_hv)
            else:
                apply_radiometric(x, hh0, hv0, valid, a, b, pol_i)
            pred = infer_pred(x, net, opts, inf_y, valid_maps).numpy().astype(np.int64)
            cm = np.bincount(gt[m] * n_cls + pred[m], minlength=n_cls * n_cls
                             ).reshape(n_cls, n_cls).astype(np.float64)
            global_cm[tag] += cm
            scene_cms[tag].append(cm)
            iou = cm_iou(cm)
            line.append('%s=%.3f' % (tag, np.mean([iou[2], iou[3]]) if m.any() else float('nan')))
            torch.cuda.empty_cache()
        del hh0, hv0, valid, x
        scene_ids.append(sid)
        print('%-44s hard-ice %s  (%dx%d)' % (sid[:44], ' '.join(line), H, W), flush=True)

    # ---- 汇总 ----
    order = np.argsort(scene_ids)
    sorted_ids = np.array(scene_ids)[order]
    base_iou = cm_iou(global_cm['id'])
    base_hard = float(np.mean([base_iou[2], base_iou[3]]))
    print('\n' + '=' * 96, flush=True)
    print('%-9s %8s %8s %9s | %s' % ('variant', 'macro', 'hard-ice', 'Δhard',
                                     ' '.join('%6s' % c for c in
                                              ('water', 'cake', 'small', 'medium', 'big', 'vast'))),
          flush=True)
    print('=' * 96, flush=True)
    for tag, a, b in variants:
        iou = cm_iou(global_cm[tag])
        hard = float(np.mean([iou[2], iou[3]]))
        print('%-9s %8.4f %8.4f %+9.4f | %s' %
              (tag, iou.mean(), hard, hard - base_hard,
               ' '.join('%6.3f' % v for v in iou)), flush=True)
        if args.dump_cm:
            root, ext = osp.splitext(args.dump_cm)
            os.makedirs(osp.dirname(args.dump_cm) or '.', exist_ok=True)
            np.savez('%s_%s%s' % (root, tag, ext or '.npz'),
                     scenes=sorted_ids, cm=np.stack(scene_cms[tag])[order])
    print('=' * 96, flush=True)
    print('参照 id: macro=%.4f hard-ice=%.4f（%d 景）' %
          (base_iou.mean(), base_hard, len(scene_ids)), flush=True)
    print('读法：Δhard 明显负且随 |b| 单调加深 ⇒ 模型依赖绝对辐射水平（H1 成立，不变性线有靶子）；'
          'Δhard ≈ 0 ⇒ 捷径不存在（H1 否掉）。参考噪声带：baseline 两 seed 的 hard-ice 差 0.005。',
          flush=True)
    print('RADIO_EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()
    sys.exit(0)
