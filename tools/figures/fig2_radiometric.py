"""Fig. 2：辐射动机图——同一场景在不同辐射水平下模型判读翻转。**需要 GPU**。

对应论文动机 B：尺度证据被场景辐射条件污染。图上直接给出"同一块冰、只把整景亮度
平移 ±0.7 z-unit（≈±4 dB），预测就从 medium 翻成 small/water"的视觉证据。

两步走
------
1) **扫描选景**（要 GPU，但只算指标、走门控，较快）：在 fold0 验证集里挑中浮冰支撑大的
   场景，测它们在 b∈{−0.7, 0, +0.7} 下的 medium IoU，排出翻转最剧烈的一个：

       python -m tools.figures.fig2_radiometric --scan

2) **出图**（要 GPU）：对选定场景关门控重跑，画 1×4 [HH+GT | b−0.7 | b=0 | b+0.7]：

       python -m tools.figures.fig2_radiometric --scene <name> [--crop r0,c0,size]

⚠️ 扰动逻辑直接复用 `_eval_radiometric.apply_radiometric`，保证与论文里报的
Δhard-ice 探针**逐位同一套口径**（只动 HH/HV 的有效像元，pol_ratio 同步重算）。
⚠️ 出图时 `gate=False`：开门控会把标签全 255 的 tile 跳过、预测留 0，画整景会出现假"水"。
"""
import argparse
import copy
import json
import os.path as osp

from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options,
                     AI4ArcticChallengeTestDataset, limit_torch_threads)

limit_torch_threads()

import torch                                          # noqa: E402
from floe.functions import get_model, fast_tiled_val_predict  # noqa: E402
from tools.analysis.radiometric import apply_radiometric, channel_layout  # noqa: E402
from tools.figures.common import *                      # noqa: E402,F401,F403

CFG = 'configs/FLOE/sota_segformer_in1k_dlr_417.py'
RUN = 'floe_baseline_417_s42'
VAL_LIST = 'datalists/val_417.json'
MEDIUM = 3

# 论文图用的三个操作点（与 08-04 因果探针同幅度）
SHIFTS = [('$-0.7$ z-unit', -0.70), ('unchanged', 0.0), ('$+0.7$ z-unit', +0.70)]


def load_net(device, ckpt=None):
    ckpt = resolve_ckpt(RUN, ckpt)
    print(f'baseline ckpt: {ckpt}')
    opts = get_variable_options(copy.deepcopy(Config.fromfile(CFG).train_options))
    assert opts['down_sample_scale'] == 1, '需原生分辨率口径'
    net = get_model(opts, device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net, val_scoped_options(opts)


def predict(x, net, opts, labels, cfv_masks, gate):
    """⚠️ valid_maps 要传 `~cfv_masks`（与 quickstart / _eval_radiometric 同口径）。
    扫描阶段 gate=True（只要指标，快）；出图阶段必须 gate=False（否则整景有假"水"）。
    """
    valid_maps = {c: ~cfv_masks[c] for c in opts['charts']}
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds, _ = fast_tiled_val_predict(x, net, opts, labels=labels,
                                          valid_maps=valid_maps, gate=gate)
    return preds['FLOE'].numpy().astype(np.int16)


def scene_iou(gt, pred, cls):
    v = gt != 255
    t, p = gt[v] == cls, pred[v] == cls
    inter = np.logical_and(t, p).sum()
    union = np.logical_or(t, p).sum()
    return float(inter / union) if union > 0 else float('nan')


def pick_files(scenes=None, min_medium_px=2e6, npz=None):
    """--scan 时按中浮冰像元数筛景（读逐景 CM，免 GPU）。"""
    all_files = json.load(open(VAL_LIST, encoding='utf-8'))
    name2file = {osp.splitext(osp.basename(f))[0]: f for f in all_files}
    if scenes:
        return [name2file[s] for s in scenes if s in name2file]
    d = np.load(npz, allow_pickle=True)
    keep = []
    for i, s in enumerate(d['scenes']):
        k = osp.splitext(str(s))[0]
        if k in name2file and d['cm'][i].astype(np.float64).sum(axis=1)[MEDIUM] >= min_medium_px:
            keep.append((d['cm'][i].sum(axis=1)[MEDIUM], name2file[k]))
    keep.sort(reverse=True)
    print(f'中浮冰像元 ≥{min_medium_px / 1e6:.0f} Mpx 的场景：{len(keep)} 个')
    return [f for _, f in keep]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scan', action='store_true', help='扫描候选场景，打印翻转幅度排序')
    ap.add_argument('--scene', default=None)
    ap.add_argument('--crop', default=None, help='"r0,c0,size"')
    ap.add_argument('--disp-width', type=int, default=900)
    ap.add_argument('--max-scan', type=int, default=12)
    ap.add_argument('--npz', default=osp.join(CM_DIR, 'baseline.npz'))
    ap.add_argument('--ckpt', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('⚠️ 没有 GPU：整景推理会非常慢，本脚本为开卡时准备。')
    net, opts = load_net(device, args.ckpt)
    hh_i, hv_i, mask_i, pol_i, n_ch = channel_layout(opts)

    files = pick_files([args.scene] if args.scene else None, npz=args.npz)
    if args.scan:
        files = files[:args.max_scan]
    if not files:
        raise SystemExit('没有候选场景')

    ds = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train')
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0, shuffle=False)
    gvm_idx = list(opts['train_variables']).index('global_valid_mask')
    summary = []

    for inf_x, inf_y, cfv_masks, tfv_mask, name, _ in dl:
        sid = osp.splitext(str(name))[0]
        x = inf_x if inf_x.dim() == 4 else inf_x.unsqueeze(0)
        hh0, hv0 = x[:, hh_i:hh_i + 1].clone(), x[:, hv_i:hv_i + 1].clone()
        valid = x[:, mask_i:mask_i + 1] > 0.5
        gt = np.squeeze(np.asarray(inf_y['FLOE'])).astype(np.int16)

        preds, ious = [], []
        for label, b in SHIFTS:
            apply_radiometric(x, hh0, hv0, valid, 1.0, b, pol_i)
            p = predict(x, net, opts, inf_y, cfv_masks, gate=args.scan)
            preds.append(p)
            ious.append(scene_iou(gt, p, MEDIUM))
        apply_radiometric(x, hh0, hv0, valid, 1.0, 0.0, pol_i)   # 还原，防跨景污染

        swing = np.nanmax(ious) - np.nanmin(ious)
        summary.append((swing, sid, list(ious)))
        print(f'{sid}  medium IoU  b-0.7 {ious[0]:.3f} | id {ious[1]:.3f} | '
              f'b+0.7 {ious[2]:.3f}   swing {swing:.3f}', flush=True)

        if args.scan:
            continue

        # ---- 出图 ----
        feats = (inf_x[0] if inf_x.dim() == 4 else inf_x).numpy()
        hh, gvm = feats[0], feats[gvm_idx] > 0.5
        arrs = [hh, gvm, gt] + preds
        if args.crop:
            r0, c0, sz = (int(v) for v in args.crop.split(','))
            arrs = [a[r0:r0 + sz, c0:c0 + sz] for a in arrs]
        step = max(1, int(round(arrs[0].shape[1] / args.disp_width)))
        hh, gvm, gt_d, p0, p1, p2 = [a[::step, ::step] for a in arrs]
        show = (gt_d != 255) & gvm
        img = stretch(hh, gvm)

        h, w = img.shape
        fig, axes = plt.subplots(1, 4, figsize=(6.9, 6.9 / 4 * h / max(w, 1) + 0.55))
        panels = [(gt_d, '(a) Reference', None)]
        for k, ((label, b), p, iou) in enumerate(zip(SHIFTS, (p0, p1, p2), ious)):
            panels.append((p, f'({chr(ord("b") + k)}) {label}', iou))
        for ax, (lab, title, iou) in zip(axes, panels):
            ax.imshow(img, cmap='gray', vmin=0, vmax=1)
            ax.imshow(label_to_rgb(np.where(show, lab, 255)),
                      alpha=np.where(show, 0.62, 0.0))
            ax.set_title(title if iou is None
                         else f'{title}\nmedium IoU {iou:.2f}', fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) for c in range(N_CLASSES)]
        fig.legend(handles, CLASS_NAMES, loc='lower center', ncol=6,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        savefig(fig, f'fig2_radiometric_{sid[:32]}')

    if args.scan:
        summary.sort(reverse=True)
        print('\n=== 翻转幅度排序（medium IoU 的最大-最小）===')
        for swing, sid, ious in summary:
            print(f'  swing {swing:.3f}   {ious[0]:.3f} / {ious[1]:.3f} / {ious[2]:.3f}   {sid}')
        print('\n挑 swing 最大、且 id 那一列本身不差的场景出图：'
              'python -m tools.figures.fig2_radiometric --scene <name>')


if __name__ == '__main__':
    main()
