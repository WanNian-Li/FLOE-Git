"""Fig. 7：定性预测对比 [HH | GT | Baseline | PGSG]。**需要 GPU**。

为什么必须有这张图
------------------
论文目前一张预测图都没有，遥感分割论文不放预测图很反常；而本文最大的单类增益
（大浮冰 +0.026，碎裂被修复）恰好是肉眼可见的差异。

两种用法
--------
1) **选景（纯 CPU，秒级，不用 GPU）**——从两个模型的逐景混淆矩阵里排出"PGSG 相对基线
   在大浮冰上改善最多"的场景：

       python -m tools.figures.fig7_qualitative --rank

2) **出图（要 GPU）**——对选定场景跑两个模型的整景推理并画 1×4：

       python -m tools.figures.fig7_qualitative --scenes <name1> <name2> [--crop r0,c0,size]

⚠️ 推理用 `fast_tiled_val_predict(..., gate=False)`：**必须关门控**。开门控时标签全 255 的
tile 会被跳过、预测留 0，算指标无妨（那些像素本来就被排除），但**画整景图会出现成片的
假"水"**——CLAUDE.md 里 `_viz_pred.py` 那条坑就是这个。
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
from tools.figures.common import *                      # noqa: E402,F401,F403

CFG_BASE = 'configs/FLOE/sota_segformer_in1k_dlr_417.py'
CFG_PGSG = 'configs/FLOE/pgsg.py'
RUN_BASE = 'floe_baseline_417_s42'
RUN_PGSG = 'floe_pgsg_417_s42'
VAL_LIST = 'datalists/val_417.json'


# ------------------------------------------------------------------ 选景（CPU）
def rank_scenes(npz_base, npz_pgsg, cls=4, top=15):
    """按"PGSG 相对基线在指定类上的逐景 IoU 增益"排序，纯读 npz。"""
    def load(p):
        d = np.load(p, allow_pickle=True)
        return {osp.splitext(str(s))[0]: d['cm'][i].astype(np.float64)
                for i, s in enumerate(d['scenes'])}
    a, b = load(npz_base), load(npz_pgsg)
    rows = []
    for k in sorted(set(a) & set(b)):
        support = a[k].sum(axis=1)[cls]
        if support < 1e5:                     # 该类像元太少，图上看不出东西
            continue
        ia, ib = iou_from_cm(a[k])[cls], iou_from_cm(b[k])[cls]
        rows.append((ib - ia, ia, ib, support, k))
    rows.sort(reverse=True)
    print(f'按 {CLASS_SHORT[cls]} 的逐景 IoU 增益排序（支撑 ≥1e5 px 的场景）：')
    print(f'{"Δ IoU":>8s}{"base":>8s}{"pgsg":>8s}{"Mpx":>8s}  scene')
    for d, ia, ib, sup, k in rows[:top]:
        print(f'{d:+8.3f}{ia:8.3f}{ib:8.3f}{sup / 1e6:8.1f}  {k}')
    if len(rows) > top:
        print(f'  …共 {len(rows)} 个场景')
    return [r[4] for r in rows]


# ------------------------------------------------------------------ 推理
def load_net(cfg_path, ckpt, device):
    opts = get_variable_options(copy.deepcopy(Config.fromfile(cfg_path).train_options))
    assert opts['down_sample_scale'] == 1, '需原生分辨率口径'
    net = get_model(opts, device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    return net, val_scoped_options(opts)


def predict(x, net, opts, labels, cfv_masks):
    """gate=False：画整景必须关门控，否则被跳过的 tile 会渲染成常数 0（假"水"）。

    ⚠️ valid_maps 要传 `~cfv_masks`（与 quickstart / _eval_radiometric 同口径），
    直接传 cfv_masks 会把有效/无效反过来。
    """
    valid_maps = {c: ~cfv_masks[c] for c in opts['charts']}
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds, _ = fast_tiled_val_predict(x, net, opts, labels=labels,
                                          valid_maps=valid_maps, gate=False)
    return preds['FLOE'].numpy().astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', action='store_true', help='只选景（CPU），不出图')
    ap.add_argument('--scenes', nargs='+', default=None)
    ap.add_argument('--crop', default=None, help='显示子窗口 "r0,c0,size"；省略=整景')
    ap.add_argument('--disp-width', type=int, default=900, help='显示时的目标宽度（抽稀）')
    ap.add_argument('--cfg-base', default=CFG_BASE)
    ap.add_argument('--cfg-pgsg', default=CFG_PGSG)
    ap.add_argument('--ckpt-base', default=None)
    ap.add_argument('--ckpt-pgsg', default=None)
    ap.add_argument('--npz-base', default=osp.join(CM_DIR, 'baseline.npz'))
    ap.add_argument('--npz-pgsg', default=osp.join(CM_DIR, 'pgsg.npz'))
    ap.add_argument('--cls', type=int, default=4, help='--rank 按哪一类排序（默认 big）')
    args = ap.parse_args()

    if args.rank or not args.scenes:
        ranked = rank_scenes(args.npz_base, args.npz_pgsg, cls=args.cls)
        if args.rank:
            return
        args.scenes = ranked[:2]
        print(f'\n未指定 --scenes，取增益最大的两个：{args.scenes}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('⚠️ 没有 GPU，整景推理会非常慢；本脚本本就是为开卡时准备的。')

    ck_b = resolve_ckpt(RUN_BASE, args.ckpt_base)
    ck_p = resolve_ckpt(RUN_PGSG, args.ckpt_pgsg)
    print(f'baseline ckpt: {ck_b}')
    print(f'pgsg     ckpt: {ck_p}')
    net_b, opts_b = load_net(args.cfg_base, ck_b, device)
    net_p, opts_p = load_net(args.cfg_pgsg, ck_p, device)

    all_files = json.load(open(VAL_LIST, encoding='utf-8'))
    name2file = {osp.splitext(osp.basename(f))[0]: f for f in all_files}
    files = [name2file[s] for s in args.scenes if s in name2file]
    missing = [s for s in args.scenes if s not in name2file]
    if missing:
        print(f'⚠️ 这些场景不在 {VAL_LIST} 里，已跳过：{missing}')
    if not files:
        raise SystemExit('没有可画的场景')

    ds = AI4ArcticChallengeTestDataset(options=opts_b, files=files, mode='train')
    dl = torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0, shuffle=False)
    gvm_idx = list(opts_b['train_variables']).index('global_valid_mask')

    for inf_x, inf_y, cfv_masks, tfv_mask, name, _ in dl:
        sid = osp.splitext(str(name))[0]
        print(f'\n--- {sid} ---', flush=True)
        pred_b = predict(inf_x, net_b, opts_b, inf_y, cfv_masks)
        pred_p = predict(inf_x, net_p, opts_p, inf_y, cfv_masks)

        gt = np.squeeze(np.asarray(inf_y['FLOE'])).astype(np.int16)
        feats = (inf_x[0] if inf_x.dim() == 4 else inf_x).numpy()
        hh, gvm = feats[0], feats[gvm_idx] > 0.5

        if args.crop:
            r0, c0, sz = (int(v) for v in args.crop.split(','))
            sl = (slice(r0, r0 + sz), slice(c0, c0 + sz))
            hh, gvm, gt, pred_b, pred_p = (a[sl] for a in (hh, gvm, gt, pred_b, pred_p))

        step = max(1, int(round(hh.shape[1] / args.disp_width)))
        hh, gvm, gt, pred_b, pred_p = (a[::step, ::step]
                                       for a in (hh, gvm, gt, pred_b, pred_p))
        show = (gt != 255) & gvm               # 三幅标签图都只画到 GT 有效区，公平对比
        img = stretch(hh, gvm)

        h, w = img.shape
        pw = 6.9 / 4
        fig, axes = plt.subplots(1, 4, figsize=(6.9, pw * h / max(w, 1) + 0.55))
        panels = [(img, None, '(a) HH'),
                  (img, gt, '(b) Reference'),
                  (img, pred_b, '(c) Baseline'),
                  (img, pred_p, '(d) PGSG')]
        for ax, (base, lab, title) in zip(axes, panels):
            ax.imshow(base, cmap='gray', vmin=0, vmax=1)
            if lab is not None:
                rgb = label_to_rgb(np.where(show, lab, 255))
                ax.imshow(rgb, alpha=np.where(show, 0.62, 0.0))
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) for c in range(N_CLASSES)]
        fig.legend(handles, CLASS_NAMES, loc='lower center', ncol=6,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))
        savefig(fig, f'fig7_qualitative_{sid[:32]}')

        for tag, p in (('baseline', pred_b), ('pgsg', pred_p)):
            v = show & (gt >= 0)
            agree = float((p[v] == gt[v]).mean()) if v.any() else float('nan')
            print(f'  {tag:9s} pixel agreement on the displayed window: {agree:.3f}')


if __name__ == '__main__':
    main()
