"""Fig. 1：任务示意（SAR 切片 + 标签叠色 + 六类米尺轴）。无卡模式安全。

内存纪律：只用 h5py 读**一个 512×512 的对齐 tile**（每变量约 1MB），绝不整景解码。
候选 tile 由 tile 索引直接挑选（索引里存了每个 tile 的逐类像元直方图）。

用法
----
    python -m tools.figures.fig1_task                      # 自动挑一个类别混合度最高的 tile
    python -m tools.figures.fig1_task --list               # 只列候选，不画图
    python -m tools.figures.fig1_task --tile 12345         # 指定索引里的第 n 个 tile
"""
import argparse

from tools.figures.common import *          # noqa: F401,F403


def pick_candidates(idx, want=(2, 3), min_valid=0.85, topk=12):
    """挑"类别多样 + 含难类"的 tile：按香农多样性打分，要求难类占比可见。"""
    hist = idx['hist']
    n_valid = hist.sum(axis=1)
    frac = np.divide(hist, np.maximum(n_valid, 1)[:, None])
    ok = (idx['label_valid_ratio'] >= min_valid) & (n_valid > 0)
    hard = frac[:, list(want)].sum(axis=1)
    ok &= hard >= 0.15                       # 难类至少占一成半，图才有说服力
    p = np.clip(frac, 1e-12, 1)
    ent = -(p * np.log(p)).sum(axis=1)       # 类别多样性
    score = np.where(ok, ent + hard, -1.0)
    order = np.argsort(-score)[:topk]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tile', type=int, default=None)
    ap.add_argument('--list', action='store_true')
    ap.add_argument('--min-valid', type=float, default=0.85)
    args = ap.parse_args()

    idx = load_tile_index()
    T = idx['tile_size']
    cands = pick_candidates(idx, min_valid=args.min_valid)

    if args.list or args.tile is None:
        print('候选 tile（索引号 / 场景 / row,col / 逐类占比 %）：')
        for t in cands:
            f = idx['files'][int(idx['scene_id'][t])]
            h = idx['hist'][t]
            frac = 100 * h / max(h.sum(), 1)
            print(f'  {t:7d}  {f[:48]:48s} r{int(idx["row"][t]):6d} '
                  f'c{int(idx["col"][t]):6d}  ' +
                  ' '.join(f'{CLASS_SHORT[c]} {frac[c]:4.1f}' for c in range(N_CLASSES)))
        if args.list:
            return
        tile_i = int(cands[0])
        print(f'\n未指定 --tile，使用得分最高的 {tile_i}')
    else:
        tile_i = args.tile

    scene = idx['files'][int(idx['scene_id'][tile_i])]
    r, c = int(idx['row'][tile_i]), int(idx['col'][tile_i])
    print(f'drawing tile {tile_i}: {scene} @ row={r} col={c} size={T}')

    d = read_tile(scene, r, c, T, ['nersc_sar_primary', 'FLOE', 'global_valid_mask'])
    hh, lab, vm = d['nersc_sar_primary'], d['FLOE'], d['global_valid_mask']
    valid = vm > 0
    img = stretch(hh, valid)
    rgb = label_to_rgb(lab)

    km = T * PIXEL_M / 1000.0
    fig = plt.figure(figsize=(6.6, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.15], wspace=0.12)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img, cmap='gray', vmin=0, vmax=1, extent=[0, km, km, 0])
    ax0.set_title('(a) Sentinel-1 EW HH')
    ax0.set_xlabel('km')
    ax0.set_ylabel('km')

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.imshow(img, cmap='gray', vmin=0, vmax=1, extent=[0, km, km, 0])
    ax1.imshow(rgb, alpha=0.55, extent=[0, km, km, 0])
    ax1.set_title('(b) Floe size label')
    ax1.set_xlabel('km')
    ax1.set_yticklabels([])

    # 第三栏：六类的物理尺度轴 + 与像元的关系
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_title('(c) Classes partition the scale axis')
    bounds_m = [(0, 20), (20, 100), (100, 500), (500, 2000), (2000, 10000)]
    for k, (lo, hi) in enumerate(bounds_m):
        cls = k + 1
        x0 = np.log10(max(lo, 12))
        ax2.barh(k, np.log10(hi) - x0, left=x0, height=0.34,
                 color=CLASS_COLORS[cls], edgecolor='none')
        # 文字放在色条**上方**，避免窄条（cake/small）里放不下而被裁掉
        ax2.text(x0, k + 0.30,
                 f'{CLASS_NAMES[cls]} ({CLASS_RANGE[cls].replace("$<$", "<").replace("--", "-")})',
                 va='bottom', ha='left', fontsize=6.2, color='#222222')
    ax2.axvline(np.log10(PIXEL_M), color='k', ls='--', lw=1)
    ax2.text(np.log10(PIXEL_M) + 0.05, -0.50, f'pixel = {PIXEL_M:.0f} m',
             fontsize=6.2, va='bottom', ha='left')
    ax2.set_yticks([])
    ax2.set_ylim(-0.6, len(bounds_m) - 0.05)
    ax2.set_xlim(np.log10(11), np.log10(22000))
    ticks = [10, 100, 1000, 10000]
    ax2.set_xticks([np.log10(t) for t in ticks])
    ax2.set_xticklabels(['10 m', '100 m', '1 km', '10 km'])
    ax2.set_xlabel('floe diameter (log scale)')
    for s in ('top', 'right', 'left'):
        ax2.spines[s].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLORS[c]) for c in range(N_CLASSES)]
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=IGNORE_COLOR))
    fig.legend(handles, CLASS_NAMES + ['ignore'], loc='lower center',
               ncol=7, frameon=False, bbox_to_anchor=(0.5, -0.16))

    savefig(fig, 'fig1_task')
    h = idx['hist'][tile_i]
    print('tile 逐类占比 %: ' +
          ' '.join(f'{CLASS_SHORT[c]} {100 * h[c] / max(h.sum(), 1):.1f}'
                   for c in range(N_CLASSES)))
    print(f'图注可用：scene {scene}, tile origin (row {r}, col {c}), '
          f'{T}x{T} pixels = {km:.1f} x {km:.1f} km')


if __name__ == '__main__':
    main()
