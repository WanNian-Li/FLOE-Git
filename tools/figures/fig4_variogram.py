"""Fig. 4：变异函数剖面示意——归一化如何扔掉幅度、只留形状。无卡模式安全。

⚠️ 口径声明（图注里必须写）
--------------------------------
论文里的 variogram 模块作用在 **encoder 的 stride-4 特征** 上；无卡模式跑不了 encoder，
所以本图把**同一个构造**作用在输入的 HH 通道上，并先做 4×4 平均池化以对齐到同样的
320 m 网格。图注写法建议：

    "Illustration of the construction of Section 4.2 applied to the (4x-pooled) HH
     channel, which shares the 320 m grid of the stride-4 encoder features on which
     the module actually operates."

图的内容
--------
(a) log10 γ_r：三个类别的原始结构函数。同类不同景之间存在明显的**整体上下平移**——
    那就是辐射条件带来的幅度差异；虚线是人为施加 f -> a·f + b 后的同一 tile。
(b) v_r（跨 lag 归一化后的剖面）：虚线与实线**完全重合**（仿射不变），且类别之间按
    形状分开——这正是模块只保留"相关长度/纹理粗细"的含义。

用法
----
    python -m tools.figures.fig4_variogram
    python -m tools.figures.fig4_variogram --classes 2 3 4 --per-class 5
"""
import argparse

from tools.figures.common import *          # noqa: F401,F403

RADII = (1, 2, 4, 8, 16)                  # 与 _VarioTexture 默认一致（池化格 = 320 m）
POOL = 4                                  # 4x4 平均池化 -> stride-4 等效网格
EPS = 1e-6
CLAMP = 6.0


def pool2d(a, k):
    """整数倍 k×k 平均池化（裁掉不足的边）。"""
    h, w = a.shape
    h, w = (h // k) * k, (w // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def gamma_profile(f, mask, radii=RADII):
    """4 方向对称位移的结构函数，在 mask 为 True 的位置上取均值。

    γ_r = mean_x [ 1/4 * Σ_{d ∈ {±r e_h, ±r e_w}} (f(x) - f(x+d))^2 ]
    只统计 x 与 x+d 都有效的位置（各方向单独配对，与实现里的窗口平均等价到常数）。
    """
    out = np.empty(len(radii), dtype=np.float64)
    for i, r in enumerate(radii):
        acc, n = 0.0, 0
        for axis in (0, 1):
            for sign in (+1, -1):
                sh = np.roll(f, sign * r, axis=axis)
                mh = np.roll(mask, sign * r, axis=axis)
                # 去掉 roll 造成的环绕带
                m = mask & mh
                if axis == 0:
                    if sign > 0:
                        m[:r, :] = False
                    else:
                        m[-r:, :] = False
                else:
                    if sign > 0:
                        m[:, :r] = False
                    else:
                        m[:, -r:] = False
                if m.sum() == 0:
                    continue
                acc += float(((f - sh)[m] ** 2).sum())
                n += int(m.sum())
        out[i] = acc / max(n, 1)
    return out


def normalise(gam):
    """v_r = clamp(log(γ_r+ε) - log(mean_r γ_r + ε), ±6)，与 _VarioTexture 一致。"""
    v = np.log(gam + EPS) - np.log(gam.mean() + EPS)
    return np.clip(v, -CLAMP, CLAMP)


def pick_pure_tiles(idx, cls, n, min_frac=0.90, min_valid=0.95):
    """挑该类占比高、标注完整的 tile，并尽量来自不同场景。"""
    hist = idx['hist']
    tot = hist.sum(axis=1)
    frac = np.divide(hist[:, cls], np.maximum(tot, 1))
    ok = (frac >= min_frac) & (idx['label_valid_ratio'] >= min_valid) & \
         (idx['valid_ratio'] >= 0.99)
    cand = np.flatnonzero(ok)
    if cand.size == 0:
        return []
    order = cand[np.argsort(-frac[cand])]
    picked, seen = [], set()
    for t in order:                       # 一景最多取一个，体现跨景差异
        s = int(idx['scene_id'][t])
        if s in seen:
            continue
        seen.add(s)
        picked.append(int(t))
        if len(picked) >= n:
            break
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--classes', type=int, nargs='+', default=[2, 3, 4, 5])
    ap.add_argument('--per-class', type=int, default=10)
    ap.add_argument('--gain', type=float, default=1.3, help='示范仿射变换的 a')
    ap.add_argument('--bias', type=float, default=0.8, help='示范仿射变换的 b')
    ap.add_argument('--style', default='band', choices=['band', 'spaghetti'],
                    help='band=逐类中位数+四分位带（推荐，看得出有没有分层）；'
                         'spaghetti=每个 tile 一条线')
    args = ap.parse_args()

    idx = load_tile_index()
    T = idx['tile_size']
    lags_m = np.array(RADII) * POOL * PIXEL_M          # 320 m ... 5.12 km

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    ax_raw, ax_norm = axes
    max_dev = 0.0
    stats = {}                                        # cls -> dict of 曲线堆叠

    for cls in args.classes:
        tiles = pick_pure_tiles(idx, cls, args.per_class)
        if not tiles:
            print(f'  ⚠️ class {CLASS_SHORT[cls]}: 没有满足纯度条件的 tile，跳过')
            continue
        print(f'class {CLASS_SHORT[cls]}: {len(tiles)} tiles')
        col = CLASS_COLORS[cls]
        acc = {'lg': [], 'lg_aff': [], 'v': [], 'v_aff': []}
        for k, t in enumerate(tiles):
            scene = idx['files'][int(idx['scene_id'][t])]
            r, c = int(idx['row'][t]), int(idx['col'][t])
            d = read_tile(scene, r, c, T,
                          ['nersc_sar_primary', 'FLOE', 'global_valid_mask'])
            hh = d['nersc_sar_primary'].astype(np.float32)
            keep = (d['FLOE'] == cls) & (d['global_valid_mask'] > 0) & np.isfinite(hh)
            hh = np.where(keep, hh, 0.0)
            f = pool2d(hh, POOL)
            m = pool2d(keep.astype(np.float32), POOL) > 0.98   # 只留全纯的池化格

            if m.sum() < 500:
                print(f'  skip tile {t} (valid pooled cells {int(m.sum())})')
                continue
            gam = gamma_profile(f, m.copy())
            gam_aff = gamma_profile(args.gain * f + args.bias, m.copy())
            v, v_aff = normalise(gam), normalise(gam_aff)
            max_dev = max(max_dev, float(np.abs(v - v_aff).max()))

            acc['lg'].append(np.log10(gam))
            acc['lg_aff'].append(np.log10(gam_aff))
            acc['v'].append(v)
            acc['v_aff'].append(v_aff)

            if args.style == 'spaghetti':
                lbl = CLASS_NAMES[cls] if k == 0 else None
                ax_raw.plot(lags_m, np.log10(gam), '-', color=col, lw=1.1,
                            alpha=0.9, label=lbl)
                ax_raw.plot(lags_m, np.log10(gam_aff), '--', color=col, lw=0.8, alpha=0.6)
                ax_norm.plot(lags_m, v_aff, '-', color=col, lw=3.0, alpha=0.30,
                             solid_capstyle='round', zorder=1)
                ax_norm.plot(lags_m, v, '-', color=col, lw=1.1, alpha=0.95,
                             label=lbl, zorder=2)
            print(f'  tile {t:7d} {scene[:44]:44s} '
                  f'log10γ {np.log10(gam[0]):+.2f}..{np.log10(gam[-1]):+.2f}  '
                  f'|v - v_affine|max {np.abs(v - v_aff).max():.2e}')

        if acc['v']:
            stats[cls] = {k: np.asarray(v) for k, v in acc.items()}

    if args.style == 'band':
        for cls, st in stats.items():
            col = CLASS_COLORS[cls]
            n = st['v'].shape[0]
            for ax, key, key_aff in ((ax_raw, 'lg', 'lg_aff'), (ax_norm, 'v', 'v_aff')):
                med = np.median(st[key], axis=0)
                q1, q3 = np.percentile(st[key], [25, 75], axis=0)
                ax.fill_between(lags_m, q1, q3, color=col, alpha=0.20, lw=0, zorder=1)
                ax.plot(lags_m, med, '-', color=col, lw=1.8, zorder=3,
                        label=f'{CLASS_NAMES[cls]} (n={n})' if ax is ax_raw else None)
                if ax is ax_norm:
                    # 仿射版画成黑色虚线压在中位线上：与彩线严丝合缝 = 不变性
                    ax.plot(lags_m, np.median(st[key_aff], axis=0), '--', color='k',
                            lw=0.9, alpha=0.9, dashes=(3, 2.5), zorder=4,
                            label='after affine change' if cls == min(stats) else None)
                else:
                    ax.plot(lags_m, np.median(st[key_aff], axis=0), '--', color=col,
                            lw=1.0, alpha=0.8, zorder=2)

        # 逐类中位剖面之间到底分不分得开：把最细 lag 上的中位值打出来，供正文引用
        print('\n=== 归一化剖面 v_r 的逐类中位数（分不分层，看这里）===')
        hdr = '  class      ' + ''.join(f'{int(l)/1000:>8.2f}km' for l in lags_m)
        print(hdr)
        for cls, st in stats.items():
            med = np.median(st['v'], axis=0)
            print(f'  {CLASS_SHORT[cls]:10s}' + ''.join(f'{m:>10.3f}' for m in med))

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(lags_m)
        ax.set_xticklabels(['320 m', '640 m', '1.3 km', '2.6 km', '5.1 km'])
        ax.set_xlabel('lag $r$')
        ax.tick_params(axis='x', labelrotation=30)
    ax_raw.set_ylabel(r'$\log_{10}\gamma_r$')
    ax_raw.set_title(r'(a) raw structure function $\gamma_r$')
    ax_norm.set_ylabel(r'$v_r$')
    ax_norm.set_title(r'(b) normalised profile $v_r$')
    ax_norm.axhline(0, color='#999999', lw=0.6, zorder=0)
    ax_raw.legend(frameon=False, loc='best')
    band_note = ('line: class median, band: IQR\n' if args.style == 'band' else '')
    ax_raw.text(0.03, 0.97,
                band_note +
                f'dashed: after $f \\mapsto {args.gain:g}f+{args.bias:g}$',
                transform=ax_raw.transAxes, va='top', ha='left', fontsize=6.5)
    ax_norm.text(0.03, 0.97,
                 'black dashed: the same profiles\nafter the affine change\n'
                 f'(they coincide to {max_dev:.0e})',
                 transform=ax_norm.transAxes, va='top', ha='left', fontsize=6.5)
    fig.tight_layout()
    savefig(fig, 'fig4_variogram')
    print(f'\n>>> 不变性自检：max |v - v_affine| over all tiles = {max_dev:.3e}'
          '（应为 ~1e-6 量级；明显大于 1e-3 说明 ε 地板起作用了，检查 γ 是否过小）')


if __name__ == '__main__':
    main()
