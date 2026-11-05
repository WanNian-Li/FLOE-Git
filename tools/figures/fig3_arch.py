"""Fig. 3：PGSG 架构图（纯绘图，无数据、无模型、无 GPU；任何装了 matplotlib 的机器都能跑）。

为什么用 matplotlib 而不是 TikZ
------------------------------
本地没有 LaTeX 环境。本脚本产出与其余六张图同规格的 PDF（矢量）+ PNG（预览），
直接 \\includegraphics 进正文即可，不引入任何新工具链。
（`docs/fig3_architecture.tex` 是同一张图的 TikZ 备选实现，**未使用**，留档备查。）

要表达的三件事（改图时别改丢）
------------------------------
1. 证据通路（红）只有两条出边，分别进入【尺度轴 transformer 的证据 token】与【门控输入】，
   **绝不并入主通路（蓝）**——这是"证据-决策解耦"的全部含义；
2. 零初始化点只有**两处**：尺度轴 transformer 的输出投影、门控的末层卷积
   （证据通路刻意不做零初始化，理由见补充材料 #1 §2.2）；
3. 主通路的形状：4 个尺度特征 → 通信 → 门控加权 → concat → fuse → 分类。

用法
----
    python -m tools.figures.fig3_arch
"""
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from tools.figures.common import *          # noqa: F401,F403

MAIN = '#1F4E79'      # 主通路
MAINF = '#DCE9F5'
EVI = '#A03020'       # 证据通路
EVIF = '#F7E3DE'
DECI = '#5C3C92'      # 决策模块（门控 / 尺度轴 transformer）
DECIF = '#E8E1F3'


def box(ax, cx, cy, w, h, lines, ec, fc, lw=1.0, fs=7.0, zorder=3):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle='round,pad=0.006,rounding_size=0.012',
                                linewidth=lw, edgecolor=ec, facecolor=fc,
                                zorder=zorder))
    ax.text(cx, cy, '\n'.join(lines), ha='center', va='center',
            fontsize=fs, zorder=zorder + 1, linespacing=1.35)
    return (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)


def arrow(ax, p0, p1, color, style='-', rad=0.0, lw=1.2, zorder=2):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle='-|>', mutation_scale=8, linewidth=lw,
        linestyle=style, color=color, zorder=zorder,
        connectionstyle=f'arc3,rad={rad}', shrinkA=1.5, shrinkB=1.5))


def zero_mark(ax, x, y):
    ax.plot([x], [y], marker='o', ms=7.5, mfc='#FFD24D', mec='k',
            mew=0.7, zorder=8)
    ax.text(x, y, '0', ha='center', va='center', fontsize=5.4,
            fontweight='bold', zorder=9)


def main():
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    YM = 0.775                       # 主通路中心高度
    H = 0.170

    b_in = box(ax, 0.048, YM, 0.082, H,
               ['Input', r'$7\times H\times W$'], MAIN, MAINF, fs=6.6)
    b_enc = box(ax, 0.158, YM, 0.088, H,
                ['MiT-B2', 'encoder'], MAIN, MAINF, fs=6.6)

    # 四个尺度特征（虚线框内的小方块）
    fx, fw, fh = 0.272, 0.074, 0.030
    for i, s in enumerate(['stride 4', 'stride 8', 'stride 16', 'stride 32']):
        y = YM + 0.050 - i * 0.0335
        ax.add_patch(FancyBboxPatch((fx - fw / 2, y - fh / 2), fw, fh,
                                    boxstyle='round,pad=0.002,rounding_size=0.004',
                                    linewidth=0.7, edgecolor=MAIN,
                                    facecolor='white', zorder=3))
        ax.text(fx, y, s, ha='center', va='center', fontsize=5.4, zorder=4)
    ax.add_patch(FancyBboxPatch((fx - 0.050, YM - 0.090), 0.100, 0.180,
                                boxstyle='round,pad=0.004,rounding_size=0.010',
                                linewidth=0.8, edgecolor=MAIN, facecolor='none',
                                linestyle=(0, (3, 2)), zorder=2))
    ax.text(fx, YM + 0.108, r'project to 256, upsample to stride 4 ($S=4$)',
            ha='center', va='bottom', fontsize=5.8, color=MAIN)

    b_xat = box(ax, 0.432, YM, 0.170, H,
                ['Scale-axis', 'transformer', r'$S{+}1$ tokens, $d=64$'],
                DECI, DECIF, lw=1.5, fs=6.2)
    b_gate = box(ax, 0.632, YM, 0.180, H,
                 ['Evidence-conditioned', 'softmax gate',
                  r'$\sum_s g_s \equiv S$,  apply $\times g_s$'],
                 DECI, DECIF, lw=1.5, fs=6.2)
    b_fuse = box(ax, 0.800, YM, 0.110, H,
                 ['concat 1024', r'$1\times1$ conv', 'BN, ReLU'], MAIN, MAINF, fs=6.2)
    b_cls = box(ax, 0.938, YM, 0.096, H, ['Classifier', '6 classes'], MAIN, MAINF, fs=6.4)

    for p, q in ((b_in, b_enc), (b_enc, (fx - 0.050,)), ):
        arrow(ax, (p[1], YM), (q[0], YM), MAIN)
    arrow(ax, (fx + 0.050, YM), (b_xat[0], YM), MAIN)
    arrow(ax, (b_xat[1], YM), (b_gate[0], YM), MAIN)
    arrow(ax, (b_gate[1], YM), (b_fuse[0], YM), MAIN)
    arrow(ax, (b_fuse[1], YM), (b_cls[0], YM), MAIN)

    # ------------------------------------------------ 证据通路
    YV, YP = 0.395, 0.180
    b_var = box(ax, 0.380, YV, 0.315, 0.125,
                ['Variogram evidence',
                 r'$\gamma_r,\ r\in\{1,2,4,8,16\}$, normalised to $v_r$',
                 '64 ch'], EVI, EVIF, fs=6.2)
    b_pol = box(ax, 0.380, YP, 0.315, 0.100,
                ['Polarimetric prior',
                 r'$p=\mathrm{HH}-\mathrm{HV}$,  32 ch'], EVI, EVIF, fs=6.2)
    b_E = box(ax, 0.610, 0.288, 0.070, 0.100,
              [r'$E$', '96 ch'], EVI, EVIF, fs=6.6)

    ds = (0, (3.5, 2.5))
    # 证据来源：encoder 的 stride-4 特征 / 输入的极化比通道
    arrow(ax, (0.158, YM - H / 2), (b_var[0], YV), EVI, style=ds, rad=-0.15)
    arrow(ax, (0.048, YM - H / 2), (b_pol[0], YP), EVI, style=ds, rad=-0.10)
    ax.text(0.150, YV + 0.118, 'stride-4 feature', ha='left', va='center',
            fontsize=5.6, color=EVI)
    ax.text(0.030, YP + 0.155, 'pol. ratio channel', ha='left', va='center',
            fontsize=5.6, color=EVI)

    arrow(ax, (b_var[1], YV), (b_E[0], 0.312), EVI, style=ds, rad=-0.10)
    arrow(ax, (b_pol[1], YP), (b_E[0], 0.264), EVI, style=ds, rad=0.10)

    # 证据 -> 两个决策模块：唯一的两条出边
    arrow(ax, (b_E[0], 0.312), (b_xat[1] - 0.035, YM - H / 2), EVI,
          style=ds, rad=0.30, lw=1.3)
    arrow(ax, (0.610, b_E[3]), (b_gate[0] + 0.045, YM - H / 2), EVI,
          style=ds, rad=-0.10, lw=1.3)

    zero_mark(ax, b_xat[1] - 0.016, b_xat[3] + 0.020)
    zero_mark(ax, b_gate[1] - 0.016, b_gate[3] + 0.020)

    # ------------------------------------------------ 图例
    y0 = 0.035
    ax.plot([0.055, 0.095], [y0, y0], color=MAIN, lw=1.6)
    ax.text(0.104, y0, 'main feature path', va='center', fontsize=6.4)
    ax.plot([0.310, 0.350], [y0, y0], color=EVI, lw=1.6, linestyle=ds)
    ax.text(0.359, y0, 'evidence (conditioning only)', va='center', fontsize=6.4)
    zero_mark(ax, 0.700, y0)
    ax.text(0.719, y0, 'zero-initialized projection', va='center', fontsize=6.4)

    savefig(fig, 'fig3_architecture')


if __name__ == '__main__':
    main()
