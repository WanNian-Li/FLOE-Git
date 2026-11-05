"""T0-0.1 + Fig. 5：fold0 混淆矩阵分析与绘图（无卡模式安全，纯 numpy）。

产出
----
1) 打印 Introduction 动机 C 需要的数字：基线上 true-big 被判成 small / medium 的比例；
   顺带打印全部逐类 IoU（与实验表对拍，防止读错文件）。
2) Fig. 5：基线 vs PGSG 的行归一化混淆矩阵热图（两联图）。

用法
----
    python -m tools.figures.fig5_confusion                                  # 用默认路径
    python -m tools.figures.fig5_confusion --baseline <npz> --pgsg <npz>

默认读 /root/autodl-tmp/cm/single/{b2dlr_f0,pgsg}.npz（evaluate_scenes --dump-cm 的输出，
键：scenes (S,)、cm (S,C,C) 行=真实列=预测）。
"""
import argparse
import os.path as osp

from tools.figures.common import *          # noqa: F401,F403


def load_cm(path):
    d = np.load(path, allow_pickle=True)
    cm = d['cm'].astype(np.float64)        # (S, C, C)
    scenes = [str(s) for s in d['scenes']]
    return cm, scenes


def row_normalise(cm):
    s = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, s, out=np.zeros_like(cm), where=s > 0)


def report(name, cm_sum):
    iou = iou_from_cm(cm_sum)
    norm = row_normalise(cm_sum)
    print(f'\n=== {name} ===')
    print('  per-class IoU : ' +
          '  '.join(f'{CLASS_SHORT[c]} {iou[c]:.3f}' for c in range(N_CLASSES)))
    print(f'  macro mIoU    : {iou.mean():.4f}'
          f'   hard-ice (small,medium) : {0.5 * (iou[2] + iou[3]):.4f}')
    print('  true-big 的去向（行归一化 %）：' +
          '  '.join(f'{CLASS_SHORT[c]} {100 * norm[4, c]:.1f}'
                    for c in range(N_CLASSES)))
    print(f'  >>> 动机 C 用数：big->small {100 * norm[4, 2]:.1f}% , '
          f'big->medium {100 * norm[4, 3]:.1f}% , '
          f'两者合计 {100 * (norm[4, 2] + norm[4, 3]):.1f}%')
    print('  true-medium 的去向（行归一化 %）：' +
          '  '.join(f'{CLASS_SHORT[c]} {100 * norm[3, c]:.1f}'
                    for c in range(N_CLASSES)))
    return iou, norm


def panel(ax, norm, title):
    im = ax.imshow(100 * norm, cmap='Blues', vmin=0, vmax=100)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_SHORT, rotation=45, ha='right')
    ax.set_yticklabels(CLASS_SHORT)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            v = 100 * norm[i, j]
            if v < 0.5:
                continue
            ax.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=6,
                    color='white' if v > 55 else '#222222')
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default=osp.join(CM_DIR, 'baseline.npz'))
    ap.add_argument('--pgsg', default=osp.join(CM_DIR, 'pgsg.npz'))
    ap.add_argument('--name-baseline', default='(a) Baseline')
    ap.add_argument('--name-pgsg', default='(c) PGSG (full)')
    args = ap.parse_args()

    cm_b, scenes_b = load_cm(args.baseline)
    cm_p, scenes_p = load_cm(args.pgsg)
    print(f'baseline: {args.baseline}  ({len(scenes_b)} scenes)')
    print(f'pgsg    : {args.pgsg}  ({len(scenes_p)} scenes)')
    if scenes_b != scenes_p:
        print('  ⚠️ 两个文件的场景列表不一致——请确认是同一折的评估结果！')

    sum_b = cm_b.sum(axis=0)
    sum_p = cm_p.sum(axis=0)
    iou_b, norm_b = report(args.name_baseline, sum_b)
    iou_p, norm_p = report(args.name_pgsg, sum_p)

    print('\n=== PGSG - Baseline（逐类 IoU 差）===')
    print('  ' + '  '.join(f'{CLASS_SHORT[c]} {iou_p[c] - iou_b[c]:+.3f}'
                           for c in range(N_CLASSES)))
    print(f'  macro {iou_p.mean() - iou_b.mean():+.4f}   hard-ice '
          f'{0.5 * (iou_p[2] + iou_p[3]) - 0.5 * (iou_b[2] + iou_b[3]):+.4f}')

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    panel(axes[0], norm_b, args.name_baseline)
    im = panel(axes[1], norm_p, args.name_pgsg)
    axes[1].set_ylabel('')
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('Share of true-class pixels (\\%)' if plt.rcParams['text.usetex']
                 else 'Share of true-class pixels (%)')
    savefig(fig, 'fig5_confusion')


if __name__ == '__main__':
    main()
