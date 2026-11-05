#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
诊断②：从 per-scene 混淆矩阵回答两个问题。不需要 GPU（几秒）。

先用 evaluate_scenes.py --dump-cm 把每个 checkpoint 的 per-scene CM 落盘，再跑本脚本。

问题 A：小/中浮冰是"找不到"还是"分不开"？
    把 small(2) 和 medium(3) 合并成一类，重算 IoU。
      - 合并类 IoU 跳到 ~0.5+，而单独的 small/medium 只有 .25/.14
        → 模型完全能"找到"小/中浮冰区域，只是分不开这两类。
        那么再换 backbone 是浪费——真问题是这两类在 80m 像元下到底可不可分
        （交给 floe_separability.py 回答）。
      - 合并类 IoU 仍然只有 ~0.3
        → 是检测问题，架构/loss 还有得调。

问题 B：模型排名有没有统计效力？
    val 只有 30 景，而 Big/Vast 只可能来自那 8 景加拿大北极图中落到 val 的那 1~2 景
    （见 analyze_val_split.py）。模型间 mIoU 的差距 93% 来自 Big 一类。
    本脚本按【场景】做 bootstrap 重采样（这才是独立同分布的单位，像素不是），
    给出 mIoU 的 95% CI；多模型时用【配对】bootstrap（同一组重采样场景喂给所有模型）
    给出 P(A > B)——配对能消掉场景难度的共同波动，是比各自 CI 是否重叠更灵敏的检验。

用法
----
    # 单模型
    python -m tools.analysis.scene_cm --cm work_dir/floe_sota_segformer/cm.npz

    # 多模型横比（推荐）
    python -m tools.analysis.scene_cm \
        --cm work_dir/floe_sota_segformer/cm.npz \
             work_dir/floe_sota_swin/cm.npz \
             work_dir/floe_pa2_swin_tile/cm.npz \
             work_dir/floe_unet_tile/cm.npz \
        --labels segformer swin pa2_swin unet
"""
import argparse
import os

def _cpu_quota():
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(int(quota) // int(period)))
    except Exception:
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


_N = str(max(1, min(8, _cpu_quota())))
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, _N)

import numpy as np  # noqa: E402

NAMES = ['water', 'cake', 'small', 'medium', 'big', 'vast']


def parse_args():
    p = argparse.ArgumentParser(description='从 per-scene 混淆矩阵做合并类 IoU 与按景 bootstrap')
    p.add_argument('--cm', nargs='+', required=True,
                   help='evaluate_scenes.py --dump-cm 产出的 .npz，可给多个做横比')
    p.add_argument('--labels', nargs='*', default=None, help='与 --cm 一一对应的模型名')
    p.add_argument('--merge', nargs='+', default=['1+2'],
                   help='要考察的合并方案，形如 1+2（cake+small）、4+5（大+巨）。可给多个。'
                        '默认 1+2：项目现行合并口径（用户 2026-07-18 定），'
                        'small+medium(2+3) 的实验线已于 07-13 关闭，不再默认分析')
    p.add_argument('--n-boot', type=int, default=5000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--drop-empty', action='store_true',
                   help='报告 mIoU 时剔除在 GT 与预测中都不出现的类'
                        '（如 2026-07-16 扩充前为空的类 1 饼状冰）。'
                        '默认关闭，以便与已报的 6 类 mIoU 直接对齐。')
    p.add_argument('--focus-class', nargs='*', default=None,
                   help='（恰好两个模型时）第④节逐景分解的焦点类，类名（medium）或下标（3），'
                        '可给多个；缺省自动取两模型全局 IoU 差最大的前 2 类')
    return p.parse_args()


def iou_from_cm(cm, drop_empty=False):
    """cm: (C, C)，行=true 列=pred。返回 (per_class_iou, miou)。"""
    tp = np.diag(cm).astype(np.float64)
    pred_sum = cm.sum(axis=0)
    true_sum = cm.sum(axis=1)
    denom = pred_sum + true_sum - tp
    iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
    if drop_empty:
        present = denom > 0
        miou = iou[present].mean() if present.any() else 0.0
    else:
        # 与 derive_metrics_from_cm 一致：空类的 IoU 记为 0 且计入分母
        miou = iou.mean()
    return iou, float(miou)


def merge_cm(cm, groups):
    """按 groups（list of list of class idx）合并混淆矩阵的行与列。"""
    g = len(groups)
    out = np.zeros((g, g), dtype=np.float64)
    for i, gi in enumerate(groups):
        for j, gj in enumerate(groups):
            out[i, j] = cm[np.ix_(gi, gj)].sum()
    return out


def groups_for(merge_spec, n_classes):
    """'2+3' -> [[0],[1],[2,3],[4],[5]]，并返回新类名。"""
    merged = [int(x) for x in merge_spec.split('+')]
    groups, names = [], []
    for c in range(n_classes):
        if c in merged:
            if c == merged[0]:
                groups.append(merged)
                names.append('+'.join(NAMES[m] for m in merged))
            continue
        groups.append([c])
        names.append(NAMES[c])
    return groups, names


def load(path):
    d = np.load(path, allow_pickle=True)
    return {
        'scenes': [str(s) for s in d['scenes']],
        'cm': d['cm'].astype(np.float64),     # (S, C, C)
        'ckpt': str(d['checkpoint']) if 'checkpoint' in d else path,
    }


def print_confusion(cm, title):
    """行归一化的混淆矩阵：每行 = 该真实类被预测成各类的比例。"""
    true_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, true_sum, out=np.zeros_like(cm), where=true_sum > 0)
    print(f'\n  {title}（行=真实，列=预测，行归一化 %）')
    print('           ' + ''.join(f'{n[:7]:>9s}' for n in NAMES))
    for i in range(cm.shape[0]):
        if true_sum[i, 0] == 0:
            print(f'  {NAMES[i]:<8s} ' + '        -' * cm.shape[1] + '   (GT 中不存在)')
            continue
        row = ''.join(f'{norm[i, j] * 100:8.1f} ' for j in range(cm.shape[1]))
        print(f'  {NAMES[i]:<8s} {row}')


def main():
    args = parse_args()
    models = [load(p) for p in args.cm]
    labels = args.labels if args.labels else [
        os.path.basename(os.path.dirname(p)) or os.path.basename(p) for p in args.cm]
    if len(labels) != len(models):
        raise ValueError(f'--labels 给了 {len(labels)} 个，--cm 给了 {len(models)} 个，必须一一对应')

    # 所有模型必须在同一批场景、同样的顺序上评估，否则配对 bootstrap 无意义
    ref = models[0]['scenes']
    for lab, m in zip(labels[1:], models[1:]):
        if m['scenes'] != ref:
            raise ValueError(f'模型 {lab} 的场景列表与第一个不一致，无法配对比较。'
                             f'请用同一个 --val-list 重新 dump。')
    S = len(ref)
    C = models[0]['cm'].shape[1]
    print(f'{S} 个验证场景，{C} 类，{len(models)} 个模型：{", ".join(labels)}')

    # ---------------------------------------------------------------- #
    # 1. 全局指标 + 逐类 IoU（自检：应能复现 CLAUDE.md 实验表里的数字）
    # ---------------------------------------------------------------- #
    print(f'\n{"=" * 90}')
    print('  ① 全局 per-class IoU（自检：应与已报数字一致）')
    print(f'{"=" * 90}')
    print(f'  {"模型":<14s} {"mIoU(6类)":>10s} ' + ''.join(f'{n[:7]:>9s}' for n in NAMES))
    print('  ' + '-' * 86)
    globals_cm = []
    for lab, m in zip(labels, models):
        g = m['cm'].sum(axis=0)
        globals_cm.append(g)
        iou, miou = iou_from_cm(g, args.drop_empty)
        print(f'  {lab:<14s} {miou:10.4f} ' + ''.join(f'{v:9.3f}' for v in iou))

    print_confusion(globals_cm[0], f'{labels[0]} 的混淆矩阵')

    # ---------------------------------------------------------------- #
    # 2. 合并类 IoU —— 核心问题："找不到" 还是 "分不开"？
    # ---------------------------------------------------------------- #
    for spec in args.merge:
        groups, mnames = groups_for(spec, C)
        merged_idx = [int(x) for x in spec.split('+')]
        new_pos = next(i for i, g in enumerate(groups) if len(g) > 1)
        title = '+'.join(NAMES[m] for m in merged_idx)

        print(f'\n{"=" * 90}')
        print(f'  ② 合并 {title} 后的 IoU —— 是"找不到"还是"分不开"？')
        print(f'{"=" * 90}')
        hdr = f'  {"模型":<14s} ' + ''.join(f'{NAMES[m][:7]:>9s}' for m in merged_idx)
        hdr += f' | {"合并后":>9s} {"提升":>8s}'
        print(hdr)
        print('  ' + '-' * (len(hdr) - 2))
        # 每个模型各自比较：合并后的 IoU vs 它自己最好的那个单类 IoU
        gains = []          # (label, merged_iou, best_single_iou)
        for lab, g in zip(labels, globals_cm):
            iou, _ = iou_from_cm(g, args.drop_empty)
            merged_iou = iou_from_cm(merge_cm(g, groups), args.drop_empty)[0][new_pos]
            best_single = max(iou[m] for m in merged_idx)
            gains.append((lab, merged_iou, best_single))
            print(f'  {lab:<14s} ' + ''.join(f'{iou[m]:9.3f}' for m in merged_idx) +
                  f' | {merged_iou:9.3f} {merged_iou - best_single:+8.3f}')

        # 判读：取合并后表现最好的那个模型，看它【自己】涨了多少
        lab_b, merged_b, single_b = max(gains, key=lambda t: t[1])
        gain_b = merged_b - single_b
        print()
        if merged_b >= 0.45 and gain_b >= 0.15:
            print(f'  ➤ {lab_b}：合并后 IoU={merged_b:.3f}，比它自己最好的单类（{single_b:.3f}）'
                  f'高 {gain_b:+.3f}。')
            print(f'    结论：模型【能找到】{title} 区域，只是【分不开】这两类。')
            print('    → 换 backbone 不会有帮助。下一步跑 floe_separability.py，')
            print('      判断这两类在 80m 像元下的纹理里到底还有没有可分的信息。')
        elif merged_b < 0.35:
            print(f'  ➤ {lab_b}：合并后 IoU 仍只有 {merged_b:.3f}。')
            print(f'    结论：模型连"这里有 {title}"都没找准，是【检测】问题而非【区分】问题。')
            print('    → loss / 采样 / 上下文仍有空间，架构实验也还有意义。')
        else:
            print(f'  ➤ {lab_b}：合并后 IoU={merged_b:.3f}（单类最好 {single_b:.3f}，'
                  f'{gain_b:+.3f}），介于两者之间，两个因素都有份。')

    # ---------------------------------------------------------------- #
    # 3. 按【场景】bootstrap —— 排名有没有统计效力？
    # ---------------------------------------------------------------- #
    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    boot_idx = rng.integers(0, S, size=(B, S))          # 所有模型共用，实现配对

    all_cm = np.stack([m['cm'] for m in models], axis=0)   # (M, S, C, C)
    M = len(models)
    boot_miou = np.zeros((M, B), dtype=np.float64)
    for b in range(B):
        # 对每个模型，用同一组重采样的场景把 CM 加起来
        cm_b = all_cm[:, boot_idx[b]].sum(axis=1)          # (M, C, C)
        tp = np.diagonal(cm_b, axis1=1, axis2=2)           # (M, C)
        denom = cm_b.sum(axis=1) + cm_b.sum(axis=2) - tp   # (M, C)
        iou = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
        if args.drop_empty:
            present = denom > 0
            boot_miou[:, b] = np.where(
                present.any(axis=1),
                (iou * present).sum(axis=1) / np.maximum(present.sum(axis=1), 1),
                0.0)
        else:
            boot_miou[:, b] = iou.mean(axis=1)

    print(f'\n{"=" * 90}')
    print(f'  ③ 按【场景】bootstrap 的 mIoU 误差棒（B={B}，重采样单位=场景，不是像素）')
    print(f'{"=" * 90}')
    print(f'  {"模型":<14s} {"点估计":>9s} {"boot均值":>10s} {"std":>8s} {"95% CI":>20s}')
    print('  ' + '-' * 68)
    for i, (lab, g) in enumerate(zip(labels, globals_cm)):
        _, point = iou_from_cm(g, args.drop_empty)
        lo, hi = np.percentile(boot_miou[i], [2.5, 97.5])
        print(f'  {lab:<14s} {point:9.4f} {boot_miou[i].mean():10.4f} '
              f'{boot_miou[i].std():8.4f} {f"[{lo:.4f}, {hi:.4f}]":>20s}')

    if M >= 2:
        print(f'\n  配对比较 P(行模型 > 列模型)：同一组重采样场景喂给两个模型，'
              f'消掉场景难度的共同波动')
        print('  （0.5 = 完全分不出；>0.95 或 <0.05 才算有证据）\n')
        print('           ' + ''.join(f'{l[:9]:>11s}' for l in labels))
        for i, li in enumerate(labels):
            row = ''
            for j in range(M):
                if i == j:
                    row += f'{"-":>11s}'
                else:
                    p = float((boot_miou[i] > boot_miou[j]).mean())
                    mark = '*' if (p > 0.95 or p < 0.05) else ' '
                    row += f'{p:10.3f}{mark}'
            print(f'  {li[:9]:<9s}{row}')
        print('\n  * = 该配对差异在场景重采样下稳定（p>0.95）。没有 * 的配对，'
              '其 mIoU 差距无法与场景抽样噪声区分。')

    # ---------------------------------------------------------------- #
    # 4. 两模型逐类差异分解 —— 差异来自哪些类、哪些场景？
    #    （2026-07-18 加入，用于 b5 vs b2 的 medium +0.10 归因）
    # ---------------------------------------------------------------- #
    if M == 2:
        la, lb = labels
        cma, cmb = all_cm[0], all_cm[1]                    # (S, C, C)
        iou_a = iou_from_cm(globals_cm[0], args.drop_empty)[0]
        iou_b = iou_from_cm(globals_cm[1], args.drop_empty)[0]
        diff = iou_a - iou_b

        # 单类配对 bootstrap：复用 ③ 的 boot_idx（同一组场景喂两个模型）
        boot_cls = np.zeros((2, B, C), dtype=np.float64)
        for k, cm_all in enumerate((cma, cmb)):
            cm_b = cm_all[boot_idx].sum(axis=1)            # (B, C, C)，~60MB 中间量
            tp = np.diagonal(cm_b, axis1=1, axis2=2)
            denom = cm_b.sum(axis=1) + cm_b.sum(axis=2) - tp
            boot_cls[k] = np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)

        print(f'\n{"=" * 90}')
        print(f'  ④ 逐类差异分解：{la} vs {lb}（配对 bootstrap 同 ③）')
        print(f'{"=" * 90}')
        print(f'  {"类别":<8s} {la[:10]:>10s} {lb[:10]:>10s} {"diff":>8s} '
              f'{f"P({la[:6]}>{lb[:6]})":>18s}')
        print('  ' + '-' * 60)
        for c in range(C):
            p = float((boot_cls[0, :, c] > boot_cls[1, :, c]).mean())
            # 差异实际为零（如两模型某空类恒同为 0）时严格 > 会给出 p=0，
            # 不应据此标"显著"——加 |diff| 门槛抑制。
            mark = '*' if (p > 0.95 or p < 0.05) and abs(diff[c]) > 1e-9 else ' '
            print(f'  {NAMES[c]:<8s} {iou_a[c]:10.3f} {iou_b[c]:10.3f} {diff[c]:+8.3f} '
                  f'{p:17.3f}{mark}')

        # 焦点类的逐景分解 + leave-one-out
        if args.focus_class:
            focus = []
            for x in args.focus_class:
                if x in NAMES:
                    focus.append(NAMES.index(x))
                    continue
                try:
                    ci = int(x)
                except ValueError:
                    raise SystemExit(f'--focus-class "{x}" 无效：须是类名（{", ".join(NAMES)}）'
                                     f'或 0..{C - 1} 的下标')
                if not 0 <= ci < C:
                    raise SystemExit(f'--focus-class 下标 {ci} 越界：须在 0..{C - 1}')
                focus.append(ci)
        else:
            focus = [int(i) for i in np.argsort(-np.abs(diff))[:2]]
        for c in focus:
            true_px = cma[:, c, :].sum(axis=1)             # 每景该类 GT 像素
            # 两个 npz 的 GT 像素应逐景一致（同 val-list 同口径）；不一致说明 dump 时
            # 分辨率/ignore 处理不同，此时"GT像素"列只代表第一个模型。
            if not np.allclose(true_px, cmb[:, c, :].sum(axis=1)):
                print(f'  ⚠ 两个 npz 在 [{NAMES[c]}] 的逐景 GT 像素数不一致——两次 dump 的'
                      f'分辨率/ignore 口径可能不同，下表 GT像素列仅为 {la} 的，解读需谨慎。')
            order = np.argsort(-true_px)
            base_diff = diff[c]
            print(f'\n  —— 焦点类 [{NAMES[c]}]：全局 diff {base_diff:+.3f}，'
                  f'GT 像素分布在 {int((true_px > 0).sum())}/{S} 景 ——')
            print(f'  {"场景":<42s} {"GT像素":>10s} {la[:8]:>8s} {lb[:8]:>8s} '
                  f'{"景diff":>8s} {"剔除后全局diff":>14s}')
            print('  ' + '-' * 96)
            loo_by_scene = {}
            for s in order:
                if true_px[s] == 0:
                    continue
                sa = iou_from_cm(cma[s])[0][c]
                sb = iou_from_cm(cmb[s])[0][c]
                loo = (iou_from_cm(globals_cm[0] - cma[s])[0][c]
                       - iou_from_cm(globals_cm[1] - cmb[s])[0][c])
                loo_by_scene[int(s)] = loo
                print(f'  {ref[s][:42]:<42s} {int(true_px[s]):>10d} {sa:8.3f} {sb:8.3f} '
                      f'{sa - sb:+8.3f} {loo:+14.3f}')
            if not loo_by_scene or base_diff == 0:
                print(f'  ➤ 全局 diff 为 0 或无含该类的场景，无单景归因可做。')
                continue
            # 判读：取对全局差影响最大的景（|base_diff - loo| 最大），不是像素最多的景——
            # 差异也可能由一个 GT 中等、但 IoU 摆动极大的景主导。
            s_top = max(loo_by_scene, key=lambda s: abs(base_diff - loo_by_scene[s]))
            loo_top = loo_by_scene[s_top]
            top_name = ref[s_top][:42]
            if loo_top != 0 and np.sign(loo_top) != np.sign(base_diff):
                # 剔除后差距变号：说明整个全局差都由该景撑起（甚至被其掩盖了反向差距）
                print(f'  ➤ 剔除 {top_name} 后 diff 从 {base_diff:+.3f} 变号为 {loo_top:+.3f}：'
                      f'差异完全由该景主导，全局结论不可靠。')
            elif abs(loo_top) < 0.4 * abs(base_diff):
                print(f'  ➤ 剔除影响最大的景（{top_name}）后 diff 从 {base_diff:+.3f} '
                      f'缩到 {loo_top:+.3f}：差异主要由单景驱动，谨慎解读。')
            else:
                print(f'  ➤ 剔除影响最大的景（{top_name}）后 diff 仍有 {loo_top:+.3f}'
                      f'（原 {base_diff:+.3f}，同向）：提升是跨场景的，不是单景假象。')

    print(f'\n{"=" * 90}')
    print('  提示：如果连排名第一和最后的模型都分不出（P 落在 0.05~0.95），')
    print('        说明 val 场景数不足以支撑架构横比 —— 先重划数据，再谈模型。')
    print(f'{"=" * 90}')


if __name__ == '__main__':
    main()
