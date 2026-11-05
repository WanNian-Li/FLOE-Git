#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全库辐射侧写（2026-08-08，CPU-only）：每景抽若干 tile 算 HH（nersc_sar_primary）
有效像元的 mean/std（z-unit），回答"暗景/异常对比度景在 train 里是否欠代表"。

背景：08-04 逐景分解把 medium 失效点名到暗景（HH 均值比优等景低 0.7~0.9 z）与异常
对比度景（20240624 HH std 1.32 vs 典型 0.55-0.8）；补数据前先确认这些条件在 train
里的覆盖密度。tile 坐标来自 tile 索引（chunk 对齐、只挑 label_valid_ratio 高的），
每景 ~4 tile，与 08-04 的 6-chunk 抽样同族（±0.1 量级噪声）。

用法: python -m tools.analysis.radiometric_profile [--tiles-per-scene 4]
"""
import argparse
import json
import os
import os.path as osp

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '2')

import h5py
import numpy as np

DATA_DIR = '/root/autodl-tmp/FLOE_datanc'
INDEX = '/root/autodl-tmp/floe_tile_index_417.npz'
HH_VAR = 'nersc_sar_primary'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tiles-per-scene', type=int, default=4)
    args = ap.parse_args()

    with open('datalists/train_417.json') as f:
        train = set(json.load(f))

    idx = np.load(INDEX, allow_pickle=False)
    files = [str(x) for x in idx['files']]
    sid, row, col = idx['scene_id'], idx['row'], idx['col']
    lvr, ts = idx['label_valid_ratio'], int(idx['tile_size'])

    stats = []   # (name, split, hh_mean, hh_std)
    for i, name in enumerate(files):
        m = sid == i
        if not m.any():
            continue
        order = np.argsort(-lvr[m])[:args.tiles_per_scene]
        rows, cols = row[m][order], col[m][order]
        vals = []
        with h5py.File(osp.join(DATA_DIR, name), 'r') as f:
            hh, gv = f[HH_VAR], f['global_valid_mask']
            for r, c in zip(rows, cols):
                x = hh[r:r + ts, c:c + ts]
                v = gv[r:r + ts, c:c + ts] == 1
                if v.any():
                    vals.append(x[v])
        if not vals:
            continue
        allv = np.concatenate(vals).astype(np.float64)
        stats.append((name, 'train' if name in train else 'val',
                      float(allv.mean()), float(allv.std())))
        if (i + 1) % 100 == 0:
            print(f'  ... {i + 1}/{len(files)}')

    means = np.array([s[2] for s in stats])
    stds = np.array([s[3] for s in stats])
    splits = np.array([s[1] for s in stats])

    print(f'\n# 全库 HH 辐射侧写（{len(stats)} 景，每景 {args.tiles_per_scene} tile 抽样）')
    for tag in ('train', 'val'):
        mm = means[splits == tag]
        print(f'  {tag}: mean 分位数 p5={np.percentile(mm, 5):+.2f} p25={np.percentile(mm, 25):+.2f} '
              f'p50={np.percentile(mm, 50):+.2f} p75={np.percentile(mm, 75):+.2f} p95={np.percentile(mm, 95):+.2f}')

    print('\n## 暗端覆盖（08-04 崩溃暗景在 HH 均值 ≈ -0.7 z 以下）')
    for th in (-0.4, -0.6, -0.8, -1.0):
        n_tr = int(((means < th) & (splits == 'train')).sum())
        n_va = int(((means < th) & (splits == 'val')).sum())
        print(f'  HH mean < {th:+.1f}: train {n_tr} 景（{n_tr / (splits == "train").sum() * 100:.1f}%）'
              f' / val {n_va} 景')

    print('\n## 异常对比度覆盖（20240624 的 HH std=1.32；典型 0.55-0.8）')
    for th in (1.0, 1.2):
        n_tr = int(((stds > th) & (splits == 'train')).sum())
        n_va = int(((stds > th) & (splits == 'val')).sum())
        print(f'  HH std > {th:.1f}: train {n_tr} 景（{n_tr / (splits == "train").sum() * 100:.1f}%）'
              f' / val {n_va} 景')

    print('\n## train 最暗 10 景（补数据决策参考）')
    order = np.argsort(means)
    shown = 0
    for j in order:
        if stats[j][1] != 'train':
            continue
        print(f'  HHmean {means[j]:+.2f} std {stds[j]:.2f}  {stats[j][0]}')
        shown += 1
        if shown >= 10:
            break

    out = 'docs/_radiometric_profile_417.csv'
    with open(out, 'w') as f:
        f.write('scene,split,hh_mean,hh_std\n')
        for name, sp, m, s in stats:
            f.write(f'{name},{sp},{m:.4f},{s:.4f}\n')
    print(f'\n逐景 CSV -> {out}')


if __name__ == '__main__':
    main()
