#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
扫描所有 .nc 场景，建立 chunk 对齐的 tile 索引表，供训练时做加权采样。

背景（2026-07-09）：.nc 底层 HDF5 的二维变量是 chunks=(512,512)+gzip4 存储的，
可以只解压覆盖到的 chunk。实测读一个 chunk 对齐的 512x512 tile（8 个变量）约 27ms，
而当前 loaders.py 走的 `xarray.to_array()` 整景解码约 9.2s / 景（输出 2.8~3.1GB
float32）。同时全图约 67% 的 512 tile 是"全无效"（黑边 + 陆地，标签 255，在 loss 里
被 ignore_index 屏蔽，梯度贡献恒为 0），解码它们纯属浪费。

本脚本只读每景的 FLOE + global_valid_mask 两个 uint8 通道（约 0.29s / 景），按 512
对齐网格枚举 tile，剔除全无效 tile，为剩下的 tile 记录有效像元比例和类别直方图。
训练时据此直接加权采样 tile，把原来的"随机裁 + 概率拒绝 + 重试"换成零拒绝的索引采样。

用法：
    python -m tools.data.scan_tiles --data-dir /root/autodl-tmp/FLOE_datanc \
        --out /root/autodl-tmp/floe_tile_index.npz --workers 16

输出 .npz（几 MB）：
    files             (S,)   str    场景文件名，与 scene_id 对应
    scene_id          (N,)   int16
    row, col          (N,)   int32  tile 左上角在原景中的像素坐标（512 的整数倍）
    valid_ratio       (N,)   float32  tile 内 global_valid_mask==1 的比例
    label_valid_ratio (N,)   float32  tile 内 FLOE != 255 的比例
    hist              (N, 6) int32    tile 内 FLOE 各类别（0..5）的像素数
    tile_size         ()     int32
"""
import argparse
import os
import os.path as osp
import time
from multiprocessing import Pool

import h5py
import numpy as np

N_CLASSES = 6          # FLOE_LOOKUP: 0=水 1=饼状冰 2=小 3=中 4=大 5=巨
IGNORE_VAL = 255
# 数据集标签 6(冰山) 在 loader 里被重映射为 255；扫描时同样并入 ignore 统计，
# 否则索引表里的类别直方图会和训练时实际看到的标签不一致。
ICEBERG_VAL = 6


def parse_args():
    p = argparse.ArgumentParser(description='Build a chunk-aligned tile index for FLOE scenes.')
    p.add_argument('--data-dir', type=str, default='/root/autodl-tmp/FLOE_datanc')
    p.add_argument('--out', type=str, default='/root/autodl-tmp/floe_tile_index.npz')
    p.add_argument('--tile-size', type=int, default=512,
                   help='必须等于 .nc 的 HDF5 chunk 边长，否则读 tile 会跨 chunk、变慢')
    p.add_argument('--patch-size', type=int, default=256,
                   help='训练 patch 边长，必须与 config 的 patch_size 一致（决定裁剪覆盖核）')
    p.add_argument('--workers', type=int, default=16)
    p.add_argument('--limit', type=int, default=None, help='只扫前 N 景（调试用）')
    return p.parse_args()


def crop_coverage_kernel(tile, patch):
    """tile 内每个像素被"随机裁的 patch"覆盖到的次数（未归一化）。

    训练时 patch 原点均匀取自 [0, tile-patch] 的整数格点，所以 tile 中心的像素比边缘的
    像素更容易被采到（一维上是个梯形/三角核）。不把这个权重算进直方图的话，索引里的
    类别占比描述的是"整个 tile"，而训练实际看到的是"按此核加权后的 tile"——两者在
    空间分布不均匀的类别上会系统性错开（实测 big/vast 各被高估约 0.013）。

    一维：位置 u 被覆盖的原点数 = min(u, tile-patch) - max(0, u-patch+1) + 1
    二维取外积。
    """
    u = np.arange(tile)
    k1 = np.minimum(u, tile - patch) - np.maximum(0, u - patch + 1) + 1
    k1 = k1.astype(np.float64)
    return np.outer(k1, k1)


def scan_one(job):
    """扫描单景，返回该景所有"非全无效" tile 的记录 + 全景像素直方图。"""
    scene_id, path, tile, kernel = job
    with h5py.File(path, 'r') as f:
        lab = f['FLOE'][:]
        vmask = f['global_valid_mask'][:]

    # 全景像素直方图（把 >=6 的标签，含冰山 6 和 mask 255，一并算作 ignore）
    flat = lab.ravel()
    px = np.bincount(flat, minlength=256).astype(np.int64)
    px_cls = px[:N_CLASSES].copy()
    px_ignore = int(px.sum() - px_cls.sum())

    kflat = kernel.ravel()
    H, W = lab.shape
    rows, cols, vrs, lvrs, hists, hists_w = [], [], [], [], [], []
    n_tiles_total = 0
    # 只枚举完整的对齐 tile；边缘不足 512 的残条丢弃（几乎全是黑边，且不对齐会跨 chunk）
    for r in range(0, H - tile + 1, tile):
        for c in range(0, W - tile + 1, tile):
            n_tiles_total += 1
            t = lab[r:r + tile, c:c + tile]
            tflat = t.ravel()
            h = np.bincount(tflat, minlength=256)
            hc = h[:N_CLASSES]
            n_label_valid = int(hc.sum())
            # 全无效 tile：没有任何一个可参与 loss 的像元。直接丢弃。
            if n_label_valid == 0:
                continue
            # 裁剪覆盖核加权的直方图：训练时实际期望看到的各类像素数（相对量）。
            hw = np.bincount(tflat, weights=kflat, minlength=256)[:N_CLASSES]
            rows.append(r)
            cols.append(c)
            lvrs.append(n_label_valid / float(tile * tile))
            vrs.append(float(vmask[r:r + tile, c:c + tile].mean()))
            hists.append(hc.astype(np.int32))
            hists_w.append(hw.astype(np.float32))

    n_kept = len(rows)
    return dict(
        scene_id=np.full(n_kept, scene_id, dtype=np.int16),
        row=np.asarray(rows, dtype=np.int32),
        col=np.asarray(cols, dtype=np.int32),
        valid_ratio=np.asarray(vrs, dtype=np.float32),
        label_valid_ratio=np.asarray(lvrs, dtype=np.float32),
        hist=(np.stack(hists) if n_kept else np.zeros((0, N_CLASSES), np.int32)),
        hist_w=(np.stack(hists_w) if n_kept else np.zeros((0, N_CLASSES), np.float32)),
        px_cls=px_cls,
        px_ignore=px_ignore,
        n_tiles_total=n_tiles_total,
        shape=(H, W),
    )


def main():
    args = parse_args()
    files = sorted(f for f in os.listdir(args.data_dir) if f.endswith('.nc'))
    if args.limit:
        files = files[:args.limit]
    print(f'Found {len(files)} .nc scenes in {args.data_dir}')

    # 校验 tile_size 与 HDF5 chunk 对齐——不对齐的话读 tile 会跨 chunk，速度打折
    with h5py.File(osp.join(args.data_dir, files[0]), 'r') as f:
        ch = f['FLOE'].chunks
    if ch is None or ch[0] != args.tile_size or ch[1] != args.tile_size:
        print(f'[WARN] FLOE chunks={ch} != tile_size={args.tile_size}; '
              f'reads will straddle chunks and be slower than necessary.')

    if args.patch_size > args.tile_size:
        raise ValueError(f'patch_size={args.patch_size} 不能大于 tile_size={args.tile_size}')
    kernel = crop_coverage_kernel(args.tile_size, args.patch_size)

    jobs = [(i, osp.join(args.data_dir, f), args.tile_size, kernel) for i, f in enumerate(files)]
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(scan_one, jobs), 1):
            results.append(res)
            if i % 40 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} scenes  ({time.time() - t0:.0f}s)')
    print(f'Scan done in {time.time() - t0:.0f}s')

    out = {k: np.concatenate([r[k] for r in results])
           for k in ('scene_id', 'row', 'col', 'valid_ratio', 'label_valid_ratio')}
    out['hist'] = np.concatenate([r['hist'] for r in results], axis=0)
    out['hist_w'] = np.concatenate([r['hist_w'] for r in results], axis=0)
    # imap_unordered 打乱了顺序，按 (scene_id, row, col) 排回确定序，便于复现和调试
    order = np.lexsort((out['col'], out['row'], out['scene_id']))
    for k in list(out):
        out[k] = out[k][order]

    out['files'] = np.array(files)
    out['tile_size'] = np.int32(args.tile_size)
    # loaders.py 会校验：hist_w 是按 patch_size 的裁剪覆盖核加权的，换 patch_size 必须重扫。
    out['patch_size'] = np.int32(args.patch_size)

    np.savez_compressed(args.out, **out)
    sz = os.path.getsize(args.out) / 1e6

    # ---------------- 统计报告 ----------------
    px_cls = np.sum([r['px_cls'] for r in results], axis=0)
    px_ignore = sum(r['px_ignore'] for r in results)
    px_total = px_cls.sum() + px_ignore
    n_tiles_total = sum(r['n_tiles_total'] for r in results)
    n_kept = len(out['row'])

    hist = out['hist']
    tile_valid = hist.sum(axis=1)
    # 冰类含类 1（ice cake，2026-07-16 数据集扩充后不再为空），从 1 起切
    n_ice = int((hist[:, 1:].sum(axis=1) > 0).sum())
    n_water_only = n_kept - n_ice

    names = ['water', 'cake', 'small', 'medium', 'big', 'vast']
    print()
    print(f'Wrote {args.out}  ({sz:.1f} MB)')
    print()
    print('--- PIXEL level (all scenes) ---')
    for i, n in enumerate(names):
        print(f'  {n:<9s} {px_cls[i] / px_total * 100:6.2f}%  of image')
    print(f'  {"255/ignore":<9s} {px_ignore / px_total * 100:6.2f}%  of image')
    valid = px_cls.sum()
    print(f'  -> valid pixels = {valid / px_total * 100:.2f}% of image')
    if valid:
        print(f'  -> among valid: water {px_cls[0] / valid * 100:.2f}%, '
              f'ice(1-5) {px_cls[1:].sum() / valid * 100:.2f}%')
    print()
    print(f'--- TILE level ({args.tile_size}x{args.tile_size} aligned) ---')
    print(f'  total tiles enumerated : {n_tiles_total}')
    print(f'  all-invalid (dropped)  : {n_tiles_total - n_kept:7d}  {(n_tiles_total - n_kept) / n_tiles_total * 100:5.1f}%')
    print(f'  kept                   : {n_kept:7d}  {n_kept / n_tiles_total * 100:5.1f}%')
    print(f'    of which contain ice : {n_ice:7d}  {n_ice / n_tiles_total * 100:5.1f}% of all tiles')
    print(f'    water/ignore only    : {n_water_only:7d}  {n_water_only / n_tiles_total * 100:5.1f}% of all tiles')
    print()
    print('  per-class tile coverage (tiles containing >=1 px of class):')
    for i, n in enumerate(names):
        c = int((hist[:, i] > 0).sum())
        print(f'    {n:<9s} {c:7d} tiles  ({c / max(n_kept, 1) * 100:5.1f}% of kept)')
    print()
    print(f'  label_valid_ratio of kept tiles: '
          f'mean={out["label_valid_ratio"].mean():.3f} '
          f'median={np.median(out["label_valid_ratio"]):.3f} '
          f'>=0.5: {(out["label_valid_ratio"] >= 0.5).sum()} tiles')
    print(f'  valid_ratio (global_valid_mask) of kept tiles: '
          f'mean={out["valid_ratio"].mean():.3f} '
          f'>=0.5: {(out["valid_ratio"] >= 0.5).sum()} tiles')


if __name__ == '__main__':
    main()
