#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""临时诊断：验证「按需读」与整景读在【会被前向的 tile 内】逐位一致。

对若干真实验证景，分别用 val_ondemand_read=False / True 走一遍 dataset，比较：
  - y（标签）           : 必须【逐位相同】（两条路都整景读标签）
  - cfv_masks / tfv_mask: 必须【逐位相同】
  - x（输入）           : 在「有有效标签的 tile」内必须【逐位相同】；这些 tile 之外
                          按需读留 0，本就不会进网络（functions.compute_valid_tile_mask）
并报告两条路径各自的耗时。

用法：python -m tools.checks.ondemand_read --config <cfg> [--scenes 3]
"""
import argparse
import json
import time

import numpy as np
import torch
from mmcv import Config

from floe.loaders import AI4ArcticChallengeTestDataset, get_variable_options, limit_torch_threads
from floe.functions import compute_valid_tile_mask


def main():
    limit_torch_threads()
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--scenes', type=int, default=3)
    args = p.parse_args()

    opts = get_variable_options(Config.fromfile(args.config).train_options)
    val_list = opts.get('path_to_env', './') + opts['val_path']
    files = json.loads(open(val_list).read())
    step = max(1, len(files) // args.scenes)
    files = files[::step][:args.scenes]

    ps = opts['patch_size']
    charts = opts['charts']

    # 按需读是构造参数级的 opt-in（config 的 val_ondemand_read 只是全局 kill switch）
    ds_full = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train',
                                            ondemand_read=False)
    ds_od = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train',
                                          ondemand_read=True)
    from floe.loaders import val_ondemand_ok
    ok, why = val_ondemand_ok(opts)
    print(f'val_ondemand_ok(config) = {ok}（{why}）\n')

    t_full = t_od = 0.0
    for i, fn in enumerate(files):
        t = time.time(); xf, yf, cf, tf, nf, sf = ds_full[i]; dt_f = time.time() - t
        t = time.time(); xo, yo, co, to, no, so = ds_od[i];   dt_o = time.time() - t
        t_full += dt_f; t_od += dt_o

        assert nf == no and tuple(sf) == tuple(so), f'name/original_size 不一致: {sf} vs {so}'
        assert xf.shape == xo.shape, f'x 形状不一致 {xf.shape} vs {xo.shape}'
        # dataset 直接取时 cfv/tfv 还是 numpy（走 DataLoader 才会被 default_convert 转成
        # tensor），统一用 np.asarray 比较。
        for c in charts:
            assert np.array_equal(yf[c], yo[c]), f'{fn}: 标签 {c} 不一致'
            assert np.array_equal(np.asarray(cf[c]), np.asarray(co[c])), \
                f'{fn}: cfv_mask {c} 不一致'
        assert np.array_equal(np.asarray(tf), np.asarray(to)), f'{fn}: tfv_mask 不一致'

        # x：在「会被前向的 tile」内必须逐位相同
        valid = None
        for c in charts:
            v = ~cf[c]
            valid = v if valid is None else (valid | v)
        tm = compute_valid_tile_mask(valid, ps)
        H, W = xf.shape[-2], xf.shape[-1]
        n_bad, n_tiles_checked, n_px = 0, 0, 0
        for (ti, tj) in torch.nonzero(tm).tolist():
            y0, x0 = ti * ps, tj * ps
            y1, x1 = min(y0 + ps, H), min(x0 + ps, W)
            a = xf[0, :, y0:y1, x0:x1]
            b = xo[0, :, y0:y1, x0:x1]
            n_tiles_checked += 1
            n_px += a.numel()
            if not torch.equal(a, b):
                n_bad += 1
        read_frac = float((xo[0] != 0).float().mean())
        print(f'[{i+1}/{len(files)}] {fn[:46]:<46s} {H}x{W}  '
              f'整景 {dt_f:6.2f}s  按需 {dt_o:6.2f}s  (×{dt_o/dt_f:.2f})  '
              f'有效 tile {n_tiles_checked}/{tm.numel()}  不一致 tile {n_bad}  '
              f'按需读非零占比 {read_frac*100:.1f}%')
        assert n_bad == 0, f'{fn}: {n_bad} 个有效 tile 的输入像素不一致！'
        del xf, xo, yf, yo

    n = len(files)
    print(f'\n全部通过。平均 整景 {t_full/n:.2f}s/景  按需 {t_od/n:.2f}s/景  '
          f'(×{t_od/t_full:.2f}，外推 42 景 {t_full/n*42:.0f}s → {t_od/n*42:.0f}s)')


if __name__ == '__main__':
    main()
