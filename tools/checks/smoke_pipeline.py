#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FLOE-SOTA config 冒烟测试：在【启动训练之前】验证一个模型 config 能端到端跑通。

为什么必须先冒烟（2026-07-10 教训）：多模型 benchmark 里每个 config 都是新架构 + 新
输入通道口径，任何一处对不上（通道数、输出不是 {chart:tensor} dict、n_classes 传错）
都会在训练启动后几分钟才崩，浪费排队时间。本脚本一次性验证：

  1. config 能加载 + 与 _sota_base / base 正确合并；
  2. get_model 能实例化，参数量、期望输入通道数打印出来；
  3. 真实 tile 采样出的一个 batch，通道数与模型期望一致；
  4. 前向 + 反向 + AMP（GradScaler + clip_grad_norm_，与 quickstart 完全一致）loss 有限；
  5. 模型输出是 {'FLOE': (B, 6, H, W)} dict；
  6. 验证推理路径（fast_tiled_val_inference → class_decider → 混淆矩阵累加）跑通、形状对。

设备自适应：有 GPU 就用 GPU（顺带给时序），无卡模式下用 CPU（只验证正确性，较慢但够用）。
所以本脚本在无卡模式下也能跑，用于开卡前先把所有 config 过一遍。

用法：
    python -m tools.checks.smoke_pipeline configs/FLOE/sota_deeplabv3.py
    python -m tools.checks.smoke_pipeline configs/FLOE/sota_deeplabv3.py configs/FLOE/sota_segformer.py configs/FLOE/sota_segnet.py
    python -m tools.checks.smoke_pipeline            # 不给参数则默认测这三个
"""
import contextlib
import copy
import json
import math
import sys
import time
import warnings

import numpy as np
import torch
from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeDataset,
                     limit_torch_threads)
from floe.functions import (get_model, get_optimizer, get_loss, fast_tiled_val_inference,
                       fast_tiled_val_predict, compute_valid_tile_mask,
                       batched_slide_inference, class_decider,
                       new_confusion_matrix, accumulate_confusion_matrix)

# 当前批次（2026-07-31 T 批：换 loss / 加辅助任务 / 换架构族）。
# 三枪各有专属风险点，冒烟必须逐个看日志而不只看 PASS：
#   focal          : FocalLoss 的 get_loss 分派 + val loss 按批聚合兼容性（alpha 权重不
#                    进分母，_loss_is_mean_over_valid 应判 True → val loss 不该是 nan）；
#   mtsic          : aux_charts 机制全链路（8 通道输入 + SIC 标签读取、模型双头输出、
#                    验证只跑 FLOE 单任务）。⚠️ 开跑前必须先过 _audit_sic_vars.py，
#                    确认 417 景里 SIC 变量齐全且取值口径正确；
#   upernet_swinv2 : 触发第 [7b] 步 fp32 仲裁的那一枪（fp16 下门控差异 margin 1.24e-2
#                    略超阈值，[7a] 已证索引无 bug）——CPU/fp32 仲裁通过即放行。
# 上一批（2026-07-30 S 批）：sota_upernet_swinv2_in1k_dlr_417 / sota_dpt_dino_dlr_417 /
# sota_upernet_mitb2_in1k_dlr_417；再上一批（2026-07-10 从零 benchmark）：
# sota_deeplabv3 / sota_segformer / sota_segnet / sota_swin。都可作命令行参数回归。
DEFAULT_CONFIGS = ['configs/FLOE/sota_segformer_in1k_dlr_417.py', 'configs/FLOE/pgsg.py']


def smoke_one(cfg_path, device, cuda):
    print('=' * 72, flush=True)
    print('SMOKE:', cfg_path, flush=True)
    print('=' * 72, flush=True)

    opts = get_variable_options(Config.fromfile(cfg_path).train_options)
    opts['diag_log_every_batches'] = 0
    # 冒烟只需少量 CPU 读取；无卡模式 CPU 很弱，别开一堆 worker。
    # num_workers_val 也置 0：swin 的 batched_slide_inference 内部会起 DataLoader。
    opts['num_workers'] = 0
    opts['num_workers_val'] = 0

    # 验证侧 options：辅助任务（aux_charts）是 train-only 的，验证只跑 target chart。
    # 无 aux_charts 时 opts_val 就是 opts 本身（同一对象），不带 aux 的 config 行为不变。
    opts_val = val_scoped_options(opts)

    target = opts.get('target_chart', 'FLOE')
    n_cls = opts['n_classes'][target]
    n_tex = 0
    if opts.get('texture_channels', False):
        from floe.loaders import N_TEXTURE_CHANNELS  # B1：训练 _augment_x 追加 N 个纹理通道
        n_tex = N_TEXTURE_CHANNELS
    exp_in_ch = (len(opts['train_variables'])
                 + (2 if opts.get('month_encoding', False) else 0)
                 + (1 if opts.get('pol_ratio_channel', False) else 0)
                 + n_tex)
    print(f'model_selection = {opts["model_selection"]} | target_chart = {target} | '
          f'n_classes = {n_cls} | expected in_channels = {exp_in_ch} '
          f'(pol_ratio={opts.get("pol_ratio_channel", False)}, '
          f'texture={opts.get("texture_channels", False)})', flush=True)

    # --- 2. 建模型 ---
    net = get_model(opts, device)
    n_param = sum(p.numel() for p in net.parameters())
    print(f'[2] model built OK | params = {n_param / 1e6:.2f} M', flush=True)

    # --- 3+4+5. 真实 batch 前向/反向 ---
    net.train()
    optimizer = get_optimizer(opts, net)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    loss_fn = {c: get_loss(opts['chart_loss'][c]['type'], chart=c, **opts['chart_loss'][c]).to(device)
               for c in opts['charts']}
    grad_clip = opts.get('grad_clip_max_norm', 5.0)

    train_list = json.load(open(opts['path_to_env'] + opts['train_list_path']))
    files = [f if f.endswith('.nc') else f + '.nc' for f in train_list]
    ds = AI4ArcticChallengeDataset(opts, files, do_transform=True)

    autocast = (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext())
    for step in range(2):
        bx, by = ds[step]
        assert bx.shape[1] == exp_in_ch, \
            f'通道数不符：batch 有 {bx.shape[1]} 通道，模型期望 {exp_in_ch}'
        bx = bx.to(device)
        with autocast:
            out = net(bx)
            assert isinstance(out, dict) and target in out, \
                f'模型输出必须是含 {target!r} 键的 dict，实际得到 {type(out)}'
            # 多任务（aux_charts）时每个 chart 都要有自己的头，逐个查形状与类数。
            for c in opts['charts']:
                assert c in out, f'模型输出缺少 chart {c!r}（charts={opts["charts"]}）'
                assert out[c].shape[1] == opts['n_classes'][c], \
                    f'{c} 输出类别数 {out[c].shape[1]} != n_classes {opts["n_classes"][c]}'
                assert out[c].shape[-2:] == bx.shape[-2:], \
                    f'{c} 输出空间尺寸 {tuple(out[c].shape[-2:])} != 输入 {tuple(bx.shape[-2:])}'
            loss = sum(w * loss_fn[c](out[c], by[c].to(device))
                       for c, w in zip(opts['charts'], opts['task_weights']))
        assert torch.isfinite(loss), f'loss 非有限值：{loss.item()}'
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip, error_if_nonfinite=False)
        scaler.step(optimizer)
        scaler.update()
    print(f'[3-5] train step OK | batch_x {tuple(bx.shape)} -> out[{target}] '
          f'{tuple(out[target].shape)} | loss = {loss.item():.4f} (finite)', flush=True)

    # --- 6. 验证推理路径（合成小场景，避免在 CPU 上跑整景）---
    # 走与 quickstart 完全一致的分派：'swin' 用 batched_slide_inference（滑窗），
    # 其余大图用 fast_tiled_val_inference。这样 swin 的真实验证路径也被覆盖到。
    # ⚠️ 合成景仅 700×900，不足以暴露 batched_slide 在最大验证景(176 Mpx)上的显存峰值——
    # swin 的显存风险点是首次真实验证，本冒烟测不出，见 sota_swin.py 顶部说明。
    net.eval()
    H, W = 700, 900   # 会被 pad 到 256 的倍数
    # 验证合成输入用【基础通道数】(不含纹理)：真实 prep_scene 整景是基础通道数，B1 纹理由
    # fast_tiled_val_inference/predict 里 per-tile 现算补上。若用 exp_in_ch(含纹理)会被验证
    # 函数再补一次纹理→双计(12+4=16)而崩——这正是修此测试脚本口径的原因。
    val_in_ch = exp_in_ch - n_tex
    fake_x = torch.randn(1, val_in_ch, H, W, device=device)
    fake_true = torch.randint(0, n_cls, (H, W), device=device)
    cm = new_confusion_matrix(n_cls, device)
    with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
        if opts['model_selection'] == 'swin':
            vout = batched_slide_inference(fake_x, net, opts_val, 'val')
            val_path = 'batched_slide_inference'
        else:
            vout = fast_tiled_val_inference(fake_x, net, opts_val)
            val_path = 'fast_tiled_val_inference'
        assert vout[target].shape == (1, n_cls, H, W), \
            f'验证输出形状 {tuple(vout[target].shape)} != (1,{n_cls},{H},{W})'
        pred = class_decider(vout[target], opts_val, target)
    accumulate_confusion_matrix(cm, fake_true.reshape(-1), pred.reshape(-1), n_cls)
    assert int(cm.sum().item()) == H * W, f'混淆矩阵像素数 {int(cm.sum())} != {H * W}'
    print(f'[6] val inference OK | {val_path} -> {tuple(vout[target].shape)} | '
          f'confmat total = {int(cm.sum())}', flush=True)

    # --- 7. tile 门控等价性（2026-07-25 新增）---
    # 造一个「大片 255」的标签，使一部分 tile 完全无效，然后把两条路径对拍：
    #   旧：fast_tiled_val_inference（全量前向）+ class_decider
    #   新：fast_tiled_val_predict（门控 + 流式 argmax）
    #
    # 分两级断言，因为两者的性质不同：
    #   7a) gate=False：tile 顺序和分批与旧路径【完全一致】，只是换了取 tile / 写回的
    #       写法 → 必须【全图逐像素相同】。这一项挂了就是索引 bug。
    #   7b) gate=True：跳过 tile 会改变每批的组成，cuDNN 可能因此选到不同算法，fp16 下
    #       末位差 1 ULP，恰好碰上 argmax 平局就会翻。所以只要求「差异比例极小、且差异
    #       像素处 top1-top2 的 logit 间隔接近 0（= 确实是平局）」。实测 b5 在随机噪声
    #       输入上翻 0.03%、间隔 max 1.2e-4（全体中位数 2.5e-2）；真实数据上间隔更大、
    #       翻得更少。同样的抖动在今天改 val_batch_size 时就存在。
    # swin 走滑窗、不经过 tile 路径，跳过本项。
    if opts['model_selection'] != 'swin':
        ps = opts['patch_size']
        lab = torch.full((H, W), 255, dtype=torch.long)
        # 左上一块（整 tile）+ 右下角一块（同时落在【右、下都不足 patch_size】的残缺
        # tile 里），后者专门覆盖门控路径下的边缘补齐分支。H=700/W=900、ps=256 时
        # tile 网格是 3×4，这样恰好留下 5 个有效 tile、7 个全无效 tile。
        lab[0:500, 0:400] = torch.randint(0, n_cls, (500, 400))
        lab[600:700, 800:900] = torch.randint(0, n_cls, (100, 100))
        valid = lab != 255
        tile_mask = compute_valid_tile_mask(valid, ps)
        n_tiles = int(tile_mask.numel())
        n_kept = int(tile_mask.sum())
        assert 0 < n_kept < n_tiles, \
            f'门控没起作用：kept={n_kept}/{n_tiles}（构造的标签应当留下部分全无效 tile）'

        with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
            ref_logits = fast_tiled_val_inference(fake_x, net, opts_val)[target]
            ref_pred = class_decider(ref_logits, opts_val, target).reshape(H, W).cpu()
            ref_loss = loss_fn[target](ref_logits, lab.unsqueeze(0).to(device)).item()
            ungated, _ = fast_tiled_val_predict(
                fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                gate=False)
            gated, gated_loss = fast_tiled_val_predict(
                fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                loss_fns=loss_fn, task_weights=opts_val['task_weights'])

        # 7a) 分批完全一致 ⇒ 必须逐像素相同（全图，含被标签排除的区域）
        n_diff_ungated = int((ungated[target].long() != ref_pred).sum())
        assert n_diff_ungated == 0, \
            f'gate=False 时预测就与旧路径不同（{n_diff_ungated}/{H*W} 像素）——取 tile 或写回索引有 bug'

        # 7b) 门控后允许极小比例的 argmax 平局翻转，但必须确认「确实是平局」
        n_valid_px = int(valid.sum())
        diff = gated[target][valid].long() != ref_pred[valid]
        n_diff = int(diff.sum())
        margin = 0.0
        if n_diff:
            lg = ref_logits[0].float().cpu().permute(1, 2, 0)[valid]   # (n_valid, C)
            top2 = lg[diff].topk(2, dim=1).values
            margin = float((top2[:, 0] - top2[:, 1]).max())
        # 阈值卡在边界上时（实测 UPerNet+SwinV2 在 fp16 下 margin 1.24e-2 刚超 1e-2，而
        # [7a] 的逐像素相同已经证明索引没 bug），转 CPU/fp32 仲裁把「数值噪声」与「真 bug」
        # 就地分开：fp32 下门控只改变每批的组成，不该产生【任何】差异。
        try:
            assert n_diff <= 0.005 * n_valid_px, \
                f'门控前后预测差异过大：{n_diff}/{n_valid_px}（>0.5%，不像平局翻转）'
            assert margin <= 1e-2, \
                f'门控前后差异像素的 top1-top2 间隔达 {margin:.3e}，不是平局翻转，是真的算错了'
        except AssertionError as _e:
            if not cuda:
                raise      # CPU 本来就是 fp32，这里失败就是真失败，没有仲裁余地
            print(f'[7b] fp16 断言未通过（{_e}）→ 转 CPU/fp32 仲裁', flush=True)
            # 不动 net 本身的 device/dtype/状态：仲裁用一份独立的 fp32 CPU 副本。
            net32 = copy.deepcopy(net).float().cpu().eval()
            x32 = fake_x.detach().float().cpu()
            with torch.no_grad():        # 刻意不进 autocast：全程 fp32
                ref_logits32 = fast_tiled_val_inference(x32, net32, opts_val)[target]
                ref_pred32 = class_decider(ref_logits32, opts_val, target).reshape(H, W).cpu()
                gated32, _ = fast_tiled_val_predict(
                    x32, net32, opts_val, labels={target: lab},
                    valid_maps={target: valid}, gate=True)
            n_diff32 = int((gated32[target][valid].long() != ref_pred32[valid]).sum())
            del net32, x32, ref_logits32, ref_pred32, gated32
            if n_diff32 != 0:
                raise AssertionError(
                    f'CPU/fp32 仲裁【未通过】：fp32 下门控前后仍有 {n_diff32}/{n_valid_px} '
                    f'像素不同（fp32 只改批组成、理论上不该有差异）——大概率是索引 bug。'
                    f'⚠️ 但 CPU fp32 下不同批尺寸也可能让 MKL 选到不同分块、产生极小数值差；'
                    f'若差异只有个位数像素，需人工看一眼这些像素的 top1-top2 间隔再定性，'
                    f'不必然是索引 bug。原 fp16 断言：{_e}')
            # 仲裁通过 ≠ 无条件放行：fp32 证明了"不是索引 bug"，但 fp16 若不稳到翻掉 5%
            # 以上的有效像元，验证指标本身就会被数值噪声污染，这枪照样不能上。
            if n_diff > 0.05 * n_valid_px:
                raise AssertionError(
                    f'CPU/fp32 仲裁通过（fp32 下逐像素相同 ⇒ 不是索引 bug），但 fp16 下门控'
                    f'翻转了 {n_diff}/{n_valid_px}（{n_diff / max(n_valid_px, 1) * 100:.1f}% '
                    f'> 5%）——fp16 数值不稳到会污染验证指标的程度，拒绝放行。')
            print(f'[7b] CPU/fp32 仲裁通过：fp16 数值噪声（原 diff {n_diff}/{n_valid_px}、'
                  f'margin {margin:.3e}），放行', flush=True)
        if math.isnan(gated_loss):
            # 不可按批聚合的 loss（Dice/Lovász/带类权重 CE）按设计记 nan（见
            # functions._loss_is_mean_over_valid），跳过数值对拍——否则 abs(nan-nan)
            # 恒 False，冒烟会以「loss 不一致：nan vs nan」的误导面目失败（code-review
            # P2-2/N2）。[7c]/[7d] 会继续断言两条路径的 nan 判定一致。
            print('[7b] val loss 为 nan（该 loss 不满足「Σ逐像素/有效像元数」形式，'
                  '按设计不做按批聚合）→ 跳过 loss 对拍', flush=True)
        else:
            assert abs(gated_loss - ref_loss) <= 2e-3 * max(1.0, abs(ref_loss)), \
                f'门控前后 val loss 不一致：{gated_loss:.6f} vs {ref_loss:.6f}'
        print(f'[7] tile gate OK | tiles {n_kept}/{n_tiles} (跳过 '
              f'{(1 - n_kept / n_tiles) * 100:.0f}%) | ungated diff 0 | '
              f'gated diff {n_diff}/{n_valid_px} (max margin {margin:.1e}) | '
              f'loss {gated_loss:.4f} vs {ref_loss:.4f}', flush=True)

        # 7c) confmat 模式（2026-08-06 新增，quickstart 验证循环默认走它）：同 gate 值下
        #     批组成与 preds 模式完全一致 ⇒ logits 逐位相同 ⇒ GPU 逐批 bincount 出的混淆
        #     矩阵应与「preds 模式 + valid 掩膜筛选 + accumulate_confusion_matrix」
        #     【整数级完全相等】。两次独立前向之间若存在非确定性算子（atomicAdd 类
        #     conv 回退），允许退一步：行和（= true 支撑，与预测无关）必须精确相等、
        #     翻转像素 ≤ 0.5%（平局翻转，与 [7b] 同一性质）。loss 聚合算式逐位一致。
        #     顺带做「无逐批同步」的经验性验证（code-review P1-1 的教训：bincount 版每批
        #     同步两次，注释吹的重叠根本不存在）：sync_debug_mode('warn') 下数同步次数。
        #     预期只有函数末尾取回 loss 累加器的少量同步；只打印不断言（模型前向内部
        #     可能有架构相关的合法同步，硬阈值会误伤）。
        _sync_n = None
        _has_sync_dbg = cuda and hasattr(torch.cuda, 'set_sync_debug_mode')
        if _has_sync_dbg:
            torch.cuda.synchronize()
            torch.cuda.set_sync_debug_mode('warn')
        try:
            with warnings.catch_warnings(record=True) as _sync_w:
                warnings.simplefilter('always')
                with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
                    cms, cm_loss = fast_tiled_val_predict(
                        fake_x, net, opts_val, labels={target: lab}, valid_maps={target: valid},
                        loss_fns=loss_fn, task_weights=opts_val['task_weights'], out='confmat')
            if _has_sync_dbg:
                # 只数 sync 警告本身；同一 catch 里可能混进别的警告（如 val loss 不可
                # 按批聚合的 RuntimeWarning），不能一锅端。
                _sync_n = len([w for w in _sync_w
                               if 'synchron' in str(w.message).lower()])
        finally:
            if _has_sync_dbg:
                torch.cuda.set_sync_debug_mode('default')
        ref_cm = new_confusion_matrix(n_cls, device)
        accumulate_confusion_matrix(
            ref_cm, lab[valid].to(device), gated[target][valid].to(device).long(), n_cls)
        cm_new = cms[target].long().cpu()
        cm_ref = ref_cm.long().cpu()
        assert int(cm_new.sum()) == n_valid_px, \
            f'confmat 统计的像素数 {int(cm_new.sum())} != 有效像素数 {n_valid_px}——掩膜口径错了'
        cm_flip = 0
        if not torch.equal(cm_new, cm_ref):
            row_ok = bool(torch.equal(cm_new.sum(dim=1), cm_ref.sum(dim=1)))
            cm_flip = int((cm_new - cm_ref).abs().sum()) // 2
            assert row_ok and cm_flip <= 0.005 * n_valid_px, \
                (f'confmat 模式与 preds 模式不等价：行和一致={row_ok}、差异像素 '
                 f'{cm_flip}/{n_valid_px}——超出两次前向间数值噪声的量级，是真 bug')
        # loss 对拍：不可按批聚合的 loss（Dice/带类权重 CE 等）两边都记 nan，是设计行为
        # 而非等价性失败（code-review P2-2）。
        if math.isnan(cm_loss) or math.isnan(gated_loss):
            assert math.isnan(cm_loss) and math.isnan(gated_loss), \
                (f'nan 不对称：confmat {cm_loss} vs preds {gated_loss}——两条路径对'
                 f'「该 loss 是否可按批聚合」的判定不一致')
        else:
            assert abs(cm_loss - gated_loss) <= 2e-3 * max(1.0, abs(gated_loss)), \
                f'confmat 模式 loss 与 preds 模式不一致：{cm_loss:.6f} vs {gated_loss:.6f}'
        print(f'[7c] confmat mode OK | CM {"整数级相等" if cm_flip == 0 else f"平局翻转 {cm_flip} px（行和精确相等）"} | '
              f'loss {cm_loss:.4f} vs {gated_loss:.4f}'
              + (f' | CUDA syncs {_sync_n}（预期为个位数：仅函数末尾取回 loss）' if _sync_n is not None else ''),
              flush=True)

        # 7d) n_valid==0 守卫覆盖（code-review P2-1）：val_batch_size=1 + gate=False ⇒
        #     7 个全 255 的 tile 各自成批，逐一命中「整批无有效像元」分支。where 守卫
        #     失效的话 loss 会被 nan 污染且【不报错】——这里显式断言 loss 有限且与整景
        #     一次算完的 ref_loss 一致。CM 与 ungated preds 模式对拍（批组成不同，允许
        #     平局翻转但行和必须精确相等，与 [7b]/[7c] 同性质）。
        opts_val_b1 = dict(opts_val)
        opts_val_b1['val_batch_size'] = 1
        with torch.no_grad(), (torch.cuda.amp.autocast() if cuda else contextlib.nullcontext()):
            cms0, loss0 = fast_tiled_val_predict(
                fake_x, net, opts_val_b1, labels={target: lab}, valid_maps={target: valid},
                loss_fns=loss_fn, task_weights=opts_val['task_weights'], gate=False,
                out='confmat')
        if math.isnan(gated_loss):
            assert math.isnan(loss0), \
                'preds 模式 loss 为 nan（不可按批聚合）时 confmat 模式也应为 nan'
        else:
            assert math.isfinite(loss0), \
                f'gate=False/vbs=1 下 confmat loss = {loss0}——n_valid==0 批的 where 守卫失效'
            assert abs(loss0 - ref_loss) <= 2e-3 * max(1.0, abs(ref_loss)), \
                (f'gate=False confmat loss 与整景 loss 不一致：{loss0:.6f} vs {ref_loss:.6f}'
                 f'——vbs=1 的批组成与整景差异极大，fp16 噪声大的架构（如 SwinV2）若在此'
                 f'翻车，先按 [7b] 先例做 CPU/fp32 仲裁判定是否数值噪声，别直接当等价性 bug')
        cm0 = cms0[target].long().cpu()
        assert int(cm0.sum()) == n_valid_px, \
            f'[7d] confmat 统计的像素数 {int(cm0.sum())} != {n_valid_px}'
        ref_cm0 = new_confusion_matrix(n_cls, device)
        accumulate_confusion_matrix(
            ref_cm0, lab[valid].to(device), ungated[target][valid].to(device).long(), n_cls)
        cm0_ref = ref_cm0.long().cpu()
        cm0_flip = 0
        if not torch.equal(cm0, cm0_ref):
            row_ok0 = bool(torch.equal(cm0.sum(dim=1), cm0_ref.sum(dim=1)))
            cm0_flip = int((cm0 - cm0_ref).abs().sum()) // 2
            assert row_ok0 and cm0_flip <= 0.005 * n_valid_px, \
                (f'[7d] vbs=1 confmat 与 ungated preds 不等价：行和一致={row_ok0}、'
                 f'差异像素 {cm0_flip}/{n_valid_px}')
        print(f'[7d] n_valid==0 guard OK | loss {loss0:.4f} vs ref {ref_loss:.4f} | '
              f'CM flip {cm0_flip}', flush=True)

    if cuda:
        print(f'    peak GPU mem = {torch.cuda.max_memory_allocated() / 1e9:.2f} GB', flush=True)
        torch.cuda.reset_peak_memory_stats()

    print(f'PASS  {cfg_path}\n', flush=True)


def main():
    limit_torch_threads()
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'device = {device}  (无卡模式下用 CPU 只验证正确性，不代表真实速度)\n', flush=True)

    cfgs = sys.argv[1:] or DEFAULT_CONFIGS
    results = {}
    t0 = time.time()
    for c in cfgs:
        try:
            smoke_one(c, device, cuda)
            results[c] = 'PASS'
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[c] = f'FAIL: {type(e).__name__}: {e}'
            print(f'FAIL  {c}\n', flush=True)

    print('=' * 72, flush=True)
    print(f'SUMMARY ({time.time() - t0:.0f}s, device={device}):', flush=True)
    for c, r in results.items():
        print(f'  {"OK " if r == "PASS" else "XX "} {c:48s} {r}', flush=True)
    print('=' * 72, flush=True)
    sys.exit(0 if all(r == 'PASS' for r in results.values()) else 1)


if __name__ == '__main__':
    main()
