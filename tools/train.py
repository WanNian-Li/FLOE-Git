import argparse
import copy
import csv
import json
import random
import os
import os.path as osp
import re
import shutil
from icecream import ic
import pathlib
import warnings
import torchmetrics

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config, mkdir_or_exist
from tqdm import tqdm  # Progress bar

import wandb
# Functions to calculate metrics and show the relevant chart colorbar.
from floe.functions import save_best_model, load_model, slide_inference, \
    batched_slide_inference, fast_tiled_val_inference, fast_tiled_val_predict, class_decider, \
    create_train_validation_and_test_scene_list, compute_combined_score, \
    get_scheduler, get_optimizer, get_loss, get_model, supcon_region_loss, ema_update, \
    new_confusion_matrix, accumulate_confusion_matrix, cm_weighted_f1, cm_classwise_f1, \
    cm_overall_accuracy, cm_classwise_iou, cm_miou, cm_precision_recall, cm_normalized, \
    cm_class_proportions, cm_ordinal_cost

# Custom dataloaders for regular training and validation.
from floe.loaders import (get_variable_options, val_scoped_options, AI4ArcticChallengeDataset,
                     AI4ArcticChallengeTestDataset, preload_scene_cache, limit_torch_threads,
                     val_ondemand_ok)
#  get_variable_options

# -- Built-in modules -- #
from floe.utils import colour_str
from floe.evaluation import test


def parse_args():
    parser = argparse.ArgumentParser(description='Train Default U-NET segmentor')

    # Mandatory arguments
    parser.add_argument('config', type=pathlib.Path, help='train config file path',)
    parser.add_argument('--wandb-project', required=True, help='Name of wandb project')
    parser.add_argument('--wandb-name', default='MyDS', help='Name of wandb run (default: MyDS)')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--seed', default=None,
                        help='the seed to use, if not provided, seed from config file will be taken')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume-from', type=pathlib.Path, default=None, # 恢复训练
                       help='Resume Training from checkpoint, it will use the \
                        optimizer and schduler defined on checkpoint')
    group.add_argument('--finetune-from', type=pathlib.Path, default=None,  # 微调/迁移学习
                       help='Start new tranining using the weights from checkpoitn')

    args = parser.parse_args()

    return args


def save_epoch_sod_distribution(save_path, epoch, class_counts, mask_count, chart='FLOE'):
    """Append one row of epoch-level label pixel distribution to CSV."""
    prefix = chart.lower()
    fieldnames = ['epoch'] + [f'{prefix}_{c}' for c in range(len(class_counts))] + [f'{prefix}_mask']
    write_header = not osp.exists(save_path)

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        row = {'epoch': epoch, f'{prefix}_mask': int(mask_count)}
        for c, count in enumerate(class_counts):
            row[f'{prefix}_{c}'] = int(count)
        writer.writerow(row)


# U2（2026-07-31）逐像素 log10-尺寸回归辅助任务的类别刻度：类 1..5 的
# log10(几何均值浮冰尺寸 / m)。粒度分界见 utils.FLOE_LOOKUP：
#   cake <20m → 10m(1.0) | small 20-100m → 45m(1.65) | medium 100-500m → 224m(2.35)
#   big 0.5-2km → 1000m(3.0) | vast 2-10km → 4472m(3.65)
# 水（类 0）与 255 不监督。改动此表即口径断代，须在实验表注明。
_SCALE_REG_MU = (1.0, 1.65, 2.35, 3.0, 3.65)


def scale_reg_aux_loss(pred, labels, mu=_SCALE_REG_MU, beta=0.5, return_stats=False):
    """U2 辅助损失：逐 cell 回归"该处浮冰的 log10 尺寸"。

    Args:
        pred:   [B, 1, h, w]，模型的 '_scale_reg' 输出（h = H // 4，stride-4 网格）。
        labels: [B, H, W] long，target chart 标签（0=水、1..5=冰、255=ignore）。
        mu:     长度 5 的类别刻度（见 _SCALE_REG_MU）。
        beta:   Smooth L1 的转折点。
    Returns:
        loss（标量 tensor，fp32）；return_stats=True 时额外返回 dict：
          'cls_sum'/'cls_cnt'（[5] GPU tensor）—— 按"主导冰类"分组的预测均值累加器，
          用于判"学到了序但主任务无补"。**只统计 max_c f_c > 0.5 的 cell**（主导类自身
          占满半格；"冰占比>0.5"不够——0.45/0.45 的 small-medium 混合 cell 会把两条曲线
          糊在一起）。**刻意不 .item()**（每 batch 一次 GPU 同步太贵），由调用方 epoch
          末统一取值。

    目标构造（设计文档 §4，评审 #9）：one-hot 后 `avg_pool2d(4)` 得每个 cell 的**类别
    占比** f_c，目标 μ̄ = Σ_c f_c·μ_c / Σ_c f_c（只算冰类），像素权重 w = 冰类占比。
    混合 cell 因此不是"中心像素"的高方差噪声，water/255 自然被排除（它们不进 f）。
    整批没有任何含冰 cell 时返回恰好 0（不是 nan），梯度为 0。
    """
    pred = pred.float()
    y = labels.long()
    n_ice = len(mu)
    B, _, h, w = pred.shape
    assert labels.shape[0] == B and labels.dim() == 3, \
        f'labels 形状应为 [B,H,W]，实际 {tuple(labels.shape)}（pred {tuple(pred.shape)}）'
    H, W = y.shape[-2:]
    assert H % h == 0 and W % w == 0 and H // h == W // w, \
        f'标签 {H}×{W} 与预测网格 {h}×{w} 的下采样倍数不一致'
    pool = H // h
    # 255 先 clamp 进合法区间再用 valid 掩膜清零（clamp 只是为了让 one_hot 不越界）。
    valid = (y >= 1) & (y <= n_ice)
    onehot = F.one_hot(y.clamp(0, n_ice), n_ice + 1)[..., 1:].to(pred.dtype)  # [B,H,W,5]
    onehot = onehot * valid.unsqueeze(-1).to(pred.dtype)
    f = F.avg_pool2d(onehot.permute(0, 3, 1, 2), pool)          # [B,5,h,w] 各冰类占比
    ice_frac = f.sum(dim=1, keepdim=True)                        # [B,1,h,w]
    mu_vec = torch.as_tensor(mu, dtype=pred.dtype, device=pred.device).view(1, -1, 1, 1)
    target = (f * mu_vec).sum(dim=1, keepdim=True) / ice_frac.clamp_min(1e-6)
    weight = ice_frac.squeeze(1)                                 # [B,h,w]
    per_cell = F.smooth_l1_loss(pred, target, beta=beta, reduction='none').squeeze(1)
    # 加权均值直接写成 Σ(l·w)/Σw：不含冰的 cell 的 w 恰为 0（浮点 0，不是"很小"），
    # 分子里被精确抹掉 ⇒ 与"先按 w>0 掩膜再平均"逐位等价，但**不需要布尔掩膜索引**
    # （那会每 batch 触发一次 D2H 同步）。整批无冰时分子恰为 0、分母被 clamp 住 ⇒
    # loss 恰为 0 且梯度为 0，不产生 nan。
    loss = (per_cell * weight).sum() / weight.sum().clamp_min(1e-6)
    if not return_stats:
        return loss
    with torch.no_grad():
        # 按类读数：只统计"主导冰类明确"的 cell —— 门是**主导类自身占满半格**
        # （max_c f_c > 0.5），不是"冰占比 > 0.5"。后者不蕴含前者：0.45 small + 0.45
        # medium 的 cell 冰占比 0.9 却没有明确主导类，argmax 还会在平局时全记给 small，
        # 把预注册要看的 cls2/cls3 两条曲线系统性糊在一起 —— 而 320 m 的 cell 恰好横跨
        # small(20-100m)/medium(100-500m) 的混合带，这类 cell 一点都不罕见。
        # 同样刻意避开布尔索引（会触发同步）——用 one-hot 求和代替 index_add_。
        # cls_sum/cls_cnt 留在 GPU 上，由调用方 epoch 末统一同步。
        dom_frac, dom = f.max(dim=1)                             # [B,h,w]，0..4 → 类 1..5
        pure = (dom_frac > 0.5).to(pred.dtype).unsqueeze(-1)     # [B,h,w,1]
        sel = F.one_hot(dom, n_ice).to(pred.dtype) * pure        # [B,h,w,5]
        cls_cnt = sel.sum(dim=(0, 1, 2))
        cls_sum = (sel * pred.squeeze(1).unsqueeze(-1)).sum(dim=(0, 1, 2))
    return loss, {'cls_sum': cls_sum, 'cls_cnt': cls_cnt}


def dann_lambda_schedule(epoch, weight, warmup):
    """V2 DANN 的对抗强度 λ：前 `warmup` 个 epoch 从 0 线性升到 `weight`，之后恒定。

    DANN 标准做法——判别器先热身，别在它还乱猜时就压 encoder（那时反转的是噪声梯度）。
    epoch 0 恰为 0（判别器纯学习、encoder 不受对抗压力），epoch == warmup 起为满值。
    warmup<=0 视为不预热（首个 epoch 就满 λ）。抽成模块级函数是为了让 smoke_dann.py
    能直接断言这条曲线（内联在训练循环里就测不到）。
    """
    if warmup <= 0:
        return float(weight)
    return float(weight) * min(1.0, epoch / float(warmup))


def train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer, scheduler, start_epoch=0,
          resume_state=None):
    '''
    Trains the model.

    resume_state: --resume-from 时从 checkpoint 取出的断点状态（functions.save_best_model
    存入），含 best_primary_miou / val_hist_tail / early_stop_counter / val_smoothing_window。
    还原它使恢复后的 run ① 第一次验证不再必定覆盖 best；② smsel 平滑窗与早停计数从
    best 那次验证【语义连续】地接上。注意不是逐位复现：只存 best、无中间 checkpoint，
    续跑会重跑 best 那个 epoch ⇒ 相比不崩溃的连续跑多插入一个验证点、此后各验证点的
    训练量少一个 epoch、LR 超前一个 epoch（旧已知项）、早停预算实际多一次验证。
    None（旧 checkpoint 或非恢复）= 旧行为。仅 model_save_criterion=='miou' 时保存/还原。
    '''
    best_primary_miou = -np.Inf  # Best validation mIoU for the target chart.
    best_ordinal_score = -np.Inf  # Best ordinal score (方案7使用)
    target_chart = train_options.get('target_chart', 'FLOE')
    model_save_criterion = train_options.get('model_save_criterion', 'miou')  # 'miou' | 'ordinal_score'

    # Early stopping
    early_stop_patience = train_options.get('early_stop_patience', 10)
    early_stop_counter = 0

    # 平滑 best 选择 / 早停（2026-07-20，评估协议修复）：per-epoch val mIoU 的噪声 std
    # 高达 ~0.02（见 docs/2026-07-20_seed_variance.md §6），单 epoch max 选 best 会挑到
    # 幸运噪声尖峰、且早尖峰触发早停提前杀 run。val_smoothing_window>1 时，用最近 k 次
    # 验证 mIoU 的滑动均值作为 best 选择与早停的准则（默认 1 = 单 epoch，与历史逐位一致）。
    val_smoothing_window = train_options.get('val_smoothing_window', 1)
    # 防 footgun：window=0 会让 _val_hist[-0:] 退化为全历史切片（静默错误），负数同理。
    assert val_smoothing_window >= 1, \
        f'val_smoothing_window 必须 >=1，得到 {val_smoothing_window}'
    _val_hist = []  # 记录每次验证的 primary_miou.item()，按验证顺序

    # 断点续跑（2026-08-06）：还原 best 记录 / 平滑窗尾部 / 早停计数。
    # best 保存的 checkpoint 都带 resume_state（save_best_model），恢复后从 best 那次
    # 验证的状态无缝继续；旧 checkpoint 无此键 → 打印告警走旧行为（best=-inf 会导致
    # 恢复后首验必覆盖 best、平滑窗欠平滑，见 CLAUDE.md 历史坑）。
    _expected_best_path = osp.join(cfg.work_dir,
                                   f'best_model_{osp.basename(cfg.work_dir)}.pth')
    model_path = None   # 从未刷新 best 时的返回值（resume 且恢复期无提升的合法情形）
    if resume_state is not None:
        assert model_save_criterion == 'miou', (
            f'resume_state 仅支持 model_save_criterion=miou（当前 {model_save_criterion}）')
        _rs_w = resume_state.get('val_smoothing_window')
        assert _rs_w == val_smoothing_window, (
            f'resume_state 的 val_smoothing_window={_rs_w} 与当前配置 '
            f'{val_smoothing_window} 不一致——平滑口径不同禁止续跑')
        best_primary_miou = float(resume_state['best_primary_miou'])
        assert np.isfinite(best_primary_miou), (
            f'resume_state.best_primary_miou={best_primary_miou} 非有限值——checkpoint 来源可疑')
        _val_hist = [float(v) for v in resume_state.get('val_hist_tail', [])]
        early_stop_counter = int(resume_state.get('early_stop_counter', 0))
        if osp.exists(_expected_best_path):
            model_path = _expected_best_path
        if resume_state.get('ema'):
            print('\033[93m[Resume][WARN] 该 best 由 EMA 轨迹产生，而 EMA 影子权重不进 '
                  'checkpoint——续跑无法复现该轨迹（欠平均若干步），结果解读要打折\033[0m')
        print(f"\033[92m[Resume] 断点状态已还原：best_smoothed={best_primary_miou:.6f}  "
              f"val_hist_tail={[f'{v:.4f}' for v in _val_hist]}  "
              f"early_stop_counter={early_stop_counter}/{early_stop_patience}\033[0m")
    elif start_epoch > 0:
        print('\033[91m[Resume][WARN] checkpoint 无 resume_state（旧格式）：best 记录/平滑窗/'
              '早停计数从零开始，恢复后第一次验证将覆盖 best_model——如需保留旧 best 先自行备份\033[0m')

    # 验证时使用原始（未编译）模型，避免动态形状触发 recompile 警告
    net_val = net._orig_mod if train_options.get('compile_model') and hasattr(net, '_orig_mod') else net

    # H3 EMA（2026-07-29）：权重指数移动平均。15 枪全在改"学什么"，没人碰过"怎么取
    # 最终权重"；smsel 证明逐次验证噪声 ±0.02 是真问题，EMA 从权重侧直接压噪声，与
    # smsel（选择准则侧）叠加不冲突。验证/best 保存全部用 EMA 权重（选择与部署一致）。
    # ⚠️ --resume-from 后 EMA 从 checkpoint 权重重新初始化（EMA 影子不进 checkpoint），
    #   前 ~1/(1-decay) 步欠平均——smsel run 本就尽量不 resume，可接受。
    ema_enabled = bool(train_options.get('ema', False))
    ema_decay = float(train_options.get('ema_decay', 0.999))
    ema_net = None
    ema_updates = 0  # 全局 update 计数，驱动 decay warmup ramp
    if ema_enabled:
        assert 0.0 < ema_decay < 1.0, f'ema_decay 应在 (0,1)，得到 {ema_decay}'
        assert not train_options.get('compile_model'), \
            'ema 与 compile_model 同开未适配（checkpoint 键名前缀不一致，code-review 🟢9）'
        ema_net = copy.deepcopy(net_val)
        ema_net.eval()
        for p in ema_net.parameters():
            p.requires_grad_(False)
        print(f'[EMA] enabled: decay={ema_decay}（时间尺度 ~{1.0/(1.0-ema_decay):.0f} 步 '
              f'≈ {1.0/(1.0-ema_decay)/max(train_options["epoch_len"],1):.1f} epoch），'
              'warmup ramp d_t=min(decay,(1+t)/(10+t))；验证与 best 保存均使用 EMA 权重')

    # 验证侧只跑 target chart（辅助任务是 train-only 的，见 loaders.val_scoped_options）。
    # 无 aux_charts 时 val_options is train_options、val_charts 就是 train_options['charts']。
    val_options = val_scoped_options(train_options)
    val_charts = val_options['charts']
    # 有辅任务时，总 train loss 含 aux 分量、而 val loss 只含 target chart，两条曲线量纲
    # 不同 —— 必须额外记一条「target 分量」的曲线，否则 train/val gap 判读会失真。
    aux_present = len(train_options['charts']) > 1

    # 训练集 IoU（仅对有效任务）
    primary_idx = train_options['charts'].index(target_chart)
    compute_train_primary_iou = (train_options['task_weights'][primary_idx] != 0)
    primary_n_classes = train_options['n_classes'][target_chart]
    if compute_train_primary_iou:
        train_iou_metric = torchmetrics.classification.MulticlassJaccardIndex(
            num_classes=primary_n_classes, average='none', ignore_index=255).to(device)

    loss_ce_functions = {chart: get_loss(train_options['chart_loss'][chart]['type'], chart=chart, **train_options['chart_loss'][chart]).to(device)
                         for chart in train_options['charts']}

    # F3 跨景区域级 SupCon 辅助损失（2026-07-28）：模型开了 segformer_supcon 时，
    # forward 在训练模式下多返回 '_supcon' 键（stride-4 投影特征），这里配上标签算
    # SupCon 并以 λ 加权进总 loss。验证 loss 不含本项（不碰按需读的按批加权聚合前提）。
    supcon_enabled = bool(train_options.get('segformer_supcon', False))
    supcon_weight = float(train_options.get('segformer_supcon_weight', 0.1))
    supcon_temp = float(train_options.get('segformer_supcon_temp', 0.1))
    supcon_min_px = int(train_options.get('segformer_supcon_min_px', 16))
    if supcon_enabled:
        print(f'[SupCon] enabled: weight={supcon_weight} temp={supcon_temp} '
              f'min_px={supcon_min_px}（区域嵌入取自 encoder stride-4 投影特征）')

    # U2 逐像素 log10-尺寸回归辅助损失（2026-07-31）：模型开了 segformer_scale_reg 时，
    # forward 在训练模式下多返回 '_scale_reg' 键（encoder stride-4 上的 1 通道回归图），
    # 这里配上标签算 Smooth L1 并以 λ 加权进总 loss。验证 loss 不含本项（不碰按需读的
    # 按批加权聚合前提），验证路径零改动。
    scale_reg_enabled = bool(train_options.get('segformer_scale_reg', False))
    scale_reg_weight = float(train_options.get('segformer_scale_reg_weight', 0.1))
    if scale_reg_enabled:
        assert primary_n_classes == len(_SCALE_REG_MU) + 1, \
            (f'segformer_scale_reg 假定 6 类（水 + 5 个冰尺寸类），但 '
             f"n_classes[{target_chart}]={primary_n_classes}；μ 刻度表需同步修订")
        print(f'[ScaleReg] enabled: weight={scale_reg_weight} '
              f'μ(log10 m, 类1..5)={_SCALE_REG_MU}（目标=cell 内冰类占比加权均值，'
              '水/255 不监督）')

    # V2 DANN 场景对抗（2026-08-05）：模型开了 segformer_dann 时，forward 在训练模式下
    # 多返回 '_dann' 键（判别器对"这个 patch 来自哪一景"的 logits，已过梯度反转层），
    # loader 在 y dict 里配一个 '_domain' 键（LongTensor [B]）。验证 loss 不含本项
    # （不碰按需读的按批加权聚合前提），验证/推理路径零改动。
    # ⚠️ **权重语义与 SupCon/ScaleReg 不同**：判别器损失按 **1.0** 权重加进总 loss
    # （DANN 原论文口径——判别器要正常最小化自己的损失才学得会认景），对抗强度由 λ
    # 控制，而 λ 只作用在 GRL 反传给 encoder 的梯度上。所以下面看不到
    # "dann_weight * loss_dann" 这种写法，那是错的（会同时削弱判别器本身）。
    dann_enabled = bool(train_options.get('segformer_dann', False))
    dann_weight = float(train_options.get('segformer_dann_weight', 0.1))   # = λ 的上限
    dann_warmup = int(train_options.get('segformer_dann_warmup', 10))      # 线性升温 epoch 数
    if dann_enabled:
        # λ 是每 epoch 变的 **python float**，会被 dynamo 当成 guard 常量 ⇒ 开 compile 时
        # 每个 epoch 都会 recompile 一次整张图（几十秒起，且没有任何报错提示）。同 EMA
        # 的先例（code-review 🟢9），直接禁掉这个组合而不是留个坑。
        assert not train_options.get('compile_model'), \
            'segformer_dann 与 compile_model 同开未适配：dann_lambda 是 dynamo guard 的 ' \
            'float，每 epoch 改一次会触发整图 recompile。'
        n_domains = int(train_options['segformer_dann_domains'])
        # 一致性硬断言：判别器输出维度必须等于实际 train 景数。不一致时"域 acc"的基线
        # 1/N 会说不通，且域序号可能越界 → CUDA device-side assert（无 traceback、
        # 整个 run 白跑）。宁可在这里干净地炸。
        assert len(train_options['train_list']) == n_domains, \
            (f'segformer_dann_domains={n_domains} 与实际 train 景数 '
             f'{len(train_options["train_list"])} 不一致（train_list 会剔除出现在 '
             f'validate_list 里的景，别直接抄 datalist 的长度）。')
        print(f'[DANN] enabled: λ_max={dann_weight} warmup={dann_warmup} epochs '
              f'n_domains={n_domains}（判别器吃 encoder stride-4 特征的 GAP；'
              f'域 acc 的随机基线 = {1.0 / n_domains:.4f}，越低越说明特征去场景化）')

    # AMP 混合精度需配 GradScaler：scaler.step() 在检测到 inf/nan 梯度时会自动跳过该次
    # optimizer 更新，避免一次数值溢出的 batch 永久污染 AdamW 的动量估计（此前 FLOE PA2-Swin
    # 训练在某个 epoch 中途 loss 变 nan 后，模型权重被污染、之后所有 epoch 都坍缩为恒定输出）。
    scaler = torch.cuda.amp.GradScaler()
    grad_clip_max_norm = train_options.get('grad_clip_max_norm', 5.0)

    print('Training...')
    # -- Training Loop -- #
    for epoch in tqdm(iterable=range(start_epoch, train_options['epochs'])):
        # gc.collect()  # Collect garbage to free memory.
        train_loss_sum = torch.tensor([0.])  # To sum the training batch losses during the epoch.
        cross_entropy_loss_sum = torch.tensor([0.])  # To sum the training cross entropy batch losses during the epoch.
        primary_ce_loss_sum = torch.tensor([0.])  # target chart 的 CE 分量（仅 aux run 记录）。
        supcon_loss_sum = torch.tensor([0.])  # F3：SupCon 分量（未乘 λ）的 epoch 求和。
        # F3 预注册读数的 epoch 累加器（code-review：几何间隔打印在 best epoch 附近
        # 拿不到读数，必须落成 wandb 曲线才能与 hard-ice 对齐判读）。
        supcon_active = 0            # 产生了有效损失（存在同类跨样本对）的 batch 数
        supcon_same_sum = 0.0        # 同类跨样本平均余弦（仅 active 且非 nan 的 batch）
        supcon_diff_sum = 0.0
        supcon_cos_batches = 0
        supcon_cls_regions = np.zeros(primary_n_classes, dtype=np.float64)  # 各类区域嵌入数
        # U2 的 epoch 累加器：回归 loss（未乘 λ）与"按主导冰类分组的预测均值"。后者是
        # 预注册读数——回归输出若按 cake<small<medium<big<vast 排开，说明辅助任务学到了
        # 尺度序；此时主任务若仍平，结论是"序学到了但对分割无补"，与"机制没咬合"不同。
        scale_reg_loss_sum = torch.tensor([0.])
        scale_reg_cls_sum = torch.zeros(len(_SCALE_REG_MU), device=device)
        scale_reg_cls_cnt = torch.zeros(len(_SCALE_REG_MU), device=device)
        # 梯度范数 / 裁剪命中率的 epoch 累加器（2026-08-05 起对**所有 run** 生效，不只
        # DANN）：`clip_grad_norm_` 返回的裁剪前全局范数以前被丢弃，"有没有被 5.0 削过"
        # 无从观测。三条曲线（Grad Norm 均值、Clip Hit Frac、Grad Nonfinite Frac）是新增
        # 的全局口径，锚 run 也会有，跨 run 可比。累加留 GPU，epoch 末同步一次。
        # ⚠️ 必须**只统计有限的范数**：AMP 下 GradScaler 会周期性地把 scale 探到溢出
        # （这是它标定 scale 的正常工作方式），那些 batch 的范数是 inf ⇒ 直接求和会让
        # 整条曲线恒为 inf/nan、读数报废。非有限的占比单独记一条，它本身就是"这个 run
        # 的数值稳定性"读数（S 批 dpt 那次 fp16 溢出若有这条曲线会当场看见）。
        gn_sum = torch.zeros((), device=device)
        gn_fin_cnt = torch.zeros((), device=device)
        clip_hits = torch.zeros((), device=device)
        # V2 的 epoch 累加器：判别器损失与域判别准确率（预注册的"机制咬合与否"读数）。
        # 域 acc 留在 GPU 上按 batch 累加，epoch 末才同步一次。
        dann_loss_sum = torch.tensor([0.])
        dann_correct = torch.zeros((), device=device)
        dann_seen = torch.zeros((), device=device)
        if dann_enabled:
            # λ 线性升温（DANN 标准做法：判别器先热身，别在它还乱猜时就压 encoder）。
            # 写在 net_val 上而不是 net：compile_model 时 net 是 OptimizedModule 包装，
            # 属性要落到真正的模块上（net_val = net._orig_mod）。
            # resume 时 epoch 从 checkpoint 续，ramp 自然接上（不会退回 0）。
            net_val.dann_lambda = dann_lambda_schedule(epoch, dann_weight, dann_warmup)
        # To sum the training edge consistency batch losses during the epoch.
        val_loss_sum = torch.tensor([0.])  # To sum the validation batch losses during the epoch.
        # To sum the validation cross entropy batch losses during the epoch.
        val_cross_entropy_loss_sum = torch.tensor([0.])
        # 实际参与 backward/step 的 batch 数（跳过的坏 batch 不计入，避免平均 loss 分母虚高）。
        processed_batches = 0

        # Aggregate target chart label distribution over all training pixels in this epoch.
        primary_num_classes = train_options['n_classes'][target_chart]
        primary_epoch_class_counts = np.zeros(primary_num_classes, dtype=np.int64)
        primary_epoch_mask_count = 0
        # Online confusion matrix for train-set diagnostics.
        train_primary_confmat = torch.zeros((primary_num_classes, primary_num_classes), dtype=torch.float32, device=device)

        net.train()  # Set network to evaluation mode.


        #===============================================================#
        #============================训练循环============================#
        #===============================================================#
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader_train, total=train_options['epoch_len'],
                                                    colour='red')):
            # torch.cuda.empty_cache()  # Empties the GPU cache freeing up memory.
            train_loss_batch = torch.tensor([0.]).to(device)  # Reset from previous batch.
            cross_entropy_loss = torch.tensor([0.]).to(device)
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)

            # Epoch-level SOD pixel stats are computed in the main process, so
            # this works with num_workers > 0.
            if target_chart in batch_y:
                primary_batch = batch_y[target_chart]
                for c in range(primary_num_classes):
                    primary_epoch_class_counts[c] += int((primary_batch == c).sum().item())
                primary_epoch_mask_count += int((primary_batch == 255).sum().item())

            # - Mixed precision training. (Saving memory)
            with torch.cuda.amp.autocast():

                #=========模型前向传播
                output = net(batch_x)
                # breakpoint()

                #==========计算损失（逐 chart 累加）
                # ⚠️ 辅任务（aux_charts）的标签覆盖【没有任何保证】：tile 采样只按 target
                # chart 的直方图选 tile、只检查 target 的标签覆盖，某个 batch 里 aux chart
                # 可能整批全是 255 → CE(ignore_index=255) 返回 nan → 下面的 isfinite 守卫会
                # 把整个 batch（含 target chart 完好的梯度）一起丢掉。所以非 target chart
                # 先查有效像元，没有就跳过该项——这是常态，不打日志（否则会刷屏）。
                for chart, weight in zip(train_options['charts'], train_options['task_weights']):
                    y_chart = batch_y[chart].to(device)
                    if chart != target_chart:
                        # 值域硬校验（只对 aux chart 做，target 的类别码由 loader 保证）：
                        # 越界类别码在 CUDA 上会变成 device-side assert —— 无 traceback、
                        # CUDA context 报废、整个 run 白跑。宁可在这里干净地 raise。
                        # 刻意放训练循环而不是 loaders 的 tile 路径：`_read_tile` 抛异常会
                        # 触发无限重试 + 每次打印（见 CLAUDE.md 已知坑）。
                        # ⚠️ 下界也要查：本仓约定负值=无效（FLOE/SOD_LOOKUP['invalid']=-9，
                        # 遗留 SIC 走同一套生成脚本很可能存了 -9），NaN 经 .type(torch.long)
                        # 也会变成极大负数——两者都躲得过 `>= n_classes` 的单边校验。
                        _valid_c = (y_chart != 255)
                        _n_cls_c = train_options['n_classes'][chart]
                        _oob = _valid_c & ((y_chart < 0) | (y_chart >= _n_cls_c))
                        # 两个标志一次同步拿齐（每 batch 两次 .any() 同步会把 loss 段与
                        # 前向串行化；aux run 每 epoch 省 ~300 次同步，无 aux 不走这里）
                        _has_valid, _has_oob = torch.stack(
                            [_valid_c.any(), _oob.any()]).tolist()
                        if _has_oob:
                            raise RuntimeError(
                                f'aux chart {chart!r} 的标签越界：n_classes[{chart}]='
                                f'{_n_cls_c}，但 epoch {epoch} batch {i} 里出现了类别码 '
                                f'{y_chart[_oob].unique()[:10].tolist()}（只列前 10 个；'
                                f'255=ignore 已排除）。检查 .nc 里该变量的取值口径 '
                                f'（_audit_sic_vars.py）或 n_classes 配置。')
                        if not _has_valid:
                            continue  # 本批 aux 无有效标签：CE 会返回 nan，跳过该项（常态，不打日志）
                    _ce_chart = loss_ce_functions[chart](output[chart], y_chart)
                    cross_entropy_loss += weight * _ce_chart
                    if chart == target_chart:
                        primary_ce_batch = _ce_chart.detach()
            train_loss_batch = cross_entropy_loss

            # F3 SupCon：刻意放在 autocast 【之外】——bmm/normalize/logsumexp 在 fp16
            # 下有溢出与精度风险，这里全程 fp32（'_supcon' 特征在函数内 .float()）。
            # 梯度照常回传（GradScaler 缩放的是最终总 loss）。
            supcon_loss, supcon_stats = None, None
            if supcon_enabled:
                # 硬断言而非静默跳过（code-review 🟡1）：开了 supcon 却拿不到特征键
                # （模型不是 sota_segformer / 键名不符），整枪会静默退化成纯 CE 空跑，
                # 且曲线恒 0 会被误读成"机制没咬合"。宁可当场炸。
                if '_supcon' not in output:
                    raise RuntimeError(
                        "segformer_supcon=True 但模型输出没有 '_supcon' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了 SupCon 投影头）")
                supcon_loss, supcon_stats = supcon_region_loss(
                    output['_supcon'], batch_y[target_chart].to(device),
                    n_classes=primary_n_classes,
                    temperature=supcon_temp, min_pixels=supcon_min_px,
                    return_stats=True)
                train_loss_batch = train_loss_batch + supcon_weight * supcon_loss

            # U2 尺寸回归：同样刻意放在 autocast 【之外】走 fp32（目标是 log10 尺度上的
            # 连续值，Smooth L1 在 fp16 下的精度没必要冒险）。'_scale_reg' 只在训练态存在。
            scale_reg_loss, scale_reg_stats = None, None
            if scale_reg_enabled:
                # 硬断言而非静默跳过（纪律同 SupCon）：开了开关却拿不到键，整枪会静默
                # 退化成纯 CE 空跑，曲线恒 0 会被误读成"机制没咬合"。宁可当场炸。
                if '_scale_reg' not in output:
                    raise RuntimeError(
                        "segformer_scale_reg=True 但模型输出没有 '_scale_reg' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了尺寸回归辅助头）")
                scale_reg_loss, scale_reg_stats = scale_reg_aux_loss(
                    output['_scale_reg'], batch_y[target_chart].to(device),
                    return_stats=True)
                train_loss_batch = train_loss_batch + scale_reg_weight * scale_reg_loss

            # V2 DANN：判别器 CE。同样放在 autocast 【之外】走 fp32 —— 375 路 softmax 在
            # fp16 下 logsumexp 的精度没必要冒险；'_dann' 只在训练态存在。
            loss_dann = None
            if dann_enabled:
                # 硬断言而非静默跳过（纪律同 SupCon/ScaleReg）：开了开关却拿不到键，整枪
                # 会静默退化成纯 CE 空跑，曲线恒 0 会被误读成"机制没咬合"。宁可当场炸。
                if '_dann' not in output:
                    raise RuntimeError(
                        "segformer_dann=True 但模型输出没有 '_dann' 键——"
                        f"model_selection={train_options.get('model_selection')!r} "
                        "是否为 sota_segformer？（只有它实现了 DANN 判别器）")
                if '_domain' not in batch_y:
                    raise RuntimeError(
                        "segformer_dann=True 但 batch 里没有 '_domain' 键——"
                        'dataset 是否走的 tile 采样路径（loaders 只在那条路上记账域序号）？')
                _dom = batch_y['_domain'].to(device, non_blocking=True)
                _dlogits = output['_dann'].float()
                # ⚠️ 按 **1.0** 权重加（不是 dann_weight）：对抗强度全部由 GRL 里的 λ 承担，
                # 见上方 [DANN] 说明。乘 λ 会连判别器自身的学习一起削弱，那不是 DANN。
                loss_dann = F.cross_entropy(_dlogits, _dom)
                train_loss_batch = train_loss_batch + loss_dann
                with torch.no_grad():
                    # 这里只【算】命中数，**累加放到 isfinite 跳批守卫之后**（与
                    # dann_loss_sum 同侧同分母）：被跳过的坏 batch 不参与更新，也就不该
                    # 进域 acc 的统计，否则两条曲线的样本集不一致、判读会错位。
                    dann_hits_b = (_dlogits.argmax(dim=1) == _dom).sum()
                    dann_seen_b = _dom.numel()

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            if not torch.isfinite(train_loss_batch):
                # 坏 batch（fp16 溢出 / 异常输入导致 loss 非有限值）：跳过本次更新，不参与
                # 梯度和优化器状态，避免污染 AdamW 的动量估计。
                print(f'[WARN] epoch {epoch} batch {i}: non-finite train loss '
                      f'({train_loss_batch.item()}), skipping this batch.')
                continue

            # - Backward pass（经 GradScaler 缩放，避免 fp16 下溢；scaler.step 会在梯度出现
            #   inf/nan 时自动跳过 optimizer.step()）。
            scaler.scale(train_loss_batch).backward()
            scaler.unscale_(optimizer)
            # error_if_nonfinite=False：inf/nan 梯度交给 scaler.step 处理（自动跳过该次更新），
            # 而不是在这里直接抛异常中断训练。
            # 返回值 = **裁剪前**的全局梯度范数。以前直接丢掉，于是"这次训练到底有没有被
            # 5.0 的上限削过、削了多少"是完全不可观测的（V2 DANN 把判别器 CE 加进同一份
            # 总 loss，这个问题第一次变成判读上的实际障碍）。累加留在 GPU 上、epoch 末才
            # 同步一次（纪律同 dann_correct / scale_reg_cls_sum，不引入 per-batch D2H）。
            _gn = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=grad_clip_max_norm,
                                                 error_if_nonfinite=False).detach()
            _gn_fin = torch.isfinite(_gn)
            gn_sum += torch.where(_gn_fin, _gn, torch.zeros_like(_gn))
            gn_fin_cnt += _gn_fin
            clip_hits += _gn_fin & (_gn > grad_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
            # - Scheduler step（ReduceLROnPlateau 改为 per-epoch，其余类型仍 per-batch）
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()

            # H3 EMA 影子更新（functions.ema_update）。decay warmup ramp（timm 惯例）：
            # 没有 ramp 时 epoch 0 结束 EMA 里仍有 74% 初始权重（0.999^300），前 2-3 次
            # 验证等于在验证半初始化模型（code-review 🟡2）。scaler.step 因 inf/nan 跳过
            # 更新时权重未变，本步 lerp 等于向原地收缩、无害。
            if ema_net is not None:
                ema_updates += 1
                _d = min(ema_decay, (1.0 + ema_updates) / (10.0 + ema_updates))
                ema_update(ema_net, net_val, _d)

            # - Add batch loss.
            processed_batches += 1
            train_loss_sum += train_loss_batch.detach().item()
            cross_entropy_loss_sum += cross_entropy_loss.detach().item()
            if aux_present:
                primary_ce_loss_sum += primary_ce_batch.item()
            if supcon_loss is not None:
                supcon_loss_sum += supcon_loss.detach().item()
                supcon_cls_regions += np.asarray(supcon_stats['regions_per_class'])
                if supcon_stats['active']:
                    supcon_active += 1
                    if (np.isfinite(supcon_stats['same_cos'])
                            and np.isfinite(supcon_stats['diff_cos'])):
                        supcon_same_sum += supcon_stats['same_cos']
                        supcon_diff_sum += supcon_stats['diff_cos']
                        supcon_cos_batches += 1
            if scale_reg_loss is not None:
                scale_reg_loss_sum += scale_reg_loss.detach().item()
                # 累加留在 GPU 上（epoch 末才同步一次），不做 per-batch .item()
                scale_reg_cls_sum += scale_reg_stats['cls_sum']
                scale_reg_cls_cnt += scale_reg_stats['cls_cnt']
            if loss_dann is not None:
                dann_loss_sum += loss_dann.detach().item()
                dann_correct += dann_hits_b
                dann_seen += dann_seen_b

            # 更新训练集 IoU 统计
            if compute_train_primary_iou:
                with torch.no_grad():
                    primary_pred_train = output[target_chart].detach().float().argmax(dim=1)
                    primary_true_train = batch_y[target_chart].to(device).long()
                    train_iou_metric.update(primary_pred_train, primary_true_train)

                    # Update confusion matrix incrementally to avoid storing full-epoch pixels.
                    valid = ((primary_true_train >= 0) & (primary_true_train < primary_num_classes) &
                             (primary_pred_train >= 0) & (primary_pred_train < primary_num_classes))
                    if valid.any():
                        t = primary_true_train[valid].view(-1)
                        p = primary_pred_train[valid].view(-1)
                        flat_idx = t * primary_num_classes + p
                        binc = torch.bincount(flat_idx, minlength=primary_num_classes * primary_num_classes).float()
                        train_primary_confmat += binc.view(primary_num_classes, primary_num_classes)

        #===========一个Epoch结束，计算平均损失
        # 分母用实际参与更新的 batch 数（processed_batches），而非总迭代数 i+1，
        # 否则被跳过的 nan/inf batch 会把分母虚高、掩盖训练不稳定的信号。
        if processed_batches < i + 1:
            print(f'[WARN] epoch {epoch}: skipped {i + 1 - processed_batches}/{i + 1} '
                  f'batches due to non-finite loss.')
        train_loss_epoch = torch.true_divide(train_loss_sum, max(processed_batches, 1)).detach().item()
        cross_entropy_epoch = torch.true_divide(cross_entropy_loss_sum, max(processed_batches, 1)).detach().item()
        supcon_epoch = torch.true_divide(supcon_loss_sum, max(processed_batches, 1)).detach().item()
        # F3 读数字典：只在开了 SupCon 时并进 wandb.log，其他 run 的曲线面板保持干净。
        # Cos Gap（同类-异类余弦间隔）与 per-class regions 是预注册判读的主读数：
        # Gap 不拉开 ⇒ 机制没咬合；某类 regions ≈0 ⇒ min_px 把它挡在门外、loss 从未约束它。
        supcon_log = {}
        if supcon_enabled:
            _nb = max(processed_batches, 1)
            _ncb = max(supcon_cos_batches, 1)
            supcon_log = {
                "Train SupCon Epoch Loss": supcon_epoch,
                "SupCon Active Batch Frac": supcon_active / _nb,
                "SupCon Same-class Cos": supcon_same_sum / _ncb,
                "SupCon Diff-class Cos": supcon_diff_sum / _ncb,
                "SupCon Cos Gap": (supcon_same_sum - supcon_diff_sum) / _ncb,
            }
            for c in range(primary_n_classes):
                supcon_log[f"SupCon Regions Class {c}"] = supcon_cls_regions[c] / _nb

        # U2 读数：回归 loss + 按主导冰类的回归输出均值（只在开了本机制时新增键）。
        scale_reg_log = {}
        if scale_reg_enabled:
            _cnt = scale_reg_cls_cnt.clamp_min(1.0)
            _mean = (scale_reg_cls_sum / _cnt).tolist()          # epoch 一次同步
            _seen = scale_reg_cls_cnt.tolist()
            scale_reg_log["ScaleReg Loss"] = torch.true_divide(
                scale_reg_loss_sum, max(processed_batches, 1)).detach().item()
            for c in range(len(_SCALE_REG_MU)):
                # 类 c+1（1..5）：该类占满半格（max_c f_c > 0.5）的 cell 上的预测均值。
                # 目标刻度是 _SCALE_REG_MU[c]，两者对比即"学到了序没有"。
                scale_reg_log[f"ScaleReg Pred cls{c + 1}"] = (
                    _mean[c] if _seen[c] > 0 else float('nan'))

        # 全局新增读数（2026-08-05，V 批起）：裁剪前梯度范数的 epoch 均值 + 命中裁剪上限的
        # batch 占比 + 非有限范数占比。**所有 run 都记**，因为它只有跨 run 对比才有意义
        # （"某个干预是不是把训练推到了裁剪墙上"必须与锚 run 的同一条曲线比）。
        # 分母：走到裁剪那一步的 batch 数 == processed_batches（loss 非有限的坏 batch 在
        # backward 之前就 continue 了，裁剪与 processed_batches++ 之间没有别的分支）；
        # 均值/命中率再收窄到**范数有限**的那些 batch（AMP 的 scale 探顶会给出 inf）。
        _nb_gn = max(processed_batches, 1)
        _nfin = gn_fin_cnt.clamp_min(1.0)
        gn_log = {
            "Grad Norm": (gn_sum / _nfin).item(),
            "Clip Hit Frac": (clip_hits / _nfin).item(),
            "Grad Nonfinite Frac": ((_nb_gn - gn_fin_cnt) / _nb_gn).item(),
        }

        # V2 读数：判别器损失 + 域判别准确率 + 当前 λ。这是"机制咬合与否"的四象限判据
        # （见 config 头）：acc 降向 1/N ⇒ 特征去场景化成功；acc 不降 ⇒ λ/结构没咬合。
        dann_log = {}
        if dann_enabled:
            # 变量名带 d 前缀：上面 U2 的块里也有一个 `_seen`，别互相覆盖。
            _dseen = float(dann_seen.item())
            dann_log = {
                "DANN Loss": torch.true_divide(
                    dann_loss_sum, max(processed_batches, 1)).detach().item(),
                "DANN Domain Acc": (float(dann_correct.item()) / _dseen) if _dseen > 0
                                   else float('nan'),
                "DANN Lambda": float(getattr(net_val, 'dann_lambda', 0.0)),
            }

        # U1 读数：SK 适配器的分支选择权重 / 选择熵 / 残差范数比（epoch 均值，只统计
        # 训练前向）。pop_* 会清零累加器，所以每个 epoch 必须【恰好取一次】——放在
        # do_validate 分支之前，两条 wandb.log 路径共用同一份。
        sk_log = {}
        if hasattr(net_val, 'pop_sk_stats'):
            _sk = net_val.pop_sk_stats()
            if _sk:
                sk_log = _sk

        # 多任务读数：只在配了 aux_charts 时新增键（无 aux 的 run 曲线面板保持不变）。
        # `Train {target} Loss` 是【target 分量】的 epoch 均值，与只含 target 的 val loss
        # 同量纲，train/val gap 要看这条而不是含 aux 的 "Train Epoch Loss"。
        aux_log = {}
        if aux_present:
            aux_log[f"Train {target_chart} Loss"] = torch.true_divide(
                primary_ce_loss_sum, max(processed_batches, 1)).detach().item()

        # 保存本epoch的像素分布统计（支持 num_workers > 0）
        patch_log_path = osp.join(cfg.work_dir, 'patch_sampling_log.csv')
        save_epoch_sod_distribution(
            patch_log_path,
            epoch,
            primary_epoch_class_counts,
            primary_epoch_mask_count,
            chart=target_chart)

        # Dynamic FocalLoss weight update (inverse frequency from actual pixel distribution)
        _dyn_enabled = train_options.get('dynamic_loss_weight', False)
        _dyn_warmup  = train_options.get('dynamic_weight_warmup', 5)
        if (_dyn_enabled
                and epoch >= _dyn_warmup
                and primary_epoch_class_counts.sum() > 0
                and target_chart in loss_ce_functions):
            _counts = primary_epoch_class_counts.astype(np.float64) + 1.0  # +1 avoid /0
            _freq   = _counts / _counts.sum()
            _raw_w  = 1.0 / _freq
            _raw_w  = _raw_w / _raw_w.mean()          # normalise to mean=1
            _max_w  = float(train_options.get('dynamic_weight_max', 5.0))
            _raw_w  = np.clip(_raw_w, 1.0 / _max_w, _max_w)
            _raw_w  = _raw_w / _raw_w.mean()          # re-normalise after clip
            _ema    = float(train_options.get('dynamic_weight_ema', 0.7))
            _loss_fn = loss_ce_functions[target_chart]
            if hasattr(_loss_fn, 'weight') and _loss_fn.weight is not None:
                _old_w = _loss_fn.weight.detach().cpu().numpy().astype(np.float64)
                _new_w = _ema * _old_w + (1.0 - _ema) * _raw_w
                _loss_fn.weight.data.copy_(torch.FloatTensor(_new_w).to(device))
                print(f"  [DynW] {target_chart} loss weights: {[f'{w:.3f}' for w in _new_w]}")
                wandb.log(
                    {f"{target_chart}/DynLossWeight_cls{c}": float(w)
                     for c, w in enumerate(_new_w)},
                    step=epoch)
            else:
                print(f"  [DynW] {target_chart} loss ({type(_loss_fn).__name__}) has no per-class weight, skipping.")

        # 计算并打印/记录训练集 mIoU
        if compute_train_primary_iou:
            train_iou_per_class = train_iou_metric.compute()  # shape: (n_classes,)
            train_miou = train_iou_per_class.mean()
            train_iou_metric.reset()

            # Derive train-set precision/recall/proportions/confusion-matrix from online CM.
            train_tp = torch.diag(train_primary_confmat)
            train_pred_sum = train_primary_confmat.sum(dim=0)
            train_true_sum = train_primary_confmat.sum(dim=1)
            train_precision_per_class = torch.where(
                train_pred_sum > 0, train_tp / train_pred_sum, torch.zeros_like(train_tp))
            train_recall_per_class = torch.where(
                train_true_sum > 0, train_tp / train_true_sum, torch.zeros_like(train_tp))
            train_row_sum = train_primary_confmat.sum(dim=1, keepdim=True)
            train_norm_confmat = torch.where(
                train_row_sum > 0, train_primary_confmat / train_row_sum, torch.zeros_like(train_primary_confmat))
            train_total = train_true_sum.sum().clamp_min(1.0)
            train_true_props = train_true_sum / train_total
            train_pred_props = train_pred_sum / train_total

            print(f"Train {target_chart} mIoU: {train_miou:.4f}")
            print(f"Train {target_chart} IoU per class: {train_iou_per_class}")
            print(f"Train {target_chart} Precision per class: {train_precision_per_class}")
            print(f"Train {target_chart} Recall per class: {train_recall_per_class}")
            print(f"Train {target_chart} True class proportion: {train_true_props}")
            print(f"Train {target_chart} Pred class proportion: {train_pred_props}")
            print(f"Train {target_chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(train_norm_confmat)

            wandb.log({f"Train {target_chart} mIoU": train_miou.item()}, step=epoch)
            for c, iou_c in enumerate(train_iou_per_class):
                wandb.log({f"Train {target_chart}/IoU Class {c}": iou_c.item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Precision Class {c}": train_precision_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Recall Class {c}": train_recall_per_class[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/True Proportion Class {c}": train_true_props[c].item()}, step=epoch)
                wandb.log({f"Train {target_chart}/Pred Proportion Class {c}": train_pred_props[c].item()}, step=epoch)
            for r in range(train_norm_confmat.shape[0]):
                for c in range(train_norm_confmat.shape[1]):
                    wandb.log({f"Train {target_chart}/ConfMatNorm r{r}c{c}": train_norm_confmat[r, c].item()}, step=epoch)
        else:
            train_iou_per_class = None

        #===============================================================#
        #===========================验证循环=============================#
        #===============================================================#
        val_freq = train_options.get('val_freq', 1)
        do_validate = (epoch % val_freq == 0) or (epoch == train_options['epochs'] - 1)

        if not do_validate:
            # 跳过验证，只记录训练指标
            wandb.log({"Train Epoch Loss": train_loss_epoch,
                       "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                       "Learning Rate": optimizer.param_groups[0]["lr"],
                       **supcon_log, **scale_reg_log, **sk_log, **aux_log,
                       **gn_log, **dann_log}, step=epoch)
            print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
            print(f"Grad Norm (pre-clip mean): {gn_log['Grad Norm']:.3f} | "
                  f"Clip Hit Frac: {gn_log['Clip Hit Frac']:.3f} | Nonfinite Frac: "
                  f"{gn_log['Grad Nonfinite Frac']:.3f} (max_norm={grad_clip_max_norm})")
            if supcon_enabled:
                print(f"Train SupCon Epoch Loss: {supcon_epoch:.3f}")
            if scale_reg_enabled:
                print(f"Train ScaleReg Epoch Loss: {scale_reg_log['ScaleReg Loss']:.4f}")
            if dann_enabled:
                print(f"Train DANN Loss: {dann_log['DANN Loss']:.4f} | Domain Acc: "
                      f"{dann_log['DANN Domain Acc']:.4f} | λ={dann_log['DANN Lambda']:.4f}")
            continue

        # 逐景累加 (n_classes × n_classes) 混淆矩阵，而不是把所有场景的像素 cat 成扁平张量。
        # 2026-07-10：原写法在 down_sample_scale=1 下必 OOM——30 个验证景约 8 亿有效像素，
        # true/pred 各一份、torchmetrics 内部再要 7.46GB，合计 27.6GB > 32GB 显存。
        # 所有验证指标都是混淆矩阵的纯函数（等价性已逐位验证），改累加后显存与像素数无关。
        torch.cuda.empty_cache()  # 释放训练残留的缓存显存，为滑窗推理腾出空间
        val_confmats = {chart: new_confusion_matrix(train_options['n_classes'][chart], device)
                        for chart in val_charts}
        net.eval()  # Set network to evaluation mode.
        # H3：开 EMA 时验证走 EMA 权重（best 选择/早停/保存与部署口径一致）。
        eval_net = ema_net if ema_net is not None else net_val
        if ema_net is not None:
            # EMA 滞后读数：‖ema−raw‖/‖raw‖。恒 0 = 影子没在更新（接线断了）；
            # 随训练先升后稳 = 正常。逐 epoch 记 wandb，与 mIoU 曲线对齐看。
            with torch.no_grad():
                _num, _den = 0.0, 0.0
                _msd = net_val.state_dict()
                for k, v in ema_net.state_dict().items():
                    if v.dtype.is_floating_point:
                        _num += (v - _msd[k]).float().pow(2).sum().item()
                        _den += _msd[k].float().pow(2).sum().item()
            _ema_lag = (_num ** 0.5) / max(_den ** 0.5, 1e-12)
            print(f'[EMA] Δ‖ema−raw‖/‖raw‖ = {_ema_lag:.5f}')
            wandb.log({"EMA Lag": _ema_lag}, step=epoch)
        print('Validating...')
        # - Loops though scenes in queue.
        for i, (inf_x, inf_y, cfv_masks, tfv_mask, name, original_size) in enumerate(tqdm(iterable=dataloader_val,
                                                                            total=len(train_options['validate_list']),
                                                                            colour='green')):
            torch.cuda.empty_cache()
            # Reset from previous batch.
            # train fill value mask
            # tfv_mask = (inf_x.squeeze()[0, :, :] == train_options['train_fill_value']).squeeze()
            # 有效像元掩膜（label != 255）。原生分辨率下最大景 2.5 亿像素，这个 bool 就有
            # 250MB，算一次给「tile 门控」和「混淆矩阵筛选」共用，别取反两遍。
            valid_masks = {c: ~cfv_masks[c] for c in val_charts}

            # - Ensures that no gradients are calculated, which otherwise take up a lot of space on the GPU.
            with torch.no_grad(), torch.cuda.amp.autocast():
                #==================推理：Swin用批量滑窗，其余大景用批量非重叠tile
                # 2026-07-25：大景改走 fast_tiled_val_predict —— ① 只前向【含有效标签】的
                # tile（val_417 实测 66.1% 的 tile 标签全是 255，对混淆矩阵和 ignore_index
                # 的 loss 贡献严格为 0，跳过它指标逐位不变）；② 整景全程留在 CPU，不再把
                # 最大 8ch×6767×37468=8.1GB 的整景搬上 GPU 再搬回来；③ 不开 n_classes×H×W
                # 的整景 logits 缓冲（最大景 3.7GB）、不逐 tile 拷 logits。
                # 等价性见 functions.fast_tiled_val_predict 的 docstring。
                use_fast_tiled = (train_options['model_selection'] != 'swin' and
                                  (inf_x.shape[2] > train_options['patch_size'] or
                                   inf_x.shape[3] > train_options['patch_size']))
                if use_fast_tiled:
                    # 2026-08-06：默认走 out='confmat'——逐批在 GPU 上累加混淆矩阵，
                    # 去掉 preds 模式的逐批 .item()/.cpu() 同步、整景 uint8 预测图、
                    # 以及下面那段「CPU 布尔掩膜筛选 + ~11.6 亿有效像素/轮 H2D」。
                    # 混淆矩阵与旧写法【整数级完全相等】（同批组成 ⇒ logits 逐位相同，
                    # smoke_sota.py 第 [7c] 步断言），loss 聚合算式逐位一致。
                    # val_confmat_batched=False 一键回退旧 preds 路径做 A-B。
                    _batched_cm = train_options.get('val_confmat_batched', True)
                    pred_maps = None
                    if _batched_cm:
                        scene_cms, val_ce = fast_tiled_val_predict(
                            inf_x, eval_net, val_options,
                            labels=inf_y,
                            valid_maps=valid_masks,
                            loss_fns=loss_ce_functions,
                            task_weights=val_options['task_weights'],
                            gate=train_options.get('val_tile_gate', True),
                            out='confmat')
                    else:
                        pred_maps, val_ce = fast_tiled_val_predict(
                            inf_x, eval_net, val_options,
                            labels=inf_y,
                            valid_maps=valid_masks,
                            # loss_ce_functions 含全部 charts（训练要用），函数内部只按
                            # options['charts'] = val_charts 取用，多余的键不会被碰。
                            loss_fns=loss_ce_functions,
                            task_weights=val_options['task_weights'],
                            # 一键回退/做 A-B 用；关掉后只剩「不开整景 logits 缓冲」的收益
                            gate=train_options.get('val_tile_gate', True))
                else:
                    inf_x = inf_x.to(device, non_blocking=True)
                    if train_options['model_selection'] == 'swin':
                        output = batched_slide_inference(inf_x, eval_net, val_options, 'val')
                    else:
                        output = eval_net(inf_x)

                    val_ce = 0.0
                    for chart, weight in zip(val_charts, val_options['task_weights']):
                        val_ce += weight * loss_ce_functions[chart](
                            output[chart],
                            inf_y[chart].unsqueeze(0).long().to(device)).item()
                    pred_maps = {c: class_decider(output[c], val_options, c)
                                 for c in val_charts}
                    del output

            # - Final output layer, and accumulation of non masked pixels into the confusion matrix.
            if use_fast_tiled and pred_maps is None:
                # confmat 模式：场景级 (n,n) 混淆矩阵已在 GPU 上算好，直接累进全局。
                for chart in val_charts:
                    val_confmats[chart] += scene_cms[chart]
                del scene_cms
            else:
                for chart in val_charts:
                    _sel = valid_masks[chart]
                    _p = pred_maps[chart]
                    # fast 路径的预测图在 CPU（uint8），先在 CPU 上做掩膜筛选、只把有效像素送上
                    # GPU；滑窗路径的预测图在 GPU 上，保持原来的索引方式。
                    _pred = (_p[_sel] if _p.device.type == 'cpu' else _p[_sel.to(_p.device)])
                    _pred = _pred.to(device, non_blocking=True)
                    _true = inf_y[chart][_sel].to(device, non_blocking=True)
                    accumulate_confusion_matrix(val_confmats[chart], _true, _pred,
                                                train_options['n_classes'][chart])
                    del _pred, _true
            # - Add batch loss.
            val_loss_sum += val_ce
            val_cross_entropy_loss_sum += val_ce

        #====================验证结束后，计算综合指标
        val_loss_epoch = torch.true_divide(val_loss_sum, i + 1).detach().item()
        val_cross_entropy_epoch = torch.true_divide(val_cross_entropy_loss_sum, i + 1).detach().item()

        # - Compute the relevant scores.  全部指标都从 val_confmats 导出（等价性已逐位验证，
        #   见 functions.py 里"基于混淆矩阵的验证指标"一节）。
        print('Computing Metrics on Val dataset')
        scores = {}
        for chart in val_charts:
            _func_name = train_options['chart_metric'][chart]['func'].__name__
            if _func_name != 'f1_metric':
                raise NotImplementedError(
                    f"chart_metric[{chart}]['func'] = {_func_name}；混淆矩阵版指标目前只实现了 "
                    f"f1_metric（加权 F1）。SIC 的 r2_metric 是连续值指标，无法从混淆矩阵导出。")
            scores[chart] = torch.round(cm_weighted_f1(val_confmats[chart]) * 100, decimals=3)
        combined_score = compute_combined_score(scores=scores, charts=val_charts,
                                                metrics=train_options['chart_metric'])

        # 始终计算 mIoU，用于按目标任务 mIoU 保存最佳模型。
        miou_scores = {c: cm_miou(val_confmats[c]) for c in val_charts}
        primary_miou = miou_scores[target_chart]

        # 有序代价指标（方案1/4/7）：临近错误-1分，远处错误-2分
        _ordinal_cfg = train_options.get('ordinal_metric', {})
        if _ordinal_cfg.get('enabled', False):
            _cpd = _ordinal_cfg.get('cost_per_distance', None)
            ordinal_raw_costs, ordinal_norm_scores = {}, {}
            for chart in val_charts:
                ordinal_raw_costs[chart], ordinal_norm_scores[chart] = cm_ordinal_cost(
                    val_confmats[chart], cost_per_distance=_cpd)
        else:
            ordinal_raw_costs, ordinal_norm_scores = None, None

        # 统一计算 per-class IoU，便于做 train-val IoU gap 分析
        classwise_iou_scores = {c: cm_classwise_iou(val_confmats[c]) for c in val_charts}

        # 计算全类别 Precision/Recall、归一化混淆矩阵、预测占比 vs 真实占比
        classwise_precision_scores, classwise_recall_scores = {}, {}
        normalized_conf_mats, true_class_props, pred_class_props = {}, {}, {}
        for chart in val_charts:
            classwise_precision_scores[chart], classwise_recall_scores[chart] = \
                cm_precision_recall(val_confmats[chart])
            normalized_conf_mats[chart] = cm_normalized(val_confmats[chart])
            true_class_props[chart], pred_class_props[chart] = cm_class_proportions(val_confmats[chart])

        if train_options['compute_classwise_f1score']:
            classwise_scores = {c: cm_classwise_f1(val_confmats[c]) for c in val_charts}
            oa_scores = {c: cm_overall_accuracy(val_confmats[c]) for c in val_charts}

        del val_confmats
        import gc; gc.collect()
        torch.cuda.empty_cache()

        print("")
        print(f"Epoch {epoch} score:")

        for chart in val_charts:
            # 跳过 task_weight=0 的任务，避免无意义的指标日志
            if train_options['chart_metric'][chart]['weight'] == 0:
                continue

            print(f"{chart} {train_options['chart_metric'][chart]['func'].__name__}: {scores[chart]}%")

            # Log in wandb the SIC r2_metric, SOD f1_metric and FLOE f1_metric
            wandb.log({f"{chart} {train_options['chart_metric'][chart]['func'].__name__}": scores[chart]}, step=epoch)

            #===============计算标签各类别的F1分数
            if train_options['compute_classwise_f1score']:
                for index, class_score in enumerate(classwise_scores[chart]):
                    wandb.log({f"{chart}/Class: {index}": class_score.item()}, step=epoch)
                print(f"{chart} F1 score:", classwise_scores[chart])

                # 打印或记录 OA
                print(f"{chart} OA:", oa_scores[chart])
                wandb.log({f"{chart}/OA": oa_scores[chart]}, step=epoch)

                # 打印/记录 mIoU
                print(f"{chart} mIoU:", miou_scores[chart])
                wandb.log({f"{chart}/mIoU": miou_scores[chart]}, step=epoch)

                # 打印/记录 per-class IoU
                print(f"{chart} IoU per class:", classwise_iou_scores[chart])
                for c, iou_c in enumerate(classwise_iou_scores[chart]):
                    wandb.log({f"{chart}/IoU Class {c}": iou_c.item()}, step=epoch)

            # 打印/记录有序代价指标（方案1/4/7）
            if ordinal_raw_costs is not None and chart in ordinal_raw_costs:
                print(f"{chart} Ordinal Mean Cost (lower=better): {ordinal_raw_costs[chart]:.4f}")
                print(f"{chart} Ordinal Score (higher=better):    {ordinal_norm_scores[chart]:.4f}")
                wandb.log({f"{chart}/OrdinalMeanCost":  ordinal_raw_costs[chart].item(),
                           f"{chart}/OrdinalScore":     ordinal_norm_scores[chart].item()}, step=epoch)

            # 打印/记录 per-class Precision
            print(f"{chart} Precision per class:", classwise_precision_scores[chart])
            for c, prec_c in enumerate(classwise_precision_scores[chart]):
                wandb.log({f"{chart}/Precision Class {c}": prec_c.item()}, step=epoch)

            # 打印/记录 per-class Recall
            print(f"{chart} Recall per class:", classwise_recall_scores[chart])
            for c, rec_c in enumerate(classwise_recall_scores[chart]):
                wandb.log({f"{chart}/Recall Class {c}": rec_c.item()}, step=epoch)

            # 打印/记录 预测占比 vs 真实占比
            print(f"{chart} True class proportion:", true_class_props[chart])
            print(f"{chart} Pred class proportion:", pred_class_props[chart])
            for c, p_true in enumerate(true_class_props[chart]):
                wandb.log({f"{chart}/True Proportion Class {c}": p_true.item()}, step=epoch)
            for c, p_pred in enumerate(pred_class_props[chart]):
                wandb.log({f"{chart}/Pred Proportion Class {c}": p_pred.item()}, step=epoch)

            # 打印/记录 归一化混淆矩阵（按真实类别行归一化）
            cm_norm = normalized_conf_mats[chart]
            print(f"{chart} Normalized Confusion Matrix (row=true, col=pred):")
            print(cm_norm)
            for r in range(cm_norm.shape[0]):
                for c in range(cm_norm.shape[1]):
                    wandb.log({f"{chart}/ConfMatNorm r{r}c{c}": cm_norm[r, c].item()}, step=epoch)

        # 计算 Train-Val IoU gap
        if compute_train_primary_iou and train_iou_per_class is not None and target_chart in classwise_iou_scores:
            train_val_iou_gap = train_iou_per_class - classwise_iou_scores[target_chart]
            print(f"{target_chart} Train-Val IoU gap per class:", train_val_iou_gap)
            for c, gap_c in enumerate(train_val_iou_gap):
                wandb.log({f"{target_chart}/Train-Val IoU Gap Class {c}": gap_c.item()}, step=epoch)

        # ReduceLROnPlateau：用目标任务 mIoU 驱动 lr 调整，与模型保存标准保持一致
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_before = optimizer.param_groups[0]['lr']
            scheduler.step(primary_miou)
            lr_after = optimizer.param_groups[0]['lr']
            if lr_after < lr_before:
                print(f"ReduceLROnPlateau: lr {lr_before:.2e} → {lr_after:.2e}")

        print(f"Combined score: {combined_score}%")
        print(f"Train Epoch Loss: {train_loss_epoch:.3f}")
        print(f"Train Cross Entropy Epoch Loss: {cross_entropy_epoch:.3f}")
        print(f"Validation Epoch Loss: {val_loss_epoch:.3f}")
        print(f"Validation Cross Entropy Epoch Loss: {val_cross_entropy_epoch:.3f}")
        print(f"Validation {target_chart} mIoU: {primary_miou.item():.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")

        # Log combine score and epoch loss to wandb
        wandb.log({"Combined score": combined_score,    # 综合得分（验证集）
                   f"Validation {target_chart} mIoU": primary_miou.item(),
                   "Train Epoch Loss": train_loss_epoch,
                   "Train Cross Entropy Epoch Loss": cross_entropy_epoch,
                   "Validation Epoch Loss": val_loss_epoch,
                   "Validation Cross Entropy Epoch Loss": val_cross_entropy_epoch,
                   "Learning Rate": optimizer.param_groups[0]["lr"],
                   **supcon_log, **scale_reg_log, **sk_log, **aux_log,
                   **gn_log, **dann_log}, step=epoch)
        print(f"Grad Norm (pre-clip mean): {gn_log['Grad Norm']:.3f} | "
              f"Clip Hit Frac: {gn_log['Clip Hit Frac']:.3f} | Nonfinite Frac: "
              f"{gn_log['Grad Nonfinite Frac']:.3f} (max_norm={grad_clip_max_norm})")
        if supcon_enabled:
            print(f"Train SupCon Epoch Loss: {supcon_epoch:.3f}")
        if scale_reg_enabled:
            print(f"Train ScaleReg Epoch Loss: {scale_reg_log['ScaleReg Loss']:.4f} "
                  f"| Pred per class 1..5: "
                  + ', '.join(f"{scale_reg_log[f'ScaleReg Pred cls{c}']:.2f}"
                              for c in range(1, len(_SCALE_REG_MU) + 1)))
        if dann_enabled:
            print(f"Train DANN Loss: {dann_log['DANN Loss']:.4f} | Domain Acc: "
                  f"{dann_log['DANN Domain Acc']:.4f}（随机基线 "
                  f"{1.0 / int(train_options['segformer_dann_domains']):.4f}）"
                  f" | λ={dann_log['DANN Lambda']:.4f}")
        if aux_present:
            print(f"Train {target_chart} Loss (target 分量): "
                  f"{aux_log[f'Train {target_chart} Loss']:.3f}")

        # -- Attention Residual alpha 监控（仅 Swin 模型）--
        # alpha 零初始化，若训练中逐渐增大说明模型学到了跨层注意力残差的价值。
        # 若始终为 0 且梯度为 0，说明残差链路存在问题（如被意外截断）。
        # 注意：alpha 变化≠效果好，最终判断依据仍是验证集 F1。
        if (train_options['model_selection'] == 'swin'
                and hasattr(net_val, 'layers')
                and len(net_val.layers) > 2):
            stage2_alphas = torch.stack([
                blk.attn.alpha.detach().cpu()
                for blk in net_val.layers[2].blocks
            ])  # [num_blocks=6, num_heads=12]
            alpha_mean = stage2_alphas.mean().item()
            alpha_max  = stage2_alphas.max().item()
            # 每个 block 在所有 head 上的均值（偶数=W-MSA，奇数=SW-MSA）
            alpha_per_block = stage2_alphas.mean(dim=1).tolist()
            print(f"[AttnResidual] Stage2 alpha | "
                  f"mean={alpha_mean:.4f}  max={alpha_max:.4f}  "
                  f"per_block(W/SW alternating)={[f'{v:.4f}' for v in alpha_per_block]}")
            wandb.log({
                "AttnResidual/alpha_mean": alpha_mean,
                "AttnResidual/alpha_max":  alpha_max,
                **{f"AttnResidual/block{i}_alpha_mean": v
                   for i, v in enumerate(alpha_per_block)},
            }, step=epoch)

        # 决定本epoch是否保存最佳模型（方案1/4用mIoU，方案7用OrdinalScore）
        if model_save_criterion == 'ordinal_score' and ordinal_norm_scores is not None:
            _current_criterion = ordinal_norm_scores.get(target_chart, torch.tensor(-np.Inf))
            _current_criterion = _current_criterion.item() if hasattr(_current_criterion, 'item') else float(_current_criterion)
            _is_best = _current_criterion > best_ordinal_score
            if _is_best:
                best_ordinal_score = _current_criterion
        else:
            # 平滑准则：最近 val_smoothing_window 次验证 mIoU 的均值（window=1 时退化为
            # 单 epoch，与历史逐位一致）。best 保存的仍是【当前 epoch】权重，但"是否最佳"
            # 由平滑分判定——即选平滑分最高那个 epoch 的模型。
            _val_hist.append(primary_miou.item())
            _current_criterion = sum(_val_hist[-val_smoothing_window:]) / len(_val_hist[-val_smoothing_window:])
            if val_smoothing_window > 1:
                print(f"Smoothed val mIoU (w={val_smoothing_window}): {_current_criterion:.4f} "
                      f"(raw {primary_miou.item():.4f})")
            _is_best = _current_criterion > best_primary_miou
            if _is_best:
                best_primary_miou = _current_criterion

        # If mIoU is better than previous epochs, save best model.
        if _is_best:
            early_stop_counter = 0

            # Log key metrics for the current best mIoU model.
            wandb.run.summary[f"While training/Best {target_chart} mIoU"] = best_primary_miou
            # scores 只在 val_charts 上算过（辅任务不参与验证），这里必须同源取值。
            for chart in val_charts:
                wandb.run.summary[f"While training/{chart} {train_options['chart_metric'][chart]['func'].__name__}"] = scores[chart]
            wandb.run.summary[f"While training/Validation {target_chart} mIoU"] = primary_miou.item()
            wandb.run.summary[f"While training/Train Epoch Loss"] = train_loss_epoch

            # Save the best model in work_dirs (仅本地落盘，不上传到 wandb)
            # H3：开 EMA 时保存 EMA 权重（它才是被验证选中的那份模型；optimizer/scheduler
            # 状态仍是 raw 训练的——resume 语义因此退化，EMA run 崩溃后倾向重跑而非续跑）。
            # 断点状态快照仅对 miou 判据有意义：ordinal_score 判据下 best_primary_miou
            # 恒为 -inf、_val_hist 恒空，存进去会在续跑时以"已还原"的假象关掉旧格式
            # WARN（code-review #1）——此时显式传 None，让旧格式告警诚实触发。
            _snap = None
            if model_save_criterion == 'miou':
                _snap = {
                    # 此刻 early_stop_counter 恰为 0（上面刚重置）
                    'best_primary_miou': best_primary_miou,
                    'val_hist_tail': list(_val_hist[-val_smoothing_window:]),
                    'early_stop_counter': early_stop_counter,
                    'val_smoothing_window': val_smoothing_window,
                    'ema': bool(ema_enabled),
                }
            model_path = save_best_model(cfg, train_options,
                                         ema_net if ema_net is not None else net,
                                         optimizer, scheduler, epoch,
                                         resume_state=_snap)
        else:
            early_stop_counter += 1
            print(f"Early stopping counter: {early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch}. Best {target_chart} mIoU: {best_primary_miou:.4f}")
                break

    return model_path


def create_dataloaders(train_options):
    '''
    Create train and validation dataloader based on the train and validation list inside train_options.

    '''
    # Custom dataset and dataloader.
    dataset = AI4ArcticChallengeDataset(
        files=train_options['train_list'], options=train_options, do_transform=True)

    prefetch = train_options.get('prefetch_factor', 2) if train_options['num_workers'] > 0 else None
    dataloader_train = torch.utils.data.DataLoader(
        dataset, batch_size=None, shuffle=True, num_workers=train_options['num_workers'],
        pin_memory=True, prefetch_factor=prefetch)
    # - Setup of the validation dataset/dataloader. The same is used for model testing in 'test_upload.ipynb'.

    # 验证按需读是【显式 opt-in】的（默认关，见 loaders.val_ondemand_ok）：这个 dataset
    # 还被 TTA / 可视化 / 推理导出等脚本使用，它们不满足「只在对齐的 tile 上前向」这个
    # 前提。这里判定一次并把原因打进日志，便于事后追溯某个 run 到底走了哪条读法。
    # 验证 dataset 只读/返回 target chart 的标签与掩膜（辅助任务 train-only），与 train()
    # 里 val_charts 的消费端同源；无 aux_charts 时 val_options is train_options。
    val_options = val_scoped_options(train_options)
    _od_ok, _od_why = val_ondemand_ok(val_options)
    print(f"[Val] 按需读 = {'ON' if _od_ok else 'OFF'}（{_od_why}）")
    dataset_val = AI4ArcticChallengeTestDataset(
        options=val_options, files=train_options['validate_list'], mode='train',
        ondemand_read=_od_ok)

    # prefetch_factor 显式钉成 1：验证的一个样本是一整景（最大 8ch×6767×37468≈8GB），
    # DataLoader 默认 prefetch_factor=2 意味着 num_workers×2 个整景同时在飞。验证期
    # 内存峰值是 07-23 两次宿主机级 SIGKILL 的怀疑对象，这里把在飞景数钉死在
    # num_workers+1，不让它随 worker 数二次放大。
    _nw_val = train_options['num_workers_val']
    dataloader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=None, num_workers=_nw_val, shuffle=False,
        **({'prefetch_factor': 1} if _nw_val > 0 else {}))

    return dataloader_train, dataloader_val
    # return dataloader_val


def _set_global_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _setup_device(train_options):
    # Get GPU resources.
    if torch.cuda.is_available():
        print(colour_str('GPU available!', 'green'))
        print('Total number of available devices: ',
              colour_str(torch.cuda.device_count(), 'orange'))

        # Check if NVIDIA V100, A100, or H100 is available for torch compile speed up
        if train_options['compile_model']:
            gpu_ok = False
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                gpu_ok = True

            if not gpu_ok:
                warnings.warn(
                    colour_str("GPU is not NVIDIA V100, A100, or H100. Speedup numbers may be lower than expected.", 'red')
                )

        # Setup device to be used
        device = torch.device(f"cuda:{train_options['gpu_id']}")
    else:
        print(colour_str('GPU not available.', 'red'))
        device = torch.device('cpu')
    print('GPU setup completed!')
    return device


def _collect_cv_folds(train_options):
    fold_dir = train_options.get('cv_fold_dir', None)
    if not fold_dir:
        return []

    fold_dir_rel = fold_dir.replace('\\', '/').rstrip('/')
    fold_dir_abs = osp.join(train_options['path_to_env'], fold_dir_rel)
    if not osp.isdir(fold_dir_abs):
        raise FileNotFoundError(f'cv_fold_dir not found: {fold_dir_abs}')

    train_pat = re.compile(r'^fold(\d+)_train\.json$')
    fold_ids = []
    for name in os.listdir(fold_dir_abs):
        m = train_pat.match(name)
        if m is not None:
            fold_ids.append(int(m.group(1)))

    fold_ids = sorted(set(fold_ids))
    if not fold_ids:
        return []

    requested_num = int(train_options.get('cv_num_folds', 0) or 0)
    if requested_num > 0:
        fold_ids = fold_ids[:requested_num]

    folds = []
    for fold_id in fold_ids:
        train_rel = f'{fold_dir_rel}/fold{fold_id}_train.json'
        val_rel = f'{fold_dir_rel}/fold{fold_id}_val.json'
        train_abs = osp.join(train_options['path_to_env'], train_rel)
        val_abs = osp.join(train_options['path_to_env'], val_rel)
        if not osp.exists(train_abs):
            raise FileNotFoundError(f'Missing fold train list: {train_abs}')
        if not osp.exists(val_abs):
            raise FileNotFoundError(f'Missing fold val list: {val_abs}')
        folds.append({'fold': fold_id, 'train_list_path': train_rel, 'val_path': val_rel})

    return folds


def _read_json_list(path):
    with open(path) as f:
        return json.loads(f.read())


def _run_training_job(cfg, args, train_options, device, fold_tag=None):
    #===========根据"model_selection"选择模型，并设置优化器和学习率调度器
    net = get_model(train_options, device)
    if train_options['compile_model']:
        net = torch.compile(net)
    optimizer = get_optimizer(train_options, net)
    scheduler = get_scheduler(train_options, optimizer)

    epoch_start = 0
    #===========恢复训练或微调模型
    # 顺序CV模式下禁止 resume/finetune（每折都是全新训练）
    resume_state = None
    if fold_tag is None:
        if args.resume_from is not None:
            print(f"\033[91m Resuming work from {args.resume_from}\033[0m")
            epoch_start, resume_state = load_model(net, args.resume_from, optimizer, scheduler,
                                                   return_resume_state=True)
        elif args.finetune_from is not None:
            print(f"\033[91m Finetune model from {args.finetune_from}\033[0m")
            _ = load_model(net, args.finetune_from)

    #===========Wandb设置
    run_name = args.wandb_name if fold_tag is None else f"{args.wandb_name}_fold{fold_tag}"
    if not train_options['cross_val_run']:
        wandb.init(name=run_name, project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)
    else:
        wandb.init(name=run_name, group=osp.splitext(osp.basename(args.config))[0], project=args.wandb_project,
                   entity="liwannian-zhejiang-university", config=train_options)

    # Define the metrics and make them such that they are not added to the summary
    wandb.define_metric("Train Epoch Loss", summary="none")     # 训练集上的总损失
    wandb.define_metric("Train Cross Entropy Epoch Loss", summary="none") # 训练集上的交叉熵损失
    wandb.define_metric("Train SupCon Epoch Loss", summary="none") # F3 跨景区域 SupCon 分量（未乘 λ）
    for _k in ("SupCon Active Batch Frac", "SupCon Same-class Cos",
               "SupCon Diff-class Cos", "SupCon Cos Gap"):
        wandb.define_metric(_k, summary="none")  # F3 预注册读数曲线
    wandb.define_metric("EMA Lag", summary="none")  # H3 EMA 滞后读数
    if len(train_options['charts']) > 1:   # 多任务：target 分量曲线（无 aux 时不新增键）
        wandb.define_metric(f"Train {train_options.get('target_chart', 'FLOE')} Loss", summary="none")
    wandb.define_metric("Train Water Consistency Epoch Loss", summary="none") # 训练集上的水体一致性损失
    wandb.define_metric("Validation Epoch Loss", summary="none") # 验证集上的总损失
    wandb.define_metric("Validation Cross Entropy Epoch Loss", summary="none") # 验证集上的交叉熵损失
    wandb.define_metric("Validation Water Consistency Epoch Loss", summary="none") # 验证集上的水体一致性损失
    wandb.define_metric("Combined score", summary="none") # 综合得分
    wandb.define_metric("SIC r2_metric", summary="none") # SIC的R2指标
    wandb.define_metric("SOD f1_metric", summary="none") # SOD的F1指标
    wandb.define_metric("FLOE f1_metric", summary="none") # FLOE的F1指标
    wandb.define_metric(f"Validation {train_options.get('target_chart', 'FLOE')} mIoU", summary="none") # 验证集目标任务的mIoU
    wandb.define_metric("Water Consistency Accuarcy", summary="none") # 水体一致性准确率
    wandb.define_metric("Learning Rate", summary="none") # 学习率
    wandb.save(str(args.config))
    print(colour_str('Save Config File', 'green'))

    # ===========创建数据加载器
    create_train_validation_and_test_scene_list(train_options)
    dataloader_train, dataloader_val = create_dataloaders(train_options) # dataloader_val 用于训练过程中验证数据

    wandb.config['validate_list'] = train_options['validate_list']
    print('Data setup complete.')

    print('-----------------------------------')
    print('Starting Training')
    print('-----------------------------------')
    if args.resume_from is not None and fold_tag is None:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler, epoch_start, resume_state=resume_state)
    else:
        checkpoint_path = train(cfg, train_options, net, device, dataloader_train, dataloader_val, optimizer,
                                scheduler)

    print('-----------------------------------')
    print('Training Complete')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Staring Validation with best model')
    print('-----------------------------------')

    # 释放训练占用的 GPU 内存，避免最终验证时 OOM
    del dataloader_train, dataloader_val
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # this is for valset 1 visualization along with gt
    # test('val', net, checkpoint_path, device, cfg.deepcopy(), train_options['validate_list'], 'Cross Validation')

    print('-----------------------------------')
    print('Completed validation')
    print('-----------------------------------')

    print('-----------------------------------')
    print('Starting testing with best model')
    print('-----------------------------------')

    # this is for test path along with gt after the gt has been released
    # test('test', net, checkpoint_path, device, cfg.deepcopy(), train_options['test_list'], 'Test', train_options['test_list_reference'])

    print('-----------------------------------')
    print('Completed testing')
    print('-----------------------------------')

    # finish the wandb run
    wandb.finish()

    # Fold-level cleanup; global scene cache stays alive for next fold.
    del net, optimizer, scheduler
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def main():
    # 必须在任何 torch CPU 算子之前设置：AutoDL 容器的 nproc 报宿主机 208 核，而 cgroup
    # 只给 25 核配额，torch 默认按 208 建 intraop 线程池，超订导致 CPU 侧算子慢几百倍
    # （2026-07-09 实测：一个 batch 的组装 0.55s → 205s）。详见 loaders.limit_torch_threads。
    limit_torch_threads()

    args = parse_args()
    ic(args.config)
    cfg = Config.fromfile(args.config)
    train_options = copy.deepcopy(cfg.train_options)
    # Get options for variables, amsrenv grid, cropping and upsampling.
    train_options = get_variable_options(train_options)
    # generate wandb run id, to be used to link the run with test_upload
    id = wandb.util.generate_id()

    #===========设置随机种子
    if train_options['seed'] != -1 or args.seed is not None:
        # set seed for everything
        if args.seed is not None:
            seed = int(args.seed)
        else:
            seed = train_options['seed']
        _set_global_seed(seed)
        print(f"Seed: {seed}")
    else:
        print("Random Seed Chosen")

    #===========确定工作目录（保存日志和权重文件）
    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        base_work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        if not train_options['cross_val_run']:
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0])
        else:
            # from utils import run_names
            run_name = id
            base_work_dir = osp.join('./work_dir',
                                     osp.splitext(osp.basename(args.config))[0], run_name)
    else:
        base_work_dir = cfg.work_dir

    ic(base_work_dir)
    mkdir_or_exist(osp.abspath(base_work_dir))
    # dump config
    shutil.copy(args.config, osp.join(base_work_dir, osp.basename(args.config)))

    #=========== GPU设置
    device = _setup_device(train_options)

    # Sequential CV: one process, one-time preload, loop over folds.
    if train_options.get('sequential_cv', False):
        if args.resume_from is not None or args.finetune_from is not None:
            raise ValueError('sequential_cv mode does not support --resume-from/--finetune-from')

        folds = _collect_cv_folds(train_options)
        if not folds:
            raise ValueError('No folds discovered. Please set train_options["cv_fold_dir"] correctly.')

        print(f'Sequential CV mode enabled. Total folds: {len(folds)}')

        # Optional one-time preload for all fold train scenes.
        if train_options.get('preload_all_cv_scenes', True):
            all_train_files = []
            for fold in folds:
                train_json_abs = osp.join(train_options['path_to_env'], fold['train_list_path'])
                all_train_files.extend(_read_json_list(train_json_abs))
            print(f'Preloading unique train scenes across folds: {len(set(all_train_files))}')
            preload_scene_cache(train_options, all_train_files)

        for fold in folds:
            fold_id = fold['fold']
            print('===================================')
            print(f'Starting fold {fold_id}')
            print('===================================')

            fold_cfg = cfg.deepcopy()
            fold_cfg.work_dir = osp.join(base_work_dir, f'fold{fold_id}')
            mkdir_or_exist(osp.abspath(fold_cfg.work_dir))
            shutil.copy(args.config, osp.join(fold_cfg.work_dir, osp.basename(args.config)))

            fold_options = copy.deepcopy(train_options)
            fold_options['train_list_path'] = fold['train_list_path']
            fold_options['val_path'] = fold['val_path']

            _run_training_job(fold_cfg, args, fold_options, device, fold_tag=fold_id)

            print('===================================')
            print(f'Completed fold {fold_id}')
            print('===================================')
    else:
        cfg.work_dir = base_work_dir
        _run_training_job(cfg, args, train_options, device)


if __name__ == '__main__':
    main()
