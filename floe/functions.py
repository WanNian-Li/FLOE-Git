#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helping functions for 'introduction' and 'quickstart' notebooks."""

# -- File info -- #
__author__ = 'Muhammed Patel'
__contributor__ = 'Xinwwei chen, Fernando Pena Cantu,Javier Turnes, Eddie Park'
__copyright__ = ['university of waterloo']
__contact__ = ['m32patel@uwaterloo.ca', 'xinweic@uwaterloo.ca']
__version__ = '1.0.0'
__date__ = '2024-04-05'

# -- Built-in modules -- #
import os
import json
import warnings
# -- Third-party modules -- #
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data as data
# from sklearn.metrics import r2_score, f1_score
from torchmetrics.functional import r2_score, f1_score, jaccard_index  # 修改：引入 jaccard_index
from torchmetrics.functional.classification import multiclass_confusion_matrix
import segmentation_models_pytorch as smp
from tqdm import tqdm  # Progress bar
# -- Proprietary modules -- #

from floe.utils import ICE_STRINGS, GROUP_NAMES
from floe.models.legacy.unet import UNet, Sep_feat_dif_stages  # Convolutional Neural Network model
from floe.models.legacy.swin_transformer import SwinTransformer  # Swin Transformer


def chart_cbar(ax, n_classes, chart, cmap='vridis'):
    """
    Create discrete colourbar for plot with the sea ice parameter class names.

    Parameters
    ----------
    n_classes: int
        Number of classes for the chart parameter.
    chart: str
        The relevant chart.
    """
    n_labels = len(GROUP_NAMES[chart])  # number of labelled (non-mask) classes
    arranged = np.arange(0, n_labels + 1)
    cmap = plt.get_cmap(cmap, n_labels)
    # Get colour boundaries. -0.5 to center ticks for each color.
    norm = mpl.colors.BoundaryNorm(arranged - 0.5, cmap.N)
    arranged = arranged[:-1]  # Discount the boundary sentinel.
    cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    cbar.set_label(label=ICE_STRINGS[chart])
    cbar.set_ticklabels(list(GROUP_NAMES[chart].values()))


def compute_metrics(true, pred, charts, metrics, num_classes):
    """
    Calculates metrics for each chart and the combined score. true and pred must be 1d arrays of equal length.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must be numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must be numpy array.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    combined_score: float
        Combined weighted average score.
    scores: list
        List of scores for each chart.
    """
    scores = {}
    for chart in charts:
        if true[chart].ndim == 1 and pred[chart].ndim == 1:
            scores[chart] = torch.round(metrics[chart]['func'](
                true=true[chart], pred=pred[chart], num_classes=num_classes[chart]) * 100, decimals=3)

        else:
            print(f"true and pred must be 1D numpy array, got {true['SIC'].ndim} \
                and {pred['SIC'].ndim} dimensions with shape {true['SIC'].shape} and {pred.shape}, respectively")

    combined_score = compute_combined_score(scores=scores, charts=charts, metrics=metrics)

    return combined_score, scores


def r2_metric(true, pred, num_classes=None):
    """
    Calculate the r2 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels. Must by numpy array.
    pred :
        ndarray, 1d contains all predicted pixels. Must by numpy array.
    num_classes :
        Num of classes in the dataset, this value is not used in this function but used in f1_metric function
        which requires num_classes argument. The reason it was included here was to keep the same structure.  


    Returns
    -------
    r2 : float
        The calculated r2 score.

    """
    r2 = r2_score(preds=pred, target=true)

    return r2


def f1_metric(true, pred, num_classes):
    """
    Calculate the weighted f1 metric.

    Parameters
    ----------
    true :
        ndarray, 1d contains all true pixels.
    pred :
        ndarray, 1d contains all predicted pixels.

    Returns
    -------
    f1 : float
        The calculated f1 score.

    """
    f1 = f1_score(target=true, preds=pred, average='weighted', task='multiclass', num_classes=num_classes)

    return f1


def water_edge_metric(outputs, options):
    # 需要至少3个任务才能计算跨任务水体一致性；单任务模式返回0
    if len(options['charts']) < 3:
        return torch.tensor(0.0)

    # Convert ouput into water and not water
    for chart in options['charts']:
        outputs[chart] = torch.where(outputs[chart] > 0.0, 1.0, 0.0)

    water_edge_accuracy = 1 - torch.mean(torch.abs(outputs[options['charts'][0]]-outputs[options['charts'][1]])
                                         + torch.abs(outputs[options['charts'][1]]-outputs[options['charts'][2]])
                                         + torch.abs(outputs[options['charts'][2]]-outputs[options['charts'][0]]))
    return water_edge_accuracy


def water_edge_plot_overlay(output, mask, options):
    # Convert ouput into water and not water
    charts = options['charts']
    water_chart = {}
    for chart in charts:
        water_chart[chart] = np.where(output[chart] > 0.0, 0.75, 0.0)
        water_chart[chart][mask] = np.nan
        water_chart[chart] = water_chart[chart][..., np.newaxis]

    img = np.concatenate((water_chart[charts[0]], water_chart[charts[1]], water_chart[charts[2]]), axis=2,)

    return img


def compute_combined_score(scores, charts, metrics):
    """
    Calculate the combined weighted score.

    Parameters
    ----------
    scores : List
        Score for each chart.
    charts : List
        List of charts.
    metrics : Dict
        Stores metric calculation function and weight for each chart.

    Returns
    -------
    : float
        The combined weighted score.

    """
    combined_metric = 0
    sum_weight = 0
    for chart in charts:
        combined_metric += scores[chart] * metrics[chart]['weight']
        sum_weight += metrics[chart]['weight']

    return torch.round(combined_metric / sum_weight, decimals=3)


# -- functions to save models -- #
def save_best_model(cfg, train_options: dict, net, optimizer, scheduler, epoch: int,
                    resume_state: dict = None):
    '''
    Saves the input model in the inside the directory "/work_dirs/"experiment_name"/
    The models with be save as best_model.pth.
    The following are stored inside best_model.pth
        model_state_dict
        optimizer_state_dict
        epoch
        train_options


    Parameters
    ----------
    cfg : mmcv.Config
        The config file object of mmcv
    train_options : Dict
        The dictory which stores the train_options from quickstart
    net :
        The pytorch model
    optimizer :
        The optimizer that the model uses.
    epoch: int
        The epoch number

    '''
    print('saving model....')
    config_file_name = os.path.basename(cfg.work_dir)
    # print(config_file_name)
    torch.save(obj={'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    # 2026-07-20：记录 scheduler 类型，供 load_model 在 --resume-from 时
                    # 校验结构一致（warmup=SequentialLR vs 纯 CosineAnnealingLR 的 state_dict
                    # 结构不同，反向加载会静默错调度、污染 best model）。
                    'scheduler_type': type(scheduler).__name__,
                    'epoch': epoch,
                    'train_options': train_options,
                    # 2026-08-06：断点续跑状态（best 的平滑 mIoU、平滑窗尾部 raw 值、早停
                    # 计数）。--resume-from 时由 load_model 取出、train() 还原，使恢复后
                    # ① 第一次验证不再必定覆盖 best（历史 footgun，见 CLAUDE.md）；
                    # ② smsel 平滑窗与早停计数从断点无缝继续。旧 checkpoint 无此键 =
                    # 旧行为（从零开始）并打印告警。
                    'resume_state': resume_state,
                    },
               f=os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth'))
    print(f"model saved successfully at {os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')}")

    return os.path.join(cfg.work_dir, f'best_model_{config_file_name}.pth')


def load_model(net, checkpoint_path, optimizer=None, scheduler=None,
               return_resume_state=False):
    """
    Loads a PyTorch model from a checkpoint file and returns the model, optimizer, and scheduler.
    :param model: PyTorch model to load
    :param checkpoint_path: Path to the checkpoint file
    :param optimizer: PyTorch optimizer to load (optional)
    :param scheduler: PyTorch scheduler to load (optional)
    :param return_resume_state: 为 True 时返回 (epoch, resume_state)；resume_state 为
        checkpoint 里存的断点状态 dict（旧 checkpoint 无此键时为 None）。
    :return: If optimizer and scheduler are provided, return the model, optimizer, and scheduler.
    """

    # 修改此处，添加 weights_only=False
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
    except TypeError:
        # 如果是旧版本 PyTorch 不支持 weights_only 参数，则回退到默认行为
        checkpoint = torch.load(checkpoint_path)

    net.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None:
        # 结构守卫：checkpoint 的 scheduler 类型必须与当前一致，否则 load_state_dict
        # 一个方向 KeyError、另一个方向静默错调度（见 save_best_model 注释）。
        # 旧 checkpoint 无 scheduler_type 字段时跳过校验（向后兼容），仅当两者都是
        # 纯 cosine 时才安全——warmup checkpoint 一定带该字段。
        ck_type = checkpoint.get('scheduler_type')
        cur_type = type(scheduler).__name__
        if ck_type is not None and ck_type != cur_type:
            raise ValueError(
                f"--resume-from scheduler 类型不匹配：checkpoint 是 {ck_type}，当前配置是 "
                f"{cur_type}。warmup 配置（SequentialLR）与纯 cosine 的 state_dict 结构不同，"
                f"必须用相同 scheduler.warmup_epochs 的 config 恢复。")
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']

    if return_resume_state:
        return epoch, checkpoint.get('resume_state')
    return epoch

def rand_bbox(size, lam):
    '''
    Given the 4D dimensions of a batch (size), and the ratio 
    of the spatial dimension (lam) to be cut, returns a bounding box coordinates
    used for cutmix

    Parameters
    ----------
    size : 4D shape of the batch (N, C, H, W)
    lam : Ratio (portion) of the input to be cutmix'd

    Returns 
    ----------
    Bounding box (x1, y1, x2, y2)
    '''
    H = size[2]
    W = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_h = int(H * cut_rat)
    cut_w = int(W * cut_rat)

    # uniform
    cx = np.random.randint(H)
    cy = np.random.randint(W)

    bbx1 = np.clip(cx - cut_h // 2, 0, H)
    bby1 = np.clip(cy - cut_w // 2, 0, W)
    bbx2 = np.clip(cx + cut_h // 2, 0, H)
    bby2 = np.clip(cy + cut_w // 2, 0, W)

    return bbx1, bby1, bbx2, bby2


def slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.


    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    options: configuration dictionary
    mode: either 'val' or 'test'

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    batch_size, _, h_img, w_img = img.size()
    device = img.device
    charts = options['charts']
    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    # 累积张量放 CPU，避免大场景下 GPU OOM；每个 crop 的 forward 仍在 GPU 上执行
    preds = {chart: torch.zeros((batch_size, options['n_classes'][chart], h_img, w_img),
                                dtype=img.dtype) for chart in charts}
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), dtype=img.dtype)

    total_patches = h_grids * w_grids
    patch_bar = tqdm(total=total_patches, desc='slide patches', leave=False,
                     disable=total_patches < 20)
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            patch_bar.update(1)
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_img_size = crop_img.size()
            crop_height_pad = max(options['patch_size'] - crop_img_size[2], 0)
            crop_width_pad  = max(options['patch_size'] - crop_img_size[3], 0)

            if crop_height_pad > 0 or crop_width_pad > 0:
                crop_img = torch.nn.functional.pad(
                    crop_img, (0, crop_width_pad, 0, crop_height_pad), mode='constant', value=0)

            crop_seg_logit = net(crop_img)

            for chart in charts:
                logit = crop_seg_logit[chart]
                if crop_height_pad > 0:
                    logit = logit[:, :, :-crop_height_pad, :]
                if crop_width_pad > 0:
                    logit = logit[:, :, :, :-crop_width_pad]
                # pad 到全场景尺寸后移至 CPU 累积，避免 GPU 上创建全场景临时张量
                preds[chart][:, :, y1:y2, x1:x2] += logit.cpu()

            count_mat[:, :, y1:y2, x1:x2] += 1

    patch_bar.close()
    assert (count_mat == 0).sum() == 0

    # 归一化后送回 GPU
    return {chart: (preds[chart] / count_mat).to(device) for chart in charts}


class Slide_patches_index(data.Dataset):
    def __init__(self, h_img, w_img, h_crop, w_crop, h_stride, w_stride):
        super(Slide_patches_index, self).__init__()

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

        self.patches_list = []

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                self.patches_list.append((y1, y2, x1, x2))

    def __getitem__(self, index):
        return self.patches_list[index]

    def __len__(self):
        return len(self.patches_list)


class Take_crops(data.Dataset):
    def __init__(self, img, patches):
        super(Take_crops, self).__init__()

        self.img = img
        self.patches = patches

    def __getitem__(self, index):
        y1, y2, x1, x2 = self.patches[index]

        return self.img[:, y1:y2, x1:x2]

    def __len__(self):
        return len(self.patches)


def batched_slide_inference(img, net, options, mode):
    """
    Inference by sliding-window with overlap.

    Parameters
    ----------
    img : 4D shape of the batch (N, C', H, W)
    net : PyTorch model of nn.Module 
    y_type: str, One of 'SIC', 'SOD', or 'FLOE'
    options: configuration dictionary

    Returns 
    ----------
    pred: Dictionary with SIC, SOD, and FLOE predictions of the batch  (N, C", H, W)
    """
    if mode == 'val':
        h_stride, w_stride = options['swin_hp']['val_stride']
    elif mode == 'test':
        h_stride, w_stride = options['swin_hp']['test_stride']
    else:
        raise 'Unrecognized mode'

    h_crop = options['patch_size']
    w_crop = options['patch_size']

    # ------------ Add Padding to the image to match with the patch size / stride
    _, _, h_img, w_img = img.size()
    height_pad = h_crop - h_img if h_img - h_crop < 0 else \
        (h_stride - (h_img - h_crop) % h_stride) % h_stride
    width_pad = w_crop - w_img if w_img - w_crop < 0 else \
        (w_stride - (w_img - w_crop) % w_stride) % w_stride
    if height_pad > 0 or width_pad > 0:
        img = torch.nn.functional.pad(
            img, (0, width_pad, 0, height_pad), mode='constant', value=0)

    # ------------ create dataloader and index track
    _, _, h_img, w_img = img.size()
    indexes = Slide_patches_index(h_img, w_img, h_crop, w_crop, h_stride, w_stride)
    samples = Take_crops(img.detach().cpu().numpy()[0], indexes.patches_list)
    samples_dataloader = data.DataLoader(dataset=samples, batch_size=options['batch_size']*4,
                                         shuffle=False, num_workers=options['num_workers_val'])

    n_batches = len(samples_dataloader)
    data_iterator = iter(samples_dataloader)
    idx_iterator = iter(indexes)

    # Only accumulate the charts the model actually outputs (driven by options['charts']),
    # so single-task models (e.g. FLOE-only) work without hardcoding SIC/SOD/FLOE.
    charts = options['charts']
    preds = {chart: img.new_zeros((options['n_classes'][chart], h_img, w_img))
             for chart in charts}
    count_mat = img.new_zeros((h_img, w_img))

    for i in range(n_batches):

        # ------------ Take data
        crop_imgs = next(data_iterator)
        crop_imgs = crop_imgs.to(img.device)

        # ------------ Forward
        crop_seg_logit = net(crop_imgs)

        # ------------ LOCATE PREDICTED LOGITS ON THE WHOLE SCENE
        for j in range(crop_imgs.shape[0]):
            y1, y2, x1, x2 = next(idx_iterator)

            for chart in charts:
                preds[chart][:, y1:y2, x1:x2] += crop_seg_logit[chart][j, :, 0:(y2-y1), 0:(x2-x1)]

            count_mat[y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0

    for chart in charts:
        preds[chart] = preds[chart] / count_mat

        # ------------ Remove pad (guard against 0-pad: slicing with :-0 yields empty dimension)
        if height_pad > 0:
            preds[chart] = preds[chart][:, :-height_pad, :]
        if width_pad > 0:
            preds[chart] = preds[chart][:, :, :-width_pad]

        preds[chart] = preds[chart].unsqueeze(0)

    return preds


def compute_valid_tile_mask(valid_map, patch_size):
    """把像素级有效掩膜聚合成 tile 级：(H, W) bool → (n_h, n_w) bool。

    True = 该 tile 内至少有一个参与打分的像元。验证指标只统计 label != 255 的像素
    （quickstart 里的 `~cfv_masks`），loss 又是 ignore_index=255 的 masked mean，所以
    False 的 tile 对混淆矩阵和 loss 的贡献【严格为 0】——跳过它的前向不改变任何指标。
    val_417 上实测 66.1% 的 tile 属于这一类（`_val_tile_gate_audit.py`）。

    tile 网格与 fast_tiled_val_inference 一致：右/下 pad 到 patch_size 的整数倍，
    pad 出来的部分没有标签，计为无效。
    """
    if not torch.is_tensor(valid_map):
        valid_map = torch.from_numpy(np.asarray(valid_map))
    valid_map = valid_map.detach().to('cpu').bool().reshape(
        valid_map.shape[-2], valid_map.shape[-1])
    H, W = valid_map.shape
    n_h = -(-H // patch_size)      # ceil
    n_w = -(-W // patch_size)
    if n_h * patch_size != H or n_w * patch_size != W:
        buf = torch.zeros((n_h * patch_size, n_w * patch_size), dtype=torch.bool)
        buf[:H, :W] = valid_map
        valid_map = buf
    return valid_map.reshape(n_h, patch_size, n_w, patch_size).any(dim=3).any(dim=1)


def _val_tile_ids(valid_tile_mask, n_h, n_w):
    """tile 门控掩膜 → 需要前向的 tile 编号（行优先，与旧实现的展平顺序一致）。"""
    if valid_tile_mask is None:
        return torch.arange(n_h * n_w)
    if tuple(valid_tile_mask.shape) != (n_h, n_w):
        raise ValueError(
            f'valid_tile_mask 形状 {tuple(valid_tile_mask.shape)} 与 tile 网格 '
            f'({n_h}, {n_w}) 不符——多半是标签与输入尺寸不一致，或 patch_size 传错。')
    return torch.nonzero(valid_tile_mask.reshape(-1), as_tuple=False).squeeze(1)


def _gather_val_tiles(src, ids, n_w, patch_size, fill, pinned=False):
    """按 tile 编号从 (C, H, W) 或 (H, W) 取出一批 tile，右/下不足处用 fill 补齐。

    ⚠️ 刻意【不】预先 pad 整景：最大的验证景 8ch×6767×37468 是 8.1GB，pad 一次就要
    再复制一份，正好叠在验证的内存尖峰上（2026-07-23 两次训练在验证中途被宿主机
    SIGKILL）。这里每批只实体化 len(ids) 个 tile。

    pinned=True 时直接把批分配在页锁定（pinned）内存上，供 `.to(device,
    non_blocking=True)` 做真正异步的 H2D——可分页内存的 H2D 是流序的且阻塞宿主，
    会把「CPU 取下一批」与「GPU 算当前批」重新锁死成串行（confmat 模式的收益就
    没了）。pinned 张量走 PyTorch 的 CachingHostAllocator：释放后的块要等挂起的
    异步拷贝事件完成才会被复用，无需手动双缓冲。仅在 CUDA 可用时传 True。
    """
    ps = patch_size
    H, W = src.shape[-2], src.shape[-1]
    shape = ((len(ids), ps, ps) if src.dim() == 2
             else (len(ids), src.shape[0], ps, ps))
    if pinned:
        # pinned 缓冲恒在 CPU；src 若在 GPU 会退化成「每 tile 一次 D2H」的静默慢路径。
        assert src.device.type == 'cpu', \
            f'_gather_val_tiles(pinned=True) 要求 src 在 CPU 上，得到 {src.device}'
        out = torch.full(shape, fill, dtype=src.dtype, pin_memory=True)
    else:
        out = src.new_full(shape, fill)
    for k, t in enumerate(ids.tolist()):
        i, j = divmod(int(t), n_w)
        y0, x0 = i * ps, j * ps
        y1, x1 = min(y0 + ps, H), min(x0 + ps, W)
        out[k, ..., :y1 - y0, :x1 - x0] = src[..., y0:y1, x0:x1]
    return out


# 「逐像素 loss，分母恰为有效像元数」的 loss 类名。只有这类 loss 能按「每批的有效像元
# 数」加权聚合出与整景一次算完【相同】的值。逐个查证过：
#   - losses.FocalLoss / losses.GCELoss: `loss.sum() / valid_mask.sum()`，分母恒为有效
#     像元数，即使带 alpha/类别权重也成立；
#   - losses.OrderedCrossEntropyLoss: 先按 mask 取出有效像元再 .mean()，同上；
#   - torch.nn.CrossEntropyLoss: reduction='mean' 的分母是 Σ w_{y_i}，**带类别权重时不是
#     像素数** → 单独判定，见 _loss_is_mean_over_valid；
#   - smp 的 Dice / Lovasz / Tversky / MCC: 集合级指标，**不成立**。
_VAL_LOSS_MEAN_OVER_VALID = ('FocalLoss', 'GCELoss', 'OrderedCrossEntropyLoss')
_VAL_LOSS_WARNED = set()


def _loss_is_mean_over_valid(fn):
    """该 loss 是否等于「Σ逐像素 loss / 有效像元数」（见 _VAL_LOSS_MEAN_OVER_VALID）。"""
    name = type(fn).__name__
    if name in _VAL_LOSS_MEAN_OVER_VALID:
        return True
    if name == 'CrossEntropyLoss':
        return (getattr(fn, 'weight', None) is None
                and getattr(fn, 'reduction', 'mean') == 'mean')
    return False


def fast_tiled_val_predict(img, net, options, labels, valid_maps=None,
                           loss_fns=None, task_weights=None, gate=True,
                           out='preds'):
    """验证专用的流式 tile 推理：只前向【含有效标签】的 tile，且不实体化整景 logits。

    与 `fast_tiled_val_inference` 在【指标上完全等价】（预测逐像素相同，见下），
    但去掉了三处纯开销：

      1) **tile 门控**：标签全为 ignore 的 tile 不前向。这些 tile 的所有像素都被
         `~cfv_masks` 排除在混淆矩阵之外、也被 ignore_index 排除在 loss 之外，贡献
         严格为 0。val_417 实测 66.1% 的 tile 属于此类 → 前向次数降到约 1/3。
      2) **不开整景 logits 缓冲**：旧实现要 n_classes×H×W 的 float32（最大景 3.7GB），
         逐 tile 把 logits D2H 拷回来，最后再把整景 logits 整块搬回 GPU（又一次
         3.7GB H2D）。这里逐 batch 就地 argmax，只留 (H, W) 的 uint8 类别图。
      3) **整景输入不上 GPU**：旧路径由调用方把整景搬上 GPU（最大景 8 通道 8.1GB）、
         函数内再 `.cpu()` 搬回来。这里整景全程留在 CPU，只有每批 tile 上 GPU。

    等价性（准确表述，别写成"逐位等价"）：tile 之间互不重叠、无上下文交互，所以【同一批
    组成下】逐 tile 的 logits 与整景切块后的 logits 逐位相同。但**门控会改变每批的组成
    （包括最后一批的大小），cuDNN/cuBLAS 可能因此选到不同算法**，logits 末位有 ULP 级
    差异；落在 argmax 近乎平局的像素上就会翻。实测（b5、42 景全量）：11.6 亿有效像元里
    翻了 528 个（5e-7），ΔmIoU 2e-7，差异像素的 top1-top2 间隔 max 1.2e-4 而全体中位数
    2.5e-2。同量级抖动在改 val_batch_size 时本来就存在。gate=False 时批组成与旧路径完全
    一致，此时才是【全图逐像素相同】（smoke_sota.py 第 [7a] 步断言的就是这一条）。

    Parameters
    ----------
    img : Tensor (1, C, H, W)  —— 可以在 CPU 上（推荐）也可以在 GPU 上。
    labels : dict[chart] -> (H, W) 整型张量/数组，验证标签（已完成 255 重映射）。
    valid_maps : dict[chart] -> (H, W) bool，True=参与打分的像元。省略则由
        `labels != options['class_fill_values'][chart]` 现算。传 `~cfv_masks[chart]`
        可以直接复用 loader 已经算好的掩膜。
    loss_fns : dict[chart] -> loss module。给了才算 loss。
    task_weights : list，与 options['charts'] 对齐；省略则全 1。
    gate : bool，False 可关掉 tile 门控（退化成"只是不开整景缓冲"的版本，调试对拍用）。
    out : 'preds'（默认，行为与 2026-07-25 版完全一致）或 'confmat'（2026-08-06）。

        **'confmat' 模式：逐批在 GPU 上直接累加混淆矩阵，不落整景预测图。**
        动机（quickstart 验证循环的剩余纯开销）：
          1) preds 模式每批有两次强制同步——loss 的 `.item()` 和预测 `.cpu()`——CPU
             必须停下来等 GPU，「取下一批 tile 的 CPU 拷贝」无法与 GPU 前向重叠；
          2) 调用方拿到整景 uint8 预测图后，还要对最大 2.5 亿像素做 CPU 布尔掩膜
             筛选、把 ~11.6 亿有效像素/轮搬上 GPU 再 bincount（quickstart 旧写法）。
        confmat 模式把这两段全部消掉：批内 argmax 后用 `index_add_(true*n+pred)`
        就地累进 int64 GPU 累加器（⚠️ 不是 bincount——它在 CUDA 上每次调用要做
        min/max 归约并取回 CPU、强制同步两次），loss 也在 GPU 上以 float64 累加，
        **整个场景只在函数返回前同步一次**。CPU 由此可以在 GPU 前向期间提前准备
        后续批次，D2H 拷贝完全消失。

        等价性（比门控那条更强）：同一 gate 值下批组成与 preds 模式完全一致 ⇒
        logits 逐位相同 ⇒ argmax 逐位相同 ⇒ 混淆矩阵与「preds 模式 + 调用方按
        valid_maps 筛选后 accumulate_confusion_matrix」**整数级完全相等**
        （index_add_ 的 atomicAdd 是整数加法，与顺序无关）；loss 的聚合算式
        （float64 逐批累加，同序同操作数）也逐位一致。smoke_sota.py 第 [7c] 步
        断言整数级相等，[7d] 覆盖「整批无有效像元」的 nan 守卫。

    Returns
    -------
    preds : （out='preds' 时）dict[chart] -> (H, W) uint8 CPU 张量。被门控跳过的
        tile 保持 0（这些像素在打分时一律被 `~cfv_masks` 排除，不影响任何指标；
        但若要拿它做可视化，请传 gate=False）。
        （out='confmat' 时）dict[chart] -> (n_classes, n_classes) float64 **GPU**
        张量：true 为行、pred 为列，只统计 valid_maps（缺省则 label != fill）为
        True 的像素——与 quickstart 旧写法 `pred[~cfv_masks]` 的像素集合完全相同。
        可直接 `+=` 进 new_confusion_matrix 的累加器。
    loss : float。按「有效像元数」加权聚合各 batch 的 loss：
        `Σ(loss_b · n_valid_b) / Σ n_valid_b`。这与整景一次算完恒等的前提是「该 loss =
        Σ逐像素 loss / 有效像元数」，判定见 `_loss_is_mean_over_valid`。**不满足的 loss
        （如带类别权重的 CrossEntropyLoss、Dice/Lovasz）会 warn 一次并返回 nan**，而不是
        给一个看起来正常的错数——val loss 只用于日志，best model 选择走 mIoU，所以宁可
        显式缺失也不要静默偏差。
    """
    if out not in ('preds', 'confmat'):
        raise ValueError(f"out 必须是 'preds' 或 'confmat'，得到 {out!r}")
    confmat_mode = (out == 'confmat')
    patch_size = options['patch_size']
    charts = options['charts']
    val_batch_size = options.get('val_batch_size', options['batch_size'] * 4)
    if task_weights is None:
        task_weights = [1] * len(charts)

    # 整景留在 CPU：只有每批 tile 会上 GPU。net 的参数在哪就往哪送。
    device = next(net.parameters()).device
    img_cpu = img.detach()
    if img_cpu.dim() == 4:
        img_cpu = img_cpu[0]
    img_cpu = img_cpu.to('cpu', copy=False)
    C, H, W = img_cpu.shape
    n_h = -(-H // patch_size)
    n_w = -(-W // patch_size)

    # ---- 标签与有效掩膜 ----
    lab_t, valid_t = {}, {}
    for chart in charts:
        lab = labels[chart]
        if not torch.is_tensor(lab):
            lab = torch.from_numpy(np.asarray(lab))
        lab = lab.detach().to('cpu').reshape(H, W)
        lab_t[chart] = lab
        if valid_maps is not None and valid_maps.get(chart) is not None:
            v = valid_maps[chart]
            if not torch.is_tensor(v):
                v = torch.from_numpy(np.asarray(v))
            valid_t[chart] = v.detach().to('cpu').bool().reshape(H, W)
        else:
            valid_t[chart] = lab != options['class_fill_values'][chart]

    # ---- tile 门控：任一 chart 有有效像元就得算这个 tile ----
    if gate:
        tile_mask = None
        for chart in charts:
            m = compute_valid_tile_mask(valid_t[chart], patch_size)
            tile_mask = m if tile_mask is None else (tile_mask | m)
    else:
        tile_mask = None
    ids_all = _val_tile_ids(tile_mask, n_h, n_w)

    if confmat_mode:
        preds = None
        # 展平的 int64 累加器（n² 个合法 bin + 1 个哨兵 bin），整数加法精确；返回前
        # 切掉哨兵、reshape 成 (n,n) 并转 float64 以便直接累进 new_confusion_matrix
        # 的 float64 累加器 / 喂 cm_* 系列函数。
        cms = {chart: torch.zeros(options['n_classes'][chart] ** 2 + 1,
                                  dtype=torch.int64, device=device)
               for chart in charts}
        # index_add_ 的 source 全 1 向量，按实际最大批尺寸预分配一次、逐批切片复用
        # （门控后 tile 数可能远小于 val_batch_size，别按满额白占显存）。
        ones_flat = torch.ones(
            min(val_batch_size, max(len(ids_all), 1)) * patch_size * patch_size,
            dtype=torch.int64, device=device)
        # loss 的 GPU 累加器（float64，与 preds 模式的 python float 累加同序同操作数，
        # 数值逐位一致），整景只在函数末尾同步一次。
        loss_num_t = {chart: torch.zeros((), dtype=torch.float64, device=device)
                      for chart in charts}
        loss_den_t = {chart: torch.zeros((), dtype=torch.float64, device=device)
                      for chart in charts}
    else:
        preds = {chart: torch.zeros((H, W), dtype=torch.uint8) for chart in charts}
        for chart in charts:
            if options['n_classes'][chart] > 255:
                raise NotImplementedError(
                    f"n_classes[{chart}]={options['n_classes'][chart]} > 255，预测图存不进 uint8。")

    # 逐 chart 分别累加「loss×有效像元数」与「有效像元数」，最后各自求均值再加权求和，
    # 与整景一次算完的 Σ_c w_c · mean_c 逐项对应（各 chart 的有效像元集合不同，不能混算）。
    # 只有「分母恰为有效像元数」的 loss 才能这样聚合；其余记 nan（见 docstring）。
    loss_num = {chart: 0.0 for chart in charts}
    loss_den = {chart: 0 for chart in charts}
    loss_ok = {}
    for chart in charts:
        if loss_fns is None or chart not in loss_fns:
            continue
        loss_ok[chart] = _loss_is_mean_over_valid(loss_fns[chart])
        if not loss_ok[chart] and chart not in _VAL_LOSS_WARNED:
            _VAL_LOSS_WARNED.add(chart)
            warnings.warn(
                f"[fast_tiled_val_predict] {chart} 的 loss "
                f"{type(loss_fns[chart]).__name__} 不是「Σ逐像素 loss / 有效像元数」形式"
                f"（带类别权重的 CrossEntropyLoss、Dice/Lovasz 等），无法按批聚合出与整景"
                f"一致的值 → 本次验证的 val loss 记为 nan。指标（mIoU/F1）不受影响。",
                RuntimeWarning)
    texture_on = options.get('texture_channels', False)
    # confmat 模式下用 pinned 内存 + non_blocking H2D（见 _gather_val_tiles docstring）；
    # preds 模式保持原样（它每批本来就有 .cpu()/.item() 同步，异步化没有意义）。
    pinned = confmat_mode and device.type == 'cuda'
    for start in range(0, len(ids_all), val_batch_size):
        ids = ids_all[start:start + val_batch_size]
        # non_blocking 只在 pinned 时开：可分页内存下 non_blocking 无效果（拷贝仍阻塞
        # 宿主），显式写成同一变量以免两种模式的语义看起来自相矛盾。
        batch = _gather_val_tiles(img_cpu, ids, n_w, patch_size, 0,
                                  pinned=pinned).to(device, non_blocking=pinned)
        # B1 纹理通道：per-tile 在 GPU 上对每个 256 tile 算（与训练 _augment_x 的 256 patch
        # 同口径），整景 loader 保持 8 通道 → 保住按需读+低内存，消除 train/val 边缘不一致。
        if texture_on:
            from floe.loaders import _texture_channels  # 懒导入避免顶层循环依赖
            batch = torch.cat([batch, _texture_channels(batch)], dim=1)
        batch_out = net(batch)

        for chart, weight in zip(charts, task_weights):
            logits = batch_out[chart]
            if logits.dim() != 4 or logits.shape[1] != options['n_classes'][chart]:
                raise NotImplementedError(
                    f'fast_tiled_val_predict 只支持 (B, n_classes, H, W) 的分割输出，'
                    f'{chart} 得到 {tuple(logits.shape)}（回归头请走 '
                    f'fast_tiled_val_inference）。')

            if confmat_mode:
                # 全程无同步：标签/掩膜 tile 上 GPU、loss 与 CM 都在 GPU 上累加。
                # CPU 在 GPU 前向期间即可继续 gather 后续批次（这正是本模式的收益来源）。
                fill = options['class_fill_values'][chart]
                lab_b = _gather_val_tiles(lab_t[chart], ids, n_w, patch_size, fill,
                                          pinned=pinned)
                lab_b = lab_b.to(device, non_blocking=pinned).long()
                # 打分像素集合严格按 valid_t（= valid_maps，缺省则 label != fill）取，
                # 与调用方旧写法 pred[valid_maps] 一致；pad 区 fill=False 自然被排除。
                vmap_b = _gather_val_tiles(valid_t[chart], ids, n_w, patch_size, False,
                                           pinned=pinned)
                vmap_b = vmap_b.to(device, non_blocking=pinned)

                if loss_ok.get(chart, False):
                    # loss 掩膜沿用 preds 模式的口径（label != fill，ignore_index 同源）。
                    # .detach()：preds 模式的 .item() 天然截断 autograd，这里若缺 detach，
                    # 调用方一旦忘了 no_grad，整景几百个批的计算图会被累加器全部挂住。
                    n_valid = (lab_b != fill).sum()
                    contrib = (loss_fns[chart](logits, lab_b).detach().double()
                               * n_valid.double())
                    # 整批无有效像元时 loss 是 0/0=nan，nan*0 仍是 nan → 用 where 归零，
                    # 与 preds 模式的 `if n_valid > 0` 守卫等价。
                    loss_num_t[chart] += torch.where(
                        n_valid > 0, contrib, torch.zeros_like(contrib))
                    loss_den_t[chart] += n_valid.double()

                n_c = options['n_classes'][chart]
                p = class_decider(logits, options, chart)
                p = p.reshape(-1, patch_size, patch_size).long()
                # 模型输出的空间尺寸若 ≠ patch_size，reshape 的第 0 维就不再是批大小，
                # 下游 index_add_ 会以「numel mismatch」这种指不到病灶的方式报错（preds
                # 模式更糟：p[k] 静默取错行）。一行断言把病灶钉在这里。
                assert p.shape[0] == len(ids), \
                    f'预测 tile 数 {p.shape[0]} != 批内 tile 数 {len(ids)}——模型输出的空间尺寸与 patch_size 不符？'
                # 打分口径与 accumulate_confusion_matrix 完全一致：valid 掩膜之外，还
                # 丢弃越界标签与越界预测（旧路径对 true/pred 都静默过滤；p 走 argmax 时
                # 恒 <n_c，但 class_decider 的回归分支 clamp 上界是 n_classes【不是 n_c-1】，
                # 不补 p<n_c 这项会把 p==n_c 静默错记进下一行的合法 bin）。顺带保证
                # index_add_ 的下标恒在 [0, n²] 内。被丢弃像素统一指到第 n² 个哨兵 bin
                # （where 会丢弃另一支的乘积结果，不必先把 lab 清零）。
                # ⚠️ 不用 torch.bincount：它在 CUDA 上为确定输出范围要做 min/max 归约并
                # 取回 CPU，【每批强制同步两次】，会把「CPU gather 与 GPU 前向重叠」的
                # 收益整个抵消（code-review P1-1）。index_add_ 是原子整数加：无同步、
                # 与顺序无关、结果精确。
                keep = vmap_b & (lab_b >= 0) & (lab_b < n_c) & (p < n_c)
                idx = torch.where(keep, lab_b * n_c + p,
                                  torch.full_like(lab_b, n_c * n_c))
                cms[chart].index_add_(0, idx.reshape(-1), ones_flat[:idx.numel()])
                continue

            if loss_ok.get(chart, False):
                fill = options['class_fill_values'][chart]
                lab_b = _gather_val_tiles(lab_t[chart], ids, n_w, patch_size, fill)
                lab_b = lab_b.to(device, non_blocking=True).long()
                n_valid = int((lab_b != fill).sum().item())
                if n_valid > 0:
                    loss_num[chart] += loss_fns[chart](logits, lab_b).item() * n_valid
                    loss_den[chart] += n_valid

            p = class_decider(logits, options, chart).reshape(-1, patch_size, patch_size)
            # 与 confmat 分支同一条护栏：空间尺寸不符时 p[k] 会静默取错行、整景预测错位。
            assert p.shape[0] == len(ids), \
                f'预测 tile 数 {p.shape[0]} != 批内 tile 数 {len(ids)}——模型输出的空间尺寸与 patch_size 不符？'
            p = p.to(torch.uint8).cpu()
            for k, t in enumerate(ids.tolist()):
                i, j = divmod(int(t), n_w)
                y0, x0 = i * patch_size, j * patch_size
                y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
                preds[chart][y0:y1, x0:x1] = p[k, :y1 - y0, :x1 - x0]

    if confmat_mode:
        # 整景唯一的一次同步点：把 GPU 上的 loss 累加器取回。CM 留在 GPU（float64），
        # 调用方直接 += 进全局累加器，不产生 D2H。
        for chart in charts:
            loss_num[chart] = float(loss_num_t[chart].item())
            loss_den[chart] = float(loss_den_t[chart].item())

    if loss_ok and not all(loss_ok.values()):
        loss = float('nan')       # 有 chart 的 loss 无法按批聚合，显式缺失而非静默偏差
    else:
        loss = 0.0
        for chart, weight in zip(charts, task_weights):
            if loss_den[chart] > 0:
                loss += weight * (loss_num[chart] / loss_den[chart])

    if confmat_mode:
        out_cms = {}
        for chart in charts:
            n_c = options['n_classes'][chart]
            out_cms[chart] = cms[chart][:n_c * n_c].reshape(n_c, n_c).to(torch.float64)
        return out_cms, loss
    return preds, loss


def fast_tiled_val_inference(img, net, options, valid_tile_mask=None):
    """
    Fast validation inference using non-overlapping tiles with batched GPU forward passes.

    Splits the scene into non-overlapping patch_size×patch_size tiles, batches them
    together, and runs a single net() call per batch. Dramatically faster than
    sliding-window inference for full-resolution (scale=1) scenes:
      e.g. 2560×2560, patch_size=256, val_batch_size=256 → 1 forward pass/scene
           vs. 484 forward passes with stride-128 sliding window.

    Caller is responsible for wrapping in torch.no_grad() / autocast().

    Parameters
    ----------
    img : Tensor (1, C, H, W)  on the target device
    net : nn.Module
    options : dict
        Uses 'patch_size', 'charts', 'n_classes', 'batch_size',
        and optionally 'val_batch_size' (defaults to batch_size * 4).
    valid_tile_mask : Tensor (n_h, n_w) bool, optional
        tile 门控（`compute_valid_tile_mask` 的输出）。False 的 tile 不前向，其 logits
        保持 0 —— 这些 tile 的像素在打分时被 `~cfv_masks` 全部排除，指标逐位不变，
        但 argmax 出来会是类 0。**要拿返回值做可视化或导出整景预测时不要传它。**

    Returns
    -------
    dict[str, Tensor (1, n_classes[chart], H, W)]  on the same device as img.
    """
    patch_size = options['patch_size']
    charts = options['charts']
    val_batch_size = options.get('val_batch_size', options['batch_size'] * 4)
    device = img.device

    _, C, h_img, w_img = img.size()

    # Pad H and W to exact multiples of patch_size
    h_pad = (-h_img) % patch_size
    w_pad = (-w_img) % patch_size
    if h_pad > 0 or w_pad > 0:
        img = torch.nn.functional.pad(img, (0, w_pad, 0, h_pad), mode='constant', value=0)
    _, _, h_padded, w_padded = img.size()

    n_h = h_padded // patch_size
    n_w = w_padded // patch_size
    n_tiles = n_h * n_w

    # 把整景重排成不重叠的 tile —— 但【只保留视图，不实体化】。
    # (C, h_padded, w_padded) → (n_h, n_w, C, patch_size, patch_size)
    #
    # ⚠️ 别写成 `.reshape(n_tiles, C, ps, ps).contiguous()`（2026-07-14 前的写法）：
    # permute 之后张量非连续，reshape 会退化成一次完整拷贝，等于把【整景又复制一份】。
    # 最大的验证景 6369×23975×8ch fp32 ≈ 4.9GB——纯浪费，而且正好叠在验证的内存尖峰上。
    # （2026-07-13 两个训练在验证阶段被 SIGKILL、无 traceback，根因是链上冷却不足导致的
    #  页缓存挤压，但这份多余拷贝是压垮骆驼的一部分。）
    # img_cpu 本身连续，所以 reshape+permute 都是零拷贝视图；每个 batch 再按需实体化。
    img_cpu = img[0].cpu()
    tiles_view = (img_cpu
                  .reshape(C, n_h, patch_size, n_w, patch_size)
                  .permute(1, 3, 0, 2, 4))          # (n_h, n_w, C, ps, ps)，视图

    # Prediction buffers on CPU; only each batch touches the GPU
    preds = {chart: torch.zeros(options['n_classes'][chart], h_padded, w_padded)
             for chart in charts}

    # tile 门控：只前向掩膜为 True 的 tile（详见 compute_valid_tile_mask）。
    # 未传掩膜时退化成全量 tile，与旧行为完全一致。
    tile_ids = _val_tile_ids(valid_tile_mask, n_h, n_w)

    for start in range(0, len(tile_ids), val_batch_size):
        end = min(start + val_batch_size, len(tile_ids))
        # tile 编号按行优先：tile_idx = h_idx * n_w + w_idx，与旧实现的展平顺序完全一致
        idx = tile_ids[start:end]
        h_idx = torch.div(idx, n_w, rounding_mode='floor')
        w_idx = idx % n_w
        # 高级索引只实体化【这一批】（val_batch_size 个 tile，约 134MB），不是整景
        batch = tiles_view[h_idx, w_idx].contiguous().to(device)
        # B1 纹理通道：per-tile 算（与 fast_tiled_val_predict / 训练同口径），
        # 供 TTA/集成/inference 路径复用，避免整景 8ch→模型 12ch 的静默错位。
        if options.get('texture_channels', False):
            from floe.loaders import _texture_channels  # 懒导入避免顶层循环依赖
            batch = torch.cat([batch, _texture_channels(batch)], dim=1)
        batch_out = net(batch)

        for local_j in range(end - start):
            y1 = int(h_idx[local_j]) * patch_size
            x1 = int(w_idx[local_j]) * patch_size
            for chart in charts:
                preds[chart][:, y1:y1 + patch_size, x1:x1 + patch_size] = \
                    batch_out[chart][local_j].detach().cpu()

    # Trim padding, restore batch dim, return on original device
    return {chart: preds[chart][:, :h_img, :w_img].unsqueeze(0).to(device)
            for chart in charts}


def class_decider(output, train_options, chart):

    # normal
    if (train_options['binary_water_classifier'] == False):
        if output.size(3) == 1:
            output = torch.round(output.squeeze())
            output = torch.clamp(output, min=0, max=train_options
                                 ['n_classes'][chart])
        else:
            output = torch.argmax(output, dim=1).squeeze()
        return output

    # if regression head    return output
    # class water
    else:
        probability = torch.nn.Softmax(dim=1)(output)
        water = probability[:, 0, :, :]
        not_water = torch.sum(probability, dim=1) - water
        class_output = water <= not_water
        without_water = probability[:, 1:, :, :]
        class_output_without_water = torch.argmax(without_water, dim=1) + 1
        class_output = class_output_without_water * class_output

        return class_output.squeeze()


def compute_classwise_f1score(true, pred, charts, num_classes):
    """ This function computes the classwise evaluation score for each task and stores them in a dic

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns score_dictionary
    """
    score = {}
    for chart in charts:
        score[chart] = f1_score(target=true[chart], preds=pred[chart], average='none',
                                task='multiclass', num_classes=num_classes[chart])
    return score


def compute_overall_accuracy(true, pred, charts):
    """ Computes Overall Accuracy (OA) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts

    Returns:
        dictionary: returns oa_score_dictionary
    """
    scores = {}
    for chart in charts:
        if true[chart].numel() == 0:
            scores[chart] = 0.0
        else:
            # 计算预测正确的像素数占总像素数的比例
            correct = (true[chart] == pred[chart]).sum()
            total = true[chart].numel()
            scores[chart] = (correct.float() / total).item()
    return scores


def compute_mIoU(true, pred, charts, num_classes):
    """ Computes Mean Intersection over Union (mIoU) for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: returns mIoU_score_dictionary
    """
    scores = {}
    for chart in charts:
        # jaccard_index 即为 IoU。task='multiclass' 适用于多分类，默认 average='macro' 即计算 mIoU
        scores[chart] = jaccard_index(target=true[chart], preds=pred[chart], task='multiclass', num_classes=num_classes[chart])
    return scores


def compute_ordinal_cost_metric(true, pred, charts, num_classes, cost_per_distance=None):
    """Ordinal Cost Metric for ordered classification (e.g. SOD thickness stages).

    Per-pixel cost is looked up from cost_per_distance[|pred - true|].
    Default is linear: cost = |pred - true|, which is consistent with EMDLoss.

    Parameters
    ----------
    cost_per_distance : list of numbers, length >= n_classes.
        cost_per_distance[d] is the penalty for a prediction that is d classes away.
        Must start with 0 (correct prediction costs nothing).
        Example (linear):  [0, 1, 2, 3, 4]
        Example (custom):  [0, 1, 2, 3, 3]  (cap at 3 for very distant errors)

    Returns
    -------
    raw_costs : dict[chart -> Tensor scalar]  mean cost per pixel (lower = better)
    norm_scores : dict[chart -> Tensor scalar]
        1 - mean_cost / max_cost, normalised to [0, 1] (higher = better).
    """
    raw_costs = {}
    norm_scores = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            raw_costs[chart] = torch.tensor(0.0)
            norm_scores[chart] = torch.tensor(1.0)
            continue

        # Build cost lookup table: index = distance, value = penalty
        if cost_per_distance is None:
            # Linear default: cost = |pred - true|, aligned with EMDLoss
            cost_table = torch.arange(n_classes, dtype=torch.float32, device=true_c.device)
        else:
            cost_table = torch.tensor(cost_per_distance, dtype=torch.float32, device=true_c.device)

        diff = torch.abs(true_valid - pred_valid)          # [N], integer distances
        cost = cost_table[diff]                            # lookup per-pixel cost

        max_cost = cost_table.max().clamp(min=1e-6)
        mean_cost = cost.mean()
        raw_costs[chart] = mean_cost
        norm_scores[chart] = 1.0 - mean_cost / max_cost

    return raw_costs, norm_scores


def compute_classwise_IoU(true, pred, charts, num_classes):
    """Computes per-class IoU for each task.

    Args:
        true (dictionary): The true tensor as value and chart tensor as key
        pred (dictionary): The pred tensor as value and chart tensor as key
        charts (list): list of charts
        num_classes (dictionary): key = chart , value = num_class

    Returns:
        dictionary: per-class IoU tensor for each chart
    """
    scores = {}
    for chart in charts:
        scores[chart] = jaccard_index(
            target=true[chart], preds=pred[chart],
            task='multiclass', num_classes=num_classes[chart],
            average='none'
        )
    return scores


def compute_classwise_precision_recall(true, pred, charts, num_classes):
    """Computes per-class precision and recall for each chart.

    Invalid labels (outside [0, num_classes-1]) are ignored.
    """
    precision_scores = {}
    recall_scores = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            precision_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            recall_scores[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        tp = torch.diag(cm)
        pred_sum = cm.sum(dim=0)
        true_sum = cm.sum(dim=1)

        precision_scores[chart] = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
        recall_scores[chart] = torch.where(true_sum > 0, tp / true_sum, torch.zeros_like(tp))

    return precision_scores, recall_scores


def compute_normalized_confusion_matrix(true, pred, charts, num_classes):
    """Computes row-normalized confusion matrix for each chart.

    Normalization is done by true-class row sums.
    """
    norm_cms = {}
    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            norm_cms[chart] = torch.zeros((n_classes, n_classes), dtype=torch.float32, device=true_c.device)
            continue

        cm = multiclass_confusion_matrix(
            preds=pred_valid,
            target=true_valid,
            num_classes=n_classes,
        ).float()

        row_sum = cm.sum(dim=1, keepdim=True)
        norm_cms[chart] = torch.where(row_sum > 0, cm / row_sum, torch.zeros_like(cm))

    return norm_cms


def compute_class_proportions(true, pred, charts, num_classes):
    """Computes per-class true/pred pixel proportions for each chart."""
    true_props = {}
    pred_props = {}

    for chart in charts:
        n_classes = num_classes[chart]
        true_c = true[chart].long()
        pred_c = pred[chart].long()

        valid = (true_c >= 0) & (true_c < n_classes) & (pred_c >= 0) & (pred_c < n_classes)
        true_valid = true_c[valid]
        pred_valid = pred_c[valid]

        if true_valid.numel() == 0:
            true_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            pred_props[chart] = torch.zeros(n_classes, dtype=torch.float32, device=true_c.device)
            continue

        true_hist = torch.bincount(true_valid, minlength=n_classes).float()
        pred_hist = torch.bincount(pred_valid, minlength=n_classes).float()
        denom = true_hist.sum().clamp_min(1.0)

        true_props[chart] = true_hist / denom
        pred_props[chart] = pred_hist / denom

    return true_props, pred_props


# --------------------------------------------------------------------------- #
# 基于混淆矩阵的验证指标（2026-07-10 引入）
#
# 原先的验证循环把 30 个场景的全部有效像素在 GPU 上 torch.cat 成一个扁平张量，再逐个
# 调用上面那些 compute_* 函数。在 down_sample_scale=4 时这只有约 5000 万像素、1.7GB，
# 尚可；改到原生分辨率(scale=1)后是 8 亿像素，true/pred 各一份，且 torchmetrics 内部的
# `target.to(long) * num_classes + preds.to(long)` 还要再要 7.46GB —— 合计 27.6GB，
# 32GB 的卡直接 CUDA OOM（首次验证即崩）。
#
# 上面 9 个指标函数无一例外都是 (num_classes × num_classes) 混淆矩阵的纯函数。改为逐景
# 增量累加一个 6×6 的矩阵后：显存占用与像素总数无关，指标计算从秒级降到微秒级，且数值
# 与原实现逐位一致（见 tests 里的等价性检查）。
#
# 约定：cm[t, p] = 真实类别 t 被预测为 p 的像素数；行=true，列=pred。
# 调用方需保证传入的 true/pred 已剔除 255（验证循环里由 ~cfv_masks 完成）。
# --------------------------------------------------------------------------- #

def new_confusion_matrix(num_classes, device):
    """float64 累加器：像素数可达 1e9，float32 的尾数只有 24 bit，累加会丢精度。"""
    return torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)


def accumulate_confusion_matrix(cm, true, pred, num_classes, chunk=1 << 24):
    """把一景的 (true, pred) 累加进混淆矩阵。就地修改并返回 cm。

    分块 bincount：`true * num_classes + pred` 是 int64，单景有效像素可达 5000 万，
    一次性构造要 400MB。按 16M 元素分块把峰值压到 128MB 以内。
    """
    true = true.reshape(-1).long()
    pred = pred.reshape(-1).long()
    if true.numel() == 0:
        return cm

    valid = (true >= 0) & (true < num_classes) & (pred >= 0) & (pred < num_classes)
    if not bool(valid.all()):
        true = true[valid]
        pred = pred[valid]

    for i in range(0, true.numel(), chunk):
        t = true[i:i + chunk]
        p = pred[i:i + chunk]
        binned = torch.bincount(t * num_classes + p, minlength=num_classes ** 2)
        cm += binned.reshape(num_classes, num_classes).to(cm.dtype)
    return cm


def _cm_parts(cm):
    """返回 (tp, pred_sum, true_sum, total)，均为 float64 一维/标量张量。"""
    tp = torch.diag(cm)
    return tp, cm.sum(dim=0), cm.sum(dim=1), cm.sum()


def cm_classwise_f1(cm):
    """等价于 f1_score(average='none', task='multiclass')。分母为 0 的类记 0。"""
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    denom = pred_sum + true_sum
    f1 = torch.where(denom > 0, 2 * tp / denom, torch.zeros_like(tp))
    return f1.float()


def cm_weighted_f1(cm):
    """等价于 f1_score(average='weighted')：按 true 支持度加权的 per-class F1。"""
    _, _, true_sum, total = _cm_parts(cm)
    if total <= 0:
        return torch.tensor(0.0)
    f1 = cm_classwise_f1(cm).double()
    return (f1 * true_sum / total).sum().float()


def cm_overall_accuracy(cm):
    total = cm.sum()
    if total <= 0:
        return 0.0
    return (torch.diag(cm).sum() / total).item()


def cm_classwise_iou(cm):
    """等价于 jaccard_index(average='none')。TP+FP+FN==0 的类（数据集中不存在）记 0。"""
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    denom = pred_sum + true_sum - tp
    iou = torch.where(denom > 0, tp / denom, torch.zeros_like(tp))
    return iou.float()


def cm_miou(cm):
    """等价于 jaccard_index(average='macro')：对所有类取均值，含空类（记 0）。

    注意：2026-07-16 数据集扩充前类 1（饼状冰 ice cake）在数据集中不存在，此前所有
    run 的 macro mIoU 绝对值都被这个空类稀释；扩充后类 1 有真实像素、正常参与均值。
    新旧数据集上的 macro mIoU 绝对值因此不可直接横比（见 CLAUDE.md 指标口径）。
    """
    return cm_classwise_iou(cm).mean()


def cm_precision_recall(cm):
    tp, pred_sum, true_sum, _ = _cm_parts(cm)
    precision = torch.where(pred_sum > 0, tp / pred_sum, torch.zeros_like(tp))
    recall = torch.where(true_sum > 0, tp / true_sum, torch.zeros_like(tp))
    return precision.float(), recall.float()


def cm_normalized(cm):
    """按真实类别行归一化。"""
    row_sum = cm.sum(dim=1, keepdim=True)
    return torch.where(row_sum > 0, cm / row_sum, torch.zeros_like(cm)).float()


def cm_class_proportions(cm):
    """true / pred 各类占比，分母统一为 true 像素总数（与 compute_class_proportions 一致）。"""
    _, pred_sum, true_sum, total = _cm_parts(cm)
    denom = total.clamp_min(1.0)
    return (true_sum / denom).float(), (pred_sum / denom).float()


def cm_ordinal_cost(cm, cost_per_distance=None):
    """有序代价指标。mean_cost = Σ_{t,p} cm[t,p]·cost[|t-p|] / Σ cm。"""
    n = cm.shape[0]
    total = cm.sum()
    if total <= 0:
        return torch.tensor(0.0), torch.tensor(1.0)

    if cost_per_distance is None:
        cost_table = torch.arange(n, dtype=torch.float64, device=cm.device)
    else:
        cost_table = torch.tensor(cost_per_distance, dtype=torch.float64, device=cm.device)

    idx = torch.arange(n, device=cm.device)
    dist = (idx.view(-1, 1) - idx.view(1, -1)).abs()
    cost_matrix = cost_table[dist]

    mean_cost = (cm * cost_matrix).sum() / total
    max_cost = cost_table.max().clamp(min=1e-6)
    return mean_cost.float(), (1.0 - mean_cost / max_cost).float()


def create_train_validation_and_test_scene_list(train_options):
    '''
    Creates the train, validation, and test scene lists from JSON datalist files.
    For MyDS, the filenames in the JSON files are the actual .nc filenames
    (no string manipulation needed).
    '''

    # Train ------------
    with open(train_options['path_to_env'] + train_options['train_list_path']) as file:
        train_options['train_list'] = json.loads(file.read())

    # Validation ---------
    if train_options['cross_val_run']:
        train_options['validate_list'] = list(np.random.choice(np.array(
            train_options['train_list']), size=train_options['p-out'], replace=False))
    else:
        with open(train_options['path_to_env'] + train_options['val_path']) as file:
            train_options['validate_list'] = json.loads(file.read())

    # Remove the validation scenes from the train list.
    train_options['train_list'] = [scene for scene in train_options['train_list']
                                   if scene not in train_options['validate_list']]

    # Test ----------
    with open(train_options['path_to_env'] + train_options['test_path']) as file:
        train_options['test_list'] = json.loads(file.read())
    # For MyDS, labels are embedded in the same file - reference list equals test list.
    train_options['test_list_reference'] = train_options['test_list']

    print('Options initialised')


def get_scheduler(train_options, optimizer):
    if train_options['scheduler']['type'] == 'CosineAnnealingLR':
        T_max_epochs = train_options['scheduler'].get('T_max_epochs', train_options['epochs'])
        T_max = T_max_epochs * train_options['epoch_len']
        # 线性 warmup（2026-07-20，训练稳定化实验）：scheduler 在本项目是【逐 batch】
        # 步进的（quickstart 训练循环内 scheduler.step()），故 warmup 长度 =
        # warmup_epochs × epoch_len 个 step，平滑爬坡；所有参数组（含判别式 LR 的
        # encoder 组）按同一比例从 warmup_start_factor 爬到各自基准 LR。cosine 的
        # T_max 相应扣除 warmup 段，总调度长度不变。默认 warmup_epochs=0 = 关闭，
        # 与历史行为逐位一致。⚠️ SequentialLR 的 state_dict 结构与纯 cosine 不同：
        # --resume-from 只能用同 warmup 配置的 checkpoint。
        warmup_epochs = train_options['scheduler'].get('warmup_epochs', 0)
        if warmup_epochs > 0:
            warmup_iters = warmup_epochs * train_options['epoch_len']
            assert warmup_iters < T_max, 'warmup_epochs 不能 >= 总调度长度'
            start_factor = train_options['scheduler'].get('warmup_start_factor', 0.01)
            linear = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_iters)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=T_max - warmup_iters, eta_min=train_options['scheduler']['lr_min'])
            print(f'[Scheduler] linear warmup {warmup_epochs} epochs '
                  f'({warmup_iters} steps, start_factor={start_factor}) -> cosine '
                  f'T_max={T_max - warmup_iters} steps', flush=True)
            return torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[linear, cosine], milestones=[warmup_iters])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max,
                                                               eta_min=train_options['scheduler']['lr_min'])
    elif train_options['scheduler']['type'] == 'CosineAnnealingWarmRestartsLR':
        # T_max = train_options['epochs']*train_options['epoch_len']
        T_0 = train_options['scheduler']['EpochsPerRestart']*train_options['epoch_len']
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0,
                                                                         T_mult=train_options['scheduler']['RestartMult'],
                                                                         eta_min=train_options['scheduler']['lr_min'],
                                                                         last_epoch=-1
                                                                        #  verbose=False
                                                                         )
    elif train_options['scheduler']['type'] == 'ReduceLROnPlateau':
        # 每 epoch 验证后调用 scheduler.step(val_score)，无需 per-batch 步进
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',                                               # 监控的指标越大越好（combined_score）
            factor=train_options['scheduler'].get('factor', 0.5),    # 触发时 lr *= factor
            patience=train_options['scheduler'].get('patience', 5),  # 连续多少个验证epoch无改善后触发
            min_lr=train_options['scheduler'].get('min_lr', 1e-5),   # lr 下限
            threshold=train_options['scheduler'].get('threshold', 1e-4),  # 判定"改善"的最小变化量
        )
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1, total_iters=5, last_epoch=- 1,
                                                        # verbose=False
                                                        )
    return scheduler


def _optimizer_param_groups(train_options, net):
    """按 optimizer.encoder_lr_scale 把 encoder 参数单独分组、LR 缩放。

    用途：预训练 encoder + 随机初始化 head 时给 encoder 更小的 LR（判别式 LR，
    mmseg 官方 SegFormer 配方即 backbone 1x / head 10x）。默认 1.0 = 单组、
    与历史行为逐位一致（所有已跑 run 都是单组统一 LR）。
    识别规则：参数名以 'encoder.' 开头或含 '.encoder.'（SegFormer 的 self.encoder、
    SMP 系模型的 model.encoder 都命中）。启用时打印分组统计，启动日志须核对。
    ⚠️ checkpoint 的 optimizer.state_dict 与分组结构绑定：--resume-from 必须用
    与原 run 相同的 encoder_lr_scale。
    """
    base_lr = train_options['optimizer']['lr']
    scale = train_options['optimizer'].get('encoder_lr_scale', 1.0)
    if scale == 1.0:
        return [{'params': list(net.parameters())}]
    enc_params, rest_params = [], []
    for name, p in net.named_parameters():
        if name.startswith('encoder.') or '.encoder.' in name:
            enc_params.append(p)
        else:
            rest_params.append(p)
    if not enc_params:
        raise ValueError(
            "optimizer.encoder_lr_scale != 1.0 但没有任何参数名匹配 'encoder.'——"
            "该模型不适用此选项，检查 model_selection 或参数命名。")
    n_enc = sum(p.numel() for p in enc_params)
    n_rest = sum(p.numel() for p in rest_params)
    print(f"[Optimizer] encoder_lr_scale={scale}: encoder {n_enc/1e6:.1f}M params "
          f"lr={base_lr * scale:.2e} | rest {n_rest/1e6:.1f}M params lr={base_lr:.2e}")
    # base LR 组放首位：quickstart 的日志/wandb 记的是 param_groups[0]['lr']，
    # 保持其语义（=config 的 lr）不因分组而变。
    return [{'params': rest_params},
            {'params': enc_params, 'lr': base_lr * scale}]


def get_optimizer(train_options, net):
    param_groups = _optimizer_param_groups(train_options, net)
    if train_options['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(param_groups, lr=train_options['optimizer']['lr'],
                                     betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                     weight_decay=train_options['optimizer']['weight_decay'])

    elif train_options['optimizer']['type'] == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=train_options['optimizer']['lr'],
                                      betas=(train_options['optimizer']['b1'], train_options['optimizer']['b2']),
                                      weight_decay=train_options['optimizer']['weight_decay'])
    else:
        optimizer = torch.optim.SGD(param_groups, lr=train_options['optimizer']['lr'],
                                    momentum=train_options['optimizer']['momentum'],
                                    dampening=train_options['optimizer']['dampening'],
                                    weight_decay=train_options['optimizer']['weight_decay'],
                                    nesterov=train_options['optimizer']['nesterov'])
    return optimizer


def get_loss(loss, chart=None, **kwargs):
    # TODO Fix Dice loss, Jacard loss,  MCC loss, SoftBCEWithLogitsLoss,
    """_summary_

    Args:
        loss (str): the name of the loss
    Returns:
        loss: The corresponding
    """
    if loss == 'DiceLoss':
        kwargs.pop('type')
        loss = smp.losses.DiceLoss(**kwargs)
    elif loss == 'FocalLoss':
        from floe.losses import FocalLoss as _FocalLoss
        kwargs.pop('type')
        loss = _FocalLoss(**kwargs)
    elif loss == 'JaccardLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.JaccardLoss(**kwargs)
    elif loss == 'LovaszLoss':
        kwargs.pop('type')
        loss = smp.losses.LovaszLoss(**kwargs)
    elif loss == 'MCCLoss':
        kwargs.pop('type')
        loss = smp.losses.MCCLoss(**kwargs)
    elif loss == 'SoftBCEWithLogitsLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftBCEWithLogitsLoss(**kwargs)
    elif loss == 'SoftCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = smp.losses.SoftCrossEntropyLoss(**kwargs)
    elif loss == 'TverskyLoss':
        kwargs.pop('type')
        loss = smp.losses.TverskyLoss(**kwargs)
    elif loss == 'CrossEntropyLoss':
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = torch.nn.CrossEntropyLoss(**kwargs)
    elif loss == 'BinaryCrossEntropyLoss':
        raise NotImplementedError
        kwargs.pop('type')
        loss = torch.nn.BCELoss(**kwargs)
    elif loss == 'OrderedCrossEntropyLoss':
        from floe.losses import OrderedCrossEntropyLoss
        kwargs.pop('type')
        loss = OrderedCrossEntropyLoss(**kwargs)
    elif loss == 'GCELoss':
        from floe.losses import GCELoss
        kwargs.pop('type')
        if 'weight' in kwargs and isinstance(kwargs['weight'], (list, tuple)):
            kwargs['weight'] = torch.FloatTensor(kwargs['weight'])
        loss = GCELoss(**kwargs)
    elif loss == 'EMDLoss':
        from floe.losses import EMDLoss
        kwargs.pop('type')
        loss = EMDLoss(**kwargs)
    elif loss == 'MSELossFromLogits':
        from floe.losses import MSELossFromLogits
        kwargs.pop('type')
        loss = MSELossFromLogits(chart=chart, **kwargs)
    elif loss == 'MSELoss':
        kwargs.pop('type')
        loss = torch.nn.MSELoss(**kwargs)
    elif loss == 'MSELossWithIgnoreIndex':
        from floe.losses import MSELossWithIgnoreIndex
        kwargs.pop('type')
        loss = MSELossWithIgnoreIndex(**kwargs)
    else:
        raise ValueError(f'The given loss \'{loss}\' is unrecognized or Not implemented')

    return loss


def get_model(train_options, device):
    if train_options['model_selection'] in ['dbunet', 'DBUNet']:
        from DBU_Net import DBUNet_ASPP
        net = DBUNet_ASPP(options=train_options).to(device)
    elif train_options['model_selection'] == 'unet':
        net = UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'swin':
        from floe.models.legacy.swin_transformer import SwinTransformer
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'PA2':
        from PA2_Swin import SwinTransformer
        net = SwinTransformer(options=train_options).to(device)
    elif train_options['model_selection'] == 'PA2_swin_improved':
        from PA2_Swin_Improved import SwinTransformerImproved
        net = SwinTransformerImproved(options=train_options).to(device)
    # 新增 SegNetXt 支持
    elif train_options['model_selection'] == 'SegNetXt':
        from SegNetXt import SegNeXt
        # 根据输入变量数量计算输入通道数
        input_channels = len(train_options['train_variables'])
        # 调整 embed_dims 的第一个维度为实际输入通道数
        embed_dims = [input_channels, 32, 64, 160, 256]
        
        net = SegNeXt(num_classes=train_options['n_classes'], embed_dims=embed_dims).to(device)
        
    elif train_options['model_selection'] == 'h_unet':
        from floe.models.legacy.unet import H_UNet
        net = H_UNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'h_unet_argmax':
        from floe.models.legacy.unet import H_UNet_argmax
        net = H_UNet_argmax(options=train_options).to(device)
    elif train_options['model_selection'] == 'Separate_decoder':
        net = Sep_feat_dif_stages(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression', 'unet_regression']:
        from floe.models.legacy.unet import UNet_regression
        net = UNet_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_regression_all']:
        from floe.models.legacy.unet import UNet_regression_all
        net = UNet_regression_all(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_regression', 'unet_sep_dec_regression']:
        from floe.models.legacy.unet import UNet_sep_dec_regression
        net = UNet_sep_dec_regression(options=train_options).to(device)
    elif train_options['model_selection'] in ['UNet_sep_dec_mse']:
        from floe.models.legacy.unet import UNet_sep_dec_mse
        net = UNet_sep_dec_mse(options=train_options).to(device)
    elif train_options['model_selection'] == 'smp':
        from floe.models.smp import SMPWrapper
        net = SMPWrapper(options=train_options).to(device)
    elif train_options['model_selection'] == 'mmseg':
        from floe.models.mmseg import MMSegWrapper
        net = MMSegWrapper(options=train_options).to(device)
    elif train_options['model_selection'] == 'pa2_fpn_mit_b2':
        from PA2_MiT_FPN import PA2FPN_MiT_B2
        net = PA2FPN_MiT_B2(options=train_options).to(device)
    # ------------------------------------------------------------------ #
    # SOTA comparison models
    # ------------------------------------------------------------------ #
    elif train_options['model_selection'] == 'sota_unet':
        from floe.models.sota.unet import UNet as _SOTAUNet
        net = _SOTAUNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segnet':
        from floe.models.sota.segnet import SegNet as _SegNet
        net = _SegNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_resnet':
        from floe.models.sota.resnet import ResNetFPN
        net = ResNetFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_densenet':
        from floe.models.sota.densenet import DenseNetFPN
        net = DenseNetFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_pspnet':
        from floe.models.sota.pspnet import PSPNet as _PSPNet
        net = _PSPNet(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_deeplabv3':
        from floe.models.sota.deeplabv3 import DeepLabV3Plus
        net = DeepLabV3Plus(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segformer':
        from floe.models.sota.segformer import SegFormer
        net = SegFormer(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_poolformer':
        from floe.models.sota.poolformer import PoolFormerFPN
        net = PoolFormerFPN(options=train_options).to(device)
    elif train_options['model_selection'] == 'sota_segnext':
        from floe.models.sota.segnext import SegNeXt
        net = SegNeXt(options=train_options).to(device)
    else:
        raise 'Unknown model selected'
    return net


def ema_update(ema_net, net, decay):
    """H3（2026-07-29）EMA 影子更新：float 张量 lerp（v ← d·v + (1−d)·m），整型
    buffer（BN num_batches_tracked 等）直拷。state_dict 级操作 ⇒ 参数与 BN running
    stats 共用同一 EMA 视野（timm ModelEmaV2 同语义）。

    抽成独立函数的原因（code-review 🟡4）：训练主循环里的内联版无法被冒烟单测；
    smoke_h_batch.py 直接对它断言（lerp 与手写参考逐位一致、raw 不被污染、整型直拷）。
    """
    with torch.no_grad():
        msd = net.state_dict()
        for k, v in ema_net.state_dict().items():
            m = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(decay).add_(m, alpha=1.0 - decay)
            else:
                v.copy_(m)


def supcon_region_loss(feat, labels, n_classes, temperature=0.1, min_pixels=16,
                       return_stats=False):
    """F3（2026-07-28）跨景区域级监督对比辅助损失（M3 落地版）。

    机理：campaign 最确认的失败签名是跨景不迁移（cake train IoU 0.53~0.70 vs val
    0.02~0.04；M1 证明白送不变特征模型也不放弃亮度依赖）——所有 14 枪的监督都是逐
    像素 CE 变体，没有任何一项直接惩罚"依赖场景身份的表征"。本损失把 batch 里每个
    (样本, 类别) 的 stride-4 投影特征做掩膜均值池化成一个"区域嵌入"，同类跨样本拉近、
    异类推远（SupCon, Khosla et al. 2020），直接优化 Stage 0 探针里那个失败的量
    （跨景区域可分性）。每个样本每类至多产出 1 个嵌入 ⇒ 正样本对**天然全部跨 patch**
    （多数跨 tile/景；tile 采样 batch=16 patch 来自 8 个随机加权 tile、横跨 375 景）。

    参数：
        feat: [B, D, h, w] —— SegFormer forward 输出的 '_supcon' 投影特征
              （autocast 下是 fp16；本函数内部全程 fp32 计算）。
        labels: [B, H, W]（device 上），255=ignore；H/W 为 patch 分辨率，内部
              nearest 下采样到 (h, w)。
        n_classes: 类别数（FLOE=6）。
        temperature: SupCon 温度 τ。
        min_pixels: (样本, 类别) 在 stride-4 网格上的最少像素数，低于则不产出区域
              嵌入（太小的区域池化统计不可靠）。内部钳到 >=1（0 会让空区域除 0 产 NaN）。
        return_stats: True 时返回 (loss, stats)。stats 为纯 float/list（已 .item()，
              有 GPU 同步开销但每 batch 只几次，与既有 train_loss_sum 的 .item() 同量级），
              供 quickstart 按 epoch 聚合进 wandb —— 这是预注册判读读数的主通道
              （code-review 指出几何间隔打印在 best epoch 附近拿不到读数）。键：
              active(0/1 本 batch 是否产生了有效损失)、regions、anchors、
              same_cos/diff_cos（同类跨样本 / 异类的平均余弦；inactive 时 nan）、
              regions_per_class（长度 n_classes，本 batch 各类产出的区域嵌入数——
              cake/small 若长期为 0 说明 min_pixels 过滤把正打的类挡在门外）。
    返回：标量 loss（fp32），或 (loss, stats)。batch 内无任何同类跨样本对时 loss=0
        标量（无梯度贡献；CE 分量仍有梯度，安全）。

    ⚠️ 只在训练路径调用；验证 loss 不含本项（不碰按需读路径的按批加权聚合前提）。
    """
    min_pixels = max(1, int(min_pixels))

    def _ret(loss, stats):
        return (loss, stats) if return_stats else loss

    def _empty_stats(per_class=None):
        return {'active': 0.0, 'regions': 0.0, 'anchors': 0.0,
                'same_cos': float('nan'), 'diff_cos': float('nan'),
                'regions_per_class': per_class if per_class is not None
                                     else [0.0] * int(n_classes)}

    B, D, h, w = feat.shape
    f = feat.float().flatten(2)                                   # [B, D, hw]
    with torch.no_grad():
        y = torch.nn.functional.interpolate(
            labels.float().unsqueeze(1), size=(h, w),
            mode='nearest').squeeze(1).long()                     # [B, h, w]，255 保持
        yf = y.flatten(1)                                         # [B, hw]
        valid = (yf >= 0) & (yf < n_classes)
        onehot = torch.zeros(B, n_classes, yf.shape[1],
                             device=feat.device, dtype=torch.float32)
        onehot.scatter_(1, yf.clamp(0, n_classes - 1).unsqueeze(1), 1.0)
        onehot = onehot * valid.unsqueeze(1).float()              # 255/越界 清零
        counts = onehot.sum(-1)                                   # [B, C]
        keep = counts >= float(min_pixels)                        # [B, C]
        per_class = keep.sum(dim=0).float().tolist()              # [C] 各类区域数
    if int(keep.sum().item()) < 2:
        return _ret(torch.zeros((), device=feat.device), _empty_stats(per_class))
    sums = torch.bmm(onehot, f.transpose(1, 2))                   # [B, C, D]
    emb = sums[keep] / counts[keep].unsqueeze(-1)                 # [N, D]
    cls = keep.nonzero(as_tuple=False)[:, 1]                      # [N] 类别 id
    z = torch.nn.functional.normalize(emb, dim=1)                 # fp32 归一化
    n = z.shape[0]
    sim = z @ z.t() / float(temperature)                          # [N, N]
    eye = torch.eye(n, dtype=torch.bool, device=feat.device)
    pos_mask = (cls.unsqueeze(0) == cls.unsqueeze(1)) & ~eye
    has_pos = pos_mask.any(dim=1)
    if not bool(has_pos.any()):
        return _ret(torch.zeros((), device=feat.device), _empty_stats(per_class))
    # 分母排除自身：masked_fill(-inf) + logsumexp（内部自带 max 平移，数值稳定）。
    # 相比 exp/clamp_min 地板的写法多一个好处：sim_ip 恒是分母求和项之一 ⇒
    # log_prob ≤ 0 ⇒ loss ≥ 0 有代数保证，极小 τ 下也不会因地板托底而变负。
    log_prob = sim - torch.logsumexp(
        sim.masked_fill(eye, float('-inf')), dim=1, keepdim=True)
    mean_log_prob_pos = ((pos_mask.float() * log_prob).sum(dim=1)
                         / pos_mask.sum(dim=1).clamp_min(1).float())
    loss = -mean_log_prob_pos[has_pos].mean()
    # 对齐度读数：同类跨样本 cos vs 异类 cos（cos = sim·τ 还原）。
    # 平/负结果时区分「没对齐」（间隔≈0 ⇒ 机制没咬合）和「对齐了但无益」（间隔大
    # ⇒ 表征变了、信息上限）。return_stats=True（训练路径）时每 batch 都算、由
    # quickstart 聚合成 epoch 级 wandb 曲线；几何间隔的 print 只是开箱即视的快照。
    same_m = diff_m = float('nan')
    if return_stats or getattr(supcon_region_loss, '_calls', 0) + 1 in (
            1, 100, 1000, 10000, 100000):
        with torch.no_grad():
            cos = sim * float(temperature)
            same = pos_mask
            diff = (~eye) & ~pos_mask
            same_m = cos[same].mean().item() if bool(same.any()) else float('nan')
            diff_m = cos[diff].mean().item() if bool(diff.any()) else float('nan')
    supcon_region_loss._calls = getattr(supcon_region_loss, '_calls', 0) + 1
    if supcon_region_loss._calls in (1, 100, 1000, 10000, 100000):
        print(f'[SupCon] call #{supcon_region_loss._calls}: regions={n} '
              f'(anchors={int(has_pos.sum().item())}) | 同类跨样本 cos {same_m:.3f} '
              f'vs 异类 {diff_m:.3f}（间隔≈0 = 没对齐）| loss {loss.item():.3f} | '
              f'per-class regions {[int(v) for v in per_class]}', flush=True)
    stats = {'active': 1.0, 'regions': float(n),
             'anchors': float(int(has_pos.sum().item())),
             'same_cos': same_m, 'diff_cos': diff_m,
             'regions_per_class': per_class}
    return _ret(loss, stats)
