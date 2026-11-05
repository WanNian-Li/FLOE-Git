# -- Built-in modules -- #
import os
import os.path as osp
import csv
import math
from collections import OrderedDict
from tqdm import tqdm

# -- Third-party modules -- #
import h5py
import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset, get_worker_info
import torchvision.transforms.functional as TF
from scipy.ndimage import binary_erosion as _scipy_binary_erosion

# -- Proprietary modules -- #
from floe.functions import rand_bbox


# Global LRU cache for pre-downsampled training scenes, bounded by byte size
# (see _get_or_create_cached_scene). Keyed by (data_root, filename, scale/options),
# value stores CPU tensors. OrderedDict so move_to_end()/popitem(last=False) give O(1) LRU.
_SCENE_CACHE = OrderedDict()
_SCENE_CACHE_BYTES = 0
# 40GB，此值为【跨所有 DataLoader worker 的总预算】（见 _effective_scene_cache_max_bytes），
# 按容器可用内存在 config 的 scene_cache_max_bytes 里覆盖。
_DEFAULT_SCENE_CACHE_MAX_BYTES = 40 * 1024 ** 3


# ---------------------------------------------------------------------------
# Tile 采样（2026-07-09 引入，取代整景解码 + 随机裁 + 概率拒绝的旧路径）
#
# .nc 底层 HDF5 的二维变量是 chunks=(512,512)+gzip4 存储的，可以只解压覆盖到的 chunk。
# 实测：读一个 chunk 对齐的 512×512 tile（8 个变量）约 27ms，而 xarray 整景解码约
# 9.2s/景（输出 2.8~3.1GB float32，正是 OOM 的来源）。按 patches_per_tile=2 折算，
# 每 patch 约 13.5ms，对比旧路径每 patch 摊 ~1.53s（缓存命中率约 1%）。
#
# 同时 scan_tiles.py 的全量统计显示：全图 69.7% 的像元是 255（黑边+陆地，在 loss 里被
# ignore_index 屏蔽，梯度贡献恒为 0），按 512 对齐切 tile 后 56.4% 的 tile 是"全无效"
# （没有任何一个像元能参与 loss）。这些 tile 被 scan_tiles.py 从索引里剔除，训练时永远
# 不会被读到。剩下的 tile 按类别直方图加权采样，接受率 100%，不再需要任何概率拒绝重试。
# ---------------------------------------------------------------------------

# 每个进程独立的 HDF5 句柄 LRU。句柄本身只占几 KB，缓存它是为了省掉重复 open 的
# ~10ms 以及 chunk cache 的重建。
_H5_HANDLES = OrderedDict()
_H5_HANDLES_PID = None
_H5_MAX_OPEN = 16
# HDF5 的 chunk cache 默认只有 1MB，装不下一个 512×512 float32 chunk（1MB）——
# 同一 tile 的多个变量/多次 crop 会反复解压同一个 chunk。给到 64MB 足够放下
# 一个 tile 的全部 8 个 chunk。
_H5_RDCC_NBYTES = 64 * 1024 ** 2


def _get_h5_handle(path):
    """按进程缓存打开的 HDF5 句柄。

    HDF5 库在 fork 之后使用继承来的句柄是未定义行为（父子共享同一个文件描述符的偏移
    和 chunk cache，并发读会读到错乱数据）。本函数只在 DataLoader worker 里被首次调用
    （__getitem__ 内），父进程从不打开文件，所以正常路径下不存在继承句柄；下面的 pid
    校验只是安全网：一旦发现当前 pid 与建表时不同，就丢弃整张表在本进程重新打开。
    直接 clear() 而不 close()：fork 后子进程的 fd 是独立的，GC 关掉子进程这份不会影响父进程。
    """
    global _H5_HANDLES_PID
    pid = os.getpid()
    if _H5_HANDLES_PID != pid:
        _H5_HANDLES.clear()
        _H5_HANDLES_PID = pid

    handle = _H5_HANDLES.get(path)
    if handle is not None:
        _H5_HANDLES.move_to_end(path)
        return handle

    handle = h5py.File(path, 'r', rdcc_nbytes=_H5_RDCC_NBYTES, rdcc_nslots=5003)
    _H5_HANDLES[path] = handle
    while len(_H5_HANDLES) > _H5_MAX_OPEN:
        _, evicted = _H5_HANDLES.popitem(last=False)
        try:
            evicted.close()
        except Exception:
            pass
    return handle


def cpu_quota():
    """容器实际可用的 CPU 核数（cgroup 配额），而不是宿主机核数。

    `nproc` / `os.cpu_count()` 在 AutoDL 容器里报的是宿主机核数（实测 208），
    真正的限制在 cgroup 的 cpu.max 里（实测 2500000/100000 = 25 核）。
    这和 `free -h` 报宿主机内存、真正的限制在 memory.max 里是同一类陷阱。
    """
    try:
        with open('/sys/fs/cgroup/cpu.max') as f:          # cgroup v2
            quota, period = f.read().split()
        if quota != 'max':
            return max(1, int(int(quota) / int(period)))
    except Exception:
        pass
    try:
        with open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us') as f:   # cgroup v1
            quota = int(f.read())
        with open('/sys/fs/cgroup/cpu/cpu.cfs_period_us') as f:
            period = int(f.read())
        if quota > 0:
            return max(1, quota // period)
    except Exception:
        pass
    return len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 1)


_THREADS_LIMITED = False


def limit_torch_threads():
    """把 torch 的 intraop 线程数收到与 cgroup CPU 配额相称的水平。

    torch 按 `nproc`（宿主机 208 核）建 intraop 线程池，而容器只有 25 核的配额，于是
    每个并行算子都在做严重超订的线程同步。2026-07-09 实测这不是"慢一点"而是灾难性的：
    一次 7×256×256 的 float32 cast 要 3.3 秒（正常是微秒级），一个 batch 的组装从
    0.55 秒膨胀到 205 秒（373 倍）。numpy 的 astype 也被这些自旋线程挤慢 170 倍。

    DataLoader worker 本身已经是进程级并行，每个 worker 再开多线程只会互相踩踏，设成 1。
    主进程留少量线程给 CPU 侧算子（collate、验证时在 CPU 上攒预测等）。
    """
    global _THREADS_LIMITED
    if _THREADS_LIMITED:
        return
    torch.set_num_threads(1 if get_worker_info() is not None else max(1, min(8, cpu_quota())))
    _THREADS_LIMITED = True


def _solve_tile_weights(hist, target, iters=400, eta=0.5):
    """求每个 tile 的采样概率，使训练实际看到的【像素】类别分布逼近 target。

    权重族取 w_i ∝ Σ_c a_c · frac[i, c]（frac 是 tile 内各类归一化占比，尺度无关），
    对 a 做乘法更新（RAS / IPF 式迭代）：每轮算出当前期望像素分布 p，
    再把 a_c 乘以 (target_c / p_c)^eta。target 落在 {frac[i]} 的凸包内时收敛。

    **期望像素分布必须按像素数加权，不能取各 tile 归一化占比的平均**：
        p = (Σ_i w_i · hist[i]) / Σ_i Σ_c w_i · hist[i, c]      ← 正确
        p = Σ_i w_i · frac[i]                                    ← 错误（首版如此）
    两者只在所有 tile 的有效像元数相等时才一致。tile 的有效像元数差异很大（有的一半是
    陆地/黑边），2026-07-09 实测首版的错误口径让求解器自以为命中了 target，实际分布却
    偏出 ±0.02。

    传进来的 hist 应该是 scan_tiles.py 里【按裁剪覆盖核加权】的 hist_w：训练是在 tile 内
    随机裁 patch，tile 中心的像素比边缘更容易被采到，用原始直方图求解会再偏 ±0.015。

    之所以不能直接"按类别稀有度的倒数配权重"：big/vast 的像素总量和 small/medium 相当，
    但只分布在约 1/3 的 tile 数量上（大浮冰是大块连续的，小/中浮冰是散布的），
    因此 tile 级的均匀加权会系统性地压低 big/vast 的像素份额。

    Parameters
    ----------
    hist : (N, C) float64  每个 tile 各类别的（核加权）像素数，255 已排除
    target : (C,) float64  目标像素占比，行和为 1；某类为 0 表示不参与（如当前数据里不存在的类）

    Returns
    -------
    w : (N,) float64  归一化的 tile 采样概率
    achieved : (C,) float64  该权重下的期望像素分布（供调用方与 target 核对）
    """
    frac = hist / np.maximum(hist.sum(axis=1, keepdims=True), 1e-12)
    active = target > 0
    a = np.zeros(hist.shape[1], dtype=np.float64)
    a[active] = 1.0

    def _pixel_mix(a_vec):
        s = frac @ a_vec
        total = s.sum()
        if not np.isfinite(total) or total <= 0:
            return None, None
        w_vec = s / total
        px = w_vec @ hist            # 期望各类像素数（按像素加权，不是占比平均）
        px_total = px.sum()
        if not np.isfinite(px_total) or px_total <= 0:
            return None, None
        return w_vec, px / px_total

    for _ in range(iters):
        w, p = _pixel_mix(a)
        if w is None:
            break
        ratio = np.ones_like(a)
        ok = active & (p > 0)
        ratio[ok] = target[ok] / p[ok]
        a = a * np.power(ratio, eta)
        a[~active] = 0.0
        amax = a.max()
        if amax > 0:
            a /= amax

    w, p = _pixel_mix(a)
    if w is None:
        raise RuntimeError('tile 采样权重求解失败：所有 tile 权重为 0，请检查 target_pixel_mix')
    return w, p


def _scene_cache_key(options, file):
    return (
        options['path_to_train_data'],
        file,
        options['down_sample_scale'],
        options['loader_downsampling'],
        options['patch_size'],
        tuple(options['full_variables']),
        tuple(options['charts']),
        options.get('cls2_filter_mask_dir', None),
    )


def _load_downsampled_scene(options, file, n_charts_init, cls2_mask_dir):
    """Load one scene, downsample/pad once, and return tensors ready for random crops."""
    # 2026-07-09 吞吐修复：原生分辨率下重新解析 .nc（NetCDF 解压 + xarray 拼接多个
    # 变量/GLCM 通道）是训练吞吐的主要瓶颈，单景约 9~10 秒；缓存受内存上限约束只能
    # 装下 288 景里的一小部分，200 epoch 训练里绝大多数请求都要重新走这条慢路径。
    # 若 preprocess_scenes.py 已经把该场景转成 .npz（labels=uint8/feats=float16，
    # 原生分辨率、已跳过慢的 NetCDF 解析步骤），优先读这个；数值上与直接解析 .nc
    # 完全一致，只是跳过了慢的那一步。未预处理/未配置目录时回退到原始 .nc 解析路径
    # （保持向后兼容，SOD 等未预处理的数据集仍可正常工作）。
    preproc_dir = options.get('preprocessed_data_dir', None)
    preproc_path = (os.path.join(preproc_dir, file.replace('.nc', '.npz'))
                     if preproc_dir is not None else None)

    if preproc_path is not None and os.path.exists(preproc_path):
        with np.load(preproc_path) as npz:
            labels_np = npz['labels']  # uint8, (n_charts_init, H, W)
            feats_np = npz['feats']    # float16, (n_features, H, W)
        temp_scene = torch.from_numpy(
            np.concatenate([labels_np.astype(np.float32), feats_np.astype(np.float32)], axis=0)
        ).unsqueeze(0)
    else:
        scene_path = os.path.join(options['path_to_train_data'], file)
        with xr.open_dataset(scene_path, engine='h5netcdf', mask_and_scale=False) as scene:
            temp_scene = scene[options['full_variables']].to_array()
            temp_scene = torch.from_numpy(np.expand_dims(temp_scene, 0))

    # down_sample_scale=1 时输出尺寸与输入完全相同，interpolate 是纯粹浪费——额外
    # 分配一份满尺寸临时张量、还要做插值计算。原生分辨率(FLOE)下单景可达 150M+
    # 像素，这份瞬时峰值内存是多 worker 并发解码时导致 OOM 的重要原因之一
    # （2026-07-09 复盘：即使有字节预算 LRU，稳态缓存达标后仍会在并发解码时被
    # 瞬时峰值顶爆），故 scale=1 时直接跳过，非 1 时行为不变。
    if options['down_sample_scale'] != 1:
        temp_scene = torch.nn.functional.interpolate(
            temp_scene,
            size=(temp_scene.size(2) // options['down_sample_scale'],
                  temp_scene.size(3) // options['down_sample_scale']),
            mode=options['loader_downsampling'])

    # Capture downsampled size before any padding (used for mask interpolation).
    h_down = temp_scene.size(2)
    w_down = temp_scene.size(3)

    if temp_scene.size(2) < options['patch_size']:
        height_pad = options['patch_size'] - temp_scene.size(2) + 1
    else:
        height_pad = 0

    if temp_scene.size(3) < options['patch_size']:
        width_pad = options['patch_size'] - temp_scene.size(3) + 1
    else:
        width_pad = 0

    if height_pad > 0 or width_pad > 0:
        temp_scene_y = torch.nn.functional.pad(
            temp_scene[:, :n_charts_init], (0, width_pad, 0, height_pad), mode='constant', value=255)
        temp_scene_x = torch.nn.functional.pad(
            temp_scene[:, n_charts_init:], (0, width_pad, 0, height_pad), mode='constant', value=0)
        temp_scene = torch.cat((temp_scene_y, temp_scene_x), dim=1)

    temp_scene = torch.squeeze(temp_scene)
    scene_labels = temp_scene[:n_charts_init].to(torch.uint8).contiguous()
    scene_feats = temp_scene[n_charts_init:].to(torch.float16).contiguous()

    cls2_mask = None
    if cls2_mask_dir is not None:
        mask_name = os.path.basename(file).replace('.nc', '_cls2_mask.npy')
        mask_path = os.path.join(cls2_mask_dir, mask_name)
        if os.path.exists(mask_path):
            try:
                mask_np = np.load(mask_path).astype(np.float32)
                mask_t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
                mask_ds = torch.nn.functional.interpolate(
                    mask_t, size=(h_down, w_down), mode='nearest'
                ).squeeze().numpy() > 0.5
                if height_pad > 0 or width_pad > 0:
                    mask_ds = np.pad(mask_ds,
                                     ((0, height_pad), (0, width_pad)),
                                     constant_values=False)
                cls2_mask = mask_ds
            except Exception as e:
                print(f'[WARN] Failed to load cls2 mask {mask_path}: {e}. Skipping mask for this scene.')

    return {
        'scene_labels': scene_labels,
        'scene_feats': scene_feats,
        'cls2_mask': cls2_mask,
    }


def _payload_nbytes(payload):
    """Approximate resident size of a cached scene payload, in bytes."""
    total = 0
    for k in ('scene_labels', 'scene_feats'):
        t = payload.get(k)
        if t is not None:
            total += t.element_size() * t.nelement()
    mask = payload.get('cls2_mask')
    if mask is not None:
        total += mask.nbytes
    return total


def _effective_scene_cache_max_bytes(options):
    """把配置里的 scene_cache_max_bytes 视为【跨所有 worker 的总预算】，而不是每个
    worker 各自的预算。PyTorch DataLoader 的 worker 是 fork 出的独立进程，每个 worker
    有自己独立的 _SCENE_CACHE/_SCENE_CACHE_BYTES 全局状态（互不共享）——如果直接把配置
    值当成"每个 worker 的预算"，num_workers=12 时实际峰值会是 12×配置值，完全达不到
    限制内存的目的（2026-07-06 OOM 复盘：第一版修复正是漏了这一步，12 个 worker 各自
    缓存到 35GB，合计 420GB，照样会被 cgroup OOM killer 杀掉）。
    """
    max_bytes = options.get('scene_cache_max_bytes', _DEFAULT_SCENE_CACHE_MAX_BYTES)
    worker_info = get_worker_info()
    if worker_info is not None and worker_info.num_workers > 0:
        max_bytes = max_bytes // worker_info.num_workers
    return max_bytes


def _get_or_create_cached_scene(options, file, n_charts_init, cls2_mask_dir, enable_cache=True):
    global _SCENE_CACHE_BYTES
    key = _scene_cache_key(options, file)
    if enable_cache and key in _SCENE_CACHE:
        _SCENE_CACHE.move_to_end(key)  # mark as most-recently-used
        return _SCENE_CACHE[key], True

    payload = _load_downsampled_scene(options, file, n_charts_init, cls2_mask_dir)
    if enable_cache:
        payload_bytes = _payload_nbytes(payload)
        max_bytes = _effective_scene_cache_max_bytes(options)
        # 淘汰最久未用的场景，直到腾出空间。原生分辨率(down_sample_scale=1)下单个场景
        # 可达 1~2GB，全量缓存几百个场景会远超容器内存上限（2026-07-06 OOM 崩溃教训），
        # 故按字节预算做 LRU 淘汰，而不是无限增长。
        while _SCENE_CACHE and (_SCENE_CACHE_BYTES + payload_bytes) > max_bytes:
            _, evicted = _SCENE_CACHE.popitem(last=False)
            _SCENE_CACHE_BYTES -= _payload_nbytes(evicted)
        _SCENE_CACHE[key] = payload
        _SCENE_CACHE_BYTES += payload_bytes
    return payload, False


def preload_scene_cache(options, files):
    """Preload a file list into the global scene cache (called once before CV loop)."""
    if options['down_sample_scale'] == 1:
        # Keep behavior consistent with dataset implementation.
        downsample = True
    else:
        downsample = True

    if not downsample:
        return

    unique_files = sorted(set(files))
    if len(unique_files) == 0:
        return

    n_charts_init = len(options['charts'])
    cls2_mask_dir = options.get('cls2_filter_mask_dir', None)
    cache_enabled = bool(options.get('enable_scene_cache', True))

    cache_hits = 0
    cache_misses = 0
    for file in tqdm(unique_files, desc='Preloading scenes', colour='blue'):
        _, was_hit = _get_or_create_cached_scene(
            options, file, n_charts_init, cls2_mask_dir, enable_cache=cache_enabled)
        if was_hit:
            cache_hits += 1
        else:
            cache_misses += 1

    print(f'[SceneCache] Preload complete: {len(unique_files)} files | '
          f'hits={cache_hits}, misses={cache_misses}, cache_size={len(_SCENE_CACHE)}')


def clear_scene_cache():
    global _SCENE_CACHE_BYTES
    _SCENE_CACHE.clear()
    _SCENE_CACHE_BYTES = 0


FLOE_CLASS_NAMES = ['water', 'cake', 'small', 'medium', 'big', 'vast']


def remap_floe_labels(arr, merge_map=None, ignore_val=255):
    """FLOE 标签重映射（就地）。所有读标签的路径都必须走这里，训练和验证口径才一致。

    1) 冰山(6) → 255：本数据集中该类为空，但安全网保留（DATASET_HANDBOOK §4）。
    2) 按 merge_map 合并类别，例如 {3: 2, 4: 3, 5: 4} 把【中浮冰并入小浮冰】，
       大/巨各降一位 → 0=水 1=饼状 2=小+中 3=大 4=巨，n_classes=5。

    ⚠️ 合并【必须同时完成】，不能顺序 in-place。若字典迭代顺序是 5→4 再 4→3，
    原本是 5 的像素会先变成 4、再被第二步二次映射成 3，静默串链。这里先把所有掩膜
    从原数组读出来，再统一写回，保证原子性。
    """
    arr[arr == 6] = ignore_val
    if not merge_map:
        return arr
    masks = [(arr == int(src), int(dst)) for src, dst in merge_map.items()]
    for m, dst in masks:
        arr[m] = dst
    return arr


def val_ondemand_ok(options, n_models=1):
    """这个 config 的验证路径能不能安全启用「按需读」。返回 (bool, 原因字符串)。

    按需读只解码「含有效标签的 tile 所覆盖的 chunk」，其余留 0。**这要求消费者只在那些
    tile 上做前向**，也就是推理必须走与 patch 网格对齐的不重叠 tile（
    `functions.fast_tiled_val_predict` / `fast_tiled_val_inference` + tile 门控）。任何会
    读到别处像素的用法都必须走整景读，否则会静默读到 0、且不会有任何报错：
      - **重叠滑窗**（`batched_slide_inference`）：窗口位置由 `swin_hp.val_stride` 决定，
        `configs/_base_/base.py` 的默认值是 [56,56]，只有 `sota_swin.py` 覆盖成
        [256,256] 才恰好与 tile 网格重合。新的 swin 系 config 忘了覆盖就会踩。
      - **翻转/偏移 TTA**（`_eval_tta.py`）：整景翻转或 pad 128 之后 tile 网格整体错位。
      - **整景可视化 / 预测导出**（`_viz_pred.py`、`inference.py`、`test_upload_function.py`）：
        未读区会被渲染/导出成常数 0。
      - **不同 patch_size 的集成**：`need` 网格只按第一个模型的 patch_size 算。
    所以按需读是【调用方显式 opt-in】的（`AI4ArcticChallengeTestDataset(..., ondemand_read=)`），
    不是靠 config 键默认打开；config 的 `val_ondemand_read` 只是全局 kill switch。
    """
    if not options.get('val_ondemand_read', True):
        return False, 'config 的 val_ondemand_read=False（全局关闭）'
    if n_models != 1:
        return False, f'集成推理（{n_models} 个模型），need 网格只按第一个模型的 patch_size 算'
    if options.get('down_sample_scale', 1) != 1 or options.get('val_downsample_scale', 1) != 1:
        return False, '降采样口径下整景 interpolate 需要全部像素'
    if 'global_valid_mask' not in list(options.get('train_variables', [])):
        return False, 'train_variables 里没有 global_valid_mask，tfv_mask 会退化成"未读即黑边"'
    if options.get('model_selection') in ('swin', 'PA2'):
        stride = tuple(options.get('swin_hp', {}).get('val_stride', ()))
        ps = options['patch_size']
        if stride != (ps, ps):
            return False, (f'滑窗推理 val_stride={stride} != ({ps}, {ps})，'
                           f'重叠窗口会读到未解码的区域')
    # 注：texture_channels 与按需读【兼容】——纹理是 per-tile 在 functions 前向里对每个
    # 256 tile 独立算的（不在整景 loader 里算），tile 的像素来自已读 chunk，纹理窗口在
    # tile 内截断（与训练 256 patch 同口径），不需要 tile 之外的邻域，故无需退回整景读。
    return True, 'ok'


# ---- B1 纹理通道（2026-07-25，模型攻坚 Phase 1）---------------------------------------
# 从 HH（通道 0）在线算的密集纹理特征，拼进输入。3 个多尺度局部标准差（变异函数的密集
# 代理，测「颗粒度尺度」：小碎冰场方差在很细尺度就饱和，中浮冰场慢慢积累）+ 1 个局部
# 边界密度（连通域的密集代理，测冰水边界碎不碎）。
# ⚠️ 增删通道数必须【同步】改：① 模型 in_channels（models/sota/segformer.py 等按
#   options 推导处 +N_TEXTURE_CHANNELS）；② 本文件两处 _n_channels 预分配。
N_TEXTURE_CHANNELS = 4
_TEXTURE_STD_WINDOWS = (3, 7, 15)   # 窗口(px)，80m 像元下 = 240 / 560 / 1200 m
_TEXTURE_EDGE_WINDOW = 7            # 边界密度窗口(px) ≈ 560 m


def _texture_channels(x):
    """从输入张量的 HH(通道0)算 N_TEXTURE_CHANNELS 个纹理通道，返回 [1, 4, H, W]。

    - 3 个多尺度局部标准差：σ(w) = sqrt(E[HH²]_w − E[HH]²_w)，盒式滤波 O(N)。
    - 1 个局部边界密度：|∇HH| 的局部均值。
    count_include_pad=False → 边界像素用实际窗口内像元数(无零偏)。x 形状 [1, C, H, W]，
    在 z-score 后的 HH 上算(loader 喂的就是 z-scored HH)；新通道不再单独归一化，交由
    首层 conv/BN 吸收尺度（4 个通道量级 ~O(0.1~1)，与 z-scored 通道同数量级）。"""
    import torch.nn.functional as F
    hh = x[:, 0:1, :, :].float()
    outs = []
    for w in _TEXTURE_STD_WINDOWS:
        pad = w // 2
        mean = F.avg_pool2d(hh, w, stride=1, padding=pad, count_include_pad=False)
        meansq = F.avg_pool2d(hh * hh, w, stride=1, padding=pad, count_include_pad=False)
        outs.append((meansq - mean * mean).clamp_min(0).sqrt())
    gx = F.pad(hh[:, :, :, 1:] - hh[:, :, :, :-1], (0, 1, 0, 0))   # 水平差分，右侧补 0
    gy = F.pad(hh[:, :, 1:, :] - hh[:, :, :-1, :], (0, 0, 0, 1))   # 垂直差分，底部补 0
    gmag = gx.abs() + gy.abs()                                     # L1 梯度幅值(便宜)
    ew = _TEXTURE_EDGE_WINDOW
    outs.append(F.avg_pool2d(gmag, ew, stride=1, padding=ew // 2, count_include_pad=False))
    return torch.cat(outs, dim=1)


def _need_chunk_grid(y, fill_values, charts, H, W, patch_size, chunk):
    """验证按需读：哪些 HDF5 chunk 必须读。返回 (cg_h, cg_w) 的 bool 网格。

    判定分两步，顺序不能反：
      1) 先在【patch_size 的 tile 网格】上找出「至少含一个有效标签像元」的 tile —— 这与
         functions.compute_valid_tile_mask 用的是同一套网格，也就是真正会被喂进网络的
         那些 tile。
      2) 再把每个有效 tile 覆盖到的【所有】chunk 标为必读。

    ⚠️ 第 2 步不能简化成「chunk 内有有效像元就读」：一个有效 tile 可能跨 2×2 个 chunk，
    其中某些 chunk 自己一个有效标签都没有，但它承载的是这个 tile 的输入像素——不读的话
    喂进网络的 patch 会缺一块，预测就变了。chunk(512) 是 patch(256) 的整数倍时这个区别
    不会显现，但别把这个巧合写死进逻辑。
    """
    valid = None
    for chart in charts:
        v = y[chart] != fill_values[chart]
        valid = v if valid is None else (valid | v)

    n_h, n_w = -(-H // patch_size), -(-W // patch_size)
    pad_h, pad_w = n_h * patch_size - H, n_w * patch_size - W
    if pad_h or pad_w:
        valid = np.pad(valid, ((0, pad_h), (0, pad_w)))
    tile_ok = valid.reshape(n_h, patch_size, n_w, patch_size).any(axis=(1, 3))

    cg_h, cg_w = -(-H // chunk), -(-W // chunk)
    need = np.zeros((cg_h, cg_w), dtype=bool)
    for i, j in np.argwhere(tile_ok):
        y0, x0 = int(i) * patch_size, int(j) * patch_size
        y1, x1 = min(y0 + patch_size, H), min(x0 + patch_size, W)
        need[y0 // chunk:(y1 - 1) // chunk + 1, x0 // chunk:(x1 - 1) // chunk + 1] = True
    return need


def _merge_hist_columns(hist, merge_map, n_new):
    """把 tile 索引的类别直方图按 merge_map 合并列。

    索引 (.npz) 里的 hist/hist_w 永远是按【原始】类别 0..5 存的（scan_tiles.py 不知道
    合并方案）。而 target_pixel_mix 的下标是【合并后】的类别。两者不对齐的话，
    _solve_tile_weights 会静默解出一个完全错误的采样分布——不会报错，只会训崩。
    """
    out = np.zeros((hist.shape[0], n_new), dtype=hist.dtype)
    for c_old in range(hist.shape[1]):
        c_new = int(merge_map.get(c_old, c_old))
        if c_new < n_new:
            out[:, c_new] += hist[:, c_old]
    return out


def merged_class_names(merge_map, n_cls):
    """合并后的类别名，仅用于日志打印。"""
    if not merge_map:
        return FLOE_CLASS_NAMES[:n_cls]
    groups = [[] for _ in range(n_cls)]
    for c_old, nm in enumerate(FLOE_CLASS_NAMES):
        c_new = int(merge_map.get(c_old, c_old))
        if c_new < n_cls:
            groups[c_new].append(nm)
    return ['+'.join(g) if g else '?' for g in groups]


def _erode_sod_boundaries(sod_patch, erosion_iters, ignore_val=255):
    """将SOD标签各类别的边界像素设为 ignore_val，只保留类别核心区域参与训练。

    CIS冰蛋图polygon与Sentinel-1 SAR存在时空错位，交界处像素标签最不可信。
    对每个类别的区域向内腐蚀 erosion_iters 像素，被腐蚀掉的边界部分设为 ignore_val。

    Parameters
    ----------
    sod_patch : ndarray, shape (H, W)
        SOD标签数组，ignore_val 表示无效像素。
    erosion_iters : int
        向内腐蚀的像素数，对应CIS对齐误差估计（建议3-7）。
    ignore_val : int
        无效像素的标记值，默认255。

    Returns
    -------
    result : ndarray, shape (H, W)
        边界像素已替换为 ignore_val 的标签数组。
    """
    result = sod_patch.copy()
    valid_mask = sod_patch != ignore_val
    for cls_id in np.unique(sod_patch[valid_mask]):
        cls_mask = (sod_patch == cls_id)
        eroded = _scipy_binary_erosion(cls_mask, iterations=erosion_iters)
        boundary = cls_mask & ~eroded
        result[boundary] = ignore_val
    return result


def _month_sin_cos(filename):
    """Extract month from scene filename and return (sin, cos) cyclic encoding.

    Filename format: S1A_EW_GRDM_1SDH_YYYYMMDDTHHMMSS_...nc
    Month digits are at filename[21:23].
    """
    month = int(filename[21:23])
    angle = 2 * math.pi * month / 12
    return math.sin(angle), math.cos(angle)


class AI4ArcticChallengeDataset(Dataset):
    """Pytorch dataset for loading batches of patches of scenes from the ASID
    V2 data set."""

    def __init__(self, options, files, do_transform=False):
        self.options = options
        self.files = files
        self.do_transform = do_transform

        # If Downscaling, down sample data and put in on memory
        if (self.options['down_sample_scale'] == 1):
            self.downsample = True
        else:
            self.downsample = True

        # cls2 filter mask support
        self._cls2_mask_dir = self.options.get('cls2_filter_mask_dir', None)

        if self.downsample:
            # 场景按需懒加载 + 全局字节预算 LRU 缓存（见 random_crop_downsample /
            # _get_or_create_cached_scene），不在此处一次性把 self.files 全部读入内存。
            # 原生分辨率(down_sample_scale=1)下单个场景可达 1~2GB，训练列表几百个场景
            # 会远超容器内存上限（2026-07-06 OOM 崩溃：66/288 场景即吃满 90GB cgroup 限额）。
            self._n_charts_init = len(self.options['charts'])
            self._cache_enabled = bool(self.options.get('enable_scene_cache', True))

        # Precompute global_valid_mask channel index for black border filtering during training
        if 'global_valid_mask' in self.options['train_variables']:
            self.global_valid_mask_idx = (len(self.options['charts']) +
                                          list(self.options['train_variables']).index('global_valid_mask'))
            # Feature-only index (into scene_feats, which doesn't include chart channels).
            self.global_valid_mask_feat_idx = list(self.options['train_variables']).index('global_valid_mask')
        else:
            self.global_valid_mask_idx = None
            self.global_valid_mask_feat_idx = None

        # Channel numbers in patches, includes reference channel.
        self.patch_c = len(
            self.options['train_variables']) + len(self.options['charts'])

        # Kept for backward compatibility; epoch-level logging is handled in quickstart.py.
        self.patch_log = []
        self._target_chart = self.options.get('target_chart', 'FLOE')
        self._target_chart_idx = self.options['charts'].index(self._target_chart)

        # 2026-07-09 采样吞吐诊断计数器（per-worker，不跨进程汇总，仅供临时诊断用）：
        # 统计每个 __getitem__ 调用里各拒绝原因的触发次数和场景解码(cache miss)次数，
        # 每 _diag_log_every 个 batch 打印一次汇总，定位"凑齐一个 batch 为什么慢"。
        self._diag_counts = {
            'attempts': 0, 'accepted': 0, 'cache_miss': 0,
            'reject_invalid_patch': 0,  # 黑边覆盖不足 或 patch 几乎全是 mask/255
            'reject_label_cov': 0, 'reject_water': 0,
            'reject_cls4': 0, 'reject_rare': 0,
        }
        self._diag_batches_done = 0
        # <=0 视为禁用打印，避免误配置成 0 时在取模处直接 ZeroDivisionError 崩训练。
        self._diag_log_every = max(0, self.options.get('diag_log_every_batches', 5))
        self._last_scene_was_cache_hit = None

        # M2 辐射域增强（radaug，2026-08-04；设计见 docs/2026-08-04_M2_radaug.md）：
        # 以概率 p 给 HH/HV 加一个 mask 内的常数偏移 b，逼模型少依赖绝对辐射水平
        # （08-04 双向因果探针：±0.7z 能把单景 medium IoU 在 0.01↔0.99 间搬动，而现行
        # 增强里一条辐射项都没有）。扰动本体在 _augment_x 开头，这里只做零成本读取。
        # data_augmentations 整键可能缺省（SOD 系 config），全程用 .get 链取。
        _da = self.options.get('data_augmentations') or {}
        self._radaug_p = float(_da.get('Radiometric_prob', 0) or 0)
        self._radaug_B = float(_da.get('Radiometric_b', 0.30))
        # M2a 对比度项（2026-08-05，V 批 V1）：a ~ U(1−A, 1+A)，围绕 patch 有效区均值做
        # 纯对比度缩放（见 _augment_x）。**默认 0 = 关闭**，此时连 a 都不抽一次随机数 ⇒
        # 现存 radaug-b config 的随机流与数据流逐位不变（smoke_m2_radaug [01] 断言）。
        self._radaug_A = float(_da.get('Radiometric_a', 0) or 0)
        # 通道间抖动幅度固定为 B/3，刻意不单设 config 键：来自第 0 步 0a 共变核对
        # （corr(HH, HV)=0.595、HH−HV 的景间 std 0.384 vs 单通道 0.427 = 部分共模）
        # ——共模分量给满 B，通道差分只注入 B/3 的少量抖动，保护 pol_ratio(HH−HV)
        # 这一路真信号不被差分噪声淹掉。
        self._radaug_jit = self._radaug_B / 3.0
        self._radaug_gen = None        # 独立 Generator，惰性构造（见 _augment_x）
        self._radaug_mask_idx = None
        if self._radaug_p > 0:
            _tv = list(self.options['train_variables'])
            if 'global_valid_mask' not in _tv:
                raise ValueError(
                    'Radiometric_prob>0 需要 train_variables 里含 global_valid_mask：'
                    '辐射扰动只作用于有效像元，黑边与几何 fill 区必须被排除在外。'
                    f'当前 train_variables={_tv}')
            # SAR 两通道按硬编码 0/1 位扰动（与 _eval_radiometric.channel_layout 同口径），
            # 顺序变了必须报错而不是静默扰动错通道（code-review P2）。
            assert _tv[0] == 'nersc_sar_primary' and _tv[1] == 'nersc_sar_secondary', \
                f'radaug 假定通道 0/1 为 nersc SAR primary/secondary，实际 {_tv[:2]}'
            # A 的取值域：负值会被 `if self._radaug_A > 0` 静默忽略（配了对比度却完全没
            # 生效、白跑一枪）；A>=1 时 a~U(1−A,1+A) 会取到 ≤0，对比度翻转/压平成常数，
            # 那不是"辐射不变性"而是在造无物理意义的样本。
            assert 0.0 <= self._radaug_A < 1.0, \
                f'Radiometric_a 必须落在 [0, 1)，得到 {self._radaug_A}'
            # x_patch 的通道空间就是 train_variables（chart 通道已在 _crop_from_tile /
            # random_crop 里切掉），所以这里【不能】用带 chart 偏移的
            # self.global_valid_mask_idx，只能用 feature-only 下标。
            self._radaug_mask_idx = _tv.index('global_valid_mask')
            # 生效状态打一行（dataset 在主进程构造，只打一次；防"键名笔误静默关闭、
            # 白跑 6 小时锚复制品"，与 [TileSampling] 诊断同风格，code-review P7）
            print(f'[Radaug] p={self._radaug_p} B={self._radaug_B} A={self._radaug_A} '
                  f'jit={self._radaug_jit:.3f} mask_ch={self._radaug_mask_idx}', flush=True)

        # Tile 采样（见模块顶部注释）。配置了 tile_index_path 才启用，否则走旧的
        # 整景解码 + 随机裁 + 概率拒绝路径（SOD 等未建索引的数据集继续可用）。
        self._tile_sampling = False
        if self.options.get('tile_index_path', None):
            self._init_tile_sampling()

        # V2 DANN 场景标签（2026-08-05）：开启时每个 patch 额外携带一个"来自哪一景"的
        # 域序号（**self.files 内的下标**，而 self.files 就是 train_options['train_list']
        # ⇒ 与 quickstart 里断言的 len(train_list) 同一套编号），经 prep_dataset 放进
        # y dict 的 '_domain' 键。关闭时不加键、不多读任何东西、零 RNG、零行为变化。
        self._dann_on = bool(self.options.get('segformer_dann', False))
        if self._dann_on and not self._tile_sampling:
            raise ValueError(
                'segformer_dann=True 目前只支持 tile 采样路径（_getitem_tile）：legacy '
                '路径会在场景级重试/放弃后换景，"这个 patch 来自哪一景"的记账没有实现。'
                '请给 config 配 tile_index_path（FLOE 一律走 tile 采样），或关掉 '
                'segformer_dann。')

    def _init_tile_sampling(self):
        opts = self.options
        ps = opts['patch_size']

        if opts['down_sample_scale'] != 1:
            raise ValueError(
                'tile 采样直接读原生分辨率的 chunk 对齐 tile，不支持 down_sample_scale != 1；'
                f'当前为 {opts["down_sample_scale"]}。要降采样请走旧路径（不要设 tile_index_path）。')

        index = np.load(opts['tile_index_path'], allow_pickle=False)
        tile_size = int(index['tile_size'])
        if tile_size < ps:
            raise ValueError(f'tile_size={tile_size} 小于 patch_size={ps}，无法在 tile 内裁 patch。')
        self._tile_size = tile_size

        # hist_w 是按 patch_size 的裁剪覆盖核加权的直方图（见 scan_tiles.crop_coverage_kernel）；
        # patch_size 变了核就变了，必须重扫，否则求解出的权重对不上实际采样分布。
        if 'hist_w' not in index.files:
            raise ValueError(
                f'{opts["tile_index_path"]} 是旧版索引（缺 hist_w）。请重跑 scan_tiles.py。')
        index_ps = int(index['patch_size'])
        if index_ps != ps:
            raise ValueError(
                f'tile 索引是按 patch_size={index_ps} 建的，但 config 的 patch_size={ps}。'
                f'裁剪覆盖核依赖 patch_size，请用 --patch-size {ps} 重跑 scan_tiles.py。')

        # 索引里的 scene_id 是相对于扫描时的全量文件表的；这里重映射到本 dataset 的
        # self.files 下标（train/val 用不同的 datalist，各自只取自己那部分 tile）。
        index_files = [str(f) for f in index['files']]
        file_to_index_id = {name: i for i, name in enumerate(index_files)}
        missing = [f for f in self.files if f not in file_to_index_id]
        if missing:
            raise ValueError(
                f'{len(missing)} 个场景不在 tile 索引里（如 {missing[:3]}）。'
                f'datalist 变了就要重跑 scan_tiles.py 生成新索引。')

        # index_id -> 本 dataset 的 scene 下标
        remap = np.full(len(index_files), -1, dtype=np.int32)
        for i, name in enumerate(self.files):
            remap[file_to_index_id[name]] = i

        scene_id = index['scene_id'].astype(np.int64)
        keep = remap[scene_id] >= 0

        cfg = opts.get('tile_sampling', {}) or {}

        # tile 级的黑边粗筛。patch 级还会再筛一次（在 tile 内存里做，零成本）：
        # 一个 valid_ratio=0.4 的 tile 里完全可能存在全 valid 的 256 patch。
        vmt = opts.get('valid_mask_threshold', 0.5)
        keep &= index['valid_ratio'] >= vmt

        # 标签有效率下限。这两个比例度量的是不同的东西，缺一不可：
        #   valid_ratio       = global_valid_mask==1 的比例，只排除 SAR 黑边；
        #   label_valid_ratio = FLOE != 255 的比例，还排除陆地和模糊多边形。
        # 陆地 tile 的 global_valid_mask 全是 1（陆地不是黑边），但标签几乎全是 255。
        # 不筛掉它们会有两个后果：(1) 从中裁出的 patch 几乎必然过不了标签覆盖检查，
        # 白白耗掉 max_crop_attempts 次重试；(2) 更隐蔽——下面的 frac 按【有效标签像元】
        # 归一化，一个只有几十个有效像元且恰好全是巨浮冰的陆地边缘 tile 会得到
        # frac[vast]=1.0，被权重求解器当成"最纯的巨浮冰 tile"而赋予极高采样概率。
        mlvr = float(cfg.get('min_label_valid_ratio', 0.5))
        keep &= index['label_valid_ratio'] >= mlvr

        if not keep.any():
            raise ValueError(f'tile 索引在 valid_ratio>={vmt} 且 label_valid_ratio>={mlvr} '
                             f'且属于本 datalist 的条件下为空。')

        hist = index['hist'][keep].astype(np.float64)      # 原始像素数，仅用于统计/报告
        hist_w = index['hist_w'][keep].astype(np.float64)  # 裁剪覆盖核加权，用于求解权重
        self._tile_scene = remap[scene_id[keep]].astype(np.int32)
        self._tile_row = index['row'][keep].astype(np.int32)
        self._tile_col = index['col'][keep].astype(np.int32)

        # n_classes 是按 chart 索引的字典（见 configs/_base_/base.py），不是整数
        n_cls = opts['n_classes'][self._target_chart]

        # 类别合并（floe_class_merge）时，索引里的直方图仍是按【原始】6 类存的，
        # 必须先合并成【新】类别空间，才能和 target_pixel_mix 的下标对齐。
        # 不做这一步不会报错——_solve_tile_weights 会照样解出一个数，但那是错的分布。
        _merge = opts.get('floe_class_merge') if self._target_chart == 'FLOE' else None
        if _merge:
            hist = _merge_hist_columns(hist, _merge, n_cls)
            hist_w = _merge_hist_columns(hist_w, _merge, n_cls)

        target = np.zeros(n_cls, dtype=np.float64)
        for cls, share in (cfg.get('target_pixel_mix') or {}).items():
            target[int(cls)] = float(share)
        if target.sum() <= 0:
            raise ValueError('tile_sampling.target_pixel_mix 未配置或全为 0。')
        # 本 datalist 中不存在的类无法被采到（2026-07-16 前 FLOE 的类 1 饼状冰全数据集
        # 为空即属此类；扩充后它有像素了，但换用旧 datalist/旧数据时仍可能触发）；若对
        # 空类误配了非零目标，乘法更新会把它的 a_c 推向无穷。这里直接剔除并告警。
        absent = (hist.sum(axis=0) == 0) & (target > 0)
        if absent.any():
            print(f'[TileSampling] 类别 {np.where(absent)[0].tolist()} 在本 datalist 中没有任何像素，'
                  f'已从 target_pixel_mix 中剔除。')
            target[absent] = 0.0
        target /= target.sum()

        weights, achieved = _solve_tile_weights(hist_w, target)
        # searchsorted 采样：cumsum 末位必须严格 >= 1.0，否则 rand() 极端取值会越界。
        self._tile_cum = np.cumsum(weights)
        self._tile_cum[-1] = 1.0

        self._patches_per_tile = max(1, cfg.get('patches_per_tile', 2))
        self._tile_sampling = True

        # 冰类从 1 起切（类 1 ice cake 自 2026-07-16 数据集扩充后不再为空；
        # 合并空间下第 1 列同样是非水类，口径不变）
        n_ice = int((hist[:, 1:].sum(axis=1) > 0).sum())
        names = merged_class_names(_merge, n_cls)
        print(f'[TileSampling] {len(self._tile_row)} tiles over {len(self.files)} scenes '
              f'(tile={tile_size}, patch={ps}, patches_per_tile={self._patches_per_tile}, '
              f'valid_ratio>={vmt}, label_valid_ratio>={mlvr}); {n_ice} tiles contain ice.')
        print('[TileSampling] expected batch pixel mix  : '
              + '  '.join(f'{n}={v:.3f}' for n, v in zip(names, achieved)))
        print('[TileSampling] requested target pixel mix: '
              + '  '.join(f'{n}={v:.3f}' for n, v in zip(names, target)))
        max_dev = float(np.abs(achieved - target).max())
        if max_dev > 0.02:
            print(f'[TileSampling][WARN] 期望分布与目标最大偏差 {max_dev:.3f}——目标可能落在 '
                  f'tile 类别分布的凸包之外（某类没有足够"纯"的 tile 把它的占比拉到目标值）。')

        # 采样集中度诊断（2026-07-16 引入，动机是 ice cake 这类"像素占比 0.3%、只在
        # 少数景出现"的稀有类）：ESS = 1/Σw² 是采样权重的有效样本数——所有 tile 等权时
        # 等于 tile 总数，权重全压在 k 个 tile 上时约等于 k。ESS 相对 tile 总数塌得越狠，
        # 说明为命中 target_pixel_mix 采样越是在反复抽同一小撮 tile（过拟合风险）。
        # 同时报每个类的"承载 tile 数 + 落在这些 tile 上的采样概率质量"：稀有类若以
        # 很少的 tile 承载了很大的权重和，就该考虑下调它在 target_pixel_mix 里的份额。
        ess = 1.0 / float((weights ** 2).sum())
        print(f'[TileSampling] effective tiles (ESS=1/Σw²): {ess:,.0f} / {len(weights)}')
        for c, nm in enumerate(names):
            has_c = hist[:, c] > 0
            if has_c.any():
                print(f'[TileSampling]   {nm:<8s} tiles={int(has_c.sum()):6d}  '
                      f'采样权重和={float(weights[has_c].sum()):.3f}')

    def __len__(self):
        return self.options['epoch_len']

    def _augment_x(self, x_patch, scene_id):
        """先施加 M2 辐射扰动（train-only，radaug），再追加 month/pol_ratio/texture 派生通道。

        顺序敏感：派生通道必须从【已扰动】的 HH/HV 算（详见下方 radaug 注释块与
        docs/2026-08-04_M2_radaug.md §2）。"""
        # ---- M2 辐射域增强（radaug）：必须在任何派生通道之前 ----
        # 插在这里而不是 transform() 末尾，一处修正三个一致性问题：pol_ratio / month /
        # texture 全部从【已扰动】的 HH/HV 派生（否则 pol_ratio 会成为未扰动的干净捷径，
        # 且 train 与 val 的通道间关系口径错配）。_augment_x 在 transform 之【前】被调用
        # （tile 路径 _getitem_tile、legacy 路径 _getitem_legacy，两条路径共用本函数，
        # 所以扰动只需在这里实现一次）。
        # 关闭时（键缺省或 Radiometric_prob=0）连 Generator 都不建、一次 RNG 都不抽 ⇒
        # 所有现存 config 的数据流与随机流逐位不变。
        if self.do_transform and self._radaug_p > 0:
            if self._radaug_gen is None:
                # 独立 Generator：与主随机流完全隔离 ⇒ radaug run 与锚 run 见到逐位相同的
                # patch 序列与几何增强，唯一差异是辐射扰动本身（E2-vs-N4 级干净对照）。
                # 种子从 torch.initial_seed() 派生（code-review P1）：worker 里它 =
                # base_seed + worker_id，base_seed 由主进程 torch RNG 每 epoch 新抽 ⇒
                # 扰动序列【逐 epoch 变化】，不会每 epoch 重放同一批 b（worker 非持久化、
                # 每 epoch 重建，常量种子会导致 200 epoch 只用 ~2400 个重复 b 值）。
                # 对照性不受影响：radaug 不消耗主流 ⇒ 锚 run 与 radaug run 的 base_seed
                # 序列逐位相同，radaug 流对给定 config 完全确定、可复现。
                # 不用 np.random：不是因为"worker 同流"（torch≥1.9 已按 worker/epoch 给
                # numpy 播种，08-04 review P3 纠正了这一误判），而是 numpy 流被 tile 选择
                # /裁剪位置消费——borrow 它会让 radaug run 与锚 run 的 patch 序列分叉。
                info = get_worker_info()
                wid = info.id if info is not None else 0
                self._radaug_gen = torch.Generator()
                self._radaug_gen.manual_seed(
                    (torch.initial_seed() * 6364136223846793005 + 7919 * wid + 17)
                    % (2 ** 63))
            g = self._radaug_gen
            # 触发判定也走独立 Generator（主随机流一位不动）
            if torch.rand(1, generator=g).item() < self._radaug_p:
                B = self._radaug_B
                b_common = (torch.rand(1, generator=g).item() * 2 - 1) * B
                # ⚠️ **抽取顺序定死为：触发判定 → b_common → a → 逐通道抖动**（改动即
                # 口径断代）。A=0 时【一次都不抽 a】，所以现存的 radaug-b config 消耗的
                # 随机数序列与本次改动前逐位相同 ⇒ 那些 run 的数据流不变。
                # M2a 对比度项：a 两通道**共享**（一景的对比度是整体性质，不是逐极化的），
                # 幅度 A=Radiometric_a。
                a = 1.0
                if self._radaug_A > 0:
                    a = 1.0 + (torch.rand(1, generator=g).item() * 2 - 1) * self._radaug_A
                # mask 门控：黑边（global_valid_mask=0）自动排除；几何 affine 的 fill 区
                # 也在 mask=0 内（fill 发生在本函数之后的 transform 里，fill=0 ⇒ 掩膜通道
                # 同步归零），不会被扰动到。
                mask = x_patch[:, self._radaug_mask_idx] > 0.5      # (1, H, W)
                for ch in (0, 1):                                   # 0=HH, 1=HV
                    b = b_common + (torch.rand(1, generator=g).item() * 2 - 1) * self._radaug_jit
                    # 原地写安全性：x_patch 在 tile 路径来自 _crop_from_tile 的
                    # np.ascontiguousarray 副本（patch_size < tile_size ⇒ 切片非连续 ⇒
                    # 必然复制），在 legacy 路径来自 .type(torch.float) 的新张量，两者
                    # 都不会回写 tile / 场景缓冲。
                    # ⚠️ 前提是 patch_size < tile_size：等尺寸时 ascontiguousarray 不复制，
                    #    原地写会污染同一 tile 的下一个 patch。smoke_m2_radaug.py 第 7 条
                    #    钉住这个前提（热路径不加逐 patch 断言）。
                    xc = x_patch[:, ch]
                    if a != 1.0:
                        # 围绕**本 patch 本通道 mask 内均值**的纯对比度：xc ← m + a·(xc−m) + b。
                        # a 不改变 patch 均值（E[a·(x−m)]=0）⇒ 与 b 这条加性轴精确解耦，
                        # 归因干净（否则 a 会顺带带来一个 (1−a)·m 的亮度偏移，与 b 混淆）。
                        # m_ch **逐 patch 逐通道现算**，不是全局统计量 —— 同一景的不同裁剪
                        # 位置 m_ch 不同，这是有意的：模型该学的是"局部对比度不影响粒度判读"。
                        # mask 全 False 时 xc[mask] 为空张量、m_ch=nan，但赋值目标同样为空 ⇒
                        # 不会有 nan 写进 patch。
                        m_ch = float(xc[mask].mean())
                        xc[mask] = m_ch + a * (xc[mask] - m_ch) + b
                    else:
                        xc[mask] = xc[mask] + b   # A=0：纯 b（radaug 首枪口径，逐位不变）
        if self.options.get('month_encoding', False):
            s, c = _month_sin_cos(self.files[scene_id])
            H, W = x_patch.shape[-2], x_patch.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x_patch = torch.cat([x_patch, month_ch], dim=1)
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x_patch[:, 0:1, :, :] - x_patch[:, 1:2, :, :]
            x_patch = torch.cat([x_patch, pol_ratio], dim=1)
        if self.options.get('texture_channels', False):
            x_patch = torch.cat([x_patch, _texture_channels(x_patch)], dim=1)
        return x_patch

    def random_crop(self, scene, scene_name=None):
        """
        Perform random cropping in scene.

        Parameters
        ----------
        scene :
            Xarray dataset; a MyDS scene with dimensions ('y', 'x').
        scene_name : str, optional
            Filename of the scene (used to locate the cls2 filter mask).

        Returns
        -------
        x_patch :
            torch array with shape (len(train_variables),
            patch_height, patch_width). None if empty patch.
        y_patch :
            torch array with shape (len(charts),
            patch_height, patch_width). None if empty patch.
        """
        patch = np.zeros((len(self.options['full_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        # Get random index to crop from.
        row_rand = np.random.randint(
            low=0, high=scene[self._target_chart].values.shape[0]
            - self.options['patch_size'])
        col_rand = np.random.randint(
            low=0, high=scene[self._target_chart].values.shape[1]
            - self.options['patch_size'])

        # Discard patches with too many black border (georrectification artifact) pixels.
        if 'global_valid_mask' in self.options['train_variables']:
            valid_mask_patch = scene['global_valid_mask'].isel(
                y=slice(row_rand, row_rand + self.options['patch_size']),
                x=slice(col_rand, col_rand + self.options['patch_size'])).values
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        target_patch = scene[self._target_chart].isel(
            y=slice(row_rand, row_rand + self.options['patch_size']),
            x=slice(col_rand, col_rand + self.options['patch_size'])).values
        if np.sum(target_patch != self.options['class_fill_values'][self._target_chart]) > 1:

            # Crop all full-resolution variables (MyDS: all variables at same 80m resolution).
            patch[0:len(self.options['full_variables']), :, :] = \
                scene[self.options['full_variables']].isel(
                y=range(row_rand, row_rand + self.options['patch_size']),
                x=range(col_rand, col_rand + self.options['patch_size'])).to_array().values

            # SOD-specific: remap labels in-place: merge old class 2 (young ice) into
            # class 1 (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if self._target_chart == 'SOD':
                sod_ch = patch[self._target_chart_idx]
                valid = sod_ch != 255
                sod_ch[valid & (sod_ch >= 2)] -= 1

            # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）
            elif self._target_chart == 'FLOE':
                remap_floe_labels(patch[self._target_chart_idx],
                                  self.options.get('floe_class_merge'))

            # Boundary erosion: mask out class-boundary pixels that are most likely
            # mislabeled due to spatial misalignment between CIS ice charts and SAR.
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0:
                patch[self._target_chart_idx] = _erode_sod_boundaries(
                    patch[self._target_chart_idx], erosion_iters=_erode_iters)

            # cls2 filter mask: on-the-fly load from full-resolution .npy mask.
            if (self._target_chart == 'SOD'
                    and self._cls2_mask_dir is not None
                    and scene_name is not None):
                _mn = os.path.basename(scene_name).replace('.nc', '_cls2_mask.npy')
                _mp = os.path.join(self._cls2_mask_dir, _mn)
                if os.path.exists(_mp):
                    _full_mask = np.load(_mp, mmap_mode='r')
                    _mask_crop = _full_mask[
                        row_rand:row_rand + self.options['patch_size'],
                        col_rand:col_rand + self.options['patch_size']
                    ]
                    patch[self._target_chart_idx][_mask_crop] = 255

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def random_crop_downsample(self, idx):
        """
        Perform random cropping on a pre-downsampled scene, fetched lazily via
        the bounded LRU cache (_get_or_create_cached_scene).

        Parameters
        ----------
        idx :
            Index from self.files to parse.

        Returns
        -------
        x_patch, y_patch :
            Torch tensors, or (None, None) if patch is invalid.
        """

        patch = np.zeros((len(self.options['full_variables']),
                          self.options['patch_size'],
                          self.options['patch_size']))

        payload, _was_hit = _get_or_create_cached_scene(
            self.options, self.files[idx], self._n_charts_init, self._cls2_mask_dir,
            enable_cache=self._cache_enabled)
        self._last_scene_was_cache_hit = _was_hit  # 供 __getitem__ 诊断计数用
        _label = payload['scene_labels']
        _feats = payload['scene_feats']
        _cls2_mask = payload['cls2_mask']
        _n_charts = len(self.options['charts'])
        ps = self.options['patch_size']

        # Get random index to crop from.
        row_rand = np.random.randint(low=0, high=_feats.shape[1] - ps)
        col_rand = np.random.randint(low=0, high=_feats.shape[2] - ps)

        # Discard patches with too many black border pixels using global_valid_mask.
        if self.global_valid_mask_feat_idx is not None:
            valid_mask_patch = _feats[
                self.global_valid_mask_feat_idx,
                row_rand: row_rand + ps,
                col_rand: col_rand + ps].numpy()
            valid_ratio = float(np.mean(valid_mask_patch))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        # Discard patches with too many label fill (masked) pixels.
        if np.sum(_label[self._target_chart_idx,
                         row_rand: row_rand + ps,
                         col_rand: col_rand + ps].numpy()
                  != self.options['class_fill_values'][self._target_chart]) > 1:

            # NumPy auto-upcasts uint8/float16 slices into the float64 `patch` buffer.
            patch[:_n_charts, :, :] = _label[:, row_rand:row_rand + ps,
                                             col_rand:col_rand + ps].numpy()
            patch[_n_charts:, :, :] = _feats[:, row_rand:row_rand + ps,
                                             col_rand:col_rand + ps].numpy()

            # SOD-specific: remap labels in-place: merge old class 2 (young ice) into
            # class 1 (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
            if self._target_chart == 'SOD':
                sod_ch = patch[self._target_chart_idx]
                valid = sod_ch != 255
                sod_ch[valid & (sod_ch >= 2)] -= 1

            # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）
            elif self._target_chart == 'FLOE':
                remap_floe_labels(patch[self._target_chart_idx],
                                  self.options.get('floe_class_merge'))

            # Boundary erosion: mask out class-boundary pixels that are most likely
            # mislabeled due to spatial misalignment between CIS ice charts and SAR.
            _erode_iters = self.options.get('boundary_erosion_iters', 0)
            if _erode_iters > 0:
                patch[self._target_chart_idx] = _erode_sod_boundaries(
                    patch[self._target_chart_idx], erosion_iters=_erode_iters)

            # cls2 filter mask: apply pre-loaded downsampled boolean mask.
            if (self._target_chart == 'SOD'
                    and _cls2_mask is not None):
                _mask_crop = _cls2_mask[
                    row_rand:row_rand + self.options['patch_size'],
                    col_rand:col_rand + self.options['patch_size']
                ]
                patch[self._target_chart_idx][_mask_crop] = 255

            x_patch = torch.from_numpy(
                patch[len(self.options['charts']):, :]).type(torch.float).unsqueeze(0)
            y_patch = torch.from_numpy(patch[:len(self.options['charts']), :, :]).unsqueeze(0)

        # In case patch does not contain any valid pixels - return None.
        else:
            x_patch = None
            y_patch = None

        return x_patch, y_patch

    def _read_tile(self, tile_i):
        """读一个 chunk 对齐的 tile。返回 (len(full_variables), T, T) 的 float32 数组。

        tile 的 row/col 都是 tile_size 的整数倍且 tile_size == HDF5 chunk 边长，所以每个
        变量恰好命中 1 个 chunk，不跨 chunk 解压——这是 27ms/tile 的来源。任何"抖动"读取
        （原点不对齐）都会变成 2×2 个 chunk，慢约 1.7 倍。
        """
        scene_id = int(self._tile_scene[tile_i])
        r = int(self._tile_row[tile_i])
        c = int(self._tile_col[tile_i])
        T = self._tile_size

        path = os.path.join(self.options['path_to_train_data'], self.files[scene_id])
        handle = _get_h5_handle(path)

        full_variables = self.options['full_variables']
        tile = np.empty((len(full_variables), T, T), dtype=np.float32)
        for j, var in enumerate(full_variables):
            tile[j] = handle[var][r:r + T, c:c + T]
        return tile, scene_id

    def _crop_from_tile(self, tile):
        """在 tile 内随机裁一个 patch_size 的 patch。返回 (x_patch, y_patch) 或 (None, None)。

        裁剪原点限制在 [0, T-ps]，即 patch 完全落在对齐 tile 内部（不跨 chunk）。代价是
        跨 tile 边界的浮冰在单个 patch 里只能看到一半——但相邻 tile 会从另一侧补上，
        且浮冰通常横跨多个 tile，随机采样下边界区域仍被覆盖。
        """
        ps = self.options['patch_size']
        T = self._tile_size
        n_charts = len(self.options['charts'])

        row_rand = np.random.randint(low=0, high=T - ps + 1)
        col_rand = np.random.randint(low=0, high=T - ps + 1)
        # 全程 float32：tile 本身就是 float32，标签值 0..255 在 float32 里精确可表示。
        # 旧路径用 float64 缓冲（np.zeros 的默认 dtype），到这里会逼出一次
        # torch `.type(torch.float)` 的 float64→float32 拷贝——在 torch 线程超订的容器上
        # 实测单次 3.3 秒（见 limit_torch_threads 注释），是首版 tile 采样慢 400ms/patch 的主因。
        patch = np.ascontiguousarray(tile[:, row_rand:row_rand + ps, col_rand:col_rand + ps])

        # 黑边过滤（tile 级已粗筛过 valid_ratio，这里是 patch 级精筛，纯内存操作）
        if self.global_valid_mask_idx is not None:
            valid_ratio = float(np.mean(patch[self.global_valid_mask_idx]))
            if valid_ratio < self.options.get('valid_mask_threshold', 0.5):
                return None, None

        # 标签覆盖：patch 里至少要有 2 个能参与 loss 的像元
        target_ch = patch[self._target_chart_idx]
        if np.sum(target_ch != self.options['class_fill_values'][self._target_chart]) <= 1:
            return None, None

        # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）。
        # 这是 tile 采样的训练路径——FLOE 实际走的就是这条。
        if self._target_chart == 'FLOE':
            remap_floe_labels(target_ch, self.options.get('floe_class_merge'))

        _erode_iters = self.options.get('boundary_erosion_iters', 0)
        if _erode_iters > 0:
            patch[self._target_chart_idx] = _erode_sod_boundaries(
                patch[self._target_chart_idx], erosion_iters=_erode_iters)

        # patch 已经是 float32，from_numpy 直接零拷贝共享内存，不再需要 .type() 转换。
        x_patch = torch.from_numpy(patch[n_charts:]).unsqueeze(0)
        y_patch = torch.from_numpy(patch[:n_charts]).unsqueeze(0)
        return x_patch, y_patch

    def _getitem_tile(self):
        """Tile 采样路径下组装一个 batch。

        与旧路径的本质区别：tile 已按类别直方图加权采过一次，所以不需要任何概率拒绝
        （water_rejection / rare_samplers / cls4_rejection 在本路径下全部不生效——目标
        分布由 tile_sampling.target_pixel_mix 精确指定）。剩下的 patch 级检查只有黑边和
        标签覆盖两项，都是纯内存操作，失败时在同一个已读入的 tile 里换个位置重裁即可。
        """
        _n_channels = (len(self.options['train_variables'])
                       + (2 if self.options.get('month_encoding', False) else 0)
                       + (1 if self.options.get('pol_ratio_channel', False) else 0)
                       + (N_TEXTURE_CHANNELS if self.options.get('texture_channels', False) else 0))
        batch_size = self.options['batch_size']
        ps = self.options['patch_size']
        x_patches = torch.zeros((batch_size, _n_channels, ps, ps))
        y_patches = torch.zeros((batch_size, len(self.options['charts']), ps, ps))
        # V2 DANN：逐 patch 的域序号（= self.files 下标）。关闭时为 None ⇒ 不分配、
        # 不进 y dict，行为与改动前完全一致。
        domains = torch.zeros(batch_size, dtype=torch.long) if self._dann_on else None

        max_crop_attempts = max(1, self.options.get('max_crop_attempts_per_scene', 20))
        sample_n = 0

        while sample_n < batch_size:
            tile_i = int(np.searchsorted(self._tile_cum, np.random.rand()))
            tile_i = min(tile_i, len(self._tile_cum) - 1)
            try:
                tile, scene_id = self._read_tile(tile_i)
            except Exception as e:
                print(f'[TileSampling] 读取 tile {tile_i} '
                      f'({self.files[int(self._tile_scene[tile_i])]}) 失败: {e}. 换一个 tile。',
                      flush=True)
                continue
            self._diag_counts['cache_miss'] += 1  # 这里等价于"读了一次 tile"

            for _ in range(self._patches_per_tile):
                if sample_n >= batch_size:
                    break
                for _attempt in range(max_crop_attempts):
                    self._diag_counts['attempts'] += 1
                    x_patch, y_patch = self._crop_from_tile(tile)
                    if x_patch is None:
                        self._diag_counts['reject_invalid_patch'] += 1
                        continue
                    x_patch = self._augment_x(x_patch, scene_id)
                    if self.do_transform:
                        x_patch, y_patch = self.transform(x_patch, y_patch)
                    x_patches[sample_n] = x_patch
                    y_patches[sample_n] = y_patch
                    if domains is not None:
                        domains[sample_n] = scene_id
                    sample_n += 1
                    self._diag_counts['accepted'] += 1
                    break

        return x_patches, y_patches, domains

    def prep_dataset(self, x_patches, y_patches, domains=None):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        x_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] containing only the trainable variables.
        y_patches : ndarray
            Patches sampled from ASID3 ready-to-train challenge dataset scenes [PATCH, CHANNEL, H, W] contrainng only the targets.
        domains : LongTensor [PATCH] 或 None
            V2 DANN 的逐 patch 域序号（self.files 下标）。None（默认）时 y dict 里
            **不出现** '_domain' 键 —— 所有现存 config 走的就是这条路，行为逐位不变。

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """

        # Convert training data to tensor float.
        x = x_patches.type(torch.float)

        # Store charts in y dictionary.

        y = {}
        for idx, chart in enumerate(self.options['charts']):
            y[chart] = y_patches[:, idx].type(torch.long)

        # 下划线前缀的保留键：训练循环按 charts 遍历 y，'_domain' 不在 charts 里 ⇒
        # 不会被当成一个 chart 参与 CE/指标，只由 DANN 分支显式取用（同 '_supcon' 契约）。
        if domains is not None:
            y['_domain'] = domains.long()

        return x, y
    
    def transform(self, x_patch, y_patch):
        data_aug_options = self.options['data_augmentations']
        if torch.rand(1) < data_aug_options['Random_h_flip']:
            x_patch = TF.hflip(x_patch)
            y_patch = TF.hflip(y_patch)

        if torch.rand(1) < data_aug_options['Random_v_flip']:
            x_patch = TF.vflip(x_patch)
            y_patch = TF.vflip(y_patch)

        # E1 尺度保真增强（2026-07-19）：无损 90° 旋转。torch.rot90 无重采样、无
        # fill 边界，与 h/v flip 组合覆盖完整 D4 对称群。默认 0 = 关闭，旧 config
        # 行为不变。patch 是方形（256），rot90 不改变形状。
        if torch.rand(1) < data_aug_options.get('Random_rot90_prob', 0):
            k = int(np.random.randint(1, 4))   # {1,2,3} → 90/180/270°
            x_patch = torch.rot90(x_patch, k, dims=[-2, -1])
            y_patch = torch.rot90(y_patch, k, dims=[-2, -1])

        assert (data_aug_options['Random_rotation'] <= 180)
        if data_aug_options['Random_rotation'] != 0 and \
                torch.rand(1) < data_aug_options['Random_rotation_prob']:
            random_degree = np.random.randint(-data_aug_options['Random_rotation'],
                                                data_aug_options['Random_rotation']
                                                )
        else:
            random_degree = 0

        scale_diff = data_aug_options['Random_scale'][1] - \
            data_aug_options['Random_scale'][0]
        assert (scale_diff >= 0)
        if scale_diff != 0 and torch.rand(1) < data_aug_options['Random_scale_prob']:
            random_scale = np.random.rand()*(data_aug_options['Random_scale'][1] -
                                                data_aug_options['Random_scale'][0]) +\
                data_aug_options['Random_scale'][0]
        else:
            # random_scale = data_aug_options['Random_scale'][1]
            random_scale = 1.0 

        # 恒等时跳过 affine（纯性能优化：默认 NEAREST 插值下恒等 affine 本就无损，
        # 跳过不改变数值，只省一次逐 patch 的重采样调用）。
        if random_degree != 0 or random_scale != 1.0:
            x_patch = TF.affine(x_patch, angle=random_degree, translate=(0, 0),
                                shear=0, scale=random_scale, fill=0)
            y_patch = TF.affine(y_patch, angle=random_degree, translate=(0, 0),
                                shear=0, scale=random_scale, fill=255)

        return x_patch, y_patch

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y : Dict
            Dictionary with 3D torch tensors for each chart; reference data for training data x.
        """
        limit_torch_threads()

        if self._tile_sampling:
            x_patches, y_patches, domains = self._getitem_tile()
        else:
            x_patches, y_patches, domains = self._getitem_legacy()

        if self.do_transform and torch.rand(1) < self.options['data_augmentations']['Cutmix_prob']:
            lam = np.random.beta(self.options['data_augmentations']['Cutmix_beta'],
                                  self.options['data_augmentations']['Cutmix_beta'])
            rand_index = torch.randperm(x_patches.size(0))
            bbx1, bby1, bbx2, bby2 = rand_bbox(x_patches.size(), lam)
            x_patches[:, :, bbx1:bbx2, bby1:bby2] = x_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            y_patches[:, :, bbx1:bbx2, bby1:bby2] = y_patches[rand_index, :, bbx1:bbx2, bby1:bby2]
            # ⚠️ **刻意不混 domains**：cutmix 后一个 patch 里有两个景的像素，但域标签保持
            # 原 patch 的。混入区因此是判别器的标签噪声（占比 = cutmix 触发率 × 混入面积
            # 比），这是接受的：DANN 的判别器本来就只吃 GAP 后的**整 patch**全局向量，
            # 没有"哪一块属于谁"的表达；按面积给软标签才是正解，但那要改损失形态、
            # 不值得在首枪引入。注记见 config 头。
        x, y = self.prep_dataset(x_patches, y_patches, domains)

        # 采样吞吐诊断：每 _diag_log_every 个 batch 打印一次汇总并清零（临时诊断用，
        # 2026-07-09 排查首个 batch 组装过慢问题；不跨 worker 汇总，每个 worker 独立打印）。
        # tile 采样路径下 cache_miss 计的是"读入的 tile 数"，reject_* 只剩黑边/标签覆盖两项。
        self._diag_batches_done += 1
        if self._diag_log_every > 0 and self._diag_batches_done % self._diag_log_every == 0:
            c = self._diag_counts
            worker_info = get_worker_info()
            wid = worker_info.id if worker_info is not None else 'main'
            print(f"[SamplingDiag worker={wid}] last {self._diag_log_every} batches: "
                  f"attempts={c['attempts']} accepted={c['accepted']} cache_miss={c['cache_miss']} | "
                  f"reject: invalid_patch={c['reject_invalid_patch']} label_cov={c['reject_label_cov']} "
                  f"water={c['reject_water']} cls4={c['reject_cls4']} rare={c['reject_rare']}", flush=True)
            for k in c:
                c[k] = 0

        return x, y

    def _getitem_legacy(self):
        """旧路径：随机选场景 → 整景解码(或命中 LRU 缓存) → 随机裁 → 多层概率拒绝重试。

        仅在未配置 tile_index_path 时使用（SOD 等尚未建 tile 索引的数据集）。FLOE 走
        _getitem_tile：整景解码约 9.2s/景，是这条路径无法在合理时间内跑完 200 epoch 的根因。
        """
        _n_channels = (len(self.options['train_variables'])
                       + (2 if self.options.get('month_encoding', False) else 0)
                       + (1 if self.options.get('pol_ratio_channel', False) else 0)
                       + (N_TEXTURE_CHANNELS if self.options.get('texture_channels', False) else 0))
        x_patches = torch.zeros((self.options['batch_size'], _n_channels,
                                 self.options['patch_size'], self.options['patch_size']))
        y_patches = torch.zeros((self.options['batch_size'], len(self.options['charts']),
                                 self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # 2026-07-09 吞吐教训（两轮复盘）：
        # 第一轮——原逻辑每次候选 patch 被任何过滤器拒绝，都会重新随机选一个全新场景，
        # 而 rare_samplers/water_rejection 等概率过滤器接受率本就不高，几乎每次重试都要
        # 在原生分辨率(down_sample_scale=1)下重新解码一个新场景(~9~10s)。已修：同一场景
        # 解码/取缓存一次后，换不同裁剪位置重试最多 max_crop_attempts_per_scene 次。
        # 第二轮——上面这个修复只解决了"每个 patch 内部的重试"，但**每凑够 1 个有效
        # patch 后仍立刻换新场景**，一个 batch(16 patch) 仍需要 ~16 次解码，吞吐没有
        # 本质改善（16 次解码 ÷ 3 worker × ~9.5s ≈ 每 batch 50+ 秒，200 epoch 跑不完）。
        # 现改为：同一场景一次性产出 patches_per_scene 个 patch（默认 6 个，裁剪位置各自
        # 随机、允许空间相邻）后才换新场景，把解码成本分摊到多个 patch 上，
        # 每 batch 所需解码次数从 ~16 降到 ~16/patches_per_scene。
        # 下界保护：两个参数配成 0/负数都会导致内层循环一次不执行，sample_n 永不增长、
        # 外层 while 无限空转且不打印任何诊断信息（静默卡死，极难排查）。
        max_crop_attempts = max(1, self.options.get('max_crop_attempts_per_scene', 20))
        patches_per_scene = max(1, self.options.get('patches_per_scene', 6))

        while sample_n < self.options['batch_size']:
            scene_id = np.random.randint(low=0, high=len(self.files), size=1).item()
            scene = None
            for _patch_idx in range(patches_per_scene):
                if sample_n >= self.options['batch_size']:
                    break
                got_patch = False  # 本次(同场景内)是否成功产出一个 patch
                fatal_error = False  # 场景解码/裁剪本身出错，整个场景作废
                for _attempt in range(max_crop_attempts):
                    self._diag_counts['attempts'] += 1
                    # 实时心跳（不等凑够 5 个 batch 才打印）：每 20 次尝试打印一次，用于
                    # 在第一个 batch 组装缓慢时也能实时看到进度，而不是完全没有输出。
                    if self._diag_counts['attempts'] % 20 == 0:
                        worker_info = get_worker_info()
                        wid = worker_info.id if worker_info is not None else 'main'
                        print(f"[SamplingDiag worker={wid}] heartbeat: sample_n={sample_n}/"
                              f"{self.options['batch_size']} attempts={self._diag_counts['attempts']} "
                              f"cache_miss={self._diag_counts['cache_miss']}", flush=True)
                    try:
                        if self.downsample:
                            self._last_scene_was_cache_hit = None
                            x_patch, y_patch = self.random_crop_downsample(scene_id)
                            if self._last_scene_was_cache_hit is False:
                                self._diag_counts['cache_miss'] += 1
                        else:
                            if scene is None:
                                scene = xr.open_dataset(os.path.join(
                                    self.options['path_to_train_data'], self.files[scene_id]),
                                    engine='h5netcdf', mask_and_scale=False)
                            x_patch, y_patch = self.random_crop(scene, scene_name=self.files[scene_id])
                    except FileNotFoundError:
                        print(f"File {self.files[scene_id]} not found. Skipping scene.")
                        fatal_error = True
                        break
                    except Exception:
                        if self.downsample:
                            print(f"Cropping in {self.files[scene_id]} failed. Skipping.")
                        else:
                            print(f"Cropping in {self.files[scene_id]} failed. "
                                  f"Scene size: {scene[self._target_chart].shape if scene is not None else '?'}. Skipping.")
                        fatal_error = True
                        break

                    if x_patch is None:
                        self._diag_counts['reject_invalid_patch'] += 1
                        continue  # 同一场景换个裁剪位置重试

                    x_patch = self._augment_x(x_patch, scene_id)
                    target_flat = y_patch[0, self._target_chart_idx].numpy().flatten()

                    # Filter 1 — Label coverage.
                    sod_invalid_max = self.options.get('sod_invalid_max_ratio', 1.0)
                    if sod_invalid_max < 1.0:
                        if float((target_flat == 255).sum()) / len(target_flat) > sod_invalid_max:
                            self._diag_counts['reject_label_cov'] += 1
                            continue

                    # Filter 2 — Water patch rejection.
                    water_max_ratio = self.options.get('water_patch_max_ratio', 1.0)
                    water_reject_prob = self.options.get('water_rejection_prob', 0.0)
                    if water_reject_prob > 0.0 and water_max_ratio < 1.0:
                        valid_mask = target_flat != 255
                        valid_count = valid_mask.sum()
                        if valid_count > 0:
                            water_ratio = float((target_flat[valid_mask] == 0).sum()) / valid_count
                            if water_ratio > water_max_ratio and np.random.rand() < water_reject_prob:
                                self._diag_counts['reject_water'] += 1
                                continue

                    # Filter 2b — cls4 (MYI) patch rejection (mirrors water rejection).
                    cls4_max_ratio = self.options.get('cls4_patch_max_ratio', 1.0)
                    cls4_reject_prob = self.options.get('cls4_rejection_prob', 0.0)
                    if cls4_reject_prob > 0.0 and cls4_max_ratio < 1.0:
                        valid_mask = target_flat != 255
                        valid_count = valid_mask.sum()
                        if valid_count > 0:
                            cls4_ratio = float((target_flat[valid_mask] == 4).sum()) / valid_count
                            if cls4_ratio > cls4_max_ratio and np.random.rand() < cls4_reject_prob:
                                self._diag_counts['reject_cls4'] += 1
                                continue

                    # Filter 3 — Rare-class weighted sampling.
                    # Supports two config styles (both can coexist; rare_samplers takes precedence):
                    #   Legacy:  rare_sampling_classes / rare_sampling_alpha  (single sampler)
                    #   New:     rare_samplers = [{'classes': [...], 'alpha': float}, ...]
                    #
                    # When multiple samplers are configured, acceptance uses OR-max logic:
                    #   p_combined = max(p_1, p_2, ...)
                    # Each sampler independently scores the patch; the most favorable score wins.
                    # This allows per-class tuning without classes interfering with each other.
                    _samplers = self.options.get('rare_samplers', None)
                    if _samplers is None:
                        # Fall back to legacy single-sampler config.
                        _cls = self.options.get('rare_sampling_classes', [])
                        _alpha = self.options.get('rare_sampling_alpha', 0.0)
                        _samplers = [{'classes': _cls, 'alpha': _alpha}] if _cls and _alpha > 0.0 else []
                    if _samplers:
                        _valid_target = target_flat[target_flat != 255]
                        if len(_valid_target) > 0:
                            _accept_prob = 0.0
                            for _s in _samplers:
                                _alpha_s = _s.get('alpha', 0.0)
                                _classes_s = _s.get('classes', [])
                                if not _classes_s or _alpha_s <= 0.0:
                                    continue
                                _rare_count = float(sum((_valid_target == c).sum() for c in _classes_s))
                                _rare_frac = _rare_count / len(_valid_target)
                                _p = (1.0 - _alpha_s) + _alpha_s * _rare_frac
                                if _p > _accept_prob:
                                    _accept_prob = _p
                            if _accept_prob > 0.0 and np.random.rand() > _accept_prob:
                                self._diag_counts['reject_rare'] += 1
                                continue

                    if self.do_transform:
                        x_patch, y_patch = self.transform(x_patch, y_patch)

                    x_patches[sample_n, :, :, :] = x_patch
                    y_patches[sample_n, :, :, :] = y_patch
                    sample_n += 1
                    self._diag_counts['accepted'] += 1
                    got_patch = True
                    break  # 跳出 attempt 循环，尝试从同一场景再产出下一个 patch

                if fatal_error or not got_patch:
                    # 场景解码出错，或该场景在 max_crop_attempts 次尝试内一次都没通过
                    # 过滤器——放弃这个场景，换新场景（不再在这个场景上浪费更多尝试）。
                    break

            if scene is not None:
                scene.close()

        # 第三个返回值是 DANN 的逐 patch 域序号；legacy 路径不支持（__init__ 里已 raise），
        # 恒为 None，只为与 _getitem_tile 保持同一个调用约定。
        return x_patches, y_patches, None

    def save_patch_log(self, save_path, epoch):
        """
        Append patch sampling statistics to a CSV file, then clear the log.

        Mode is controlled by train_options['patch_log_mode']:
          'per_epoch' (default): one row per epoch, columns are total pixel counts
                                 summed across ALL patches in that epoch.
          'per_patch'           : one row per patch, columns are pixel counts for
                                 that individual patch (also records the source file).

        NOTE: requires num_workers=0 to function correctly.

        Parameters
        ----------
        save_path : str
            Path to the CSV file (created on first call, appended thereafter).
        epoch : int
            Current epoch number (written into every row).
        """
        mode = self.options.get('patch_log_mode', 'per_epoch')
        prefix = self._target_chart.lower()
        n_classes = self.options['n_classes'][self._target_chart]
        real_classes = list(range(n_classes))

        if mode == 'per_epoch':
            fieldnames = ['epoch'] + [f'{prefix}_{c}' for c in real_classes] + [f'{prefix}_mask']
        else:  # per_patch
            fieldnames = ['epoch', 'file'] + [f'{prefix}_{c}' for c in real_classes] + [f'{prefix}_mask']

        write_header = not osp.exists(save_path)
        with open(save_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()

            if mode == 'per_epoch':
                # Aggregate pixel counts across all patches in this epoch into one row.
                totals = {c: 0 for c in real_classes}
                totals[255] = 0
                for entry in self.patch_log:
                    for c in real_classes:
                        totals[c] += entry['sod_dist'].get(c, 0)
                    totals[255] += entry['sod_dist'].get(255, 0)
                row = {'epoch': epoch}
                for c in real_classes:
                    row[f'{prefix}_{c}'] = totals[c]
                row[f'{prefix}_mask'] = totals[255]
                writer.writerow(row)
            else:
                # One row per patch.
                for entry in self.patch_log:
                    row = {'epoch': epoch, 'file': entry['file']}
                    for c in real_classes:
                        row[f'{prefix}_{c}'] = entry['sod_dist'].get(c, 0)
                    row[f'{prefix}_mask'] = entry['sod_dist'].get(255, 0)
                    writer.writerow(row)

        self.patch_log.clear()


class AI4ArcticChallengeTestDataset(Dataset):
    """Pytorch dataset for loading full scenes from the ASID ready-to-train challenge dataset for inference."""

    def __init__(self, options, files, files_reference=None, mode='test',
                 ondemand_read=False):
        self.options = options
        self.files = files
        self.files_reference = files_reference
        # 按需读是【显式 opt-in】的，默认关：这个 dataset 有近十个消费者（TTA / 可视化 /
        # 推理导出 / 历史脚本），其中大多数不满足「只在对齐的 tile 上前向」这个前提，
        # 默认打开会静默污染它们。安全性判定见 val_ondemand_ok()。
        self.ondemand_read = ondemand_read

        # if mode not in ["train_val", "test_val", "test"]:
        if mode not in ["train", "test", "test_no_gt"]:
            raise ValueError("String variable must be one of 'train', 'test', or 'test_no_gt'")
        self.mode = mode

    def __len__(self):
        """
        Provide the number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene, scene_y=None, filename=None):
        """
        Load a full scene for inference/validation. For MyDS, all variables are at
        the same 80m resolution, so no upsampling is needed.

        Parameters
        ----------
        scene :
            Xarray dataset for input features.
        scene_y :
            Optional separate dataset for labels (targets). If None, uses scene.

        Returns
        -------
        x :
            4D torch tensor, ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        """
        # All MyDS variables are at the same 80m resolution - load all at once.
        x = torch.from_numpy(
            scene[self.options['train_variables']].to_array().values).unsqueeze(0).float()

        # Downscale if needed
        # 验证/测试时可用 val_downsample_scale 单独控制分辨率，避免大场景 GPU OOM
        if self.mode in ('train', 'test', 'test_no_gt') and self.options.get('val_downsample_scale', 1) != 1:
            effective_scale = self.options['val_downsample_scale']
        else:
            effective_scale = self.options['down_sample_scale']

        if effective_scale != 1:
            x = torch.nn.functional.interpolate(
                x, scale_factor=1/effective_scale,
                mode=self.options['loader_downsampling'])

        if self.mode != 'test_no_gt':
            scene_for_y = scene_y if scene_y is not None else scene
            y_charts = torch.from_numpy(
                scene_for_y[self.options['charts']].isel().to_array().values).unsqueeze(0)
            if effective_scale != 1:
                y_charts = torch.nn.functional.interpolate(
                    y_charts, scale_factor=1/effective_scale, mode='nearest')

            y = {}
            for idx, chart in enumerate(self.options['charts']):
                y[chart] = y_charts[:, idx].squeeze().numpy()

            y = self._finalize_labels(y, filename, effective_scale)
        else:
            y = None

        # Append sin/cos month encoding as 2 constant spatial channels.
        if self.options.get('month_encoding', False) and filename is not None:
            s, c = _month_sin_cos(filename)
            H, W = x.shape[-2], x.shape[-1]
            month_ch = torch.tensor([s, c], dtype=torch.float).view(1, 2, 1, 1).expand(1, 2, H, W)
            x = torch.cat([x, month_ch], dim=1)

        # Append HH-HV polarization ratio channel (dB difference).
        # nersc_sar_primary (HH) is channel 0, nersc_sar_secondary (HV) is channel 1.
        if self.options.get('pol_ratio_channel', False):
            pol_ratio = x[:, 0:1, :, :] - x[:, 1:2, :, :]
            x = torch.cat([x, pol_ratio], dim=1)

        # 纹理通道（B1）【不在这里加】：验证走 per-tile——在 functions.fast_tiled_val_predict
        # 里对每个 256 tile 单独算纹理（与训练 _augment_x 的 256 patch 同口径），既避免整景
        # 多 4 通道的内存峰值、又保住按需读，且消除 train/val 边缘不一致。整景 x 保持 8 通道。
        return x.float(), y

    def _finalize_labels(self, y, filename, effective_scale):
        """标签后处理：类别重映射 / 边界腐蚀 / cls2 污染掩膜。

        ⚠️ 整景读取与「按需读取」两条路径都必须走这里，否则两条路的打分口径会漂移。
        """
        target_chart = self.options.get('target_chart', 'FLOE')

        # SOD-specific: remap labels: merge old class 2 (young ice) into class 1
        # (new/young ice). All old labels >=2 (excluding mask=255) shift down by 1.
        if target_chart == 'SOD' and 'SOD' in y:
            sod = y['SOD'].copy()
            valid = sod != 255
            sod[valid & (sod >= 2)] -= 1
            y['SOD'] = sod

        # FLOE: 冰山(6)→255，并按 floe_class_merge 合并类别（见 remap_floe_labels）。
        # 这是【验证】路径：必须与上面的训练路径用同一个 merge_map，否则训练学的是
        # 合并类、验证却按原始类打分，指标会完全错乱。
        elif target_chart == 'FLOE' and 'FLOE' in y:
            floe = y['FLOE'].copy()
            remap_floe_labels(floe, self.options.get('floe_class_merge'))
            y['FLOE'] = floe

        # Boundary erosion on validation labels: mask out class-boundary pixels that
        # are most likely mislabeled due to spatial misalignment between CIS ice
        # charts and SAR. Controlled by 'boundary_erosion_iters' (default 0 = off).
        _erode_iters = self.options.get('boundary_erosion_iters', 0)
        if _erode_iters > 0 and target_chart in y:
            y[target_chart] = _erode_sod_boundaries(
                y[target_chart], erosion_iters=_erode_iters)

        # cls2 filter mask: apply pre-generated pollution mask to validation labels.
        # Polluted cls2 pixels are set to 255 (ignore), consistent with training.
        _cls2_mask_dir = self.options.get('cls2_filter_mask_dir', None)
        if (_cls2_mask_dir is not None
                and target_chart == 'SOD' and 'SOD' in y
                and filename is not None):
            _mn = os.path.basename(filename).replace('.nc', '_cls2_mask.npy')
            _mp = os.path.join(_cls2_mask_dir, _mn)
            if os.path.exists(_mp):
                _full_mask = np.load(_mp)           # bool (H_full, W_full)
                if effective_scale != 1:
                    _mask_t = torch.from_numpy(_full_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    _h_ds, _w_ds = y['SOD'].shape[0], y['SOD'].shape[1]
                    _mask_ds = torch.nn.functional.interpolate(
                        _mask_t, size=(_h_ds, _w_ds), mode='nearest'
                    ).squeeze().numpy() > 0.5
                else:
                    _mask_ds = _full_mask
                y['SOD'][_mask_ds] = 255
        return y

    # ------------------------------------------------------------------
    # 按需读取（2026-07-25）
    # ------------------------------------------------------------------
    def _ondemand_enabled(self):
        """本次 __getitem__ 是否走「只读含有效标签的 chunk」。

        调用方必须显式传 `ondemand_read=True`（判定用 val_ondemand_ok），这里只做与
        【单个样本】有关的兜底检查。

        实测（_val_loader_timing.py）：val_417 一轮纯数据加载 463s，而带模型的完整验证
        466s —— 原生分辨率下 GPU 前向已经完全藏在加载后面，**加载是唯一瓶颈**。而
        val_417 只有 38.6% 的 chunk 含有效标签（_val_tile_gate_audit.py），其余 chunk
        解压出来的像素全部会被 `~cfv_masks` 丢掉、也被 tile 门控挡在网络之外，纯属白解。
        实测逐 chunk 按需读 5.41s/景 vs 整景 10.12s/景（_val_read_bench.py）。

        以下情形自动退回整景读取：
          - 调用方没有 opt-in（ondemand_read=False，默认）；
          - 没有标签（test_no_gt）：无从判断哪些 chunk 需要；
          - 降采样（down_sample_scale / val_downsample_scale != 1）：整景 interpolate
            需要全部像素，按需读会污染插值结果；
          - config 显式关掉 val_ondemand_read（全局 kill switch）。
        """
        return (self.ondemand_read
                and self.mode in ('train', 'test')
                and self.options.get('val_ondemand_read', True)
                and self.options.get('down_sample_scale', 1) == 1
                and self.options.get('val_downsample_scale', 1) == 1)

    def prep_scene_ondemand(self, path, filename=None):
        """用 h5py 只读「含有效标签的 chunk」，其余位置留 0。

        返回值与 prep_scene 完全同构：(x (1,C,H,W) float32, y dict)。未读区域的 x 是 0，
        这些像素对应的 tile 会被 functions.compute_valid_tile_mask 挡掉、根本不进网络，
        指标不受影响（等价性由 _val_ondemand_check.py 与 evaluate_scenes 的 CM 对拍验证）。

        读【整景】的例外：
          - 标签变量：要拿它算「哪些 chunk 需要」，且只是 uint8，很便宜；
          - `global_valid_mask`：`tfv_mask` 由它推出（__getitem__ 末尾），按需读会在未读
            区域凭空多出「无效」，与整景路径产生差异。**按变量名强制整读，不靠 dtype
            大小去猜**——将来它若从 uint8 换成别的 dtype，靠 itemsize 判断会静默出错。
          - 其余单字节变量：反正便宜，一并整读。

        ⚠️ 内存要点：所有通道（含 month/pol_ratio 派生通道）一次性分配在同一个数组里、
        且派生通道只在必读 chunk 上计算。**不要改回 `torch.cat`**：cat 会新分配一个满
        尺寸张量并写满每一页，最大景约 8GB 全部变成实打实的 anon，「未读的页不占物理
        内存」这个前提当场失效——按需读省下的内存会被它吃掉一大半。
        """
        charts = self.options['charts']
        variables = list(self.options['train_variables'])
        patch_size = self.options['patch_size']
        use_month = bool(self.options.get('month_encoding', False)) and filename is not None
        use_pol = bool(self.options.get('pol_ratio_channel', False))

        with h5py.File(path, 'r', rdcc_nbytes=64 * 1024 ** 2, rdcc_nslots=4001) as f:
            # ---- 1. 标签（整景）+ 与整景路径完全相同的后处理 ----
            y = {chart: f[chart][:] for chart in charts}
            y = self._finalize_labels(y, filename, effective_scale=1)

            # ---- 2. 由最终标签算出需要哪些 chunk ----
            first = f[variables[0]]
            H, W = first.shape[-2], first.shape[-1]
            chunk = (first.chunks[-1] if first.chunks else patch_size)
            need = _need_chunk_grid(y, self.options['class_fill_values'], charts,
                                    H, W, patch_size, chunk)

            # ---- 3. 一次性分配所有通道（顺序与 prep_scene 尾部一致：变量、month、pol）----
            n_var = len(variables)
            c_month = n_var if use_month else None
            c_pol = (n_var + (2 if use_month else 0)) if use_pol else None
            n_ch = n_var + (2 if use_month else 0) + (1 if use_pol else 0)
            # dtype 固定 float32：prep_scene 尾部也是 .float()。用 result_type 的话，任一
            # 变量是 float64 就会让整景数组翻倍、最后再被 .float() 拷一遍。
            x_np = np.zeros((n_ch, H, W), dtype=np.float32)

            dsets = [f[v] for v in variables]
            full_read = [(v == 'global_valid_mask' or d.dtype.itemsize == 1)
                         for v, d in zip(variables, dsets)]
            for k, d in enumerate(dsets):
                if full_read[k]:
                    x_np[k] = d[:]

            month_s, month_c = _month_sin_cos(filename) if use_month else (0.0, 0.0)
            for ci, cj in np.argwhere(need):
                y0, x0 = int(ci) * chunk, int(cj) * chunk
                y1, x1 = min(y0 + chunk, H), min(x0 + chunk, W)
                for k, d in enumerate(dsets):
                    if not full_read[k]:
                        x_np[k, y0:y1, x0:x1] = d[y0:y1, x0:x1]
                # 派生通道只在必读 chunk 上算：写满整景会毁掉稀疏性（见 docstring）
                if c_month is not None:
                    x_np[c_month, y0:y1, x0:x1] = month_s
                    x_np[c_month + 1, y0:y1, x0:x1] = month_c
                if c_pol is not None:
                    x_np[c_pol, y0:y1, x0:x1] = (x_np[0, y0:y1, x0:x1]
                                                 - x_np[1, y0:y1, x0:x1])

        return torch.from_numpy(x_np).unsqueeze(0), y

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            Dict with 2D numpy arrays for each chart. None if mode is 'test_no_gt'.
        cfv_masks :
            Dict of boolean masks for class fill values (label mask). None if 'test_no_gt'.
        tfv_mask :
            2D boolean mask for train fill value (input invalid region / black border).
        name : str
            Filename of scene.
        original_size : tuple
            (H, W) of the scene before any downsampling.
        """
        root = (self.options['path_to_train_data'] if self.mode == 'train'
                else self.options['path_to_test_data'])
        path = os.path.join(root, self.files[idx])

        if self._ondemand_enabled():
            # 只读「含有效标签的 chunk」，其余留 0（见 prep_scene_ondemand）。
            x, y = self.prep_scene_ondemand(path, filename=self.files[idx])
            with h5py.File(path, 'r') as _f:
                original_size = tuple(int(v) for v in _f['nersc_sar_primary'].shape)
        else:
            scene = xr.open_dataset(path, engine='h5netcdf', mask_and_scale=False)
            # For MyDS, labels are embedded in the same file as features.
            scene_y = scene if self.mode == 'test' else None
            x, y = self.prep_scene(scene, scene_y, filename=self.files[idx])
            original_size = scene['nersc_sar_primary'].values.shape
        name = self.files[idx]

        if self.mode != 'test_no_gt':
            cfv_masks = {}
            for chart in self.options['charts']:
                cfv_masks[chart] = (
                    y[chart] == self.options['class_fill_values'][chart]).squeeze()
        else:
            cfv_masks = None

        # Use global_valid_mask as the train fill value mask (more accurate than checking SAR=0).
        if 'global_valid_mask' in self.options['train_variables']:
            mask_idx = list(self.options['train_variables']).index('global_valid_mask')
            tfv_mask = (x.squeeze()[mask_idx, :, :] == 0).squeeze()
        else:
            tfv_mask = (x.squeeze()[0, :, :] == self.options['train_fill_value']).squeeze()

        return x, y, cfv_masks, tfv_mask, name, original_size


def val_scoped_options(train_options: dict):
    """返回【验证/评估侧】应当使用的 options（charts 坍缩为 val_charts = [target_chart]）。

    辅助任务（config 里的 `aux_charts`，见 get_variable_options）是 **train-only** 的：
    训练 loss 与训练 dataloader 带上它们，但验证、评估、探针、推理一律只跑 target chart。
    理由见 get_variable_options 的注释（辅任务标签几乎处处有效 → tile 门控跳过率坍缩、
    按需读失效、val loss 口径断代；且 val 景未必有辅任务变量，读了会 KeyError）。

    ⚠️ **没配 aux_charts 时返回 train_options 【本身】**（不是副本）：调用方拿到的是同一个
    对象，没有任何新分支被触发，行为逐位不变。

    所有"拿 checkpoint 在 val 上做前向"的脚本都应当在 `get_variable_options` 之后立刻
    过一遍本函数（evaluate_scenes / _eval_* / _viz_pred / inference / smoke_sota）。
    """
    val_charts = train_options.get('val_charts', train_options['charts'])
    if val_charts == train_options['charts']:
        return train_options            # 单任务：同一对象，零行为变化
    val_options = dict(train_options)   # 浅拷贝足够：只覆盖两个顶层键
    val_options['charts'] = val_charts
    val_options['task_weights'] = [1] * len(val_charts)
    return val_options


def get_variable_options(train_options: dict):
    """
    Set up variable category lists for MyDS dataset.

    In MyDS all input variables share the same 80m pixel spacing as the SAR imagery,
    so no upsampling is required. All train_variables are treated as full-resolution.

    Parameters
    ----------
    train_options: dict
        Dictionary with training options.

    Returns
    -------
    train_options: dict
        Updated with variable category lists.
    """
    # Derive active charts and task weights from target_chart.
    # Set 'target_chart' in config to 'SOD' or 'FLOE' to switch the detection target.
    target_chart = train_options.get('target_chart', 'FLOE')
    # 辅助任务（多任务学习）是【只在训练期】生效的：'aux_charts' 里的 chart 会进
    # train_options['charts']（loss 与 dataloader 会带上它们的标签），但验证/评估一律
    # 只跑 target chart（见 'val_charts'）。理由：
    #   ① SIC 这类辅任务几乎处处有标签，进了验证会让 tile 门控的跳过率从 66% 坍缩到 ~0，
    #      验证耗时约 ×3（按需读也随之失效）；
    #   ② 验证 loss 会变成"target + 辅任务"的加权和，与历史所有 run 的 val loss 口径断代。
    # 未配置 aux_charts 时 charts/task_weights 与历史逐位相同，val_charts == charts。
    aux_charts = [c for c in train_options.get('aux_charts', []) if c != target_chart]
    train_options['charts'] = [target_chart] + aux_charts
    aux_w = train_options.get('aux_task_weights', [0.2] * len(aux_charts))
    if len(aux_w) != len(aux_charts):
        raise ValueError(
            f"aux_task_weights 长度 {len(aux_w)} != aux_charts 长度 {len(aux_charts)}"
            f"（aux_charts={aux_charts}，已剔除与 target_chart 重复的项；"
            f"aux_task_weights={list(aux_w)}）")
    train_options['task_weights'] = [1] + [float(w) for w in aux_w]
    # 验证恒为 target 单任务（消费入口见上面的 val_scoped_options）。
    train_options['val_charts'] = [target_chart]
    if aux_charts:   # 只在真的配了辅任务时打印，不给现有 config 增加任何输出
        print(f"[Charts] train={train_options['charts']} "
              f"task_weights={train_options['task_weights']} | "
              f"val={train_options['val_charts']}（辅任务 train-only）")

    # Update chart_metric weights: 1 for target, 0 for others.
    for chart in train_options.get('chart_metric', {}):
        train_options['chart_metric'][chart]['weight'] = 1 if chart == target_chart else 0

    # All train variables in MyDS are full-resolution (80m) - no separate AMSR/env grids.
    train_options['sar_variables'] = list(train_options['train_variables'])
    train_options['full_variables'] = list(train_options['charts']) + train_options['sar_variables']
    train_options['amsrenv_variables'] = []
    train_options['auxiliary_variables'] = []

    return train_options
