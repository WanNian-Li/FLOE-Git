# 2026-07-09 用 tile 采样同时解决 FLOE 的 OOM 与吞吐问题

## 结论先行

放弃"整景解码 + 随机裁 + 概率拒绝"，改为"chunk 对齐 tile 加权采样 + 就地裁 patch"。

| | 旧路径 | 新路径 |
|---|---|---|
| 单次数据读取 | 整景 9.2s，输出 2.8~3.1GB float32 | 一个 512 tile 27~53ms，8.4MB |
| 每 patch 摊销成本 | ~1.53s（缓存命中率约 1%） | 34.5ms（单进程实测） |
| worker 常驻内存 | 数 GB 缓存 + 解码瞬时峰值 | 一个 tile 8.4MB + 索引 1.2MB |
| 额外磁盘 | 400GB+（放不下） | 252KB 索引 |
| 采样接受率 | 冷启动时第一个 batch 8 分钟凑不齐 | ~99%（81 次尝试接受 80 次） |
| 训练分布 | 多层概率拒绝叠出来的黑箱 | 精确指定，实测吻合到小数点后三位 |

`epoch_len=300` 下单进程 165s/epoch。数据侧不再是瓶颈。

## 问题的重新定义

之前的所有努力（字节预算 LRU、`num_workers` 12→6→3、`patches_per_scene`、
`down_sample_scale` 1↔2、离线预处理成 `.npz`）都在给一个 400GB 的工作集找容身之处。
但 `scan_tiles.py` 对全部 360 景的实测显示，这个工作集里绝大部分内容根本不需要：

- **255（陆地 + SAR 黑边 + 模糊多边形）占全图 69.70%**，在 loss 里被 `ignore_index=255`
  屏蔽，梯度贡献严格为 0；
- 有效像元里 **89.34% 是开阔水**（与 `DATASET_HANDBOOK.md` §5 的 89.67% 吻合）；
- 冰只占全图 **3.24%**。

按 512×512 对齐切 tile（111,148 个）：**56.4% 是全无效**（没有任何一个像元能参与 loss），
20.7% 是纯水，**只有 7.8% 含冰**。

旧路径的顺序是：花 9.2 秒解码整景（1.5 亿像素）→ 随机裁一个 256×256 窗口（占整景 0.04%）
→ 用概率过滤器决定要不要扔掉它。**先付昂贵的代价，再判断该不该付。**

## 关键使能条件：`.nc` 本来就是 chunked 的

```
FLOE                  (6046, 14406)  uint8    chunks=(512,512)  gzip-4
global_valid_mask     (6046, 14406)  uint8    chunks=(512,512)  gzip-4
nersc_sar_primary     (6046, 14406)  float32  chunks=(512,512)  gzip-4
...（全部 8 个变量同布局）
```

`h5py` 的 `dset[r:r+512, c:c+512]`（r/c 为 512 的整数倍）只解压 1 个 chunk。
9.2 秒是 `xarray.to_array()` 把全部变量的全部像素读进来自找的，不是 NetCDF 的固有成本。

## 实现

1. **`scan_tiles.py`**（新增，约 30 秒扫完 360 景）：每景只读 `FLOE` + `global_valid_mask`
   两个 uint8 通道，按 512 对齐网格枚举 tile，剔除全无效 tile，为剩下的记录
   `valid_ratio` / `label_valid_ratio` / 六类直方图。输出 `floe_tile_index.npz`（252KB）。

2. **`loaders.py`**：
   - `_get_h5_handle`：按进程缓存的 HDF5 句柄 LRU（`rdcc_nbytes=64MB`，够放下一个 tile 的
     8 个 chunk）。只在 worker 里首次调用时打开，父进程从不打开——HDF5 在 fork 之后使用
     继承来的句柄是未定义行为。
   - `_solve_tile_weights`：乘法迭代（RAS/IPF 式）求解 tile 采样权重。
   - `_read_tile` / `_crop_from_tile` / `_getitem_tile`：新的采样路径。
   - 旧路径保留为 `_getitem_legacy`（SOD 等未建索引的数据集仍可用）。

3. **`configs/FLOE/{baseline,pa2_swin}.py`**：新增 `tile_index_path` 与 `tile_sampling`，
   删除 `water_rejection_prob` / `rare_samplers` / `cls4_rejection_prob` /
   `patches_per_scene` / `preprocessed_data_dir` / `scene_cache_max_bytes`。

## 为什么采样权重必须对"像素"而不是"tile"求解

四个冰类的**像素总量**基本相当（各占全图约 0.8%），但**tile 覆盖数**差了三倍：

| | small | medium | big | vast |
|---|---|---|---|---|
| 含该类的 tile 数 | 3386 | 3173 | 1138 | 1224 |

大/巨浮冰是大块连续的，小/中浮冰是散布的。按 tile 均匀加权会系统性压低 big/vast 的
像素份额。`_solve_tile_weights` 直接对像素分布做定点迭代，绕开这个不一致：

```
w_i ∝ Σ_c a_c · frac[i,c]        # frac[i,c] = tile i 内类别 c 占有效标签像元的比例
a_c ← a_c · (target_c / p_c)^η   # p = Σ_i w_i · frac[i]，即当前期望像素分布
```

实测 `expected = water 0.150 / small 0.212 / medium 0.212 / big 0.212 / vast 0.212`，
与 target 一致。

## 期望分布必须按像素加权，而且要算进裁剪覆盖核（2026-07-10 补）

首版 `_solve_tile_weights` 用 `p = Σ_i w_i · frac[i]`（各 tile 归一化类别占比的加权平均）
当作"期望像素分布"。这是错的：真正落到 loss 上的是

```
p_c = (Σ_i w_i · hist[i,c]) / (Σ_i Σ_c' w_i · hist[i,c'])
```

两者只在所有 tile 的有效像元数相等时才一致。而 tile 的有效像元数差异很大（`label_valid_ratio`
均值 0.738），于是求解器自以为命中了 target，实测分布却偏出 ±0.02。

第二层偏差来自 tile 内裁 patch：patch 原点均匀取自 `[0, 256)²`，tile 中心的像素被覆盖到的
概率高于边缘（一维是梯形核，二维取外积）。不把这个核算进直方图，big/vast 会被再高估约 0.013。

两处一起修：`scan_tiles.py` 用 `crop_coverage_kernel(512, 256)` 给直方图加权，额外存一份
`hist_w`；`_solve_tile_weights` 改成接收 `hist_w` 并按像素加权计算 `p`。索引里同时存
`patch_size`，`loaders.py` 启动时校验——**换 patch_size 必须重扫索引**。

实测一个完整 epoch（4800 patch）的像素分布：

| | 修复前 | 修复后 | target |
|---|---|---|---|
| water | 0.1662 (+0.0162) | **0.1497 (−0.0003)** | 0.1500 |
| small | 0.1977 (−0.0148) | 0.2054 (−0.0071) | 0.2125 |
| medium | 0.2039 (−0.0086) | 0.2097 (−0.0028) | 0.2125 |
| big | 0.2201 (+0.0076) | 0.2231 (+0.0106) | 0.2125 |
| vast | 0.2121 (−0.0004) | 0.2121 (−0.0004) | 0.2125 |

残余的 big `+0.0106` 约 1.5σ（4800 patch ≈ 2400 个 tile，逐 tile 类别占比的标准误约 0.007），
在采样噪声量级内。

## 三个踩过的坑

**1. torch 按宿主机核数开线程池（最贵的一个）**

容器 `nproc` 报 208 核，cgroup `cpu.max` 只给 25 核。torch 按 208 建 intraop 线程池，
严重超订。实测一次 `7×256×256` 的 float64→float32 cast 要 **3.3 秒**（正常微秒级），
numpy 的 `astype` 被自旋线程挤慢 170 倍，一个 batch 的组装 **0.55s → 205s**。

修复：`loaders.limit_torch_threads()`——worker 设 1 线程，主进程设 `min(8, cpu_quota())`，
其中 `cpu_quota()` 读 cgroup 而不是 `nproc`。已在 `quickstart.main()` 开头调用。
**任何新入口脚本都要调它一次。**

顺带把 `_crop_from_tile` 的 patch 缓冲从 float64（旧代码 `np.zeros` 的默认 dtype）
改成 float32，直接消掉那次 cast——tile 本来就是 float32，标签值 0..255 在 float32 里精确。

**2. 陆地 tile 会毒化采样权重**

`valid_ratio`（`global_valid_mask==1` 的比例）只排除 SAR 黑边。**陆地的
`global_valid_mask` 是 1**，但标签是 255。只筛 `valid_ratio` 的话，陆地 tile 会进候选集：

- 从中裁出的 patch 几乎必然过不了标签覆盖检查，白白耗掉 20 次重试
  （实测某陆地 tile：20 次裁剪只接受 2 次）；
- 更隐蔽的是，`frac` 按**有效标签像元**归一化，一个只有几十个有效像元、恰好全是巨浮冰的
  陆地边缘 tile 会得到 `frac[vast]=1.0`，被求解器当成"最纯的巨浮冰 tile"而赋予极高权重。

修复：索引里同时存 `label_valid_ratio`，config 用 `min_label_valid_ratio=0.5` 筛掉。
筛完 34361 → 28356 个 tile，接受率从 2/20 变成 20/20。

**3. `n_classes` 是按 chart 索引的字典，不是整数**

见 `configs/_base_/base.py`。写 `opts['n_classes']` 会拿到一个 dict。

## 验证状态（2026-07-10 在 RTX 5090 / 25 核 / 90GB 实例上全部跑通）

- ✅ 权重求解：期望分布命中 target（水 0.150，四冰类各 0.212~0.213），train/val 两个 datalist 都是
- ✅ **`num_workers=12` 吞吐：1.9 ms/patch，0.030 s/batch，9.0 s/epoch**（`epoch_len=300`）；
  200 epoch 的数据加载总成本约 0.5 小时，GPU 成为唯一瓶颈
- ✅ 内存：加载期 anon 峰值 **2.7 GB**（上限 90 GB）。OOM 问题消失
- ✅ 实际采样分布（4800 patch = 1 epoch）：最大偏差 0.011，water 精确到 0.0003
- ✅ 采样接受率 ~99%（81 次尝试接受 80 次），`invalid_patch` 是唯一的拒绝原因
- ✅ DataLoader fork + h5py 正确性：shape/dtype/有限性/标签值全对，无死锁
- ✅ 首个 batch 0.46s（含 12 个 worker 启动），冷启动问题消失

对照旧路径的实测：整景解码 9.2s/景、每 patch 摊 ~1.53s、冷启动第一个 batch 8 分钟凑不齐。
**每 patch 快约 800 倍。**

## 顺带确认

`quickstart.py` 已有 `GradScaler`（:111）和 `clip_grad_norm_`（:185），2026-07-06 那次
`floe_pa2_swin` 在 epoch 51 的 NaN 坍缩（AMP autocast 无 GradScaler / 无梯度裁剪）已修复，
重启训练不会踩同一颗雷。
