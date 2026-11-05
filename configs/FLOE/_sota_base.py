# FLOE-SOTA 对比实验的【共享训练协议】base config。
#
# 目的：让 configs/FLOE/sota_*.py 里每个模型只改 model_selection 一行，其余训练超参
# 全部从这里继承，从而保证多模型 benchmark 的【唯一变量是网络架构】——否则 val mIoU
# 的横比没有意义。本文件本身不含 model_selection，不能直接跑，只作为 _base_ 被继承。
#
# 协议逐项对齐 configs/FLOE/pa2_swin.py（当前冠军），关键点：
#   - tile 采样（tile_index_path + tile_sampling），不再整景解码；
#   - down_sample_scale=1（原生分辨率，小/中浮冰仅 1~6px，勿下采样）；
#   - epochs=200 / epoch_len=300 / val_freq=2 / early_stop_patience=20；
#   - AdamW lr=1e-4 wd=0.05 + CosineAnnealingLR；
#   - 反不平衡完全由 tile_sampling.target_pixel_mix 负责，无概率拒绝。
#
# 数据路径、charts=['FLOE']、target_chart='FLOE'、task_weights、chart_loss（FLOE
# CrossEntropyLoss ignore_index=255）均继承自 ../_base_/base.py，此处不重复。
#
# ⚠️ 输入通道口径：这里设 pol_ratio_channel=True → 8 通道（7 个 train_variables +
# HH−HV 极化比），与 PA²-Swin 一致，使 SOTA-vs-PA² 在输入上公平。所有 sota_* 模型都
# 正确按 len(train_variables)+pol_ratio 推导 in_channels，tile 训练路径和验证路径都会
# 应用该通道（loaders._augment_x）。注意：已跑过的 floe_unet_tile 基线是 7 通道（plain
# UNet 首层卷积写死 len(train_variables)），与这批 8 通道 SOTA 不完全同口径——它是基线
# 参考点，不是 SOTA 竞争者，报告时须注明。

_base_ = ['../_base_/base.py']

train_options = {
    # ---- 数据划分（与 pa2_swin / baseline 一致）----
    # 2026-07-17 换 417 景新划分：train 375 景 / val 42 景（数据目录 417 景与 datalist
    # 双向完全对齐）。上一代为 2026-07-16 的 396 景（train.json/val.json，356/40）。
    # train 冰区构成：cake 4.8%(5019万px, 55 景，较 396 时代的 3.0%/27 景近乎翻倍)、
    # small 23.9%、medium 23.3%、big 23.4%、vast 24.6%。
    # val 有效像元：water 57.5%、cake 0.39%(448万px, 6 景)、small 1.65%、medium 2.46%、
    # big 12.2%(5 景)、vast 25.8%(4 景)；冰区内 cake 0.91%/small 3.9%/medium 5.8%/
    # big 28.7%/vast 60.7%——val 冰区仍被 Big/Vast 主导（89.4%），逐类 IoU/F1 不受
    # 影响，像素加权的总体指标会偏向这两类。
    # ⚠️ 口径断代：val 集又换了，此后的 val mIoU 与 396 景时代及更早的数字不可直接
    # 横比（旧 checkpoint 用 evaluate_scenes.py --val-list datalists/val_417.json 重评）。
    'train_list_path': 'datalists/train_417.json',
    'val_path': 'datalists/val_417.json',
    # 暂无独立测试集：test_path 仅在启动时被读入、训练循环不使用 test_list，占位即可。
    'test_path': 'datalists/val_417.json',

    # ---- 输入通道（见文件顶部说明）----
    'pol_ratio_channel': True,

    # ---- 训练规模 ----
    'seed': 42,
    'epochs': 200,
    'epoch_len': 300,
    'patch_size': 256,
    'batch_size': 16,
    'down_sample_scale': 1,

    # ---- tile 采样（详见 configs/FLOE/pa2_swin.py 与 loaders.py 顶部说明）----
    # 换数据目录（新增/删除场景）后必须重跑 scan_tiles.py；换 patch_size 也要用
    # --patch-size 重扫（loaders 启动时会校验并报错）。
    # 2026-07-17 数据集扩充至 417 景后重扫：
    #   python -m tools.data.scan_tiles --data-dir /root/autodl-tmp/FLOE_datanc \
    #       --out /root/autodl-tmp/floe_tile_index_417.npz --patch-size 256 --workers 16
    # 旧索引（floe_tile_index.npz / _368.npz / _20260716.npz 等）保留勿覆盖，复现旧 run 用。
    'tile_index_path': '/root/autodl-tmp/floe_tile_index_417.npz',
    'tile_sampling': dict(
        # 2026-07-16 起类 1（饼状冰 ice cake）不再为空，纳入目标分布。
        # cake 给 0.10 而非与其他冰类均分：396 时代它仅占有效像元 0.32%/27 景（0.17 意味着
        # ~53 倍过采样、采样质量坍缩到极少数 tile），0.10 是 ~31 倍的折中。417 景后 cake
        # 占冰区 4.8%/55 景（有效像元约 0.5%），0.10 的过采样倍率降到 ~20 倍、承载 tile
        # 数近乎翻倍，保持 0.10 更稳；预检 ESS/承载 tile 数确认后再考虑是否上调。
        # 首跑必须核对启动日志：expected batch pixel mix（偏差 >0.02 会 WARN，说明目标
        # 超出 tile 分布凸包）和 effective tiles（采样权重的有效样本数，过小=极端集中），
        # 再决定 cake 份额上/下调。其余四冰类均分剩余份额（各 0.1875），水保持 0.15。
        target_pixel_mix={0: 0.15, 1: 0.10, 2: 0.1875, 3: 0.1875, 4: 0.1875, 5: 0.1875},
        patches_per_tile=2,
        min_label_valid_ratio=0.5,
    ),
    'enable_scene_cache': False,

    # ---- 验证 / 早停 ----
    'val_batch_size': 32,   # 每批上 GPU 的 tile 数（与场景分辨率无关；验证走增量混淆矩阵，
                            # val mIoU 与批大小【无关】，只影响验证峰值显存 → 改此值不影响结果、
                            # 不破坏跨 run 可比）。2026-07-23 从 64 降到 32 作保险：run① 在验证期
                            # 被外部 SIGKILL(rc=137、容器 oom_kill=0) 疑因主机内存压力，降峰值兜底。
    'val_freq': 2,          # 原生分辨率验证较慢，每 2 个 epoch 验证一次
    'early_stop_patience': 20,   # 单位是"验证次数"；val_freq=2 下 = 40 epoch 无提升才停
                                 # （2026-07-11 从 30 收紧到 20：前两次 run 都在早停前 60+
                                 # epoch 就到平台期，30 太松、浪费验证时间）
    # 2026-07-21 起标准协议（E6，docs/2026-07-20_seed_variance.md §7）：best 选择与早停
    # 均用最近 3 次验证 mIoU 的滑动均值。per-epoch val mIoU 噪声 std~0.02，单 epoch max
    # 选 best 会挑到幸运尖峰、且早尖峰触发早停提前杀 run（seed 复跑实测 run 间方差 0.033，
    # 平滑后坍缩到 0.007）。⚠️此后 run 与旧 max 口径（含幸运膨胀、如 b5 的 0.5521）不可
    # 直接横比；旧 checkpoint 只存了 max 选中那一个 epoch，无法追溯平滑。
    'val_smoothing_window': 3,

    'compile_model': False,      # 首跑先关，确认稳定后再考虑开

    # ---- 优化器 / 调度器 ----
    'optimizer': {
        'type': 'AdamW',
        'lr': 1e-4,
        'b1': 0.9,
        'b2': 0.999,
        'eps': 1e-8,
        'weight_decay': 0.05,
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'lr_min': 1e-6,
    },

    # ---- DataLoader（容器实际 25 核；瓶颈是 gzip 解压，CPU-bound）----
    'num_workers': 12,
    # 2026-07-23：验证期 num_workers_val 个预取 worker 各解码一整景原生分辨率场景（最大
    # 6369×23975）是验证阶段内存峰值来源。两次训练都在【验证中途】被宿主机级 SIGKILL
    # (rc=137、容器 oom_kill=0)，疑因验证峰值内存使本容器成为宿主机 OOM 目标 → 4→1 砍预取
    # 内存兜底（验证会慢些但不影响结果；训练侧 num_workers=12 不变，训练阶段从未崩）。
    #
    # 2026-07-25：`val_ondemand_read`（见下）把每景的解码量降到 38.6%，验证期内存峰值随之
    # 下降，预取可以适度提回来。实测（evaluate_scenes 跑满 42 景 b5，_memmon.sh 每 2s 采样）：
    #     配置                          耗时    anon   shmem   current(总)
    #     改动前 整景读+无门控 w=1        542s   23.7    18.6      34.4 GB
    #     按需读+门控 w=1                278s   10.0    18.3      25.2 GB
    #     按需读+门控 w=2                183s   13.0    20.2      26.7 GB   ← 采用
    #     按需读+门控 w=3                159s   15.0    23.9      33.5 GB
    # 取 w=2：比 w=1 快 34% 只多 1.5GB，比改动前省 7.7GB；w=3 只再快 13% 却要多花 6.8GB，
    # 把按需读省下的内存全花回去、退回改动前的水平，不划算（07-23 被宿主机 SIGKILL 杀过
    # 两次，一次损失整个 run）。若验证期再出 rc=137，调回 1 仍比改动前快 1.95 倍。
    #
    # ⚠️ 看内存【不能只看 anon】：多 worker 的 DataLoader payload（一个样本就是一整景）
    # 走 /dev/shm，在 cgroup v2 里记进 shmem 而非 anon。上表 w=3 的 anon 只有 15.0GB 但
    # shmem 有 23.9GB——只盯 anon 会把"提 worker"误判成免费。CLAUDE.md 里"只看 anon"那条
    # 是【训练】路径（tile 采样，payload 几 MB）的经验，不适用于验证。用 _memmon.sh 采全。
    'num_workers_val': 2,

    # ---- 验证提速的两个开关（2026-07-25；464s→142s，详见 CLAUDE.md「关键设计约定」）----
    # ① tile 门控：只前向「标签非全 255」的 tile（val_417 上可跳 66.1%）。
    #    设 False 可一键回退做 A/B，此时只剩「不开整景 logits 缓冲」的收益。
    'val_tile_gate': True,
    # ② 按需读：只解码「含有效 tile」的 chunk（val_417 只需 38.6%），其余留 0；那些位置
    #    对应的 tile 会被 ① 挡在网络之外。已用 _val_ondemand_check.py（标签/掩膜逐位相同、
    #    所有有效 tile 的输入像素逐位相同）+ evaluate_scenes 全量 CM 对拍双重验证。
    #    ⚠️ 这个键只是【全局 kill switch】：按需读实际是调用方显式 opt-in 的
    #    （AI4ArcticChallengeTestDataset(..., ondemand_read=)，判定见 loaders.val_ondemand_ok），
    #    因为同一个 dataset 还被 TTA / 可视化 / 推理导出等脚本使用，它们不满足「只在对齐的
    #    tile 上前向」这个前提，默认打开会静默污染。滑窗 stride 不等于 patch_size、集成、
    #    降采样、缺 global_valid_mask 时也会自动退回整景读并打印原因。
    'val_ondemand_read': True,
    # ③ 逐批 GPU 混淆矩阵（2026-08-06，第二轮提速）：quickstart 验证的 fast 路径走
    #    fast_tiled_val_predict(out='confmat')——每批 argmax 后 index_add_ 就地累加 CM、
    #    loss 在 GPU 上 float64 累加，整景只同步一次；去掉逐批 .item()/.cpu() 同步、
    #    整景 uint8 预测图、CPU 掩膜筛选与 ~11.6 亿有效像素/轮的 H2D。CM 与旧路径
    #    【整数级完全相等】（同批组成 ⇒ logits 逐位相同，smoke_sota [7c]/[7d] 断言）。
    #    设 False 一键回退旧 preds+掩膜筛选路径做 A/B；与 ①② 相互独立。
    'val_confmat_batched': True,

    # ⚠️ 验证变快了，但【不要】顺手改 val_freq / early_stop_patience / val_smoothing_window：
    # early_stop_patience 的单位是「验证次数」，val_smoothing_window 平滑的也是「验证次数」，
    # 改 val_freq 会同时改变早停的实际 epoch 跨度和平滑窗覆盖的时间跨度，破坏与 07-21 定案的
    # 平滑协议 run（smsel s42 0.5222 / s123 0.5296）的可比性。要改必须当作新的口径断代声明。
    'prefetch_factor': 4,

    # ---- 指标 ----
    'compute_classwise_f1score': True,
    'plot_confusion_matrix': True,

    # ---- patch 级过滤（tile 采样下只剩黑边 + 标签覆盖两项纯内存检查）----
    'max_crop_attempts_per_scene': 20,
    'diag_log_every_batches': 5,

    # ---- 数据增强（几何 + cutmix，与 pa2_swin 一致）----
    'data_augmentations': {
        'Random_h_flip': 0.5,
        'Random_v_flip': 0.5,
        'Random_rotation_prob': 0.5,
        'Random_rotation': 90,
        'Random_scale_prob': 0.5,
        'Random_scale': (0.9, 1.1),
        'Cutmix_beta': 1.0,
        'Cutmix_prob': 0.5,
    },

    # 注：swin_hp 仅当 model_selection=='swin' 时被 batched_slide_inference 读取；
    # sota_* 模型走 fast_tiled_val_inference，不需要它。若以后加纯 'swin' config，
    # 在那个 config 里单独补 swin_hp。
}
