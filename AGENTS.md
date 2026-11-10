# FLOE-Git 工作约定

- 当前范围：Baseline、PGSG、单次训练评估、论文，以及用户新增授权的 preprocessing 数据处理/标注源码；不包含五折实验流程。preprocessing 排除 SAR 下载、HH/标签绘图和 NC 统计脚本。
- Windows 本地只编辑和静态检查，不运行 Python。训练、评估、推理及 Python 冒烟检查在服务器 `ice-binary` 环境运行。
- 服务器连接信息须实际核实，不复用历史文档的过时端口。同步用 scp；本次本地目录整理不授权服务器部署、开关机或训练。
- 数据仍在外部目录，配置默认 NC 为 `/root/autodl-tmp/FLOE_datanc/`，tile 索引为 `/root/autodl-tmp/floe_tile_index_417.npz`。
- 保持 6 类，ignore index 255；输入实际为 8 通道，极化差分由标准化后的 HH/HV 相减。修改物理解释前核准归一化统计。
- 使用原生处理网格，训练/验证分辨率一致。训练走 chunk 对齐 tile 采样，推理走滑窗。
- 按需读取只能显式用于满足网格对齐前提的验证路径；整景可视化、偏移/翻转 TTA 不可随意启用门控。
- 新 Python 入口及时限制 torch 线程；运行前检查 GPU、cgroup CPU/内存及磁盘。验证内存同时看 anon、shmem、current。
- 训练放 tmux；日志写持久盘；不使用 tee 的退出码判断训练成败；同卡训练期间不另起 GPU 任务。
- checkpoint 不上传 wandb；只记标量和小文件。
- 以 macro mIoU、逐类 IoU 和 hard-ice（small/medium IoU 均值）为主要比较；区分 macro-F1、weighted-F1、OA。
- 历史论文 fold-0 指标不得标成新 375/42 配置成绩。目录中没有独立测试集，不能声称无偏独立测试表现。
- 不改写历史日志和论文结果；修复路径与变更模型科学定义分开处理。
- 保留 `floe/models/legacy/unet.py`、`floe/models/legacy/swin_transformer.py` 及 `floe/models/sota` 的导入依赖，移除前先修导入并验证。

- preprocessing 使用直接脚本入口（python preprocessing/run_pipeline.py），运行前读该目录 README；只整理源码不代表获准执行下载、SNAP、标注或归一化。
