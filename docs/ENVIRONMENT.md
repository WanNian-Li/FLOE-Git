# 环境与验证边界

本次未连接服务器，未导出或安装依赖，未运行 Python。没有伪造 requirements 锁定版本。

源码直接使用的主要库包括 PyTorch、torchvision、torchmetrics、segmentation-models-pytorch、timm、mmcv、mmengine、numpy、scipy、h5py、xarray、h5netcdf、matplotlib、tqdm、wandb、icecream；部分可视化使用 seaborn，计算量统计可选 fvcore/thop。此列表不是经安装验证的完整锁定清单。

实际版本需后续在服务器 `ice-binary` 环境核实导出；不可盲装最新版本覆盖原环境。ImageNet 编码器首次构造可能需要缓存权重或联网下载。

迁移检查范围：文件存在性、配置继承路径、运行默认值、数据名单 JSON、源文件与目标 SHA256；静态检查不等于 Python 语法/导入或 GPU 运行验证。

下一次服务器验证应先核实实例及资源，再执行 `tools/checks/smoke_pgsg.py`、`tools/checks/smoke_pipeline.py`，确认数据、索引、环境与输入通道。未验证前不标记“训练可运行”。
