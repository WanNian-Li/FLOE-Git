# FLOE-Git

Sentinel-1 EW 区域主导浮冰尺寸语义分割项目。保留 SegFormer-B2 Baseline、PGSG、单次训练/评估、论文诊断及正式论文；另含原始 SAR 处理与标注源码，不包含五折实验流程。

## 目录

```text
FLOE-Git/
├── floe/                  公共代码与模型
│   ├── functions.py       训练、模型构建与指标公共函数
│   ├── loaders.py         数据读取与采样
│   ├── evaluation.py      测试、导出与可视化辅助
│   └── models/            PGSG/SegFormer 及必要兼容模型
├── tools/
│   ├── train.py           训练入口
│   ├── evaluate.py        逐景评估入口
│   ├── infer.py           推理入口
│   ├── data/              tile 索引
│   ├── analysis/          指标分析与论文诊断
│   ├── figures/           论文制图
│   ├── checks/            冒烟与一致性检查
│   └── system/            资源监控
├── preprocessing/         SAR 处理、冰图标注与归一化（见该目录 README）
├── configs/               非五折配置
├── datalists/             375/42 场景名单
├── docs/                  数据手册、结果与整理记录
└── paper/                 论文源码和正式插图
```

根目录只保留 `README.md`、`AGENTS.md`、`.gitignore`。`tools/` 下的 Python 工具从项目根目录以 `python -m tools.…` 执行，不直接用 `python tools/…py`；无需修改 `PYTHONPATH`。

## 当前入口

| 配置 | 用途 |
|---|---|
| `configs/FLOE/sota_segformer_in1k_dlr_417.py` | Baseline |
| `configs/FLOE/pgsg_evidence_gate.py` | 证据分支＋竞争门控 |
| `configs/FLOE/pgsg.py` | 完整 PGSG，另含尺度轴 Transformer |

三个入口使用 `datalists/train_417.json`（375 景）和 `datalists/val_417.json`（42 景）。没有独立测试集；`test_path` 暂指验证名单。历史论文单折为 334/83 划分，不与此处 375/42 结果混用。

## 服务器运行

本地 Windows 只编辑和静态检查；Python 检查、训练、评估与推理在服务器执行。先确认实例身份、资源及 GPU 空闲，激活 `ice-binary`。新目录尚未部署到服务器，不根据旧文档猜测 SSH 端口。

以下是部署后在项目根目录运行的示例，本次迁移没有执行：

```bash
conda activate ice-binary
python -m tools.checks.smoke_pgsg
python -m tools.checks.smoke_pipeline configs/FLOE/pgsg.py
python -m tools.train configs/FLOE/pgsg.py --wandb-project MyDS-FLOE --wandb-name floe_pgsg_417_s42 --work-dir work_dir/floe_pgsg_417_s42
python -m tools.evaluate --config configs/FLOE/pgsg.py --checkpoint work_dir/floe_pgsg_417_s42/best_model_floe_pgsg_417_s42.pth --val-list datalists/val_417.json --dump-cm /root/autodl-tmp/cm/single/pgsg.npz
```

Baseline 使用对应配置和 `floe_baseline_417_s42` run 名；CM 导出为 `/root/autodl-tmp/cm/single/baseline.npz`。辐射测试使用 `tools/analysis/radiometric.py`，显式提供配置、checkpoint、名单和 `--dump-cm /root/autodl-tmp/cm/single/radio/baseline`。完整参数以脚本为准。

长训练放 tmux，重定向到持久盘日志，不用 `python ... | tee` 判断退出码。训练占卡时不启动其他 GPU 工作。

## 外部资产和环境

- NC：`/root/autodl-tmp/FLOE_datanc/`；tile 索引：`/root/autodl-tmp/floe_tile_index_417.npz`。
- checkpoint、CM 和大部分原始训练日志未包含在此目录，不存在已训练好的新 375/42 PGSG 成绩。
- 制图脚本通过 `FLOE_DATA_DIR`、`FLOE_TILE_INDEX`、`FLOE_CM_DIR`、`FLOE_FIG_DIR` 配置公共路径。默认 CM 为 `/root/autodl-tmp/cm/single`。
- 图 6 可用 `--logs` 显式指定原始训练日志。已有 `docs/figs/fig6_readouts.csv` 为历史解析数据。
- 暗景诊断另需 `docs/_radiometric_profile_417.csv`，此资产未找到；可在服务器使用保留的统计脚本生成。
- 依赖与验证边界见 `docs/ENVIRONMENT.md`。

## 资料

- `paper/`：原样保留的正式 LaTeX 与插图。迁移未改写或编译论文。
- `docs/EXPERIMENT_RESULTS.md`：历史指标及其适用范围。
- `docs/PAPER_TODO.md`：论文收尾事项。
- `docs/FLOE_Git_TRANSFER_MANIFEST.md`：转移前批准清单，保留为计划快照。
- `docs/TRANSFER_REPORT.md`、`docs/TRANSFER_FILES.csv`、`docs/TRANSFER_SHA256.csv`：实际转移记录、源文件对照与最终校验。

上述 TRANSFER 文件是首次转移时的历史快照，记录旧目录布局。当前路径以 `docs/LAYOUT_FILES.csv` 为准，当前校验见 `docs/LAYOUT_SHA256.csv`，目录调整说明见 `docs/DIRECTORY_LAYOUT.md`。数据手册在 `docs/DATASET_HANDBOOK.md`。

历史文档可能引用未转移的五折文件或旧服务器路径，仅用于追溯论文依据，不是本目录可执行指令。

## 开源许可

本仓库采用 MIT 许可证，见 [LICENSE](LICENSE)。已有第三方代码的版权声明应予保留，外部依赖、数据及预训练权重按各自许可使用。开源此代码库不代表发布外部数据集或模型权重。

## 原始数据处理

新增源码见 [preprocessing/README.md](preprocessing/README.md)。该目录以独立脚本方式调用，与 `tools/` 的模块调用方式不同。按要求不包含 SAR 下载、HH/标签绘图和 NC 统计脚本。当前已完成源码整理与静态检查，尚未执行处理流程或完成运行验证。
