# FLOE-Git 目录调整记录

日期：2026-11-05。

## 目的与范围

根目录仅保留 README、AGENTS、.gitignore。公共代码和模型集中到 `floe`，运行工具按用途放进 `tools`，数据手册放进 `docs`。本次移动 41 个既有文件，增加 Python 包标记，更新项目内绝对导入和当前命令示例。没有改动模型计算、损失、增强、采样或指标公式。

完整旧路径→新路径及整理前后哈希在 `LAYOUT_FILES.csv`。原模型中兼容代码仍保留，原先未迁入的 SMP/mmseg 等可选模型包装器仍不包含；仅支持本次保留的 Baseline/PGSG 入口，不因公共函数含历史分派分支就声称支持那些未迁入模型。

## 使用方式

在项目根目录，使用原服务器环境运行，示例：

```bash
python -m tools.train configs/FLOE/pgsg.py --wandb-project MyDS-FLOE --wandb-name floe_pgsg_417_s42 --work-dir work_dir/floe_pgsg_417_s42
python -m tools.evaluate --config configs/FLOE/pgsg.py --checkpoint <checkpoint> --val-list datalists/val_417.json
python -m tools.infer --help
python -m tools.data.scan_tiles --help
python -m tools.analysis.radiometric --help
python -m tools.analysis.evidence_ablation --help
python -m tools.analysis.parameters
python -m tools.figures.fig5_confusion --baseline <baseline.npz> --pgsg <pgsg.npz>
python -m tools.checks.smoke_pgsg
python -m tools.checks.smoke_pipeline configs/FLOE/pgsg.py
bash tools/system/memmon.sh <output.csv>
```

项目包导入例如 `from floe.loaders import ...`；制图公共代码导入 `from tools.figures.common import ...`。配置基础文件也改为导入 `floe.functions`。

## 文档与科学资产

论文、插图、既有日志和历史结果文档保持原内容；其中旧文件名属于历史上下文。首次转移的 TRANSFER 报告及 CSV 保持为当时的快照，不再作为当前路径索引。当前布局通过 LAYOUT CSV 追踪。

此前列明的 NC、索引、checkpoint、CM、原始训练日志等外部资产状态不变；没有加入数据构建源码或五折实验。

## 验证边界

本地仅执行 PowerShell 静态与哈希检查：配置父文件、关键入口、包导入路径、根目录白名单、论文/日志内容保持，以及最终目标文件对照。不运行 Python，不连接服务器，不编译论文，不启动训练或评估。

Python 语法/实际导入与 GPU 冒烟尚需在服务器验证；静态检查不能代替这些检查。
