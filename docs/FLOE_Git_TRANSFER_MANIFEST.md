# FLOE-Git 文件转移清单

更新日期：2026-11-05。

- 源目录：`F:\ZJU\11_Ice\code\FLOE-Detection`
- 目标目录：`F:\ZJU\11_Ice\code\FLOE-Git`
- 当前状态：仅制定清单，尚未复制、移动、删除文件，也未启动实验。
- 用户明确排除：数据构建源码；五折实验相关文件。
- 保留目标：Baseline、最终 PGSG、单次训练与评估、已有单折论文诊断、论文源码及必要证据。
- 下列文件路径均相对于源目录；实际迁移原则上保持相对位置。此文档自身也应保留到目标目录的 `docs/`。

## 1. 核心代码：原样转移

| 文件 | 用途 |
|---|---|
| `quickstart.py` | 训练、验证、早停、平滑选优及 checkpoint 保存 |
| `loaders.py` | NC 读取、tile 加权采样、增强、派生通道及按需读取 |
| `functions.py` | 模型、优化器、损失、滑窗预测、指标及公共功能 |
| `losses.py` | 损失函数定义及公共调用依赖 |
| `utils.py` | 类别名称、配色及公共辅助工具 |
| `evaluate_scenes.py` | 逐景评估，输出 OA、F1、IoU 及逐景混淆矩阵 |
| `inference.py` | 推理与预测导出入口 |
| `test_upload_function.py` | 被训练、推理入口直接导入的测试及可视化功能；不是可以按文件名直接删除的临时测试 |
| `scan_tiles.py` | 建立训练 tile 索引及裁剪覆盖核加权的类别直方图 |
| `unet.py` | `functions.py` 顶层直接导入，暂作兼容依赖保留 |
| `swin_transformer.py` | `functions.py` 顶层直接导入，暂作兼容依赖保留 |

说明：只整理文件，不在迁移时顺带删除这些公共文件内部的历史分支。内部简化应另做代码审查。

## 2. 模型代码：原样转移

| 文件 | 用途 |
|---|---|
| `models/__init__.py` | 模型包结构 |
| `models/sota/__init__.py` | 当前会直接导入下面多个架构，需维持完整导入链 |
| `models/sota/segformer.py` | Baseline 和 PGSG 的主体实现，包含证据分支、门控与尺度通信 |
| `models/sota/unet.py` | 模型包直接导入依赖 |
| `models/sota/segnet.py` | 同上 |
| `models/sota/resnet.py` | 同上 |
| `models/sota/densenet.py` | 同上 |
| `models/sota/pspnet.py` | 同上 |
| `models/sota/deeplabv3.py` | 同上 |
| `models/sota/poolformer.py` | 同上 |
| `models/sota/segnext.py` | 同上 |

不转移任何 `__pycache__`。`models/smp`、`models/mmseg` 暂不转移；当前 Baseline/PGSG 不走这两个包装器。

## 3. 非五折配置与名单：原样转移

| 文件 | 用途 |
|---|---|
| `configs/_base_/base.py` | 类别映射、输入变量与基础配置 |
| `configs/FLOE/_sota_base.py` | 共享训练、采样、增强与验证协议 |
| `configs/FLOE/sota_segformer_in1k_417.py` | MiT-B2 编码器及 ImageNet 初始化 |
| `configs/FLOE/sota_segformer_in1k_dlr_417.py` | 非五折 Baseline，设置 encoder/head 判别式学习率 |
| `datalists/train_417.json` | 原 375 景训练集名单 |
| `datalists/val_417.json` | 原 42 景验证集名单 |

注意：375/42 划分与论文 fold 0 的 334/83 划分不同，不能据此重现论文单折成绩，也不能直接横比绝对指标。

## 4. PGSG 配置缺口：迁移时另行整理，不能原样复制五折配置

现有最终 PGSG 的继承链为：

`sota_segformer_in1k_dlr_cv5f0_pgsg4.py`
→ `sota_segformer_in1k_dlr_cv5f0_pgsg_nodann.py`
→ `sota_segformer_in1k_dlr_cv5f0_pgsg.py`
→ 非五折 Baseline 配置。

前三个文件均包含五折实验语义，按本次要求不复制。不能因此宣称新目录已经有可直接使用的 PGSG 配置。

实际迁移时应从现有实现中提取模型参数，另建以下非五折配置；本轮仅记录方案，不新建这些配置：

| 拟新增文件 | 用途与必要参数 |
|---|---|
| `configs/FLOE/pgsg.py` | 继承非五折 Baseline；启用 `segformer_scale_gate=True`、`segformer_gate_mode='softmax'`、`segformer_gate_evidence=True`、`segformer_scale_xattn=True`；显式关闭 DANN/PBA；数据名单继承非五折配置 |
| `configs/FLOE/pgsg_evidence_gate.py` | 保留“证据＋门控”单次消融，尺度通信关闭，DANN/PBA 关闭 |

不带入五折编号、五折训练名单和 334 个域等折专属参数；不自动补做 DANN/PBA 探针的新配置。

## 5. 单次评估、论文诊断与检查脚本

| 文件 | 用途 | 转移前处理 |
|---|---|---|
| `analyze_scene_cm.py` | 单次评估的逐景混淆矩阵分析 | 检查默认输入路径 |
| `_eval_radiometric.py` | 辐射扰动敏感性测试 | 使用显式配置、checkpoint 和场景名单；检查旧默认值 |
| `_paper_gpu_ablation_evidence.py` | 推理期证据替换消融 | 默认配置、run 和名单含 fold0，需适配新的非五折配置和实际资产 |
| `_paper_t0_params.py` | 参数量、计算量统计 | 默认模型配置需适配 |
| `_train_radiometric_profile.py` | 场景辐射统计 | 检查数据目录及名单 |
| `_paper_t0_darkscenes.py` | 保留“暗景更差”假设的单次诊断方法 | 指向实际 CM 与辐射统计 CSV，不携带旧五折默认路径 |
| `_val_ondemand_check.py` | 按需读与整景读的有效 tile 输入一致性检查 | 检查配置及数据路径 |
| `_memmon.sh` | 服务器资源与内存监控 | 检查日志输出路径 |
| `smoke_sota.py` | 模型与公共管线检查 | 使用新配置显式运行，核查默认配置引用 |
| `smoke_pgsg4.py` | 最终模型的初始化等价性、接线及梯度检查 | 将两个硬编码的 fold0 配置替换为新的完整/门控配置 |

本表中“需适配”的文件不是原样复制即可用。适配只修改配置/路径引用，不能悄悄改变实验定义；本轮未修改脚本。

`smoke_pgsg.py`、`smoke_pgsg3.py` 随旧实验配置留在原项目，本次不转移。

## 6. 论文源码与正式插图：原样转移

| 文件 | 用途 |
|---|---|
| `paper/main.tex` | 当前正式论文 |
| `paper/References.bib` | 文献库 |
| `paper/elsarticle-harv.bst` | 文献样式 |
| `paper/TODO.txt` | 作者写作注意事项 |
| `paper/figures/fig1_task.pdf` | 任务示意 |
| `paper/figures/fig2_radiometric.pdf` | 辐射扰动示意 |
| `paper/figures/fig3_architecture.pdf` | PGSG 架构 |
| `paper/figures/fig4_variogram.pdf` | 变异函数剖面 |
| `paper/figures/fig5_confusion.pdf` | 混淆矩阵 |
| `paper/figures/fig6_readouts.pdf` | 组件训练读数 |
| `paper/figures/fig7a_qualitative_hardclasses.pdf` | 难类预测对比 |
| `paper/figures/fig7b_qualitative_bigfloe.pdf` | 大浮冰预测对比 |

论文及已有插图中的 fold0/cv5 表述属于既有科研证据，原样保留，不因工程文件整理而删除或改写。排除的是五折实验代码、配置、名单和专门文档。

## 7. 制图源码与数据

| 文件 | 用途 | 注意事项 |
|---|---|---|
| `_paper_figs_common.py` | 制图公共工具及资产定位 | 默认 CM 目录含 `cv5`，迁移时改为实际资产位置 |
| `_paper_fig1_task.py` | 图 1 | 核对数据路径 |
| `_paper_fig2_radiometric.py` | 图 2 | 核对模型配置、checkpoint 和场景路径 |
| `_paper_fig3_arch.py` | 图 3 | 保留架构制图能力 |
| `_paper_fig4_variogram.py` | 图 4 | 核对数据及索引路径 |
| `_paper_t0_cm.py` | 图 5 与混淆矩阵分析 | 保留单次比较用途，适配实际 CM 路径 |
| `_paper_fig6_readouts.py` | 图 6 | 输入单次训练日志，不迁入五折训练链日志 |
| `_paper_fig7_qualitative.py` | 图 7 | 核对模型配置、checkpoint 和场景路径 |
| `docs/figs/fig6_readouts.csv` | 图 6 的解析数据 | 原样保留 |
| `docs/fig3_architecture.tex` | 架构图 LaTeX 源码 | 原样保留 |

不整目录转移 `docs/figs`、`docs/paper_figures`，避免重复 PDF/PNG。正式成图统一保留在 `paper/figures`。

## 8. 文档与已有单次实验日志

### 8.1 原样保留的资料

| 文件 | 用途 |
|---|---|
| `DATASET_HANDBOOK.md` | 数据变量、标签与归一化来源；其中旧版本统计须保留版本语境 |
| `docs/2026-07-09_tile_sampling.md` | tile 采样设计与实现依据 |
| `docs/2026-07-21_cake_flow_diagnosis.md` | 难类纹理与可分性诊断背景 |
| `docs/2026-08-07_dataset_distribution.md` | 数据集地域、季节与类别分布 |
| `docs/2026-08-11_PGSG_paper_story.md` | 方法提出与设计背景；历史实验安排不视为新任务 |
| `docs/2026-08-12_PAPER_WRITING_MANUAL.md` | 论文术语、写作框架及证据边界 |
| `docs/2026-09-15_PAPER_SUPPLEMENT_01.md` | 数据统计及方法描述补正 |
| `docs/2026-09-16_PAPER_SUPPLEMENT_02_T0.md` | 已完成诊断、计算量和组件读数 |
| `docs/2026-09-17_PAPER_FIGURES_GUIDE.md` | 图注与插图使用依据 |
| `docs/logs/paper_gpu_batch.log` | 已完成的单折论文诊断批次摘要 |
| `docs/logs/paper_radio_f0.log` | Baseline 的既有单折辐射测试 |
| `docs/logs/paper_radio_pgsg4_f0.log` | PGSG 的既有单折辐射测试 |
| `docs/logs/paper_evabl_pgsg4_f0.log` | 既有单折推理证据消融 |
| `docs/logs/paper_traineval_f0.log` | 已完成的训练景评估，支撑论文泛化差距 |

说明：这些混合内容的论文资料中可能提及五折，但保留它们是为了保存论文证据；不转移专门的五折实验文件。日志文件名中的 f0 是历史证据标识，不代表迁入五折执行流程。

### 8.2 迁移时需新写的简短说明

以下是计划新增文件，本轮不生成：

- `README.md`：项目用途、Baseline/PGSG 入口、服务器执行方式、数据与资产位置。
- `AGENTS.md`：从现有约定中提取仍适用的环境和执行规则，移除五折任务及过时连接信息；不整份复制旧文件。
- `docs/EXPERIMENT_RESULTS.md`：摘录 Baseline/PGSG 的已有单次结果及来源，明确当前清单不包含五折复现材料。
- `docs/PAPER_TODO.md`：论文当前缺口与已完成事项，不迁入五折执行计划。
- 环境依赖清单：后续从实际服务器环境核实导出，不凭本地推测填写版本。

不整份复制 `CLAUDE.md`、旧 `AGENTS.md`、`实验记录.md`，以免把大量历史任务与五折实验记录重新带入新目录。

## 9. 明确不转移

1. 整个 `FLOE_only/`：所有数据下载、构建、归一化源码及批次日志，均按用户要求排除。
2. 整个 `datalists/cv5/`。
3. 所有文件名含 `cv5` 的配置与脚本，包括 `aggregate_cv.py`、`make_cv_folds.py`、`make_cv_configs.py`、`_precheck_cv5.py`、`_cv5_drop_worst.py`、`chain_cv5*.sh`、`chain_pgsg_cv5.sh`、`watchdog_cv5.sh`。
4. 专门五折文档：`docs/2026-08-06_cv5_design.md`、`docs/2026-08-08_cv5_baseline_results_table.md`。
5. 旧论文执行清单：`docs/2026-08-12_PAPER_PENDING_EXPERIMENTS.md`、`docs/2026-09-15_T0_RUNBOOK.md`、`docs/2026-09-16_GPU_RUNBOOK.md`；其中有五折或旧配置执行安排。
6. `_paper_t0_region.py`：当前依赖五折 CM 汇总；已有区域统计保留在论文资料，不转移该脚本。
7. `_paper_run_t0.sh`、`_paper_run_gpu.sh`、`watchdog_paper_gpu.sh`：现有批处理绑定旧配置/目录或包含上述被排除脚本，本次不迁入；单脚本入口足够。
8. `configs/SOD/`、`configs/SOTA/`、`configs/paper_report/` 及未在第 3 节列出的其他历史 FLOE 配置。
9. `viz/`、`viz_all/`、`examples/`、`.claude/`、`.codex/`、所有 `__pycache__/` 和 `.pyc`。
10. `docs/main (1).tex`、重复图像、无关历史探索文档、一次性上传/修复/补传/救援脚本、锁与完成标记。
11. `PA2_Swin.py`、`PA2_Swin_Improved.py`、`swin_transformer_raw.py`、`PA2_Swin_Technical_Report.md`。
12. 其他未列入本清单的文件：默认不转移；若发现实际依赖，先补充清单再处理。

## 10. 迁移前必须说明的依赖边界

- 当前清单不构成“原样复制即可完整运行”的承诺：PGSG 新配置以及部分脚本的默认路径需要按第 4、5、7 节适配。
- 数据集 NC、tile 索引、checkpoint、逐景 CM、原始训练日志及归一化统计文件不应凭空视为已包含。本轮不下载这些资产，也不执行服务器操作。
- 图 6 的解析 CSV 已在本地；重新解析仍需要对应原始训练日志。
- 暗景脚本引用的 `docs/_radiometric_profile_417.csv` 当前本地未发现，需要后续找到实际文件或在服务器按既定流程生成。
- 训练数据预处理源码被排除，不意味着可删除标签、归一化及输入通道的科学说明。
- 后续整理不得把旧单折成绩标成新 375/42 划分的成绩，不因改路径而启动新训练。
- 后续正式复制前，应先确认目标目录已有内容；不覆盖用户文件，不删除源目录。

## 11. 后续执行顺序（尚未执行）

1. 以本清单生成确切文件名单，检查源文件和目标冲突。
2. 在得到执行转移的指令后复制白名单文件，保留原目录结构。
3. 整理非五折 PGSG 配置、脚本引用及简短项目说明；不带入被排除的实验流程。
4. 对照源文件核验复制结果；静态检查本地导入和配置路径，不在本地运行 Python。
5. 单独列出仍依赖服务器资产的功能，再决定是否需要服务器检查。
