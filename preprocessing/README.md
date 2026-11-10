# 原始 SAR 处理与冰图标注

## 文件与流程

| 文件 | 用途 |
|---|---|
| `download_charts.py` | 下载 DMI 区域冰图；支持区域、年份与月份过滤 |
| `prescreen_dmi_charts.py` | DMI 冰图预筛、生成白名单和区域范围 |
| `prescreen_floe_charts.py` | CIS 冰图预筛；使用其自身区域/月份/AOI 设置 |
| `process_sar.py` | 原始 S1 ZIP → SNAP 轨道校正、去噪、定标、Refined Lee、地形校正、dB 和 GLCM |
| `make_labels.py` | 冰图匹配、重投影、栅格化、类别映射、掩膜与 NC 打包 |
| `floe_label.py` | 多边形分类、各类阈值、保留类别与目标类别 |
| `label_definitions.py` | SIGRID 编码及类别定义；原 `utils.py`，不与模型工具混用 |
| `normalize.py` | 首次 fit、apply 或 fit_apply 的全局 z-score 归一化 |
| `normalize_cake.py` | 原 `_cake_normalize.py`；新增批次沿用已知统计量，逐变量写出 |
| `normalize_batch2.py` | 原 `_batch2_finalize.py`；保留另一批次低内存归一化实现 |
| `run_pipeline.py` | 串联 3/4/5；不下载 SAR、不上传服务器 |
| `_settings.py` | 外部数据根目录公共配置 |

两份追加归一化实现保留用于来源追溯，不需要对同一批数据重复执行。其“低内存”指逐变量加载/写出，仍可能加载一整个二维变量和有效掩膜，并非完全按 HDF5 chunk 流式处理。完整源/目标映射和 SHA256 见 `SOURCE_FILES.csv`。

## 配置和调用约定

本地不运行 Python；本次整理只做静态检查。运行检查及正式处理留待明确授权的执行环境。处理环境除 `requirements.txt` 外，还需要原流程对应的 ESA SNAP/Microwave Toolbox；历史记录版本为 SNAP 12.0.0、Microwave Toolbox 12.0.1，未在本次重新核实安装。

以仓库根目录作为命令所在目录，**直接运行本目录脚本**，例如 `python preprocessing/run_pipeline.py ...`。本目录采用本地脚本导入，不使用 `python -m preprocessing.…`；与 `tools/` 的模块调用约定不同。

使用 `FLOE_PREPROCESS_ROOT` 指向外部数据目录，默认是仓库下的 `data/floe_preprocessing`；不要把实际数据提交 Git。示例变量见 `config.env.example`，需通过 shell 设置，脚本不会自动加载此示例文件。默认子目录为：

```text
<FLOE_PREPROCESS_ROOT>/
├── S1_raw/          自行准备的 Sentinel-1 EW HH/HV 原始 ZIP
├── Ice_chart/       DMI 冰图及配套 shp/shx/dbf/prj 文件
├── S1_process/      SNAP 中间及最终产物
├── data_nc/         未归一化特征和 FLOE 标签
└── data_nc_zscore/  归一化输出与 zscore_stats.json
```

`FLOE_GPT_PATH` 设置 SNAP gpt 路径，默认从 PATH 查找 `gpt`。处理脚本不再强制设置代理；DMI 下载遵循 requests 的标准代理环境变量，TLS 校验默认开启。

已有环境变量入口如 `FLOE_INPUT_DIR`、`FLOE_PROC_DIR`、`FLOE_SAR_ROOT`、`FLOE_CHART_ROOT`、`FLOE_OUTPUT_NC_DIR`、`FLOE_CHART_REGION` 继续保留，可覆盖公共目录。SAR 与冰图匹配窗口当前为 ±2 天，多区域冰图应设置区域过滤，避免错配。

## 类别规则必须显式选择

`floe_label.py` 当前默认保留 `0,1,2,3,4,5` 全六类，目标冰类为 `1,2,3,4,5`。未设置、空置或无法解析的类别环境变量均回退到这组默认值。统一入口仍要求显式选择处理模式：

```bash
# 仅示例，本次没有运行
python preprocessing/run_pipeline.py --steps 3,4 --class-profile six-class
python preprocessing/run_pipeline.py --steps 4 --class-profile hard-ice
```

六类模式与上述默认值一致；难类模式显式设置保留 `0,1,2,3`，目标类 `1,2,3`。直接运行 `make_labels.py` 且未设置覆盖变量时，使用六类默认值。需要子集时，可设置 `FLOE_KEEP_CLASSES` 与 `FLOE_TARGET_CLASSES`；已有环境变量仍优先于默认值。DMI 预筛另有 `FLOE_PS_CLASSES`（默认 `2,3`），CIS 预筛也有自身目标筛选定义，不会因标注默认值或流水线 class-profile 自动成为全类别采集。

分类阈值：Cake/Small/Medium 使用 CT≥60%、主导占比≥65%；Big/Vast 使用 CT≥90%、占比≥90%。快冰、陆地及无效/模糊区域按原实现处理，训练忽略值为 255。

SNAP 输出网格间距为 80 m；这不等于原始产品空间分辨率。GLCM 为 HH、四方向、64 级概率量化、11×11 窗口、位移 2，保留 Contrast/Dissimilarity/Homogeneity。具体处理图仍嵌在 `process_sar.py`。

## 归一化与追加数据

`normalize.py` 通过 `FLOE_NORM_MODE=fit/apply/fit_apply` 控制运行；默认 fit_apply。使用 `FLOE_NORM_INPUT`、`FLOE_NORM_OUTPUT`、`FLOE_STATS_JSON` 配置输入、输出和统计文件。SAR 先截断到 [-35,5] dB，再按统计量归一化；无效特征置 0，输出 global_valid_mask。标签不归一化。

追加数据应使用已有统计量的 apply 模式，不能把分别 fit 的批次当成同一归一化体系。`normalize_cake.py` / `normalize_batch2.py` 可用 `FLOE_APPEND_INPUT`、`FLOE_APPEND_OUTPUT` 和 `FLOE_STATS_JSON` 指定路径，默认对应 `data_nc_cake` / `data_nc_batch2` 及各自 zscore 输出。

追加脚本的锁与完成标记改放输出目录；默认不等待原采集批次标记。如确需旧等待行为，显式设置 `FLOE_SKIP_WAIT=0`、`FLOE_COLLECT_DONE`。同一输出目录不要并发启动归一化。完成标记存在时会跳过整批，增加新文件后应检查标记及输入范围，不能以旧标记认定新文件已处理。

标注默认跳过已存在 NC。修改规则后不能靠默认重跑覆盖旧标签，需要先明确重生成策略。原始 ZIP / SAR 源目录删除开关保留默认关闭；SNAP 中间产物按原 KEEP_INTERMEDIATE 规则清理，不应与原始数据混淆。