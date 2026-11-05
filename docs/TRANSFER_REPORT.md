# FLOE-Git 转移记录

日期：2026-11-05。来源：`F:\ZJU\11_Ice\code\FLOE-Detection`；目标：`F:\ZJU\11_Ice\code\FLOE-Git`。

## 范围

按批准清单保留 75 个原有文件，其中 64 个按原字节复制，11 个只适配配置、名单或默认路径。另新增两个非五折 PGSG 配置、README、简短 AGENTS、结果摘要、论文待办、环境说明和转移校验记录。

数据构建目录、五折配置/名单/链式脚本和专门文档未带入。已有论文与单折诊断日志保持原样，保留历史出处，不改写科研记录。

## 11 个适配文件

- `_paper_fig2_radiometric.py`、`_paper_fig7_qualitative.py`：配置、run、名单与 CM 名称改为单次入口。
- `_paper_gpu_ablation_evidence.py`：使用 `configs/FLOE/pgsg.py`、`val_417.json` 和单次输出名。
- `_paper_t0_params.py`：统计非五折 Baseline/PGSG 配置。
- `_paper_t0_cm.py`：默认读取单次 Baseline/PGSG 的 CM。
- `_paper_t0_darkscenes.py`：默认读取单次辐射实验 CM。
- `_paper_figs_common.py`：默认 CM 目录为 `/root/autodl-tmp/cm/single`。
- `_paper_fig6_readouts.py`：示例 run 及搜索项目路径适配新目录。
- `configs/_base_/base.py`：默认名单指向已保留的 `train_417.json`、`val_417.json`。
- `smoke_pgsg4.py`：引用新的完整/证据门控配置。
- `smoke_sota.py`：默认检查 Baseline/PGSG，移除缺失历史配置默认项。

修改未涉及模型、loss、采样算法或指标公式。部分历史注释和帮助示例仍可能提及未保留架构，当前用法以 README 和显式配置为准。

## 新配置

- `configs/FLOE/pgsg_evidence_gate.py`：继承非五折 Baseline，启用证据门控，关闭尺度通信、DANN、PBA。
- `configs/FLOE/pgsg.py`：继承上述配置，仅将尺度通信设为 True，保持 `smoke_pgsg4.py` 的单变量配置约束。

使用 375/42 划分。没有声称在该划分上获得历史论文成绩；历史 checkpoint 和 CM 需另外定位，不自动混用。

## 核验

- 75 个来源文件逐个记录源/目标 SHA256；源内容未修改。
- 六个配置的 `_base_` 父文件均存在。
- JSON 名单：训练 375、验证 42，交集为空。
- 关键脚本的配置和名单默认值指向已保留文件。
- 未包含 `FLOE_only`、五折目录/文件或 Python 缓存。
- `TRANSFER_FILES.csv` 记录来源对照；`TRANSFER_SHA256.csv` 覆盖最终交付文件（不自包含该哈希表）。最终目标已完成 86/86 文件 SHA256 对照核验；75 个来源文件也已核对源内容未改变。

以上为 PowerShell 静态及字节检查。未运行 Python、未编译 LaTeX、未启动 GPU 任务、未连接或部署到服务器。不能将其表述为运行验证通过。

## 外部依赖与后续

数据 NC、tile 索引、checkpoint、逐景 CM、归一化统计及部分原始训练日志不在迁移包中。暗景统计 CSV 未找到。环境版本未从服务器导出，见 `ENVIRONMENT.md`。

原始批准清单 `FLOE_Git_TRANSFER_MANIFEST.md` 保留为迁移前计划快照；本文件及 CSV 为实际转移记录。源目录保留，不删除原有项目文件。
