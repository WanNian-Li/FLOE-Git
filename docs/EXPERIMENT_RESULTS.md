# 已有论文结果与适用边界

以下为迁移前已完成的历史单折（fold 0，83 验证景）结果，不是本目录 375/42 名单上新运行的成绩。本次不带入五折配置、名单和训练链。

| 指标 | Baseline | 完整 PGSG |
|---|---:|---:|
| macro mIoU | 0.5031 | 0.5109 |
| hard-ice mIoU | 0.2243 | 0.2302 |
| water IoU | 0.969 | 0.968 |
| cake IoU | 0.081 | 0.088 |
| small IoU | 0.220 | 0.231 |
| medium IoU | 0.229 | 0.229 |
| big IoU | 0.650 | 0.676 |
| vast IoU | 0.870 | 0.874 |

逐类值已四舍五入。来源：原样保留的 `paper/main.tex`、写作补充材料及 `docs/logs/paper_gpu_batch.log`。macro-F1 可由逐类 IoU 换算，但正式报告应读原始 CM；本目录没有两模型的确切 OA 原始值，不补造数值。

- 原训练景评估：Baseline medium IoU 0.6693、small 0.5669；日志为 `docs/logs/paper_traineval_f0.log`。
- 辐射扰动和证据替换实验已完成，见 `paper_radio_f0.log`、`paper_radio_pgsg4_f0.log`、`paper_evabl_pgsg4_f0.log`。
- 两路证据同时替换成估计的逐通道均值后，PGSG macro 下降约 0.0009，hard-ice 约 0.0005。仅说明所测检查点推理期替换的影响，不能代替从头无证据训练对照。
- 参数量：24,738,310 → 24,972,410；记录的计算量 5.64 → 7.56 GMACs。记录来源见补充材料 #2。
- 单折改善幅度有限，medium 尚无改善；不据此宣称稳健或显著提升。

保留日志文件名及其内部旧路径，便于追溯。新脚本默认名称与历史资产名不同，重绘历史图时必须显式定位正确资产，不能将新名单与旧 CM 混配。
