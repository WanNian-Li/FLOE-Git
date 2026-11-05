# SegFormer-B2 + MiT ImageNet 预训练 + 判别式 LR —— 417 景，6 类标准口径
#
# 与 sota_segformer_in1k_417 的【单变量对照】：唯一差异是 optimizer.encoder_lr_scale=0.1
# （encoder 1e-5 / head 1e-4，方向同 mmseg 官方 SegFormer 的 backbone 1x / head 10x）。
# 动机：预训练 encoder 是"专家"、all-MLP head 是随机初始化，统一 LR 下训练初期 head 的
# 混乱梯度会反传损伤预训练特征；Transformer 骨干对此比 CNN 更敏感（注意力权重 + LN）。
# 可检验假设：in1k run 的 best epoch 都极早（e10~e12）、随后长期无提升——若系初期
# encoder 被损伤所致，保护 encoder 后峰值应更晚、更高。
#
# 机制：functions._optimizer_param_groups（2026-07-17 加入），启动日志必核对
#   [Optimizer] encoder_lr_scale=0.1: encoder ~24.2M params lr=1.00e-05 | rest ...
# ⚠️ --resume-from 必须与原 run 相同的 encoder_lr_scale（optimizer state 与分组绑定）；
#   CosineAnnealing 的 eta_min 为绝对值、两组共用，末期两组 LR 收敛到同一下限。
#
# 对照：floe_sota_segformer_in1k_417（统一 LR）——直接读它的结果比"判别式 LR 是否有效"。
#
# 用法：
# python -m tools.train configs/FLOE/sota_segformer_in1k_dlr_417.py --wandb-project MyDS-FLOE \
#     --wandb-name floe_sota_segformer_in1k_dlr_417 --work-dir work_dir/floe_sota_segformer_in1k_dlr_417

_base_ = ['./sota_segformer_in1k_417.py']

train_options = {
    # mmcv Config 对 dict 递归合并：只写差异键，type/lr/b1/b2/weight_decay 继承 base。
    'optimizer': {
        'encoder_lr_scale': 0.1,
    },
}
