# SegFormer-B2 + MiT encoder ImageNet 预训练 —— 417 景数据集，6 类标准口径
#
# 动机：预训练是目前唯一显著有效的干预（DL3+·R50 从零 0.4638 → in1k 0.5419，
# 2026-07-13，30 景 val 时代），而 SegFormer 是从零口径下最强的 base（0.4867@30 景 val）。
# 本实验给 SegFormer 的 MiT-B2 encoder 启用 ImageNet-1k 权重，检验预训练增益在
# 分层 Transformer 上是否复现。
#
# 实现：models/sota/segformer.py 的 encoder 经 smp.FPN(encoder_name='mit_b2') 构建，
# `segformer_encoder_weights` 直接透传给 encoder_weights。'imagenet' 是 SMP 对 mit_b0~b5
# 提供的唯一预训练（首层 3 通道 patch embedding 由 SMP 循环平铺到 8 通道，机制同
# DL3+·R50·in1k）。all-MLP head（~3M 参数）从零——与 DL3 的 ASPP head 从零口径一致。
# ADE20K/Cityscapes 的完整分割权重需要 mmseg/HF 实现，与本包装不兼容，不采用。
#
# ⚠️ 权重下载：~/.cache/torch/hub/checkpoints/ 里只有 mit_b2 的 .partial 残留
# （上次下载中断），启动前需联网重新下载（~100MB）；离线启动会失败。
#
# 对照：
#   - floe_sota_dl3_in1k_417（0.5042@42 景 val）：同数据同协议，比"哪个 in1k base 强"；
#   - 从零 SegFormer 仅有 30 景 val 时代结果（0.4867），与本次不可横比；如需 417 口径
#     下干净的预训练 A/B，另跑一个 encoder_weights=None 的 417 版。
#
# 判别式 LR（预训练 encoder vs 随机 head 的区别对待）：
#   functions.get_optimizer 已支持 optimizer['encoder_lr_scale']（如 0.1 = encoder 用
#   1e-5、head 用 1e-4，方向同 mmseg 官方 SegFormer 的 backbone 1x/head 10x）。
#   本 config 首跑【刻意不启用】——保持与 floe_sota_dl3_in1k_417 完全同协议（它也是
#   统一 LR），预训练增益归因干净；有效后再单变量加 'encoder_lr_scale': 0.1 对照。
#   ⚠️ 启用时注意：CosineAnnealing 的 eta_min 是绝对值、对所有参数组相同，末期两组
#   LR 会收敛到同一下限；--resume-from 必须与原 run 的 encoder_lr_scale 一致。
#
# 用法：
# python -m tools.train configs/FLOE/sota_segformer_in1k_417.py --wandb-project MyDS-FLOE \
#     --wandb-name floe_sota_segformer_in1k_417 --work-dir work_dir/floe_sota_segformer_in1k_417

_base_ = ['./_sota_base.py']

train_options = {
    'model_selection': 'sota_segformer',
    'segformer_encoder_weights': 'imagenet',
}
