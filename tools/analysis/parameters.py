"""T0-0.4：参数量统计（PGSG 主模型 vs 基线）。无卡模式可跑（CPU，只构模型不前向）。

产出论文 §4.4 / §4.6 需要的三个数里的两个：**各新增模块的参数量**与**全模型总参数量**。
FLOPs 需要一次前向，默认**不做**（无卡只有 0.5 核 / 2GB，容易超时或 OOM）；
开卡时加 --flops 即可。

⚠️ 构模型时强制 `segformer_encoder_weights=None`：参数量与是否加载预训练权重无关，
但 'imagenet' 会触发一次权重下载，无卡模式下没必要。

用法
----
    python -m tools.analysis.parameters
    python -m tools.analysis.parameters --flops                 # 开卡时再加
    python -m tools.analysis.parameters --pgsg <cfg> --baseline <cfg>
"""
import argparse
import copy

from mmcv import Config

from floe.loaders import limit_torch_threads, get_variable_options  # 新入口必须先限线程（CLAUDE.md）

limit_torch_threads()

import torch                                  # noqa: E402
from floe.functions import get_model               # noqa: E402

DEFAULT_PGSG = 'configs/FLOE/pgsg.py'
DEFAULT_BASE = 'configs/FLOE/sota_segformer_in1k_dlr_417.py'

# 报告里按名字前缀归并的模块（与论文 §4.2–4.5 一一对应）
MODULE_KEYS = [
    ('Variogram evidence',              ['gate_ev_vario']),
    ('Polarimetric prior',              ['gate_ev_pol']),
    ('Evidence-conditioned scale gate', ['scale_gate']),
    ('Scale-axis transformer',          ['scale_xattn']),
    ('Polarimetric bias attention',     ['pba']),
]


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    opts = get_variable_options(copy.deepcopy(cfg.train_options))
    opts['segformer_encoder_weights'] = None          # 别下载权重
    net = get_model(opts, torch.device('cpu'))
    net.eval()
    return net, opts


def count(net):
    return sum(p.numel() for p in net.parameters())


def by_module(net):
    out = {}
    for label, keys in MODULE_KEYS:
        n = 0
        for name, p in net.named_parameters():
            if any(f'.{k}.' in f'.{name}.' or name.startswith(k) for k in keys):
                n += p.numel()
        if n:
            out[label] = n
    return out


def input_channels(opts):
    """网络实际吃的通道数。

    ⚠️ 不等于 len(train_variables)：loaders._augment_x 会在后面追加派生通道
    （month_encoding ×2 / pol_ratio ×1），顺序与 _eval_radiometric.channel_layout 一致。
    本项目 FLOE 的实际值是 8 = 7 个 train_variables + pol_ratio。
    （论文口径的"7 通道"是另一回事：它不计 global_valid_mask、但计 pol_ratio。）
    """
    return (len(opts['train_variables'])
            + (2 if opts.get('month_encoding', False) else 0)
            + (1 if opts.get('pol_ratio_channel', False) else 0))


def try_flops(net, opts):
    ch = input_channels(opts)
    x = torch.zeros(1, ch, opts['patch_size'], opts['patch_size'])
    for lib in ('fvcore', 'thop'):
        try:
            if lib == 'fvcore':
                from fvcore.nn import FlopCountAnalysis
                f = FlopCountAnalysis(net, x)
                f.unsupported_ops_warnings(False)
                f.uncalled_modules_warnings(False)
                return f.total(), 'fvcore (MACs)'
            from thop import profile
            macs, _ = profile(net, inputs=(x,), verbose=False)
            return macs, 'thop (MACs)'
        except ImportError:
            continue
        except Exception as e:                        # noqa: BLE001
            print(f'  [{lib}] 失败：{e}')
    print('  未装 fvcore / thop，跳过 FLOPs（pip install fvcore 后重跑）')
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pgsg', default=DEFAULT_PGSG)
    ap.add_argument('--baseline', default=DEFAULT_BASE)
    ap.add_argument('--flops', action='store_true')
    args = ap.parse_args()

    print(f'building baseline: {args.baseline}')
    net_b, opt_b = build(args.baseline)
    n_b = count(net_b)
    print(f'building PGSG    : {args.pgsg}')
    net_p, opt_p = build(args.pgsg)
    n_p = count(net_p)

    print('\n=== 总参数量 ===')
    print(f'  (a) Baseline      : {n_b:,}  ({n_b / 1e6:.2f} M)')
    print(f'  (c) PGSG (full)   : {n_p:,}  ({n_p / 1e6:.2f} M)')
    print(f'  增量              : {n_p - n_b:,}  ({(n_p - n_b) / 1e6:.3f} M, '
          f'+{100 * (n_p - n_b) / n_b:.2f}%)')

    mods = by_module(net_p)
    print('\n=== PGSG 各新增模块 ===')
    tot = 0
    for label, n in mods.items():
        print(f'  {label:34s} {n:>10,}  ({n / 1e6:.3f} M)')
        tot += n
    print(f'  {"合计（按名字归并）":34s} {tot:>10,}  ({tot / 1e6:.3f} M)')
    if tot != n_p - n_b:
        print(f'  ⚠️ 与总量差 {n_p - n_b - tot:,}——说明还有未归类的新增参数，'
              '检查 MODULE_KEYS 是否漏了某个模块名')

    if args.flops:
        print('\n=== FLOPs（单张 patch，1x{}x{}x{}） ==='.format(
            input_channels(opt_p), opt_p['patch_size'], opt_p['patch_size']))
        for label, net, opt in (('Baseline', net_b, opt_b), ('PGSG', net_p, opt_p)):
            v, how = try_flops(net, opt)
            if v is not None:
                print(f'  {label:10s} {v / 1e9:.2f} G{how.split()[-1].strip("()")}'
                      f'  [{how}]')
    else:
        print('\n（未测 FLOPs：无卡模式下跳过，开卡时加 --flops）')


if __name__ == '__main__':
    main()
