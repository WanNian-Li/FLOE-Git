"""0-2：推理期证据消融——训练好的 PGSG 到底还在用证据吗？**需要 GPU**。

动机
----
Fig. 6 的读数显示"证据/特征激活贡献比"随训练衰减（(b) 0.445→0.098、(c) 0.394→0.045）。
但那是 **conv1 输出的激活 RMS 之比**，是个幅度指标，不能直接读成"证据没用了"——
也可能证据幅度小但携带关键信息。本脚本用**功能性**的方式回答：把证据换掉，指标掉不掉。

做法（三点设计，直接决定结论能不能立住）
----------------------------------------
1. **用"数据集均值"替换，不是置零**。门控第一层是 conv(1120→64) 后接 BN，BN 的 running
   stats 是带着真实证据统计训出来的；置零会整体平移 conv1 的输出分布，掉下来的分数分不清
   是"信息没了"还是"分布错位"。换成每通道均值 ⇒ 只拿掉**空间/结构信息**、保留量级。
   （`--sub zero` 保留作对照，但主结论请用 mean。）
2. **证据有两个消费者**：尺度轴 transformer 的证据 token 与门控输入。只换一路只覆盖一半，
   所以默认跑 both / 仅 gate / 仅 xattn 三档。
3. **两路证据可分别消融**：证据张量是 `cat([vario 64ch, pol 32ch])`，可单独换掉任一路
   ⇒ **免训练地回答"两条物理先验哪条在起作用"**。
   ⚠️ 解释有界：这是"训练好的模型在推理时还依赖哪一路"，**不等于**"从头去掉那一路重训会
   怎样"。论文里必须这样限定措辞。

用法
----
    python -m tools.analysis.evidence_ablation                      # 默认 6 个变体，全 83 景
    python -m tools.analysis.evidence_ablation --scenes 5           # 冒烟
    python -m tools.analysis.evidence_ablation --config <cfg> --run <run 名>
"""
import argparse
import copy
import json
import os
import os.path as osp

from mmcv import Config

from floe.loaders import (get_variable_options, val_scoped_options,
                     AI4ArcticChallengeTestDataset, limit_torch_threads)

limit_torch_threads()

import torch                                              # noqa: E402
from floe.functions import get_model, fast_tiled_val_predict    # noqa: E402
from tools.figures.common import *                          # noqa: E402,F401,F403

CFG = 'configs/FLOE/pgsg.py'
RUN = 'floe_pgsg_417_s42'
VAL_LIST = 'datalists/val_417.json'
VARIO_CH, POL_CH = 64, 32                 # 证据张量 = cat([vario, pol])

# (标签, 替换哪几路证据, 替换给哪个消费者)
VARIANTS = [
    ('control',        None,      None),
    ('sub-all/both',   'all',     'both'),
    ('sub-vario/both', 'vario',   'both'),
    ('sub-pol/both',   'pol',     'both'),
    ('sub-all/gate',   'all',     'gate'),
    ('sub-all/xattn',  'all',     'xattn'),
]


def stream_slice(streams):
    if streams == 'all':
        return slice(0, VARIO_CH + POL_CH)
    if streams == 'vario':
        return slice(0, VARIO_CH)
    if streams == 'pol':
        return slice(VARIO_CH, VARIO_CH + POL_CH)
    raise ValueError(streams)


class EvidenceAblation:
    """用 forward_pre_hook 在两个消费点替换证据；不改项目源码。

    挂载点（models/sota/segformer.py `_SegformerHead.forward`）：
      - `head.scale_xattn(projected, gate_evidence)`  -> args[1] 是证据张量
      - `head.scale_gate(gate_in)`，其中 gate_in = cat([x, evidence]) -> 末 ev_ch 个通道
    """

    def __init__(self, net, head, ev_ch):
        self.net, self.head, self.ev_ch = net, head, ev_ch
        self.mean = None                  # (ev_ch,) float32 CPU
        self._handles = []

    # ---------------------------------------------------------------- 均值估计
    def estimate_mean(self, loader, opts, n_scenes, device):
        """前置小规模 pass，累计证据的逐通道均值（不挂替换钩子）。"""
        sums = torch.zeros(self.ev_ch, dtype=torch.float64)
        cnt = 0

        def make_hook(offset):
            def hook(mod, inp, out):
                nonlocal cnt
                o = out.detach().float()
                s = o.sum(dim=(0, 2, 3)).double().cpu()
                sums[offset:offset + o.shape[1]] += s
                if offset == 0:           # 只用 vario 那一路计元素数，避免重复计
                    cnt += o.shape[0] * o.shape[2] * o.shape[3]
            return hook

        hs = [self.net.gate_ev_vario.register_forward_hook(make_hook(0)),
              self.net.gate_ev_pol.register_forward_hook(make_hook(VARIO_CH))]
        try:
            for i, batch in enumerate(loader):
                if i >= n_scenes:
                    break
                inf_x, inf_y, cfv_masks = batch[0], batch[1], batch[2]
                _run_scene(inf_x, inf_y, cfv_masks, self.net, opts)
                print(f'  [mean] scene {i + 1}/{n_scenes}', flush=True)
        finally:
            for h in hs:
                h.remove()
        if cnt == 0:
            raise SystemExit('均值估计没有采到任何证据——检查该 config 是否开了证据门控')
        self.mean = (sums / cnt).float()
        print(f'  证据逐通道均值：vario |mean| 均值 {self.mean[:VARIO_CH].abs().mean():.4f}，'
              f'pol |mean| 均值 {self.mean[VARIO_CH:].abs().mean():.4f}')
        return self.mean

    # ---------------------------------------------------------------- 替换钩子
    def enable(self, streams, consumers, sub='mean'):
        self.disable()
        if streams is None or consumers is None:
            return
        sl = stream_slice(streams)
        if sub == 'mean':
            if self.mean is None:
                raise RuntimeError('先调 estimate_mean()')
            fill = self.mean[sl]
        else:
            fill = torch.zeros(sl.stop - sl.start)

        def sub_tensor(ev):
            """把 ev 的 [sl] 通道换成常数（逐通道）。ev: [B,C,H,W]"""
            v = fill.to(device=ev.device, dtype=ev.dtype).view(1, -1, 1, 1)
            out = ev.clone()
            out[:, sl] = v
            return out

        if consumers in ('gate', 'both') and self.head.scale_gate is not None:
            ev_ch = self.ev_ch

            def gate_hook(mod, args):
                gin = args[0]
                # gate_in = cat([x, evidence])：证据恒在末 ev_ch 个通道
                ev = gin[:, -ev_ch:]
                new = gin.clone()
                new[:, -ev_ch:] = sub_tensor(ev)
                return (new,) + tuple(args[1:])

            self._handles.append(
                self.head.scale_gate.register_forward_pre_hook(gate_hook))

        if consumers in ('xattn', 'both') and self.head.scale_xattn is not None:
            def xattn_hook(mod, args):
                if len(args) < 2 or args[1] is None:
                    return None
                return (args[0], sub_tensor(args[1])) + tuple(args[2:])

            self._handles.append(
                self.head.scale_xattn.register_forward_pre_hook(xattn_hook))

    def disable(self):
        for h in self._handles:
            h.remove()
        self._handles = []


def self_test():
    """hook 接线自检（无卡可跑、不碰数据/GPU）：验证替换只命中该命中的通道。

    造两个与真模块同接口的假模块：
      - scale_gate: 吃 cat([x, evidence]) 的单入张量（证据在末 ev_ch 个通道）
      - scale_xattn: 吃 (projected_list, evidence) 两个位置参数
    断言：被选中的通道逐个等于 mean、其余通道逐位不变、未启用的消费者完全不受影响。
    """
    import torch.nn as nn
    ev_ch = VARIO_CH + POL_CH
    S_E = 1024

    class FakeXAttn(nn.Module):
        def forward(self, projected, evidence=None):
            self.seen = None if evidence is None else evidence.clone()
            return projected

    class FakeHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.evidence_ch = ev_ch
            self.scale_gate = nn.Sequential(nn.Identity())
            self.scale_xattn = FakeXAttn()

        def forward(self, x, ev):
            self.gate_seen = self.scale_gate(torch.cat([x, ev], dim=1)).clone()
            self.scale_xattn([x], ev)
            return self.gate_seen

    class FakeNet(nn.Module):
        pass

    torch.manual_seed(0)
    head, net = FakeHead(), FakeNet()
    abl = EvidenceAblation(net, head, ev_ch)
    abl.mean = torch.arange(ev_ch, dtype=torch.float32)      # 每通道一个可辨认的常数

    x = torch.randn(2, S_E, 4, 4)
    ev = torch.randn(2, ev_ch, 4, 4)

    def check(streams, consumers, expect_gate, expect_xattn):
        abl.enable(streams, consumers)
        head(x, ev)
        abl.disable()
        g_ev = head.gate_seen[:, -ev_ch:]
        g_x = head.gate_seen[:, :S_E]
        assert torch.equal(g_x, x), f'{streams}/{consumers}: 主特征通道被误改'
        sl = stream_slice(streams) if streams else slice(0, 0)
        for name, seen, hit in (('gate', g_ev, expect_gate),
                                ('xattn', head.scale_xattn.seen, expect_xattn)):
            if hit:
                got = seen[:, sl]
                want = abl.mean[sl].view(1, -1, 1, 1).expand_as(got)
                assert torch.allclose(got, want), f'{streams}/{consumers}: {name} 替换值不对'
                keep = torch.ones(ev_ch, dtype=torch.bool)
                keep[sl] = False
                assert torch.equal(seen[:, keep], ev[:, keep]), \
                    f'{streams}/{consumers}: {name} 改到了不该改的通道'
            else:
                assert torch.equal(seen, ev), f'{streams}/{consumers}: {name} 不该被改却变了'
        print(f'  [ok] streams={streams!s:6s} consumers={consumers!s:6s}')

    print('证据消融 hook 自检：')
    check(None, None, False, False)
    check('all', 'both', True, True)
    check('vario', 'both', True, True)
    check('pol', 'both', True, True)
    check('all', 'gate', True, False)
    check('all', 'xattn', False, True)
    print('  通道布局确认：vario = [0:%d)，pol = [%d:%d)' % (VARIO_CH, VARIO_CH, ev_ch))
    print('SELF_TEST_OK')


def _run_scene(inf_x, inf_y, cfv_masks, net, opts):
    valid_maps = {c: ~cfv_masks[c] for c in opts['charts']}
    with torch.no_grad(), torch.cuda.amp.autocast():
        preds, _ = fast_tiled_val_predict(inf_x, net, opts, labels=inf_y,
                                          valid_maps=valid_maps, gate=True)
    return preds['FLOE']


def cm_iou(cm):
    inter = np.diag(cm)
    union = cm.sum(0) + cm.sum(1) - inter
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=CFG)
    ap.add_argument('--run', default=RUN)
    ap.add_argument('--ckpt', default=None)
    ap.add_argument('--val-list', default=VAL_LIST)
    ap.add_argument('--scenes', type=int, default=None, help='只跑前 N 景（冒烟用）')
    ap.add_argument('--mean-scenes', type=int, default=8, help='估均值用几景')
    ap.add_argument('--sub', default='mean', choices=['mean', 'zero'])
    ap.add_argument('--dump-cm', default='/root/autodl-tmp/cm/evabl/pgsg')
    ap.add_argument('--self-test', action='store_true',
                    help='只跑 hook 接线自检后退出（无卡可跑）')
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('⚠️ 没有 GPU：本脚本是为开卡准备的，无卡模式下会非常慢。')

    ckpt = resolve_ckpt(args.run, args.ckpt)
    print(f'config: {args.config}\nckpt  : {ckpt}')
    opts = get_variable_options(copy.deepcopy(Config.fromfile(args.config).train_options))
    assert opts['down_sample_scale'] == 1, '需原生分辨率口径'
    net = get_model(opts, device)
    sd = torch.load(ckpt, map_location=device, weights_only=False)['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in sd):
        sd = {k.replace('_orig_mod.', '', 1): v for k, v in sd.items()}
    net.load_state_dict(sd)
    net.eval()
    opts = val_scoped_options(opts)

    chart = opts['charts'][0]
    head = net.heads[chart]
    ev_ch = int(getattr(head, 'evidence_ch', 0))
    if ev_ch != VARIO_CH + POL_CH:
        raise SystemExit(f'证据通道数 {ev_ch} ≠ 预期 {VARIO_CH + POL_CH}，'
                         '脚本里的 VARIO_CH/POL_CH 需同步修改')
    print(f'evidence_ch={ev_ch}  scale_gate={"有" if head.scale_gate is not None else "无"}'
          f'  scale_xattn={"有" if head.scale_xattn is not None else "无"}')

    files = json.load(open(args.val_list, encoding='utf-8'))
    if args.scenes:
        files = files[:args.scenes]
    print(f'验证集：{len(files)} 景')

    def new_loader():
        ds = AI4ArcticChallengeTestDataset(options=opts, files=files, mode='train')
        return torch.utils.data.DataLoader(ds, batch_size=None, num_workers=0,
                                           shuffle=False)

    abl = EvidenceAblation(net, head, ev_ch)
    if args.sub == 'mean':
        print(f'\n[1/2] 估计证据逐通道均值（前 {args.mean_scenes} 景）…', flush=True)
        abl.estimate_mean(new_loader(), opts, args.mean_scenes, device)

    n_cls = opts['n_classes'][chart]
    results, scene_ids = {}, []
    global_cm = {tag: np.zeros((n_cls, n_cls)) for tag, _, _ in VARIANTS}
    scene_cms = {tag: [] for tag, _, _ in VARIANTS}

    print(f'\n[2/2] 跑 {len(VARIANTS)} 个变体 × {len(files)} 景…', flush=True)
    for si, batch in enumerate(new_loader()):
        inf_x, inf_y, cfv_masks, name = batch[0], batch[1], batch[2], batch[4]
        sid = osp.splitext(str(name))[0]
        gt = np.squeeze(np.asarray(inf_y['FLOE'])).astype(np.int64)
        m = gt != 255
        line = []
        for tag, streams, consumers in VARIANTS:
            abl.enable(streams, consumers, sub=args.sub)
            pred = _run_scene(inf_x, inf_y, cfv_masks, net, opts).numpy().astype(np.int64)
            abl.disable()
            cm = np.bincount(gt[m] * n_cls + pred[m],
                             minlength=n_cls * n_cls).reshape(n_cls, n_cls).astype(np.float64)
            global_cm[tag] += cm
            scene_cms[tag].append(cm)
            iou = cm_iou(cm)
            line.append('%s=%.3f' % (tag.split('/')[0].replace('sub-', ''),
                                     float(np.mean([iou[2], iou[3]])) if m.any() else np.nan))
            torch.cuda.empty_cache()
        scene_ids.append(sid)
        print('%-44s hard-ice %s' % (sid[:44], ' '.join(line)), flush=True)

    # ---------------------------------------------------------------- 汇总
    base_iou = cm_iou(global_cm['control'])
    base_hard = float(np.mean([base_iou[2], base_iou[3]]))
    base_macro = float(base_iou.mean())
    W = 104
    print('\n' + '=' * W)
    print('%-16s %8s %9s %9s %9s | %s' %
          ('variant', 'macro', 'Δmacro', 'hard-ice', 'Δhard',
           ' '.join('%6s' % c for c in CLASS_SHORT)))
    print('=' * W)
    for tag, _, _ in VARIANTS:
        iou = cm_iou(global_cm[tag])
        hard = float(np.mean([iou[2], iou[3]]))
        print('%-16s %8.4f %+9.4f %9.4f %+9.4f | %s' %
              (tag, iou.mean(), iou.mean() - base_macro, hard, hard - base_hard,
               ' '.join('%6.3f' % v for v in iou)))
        results[tag] = {'macro': float(iou.mean()), 'hard_ice': hard,
                        'per_class': [float(v) for v in iou]}
    print('=' * W)
    print('读法：')
    print('  Δ ≈ 0（在 run 间噪声 ~0.005 内）⇒ 训练好的模型推理时【已不依赖】该路证据，')
    print('      与 Fig. 6 的贡献比衰减互相印证，构成一个完整的机制性负结果；')
    print('  Δ 明显为负 ⇒ 贡献比低估了证据的作用，证据仍在携带关键信息。')
    print('  vario vs pol 的对比回答"两条物理先验哪条还在被用"——')
    print('  ⚠️ 仅限"推理期依赖"，不等于"从头不带那一路重训"的结论。')

    if args.dump_cm:
        os.makedirs(osp.dirname(args.dump_cm), exist_ok=True)
        np.savez_compressed(
            args.dump_cm + '.npz',
            scenes=np.array(scene_ids),
            **{f'cm_{t.replace("/", "_")}': np.stack(scene_cms[t]) for t, _, _ in VARIANTS},
            sub=np.array(args.sub), config=np.array(args.config),
            checkpoint=np.array(ckpt))
        print(f'\nper-scene CM -> {args.dump_cm}.npz')
    print('\nEVIDENCE_ABLATION_DONE')


if __name__ == '__main__':
    main()
