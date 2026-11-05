#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""PGSG-v4（v2 + 尺度轴 transformer，无 PBA）结构自检 —— CPU 可跑。

  [A] 单变量机器校验：v4 相对 v2 的 config 差异只在 segformer_scale_xattn；
      state_dict 新增键恰为 heads.*.scale_xattn.* 前缀；**共有键逐位相同**
      （构造顺序保证：scale_xattn 在 gate_ev 之后构造 ⇒ 不平移共有权重 RNG 流）；
      新参数全部 rest(满 LR) 组；无 pba_projs 键（PBA 确实被拆掉）。
  [B] 初始严格等价：同 seed 下 v4 与 v2 eval logits 逐位一致（xattn 零 out_proj）。
  [C] 接线有效：xattn out_proj 置非零 → logits 变。
  [D] 梯度通路：置非零后 scale_xattn 全部参数有非零梯度。
"""
import sys

import torch
from mmcv import Config

from floe.loaders import get_variable_options, limit_torch_threads
from floe.models.sota.segformer import SegFormer

V2_CFG = 'configs/FLOE/pgsg_evidence_gate.py'
V4_CFG = 'configs/FLOE/pgsg.py'


def _load_opts(p):
    o = get_variable_options(Config.fromfile(p).train_options)
    o['segformer_encoder_weights'] = None
    return o


def _build(o, seed=0):
    torch.manual_seed(seed)
    return SegFormer(options=o)


def main():
    limit_torch_threads()
    o2, o4 = _load_opts(V2_CFG), _load_opts(V4_CFG)
    diff = {k for k in set(o2) | set(o4)
            if o2.get(k, '<absent>') != o4.get(k, '<absent>')}
    assert diff == {'segformer_scale_xattn'}, f'v4 非单变量差异：{sorted(diff)}'
    n2, n4 = _build(o2), _build(o4)
    s2, s4 = set(n2.state_dict()), set(n4.state_dict())
    assert not any(k.startswith('pba_projs.') for k in s4), 'v4 不应含 PBA 键'
    assert n4.pba_projs is None, 'v4 的 pba_projs 应为 None（PBA 已拆）'
    extra = s4 - s2
    assert extra and all('.scale_xattn.' in k for k in extra), \
        f'新增键出现在 scale_xattn 之外：{sorted(k for k in extra if ".scale_xattn." not in k)[:5]}'
    assert not (s2 - s4), f'v4 丢失 v2 键：{sorted(s2 - s4)[:5]}'
    d2, d4 = n2.state_dict(), n4.state_dict()
    bad_val = [k for k in s2 if not torch.equal(d2[k], d4[k])]
    assert not bad_val, f'共有键数值不同（RNG 流被平移 = 隐性换 seed）：{bad_val[:5]}'
    for name, p in n4.named_parameters():
        if '.scale_xattn.' in name:
            assert not (name.startswith('encoder.') or '.encoder.' in name), \
                f'新参数落进 encoder 组：{name}'
    xa = n4.heads['FLOE'].scale_xattn
    assert xa is not None and xa.out_proj.weight.abs().max().item() == 0, \
        'scale_xattn.out_proj 未零初始化'
    print('[A] PASS：单变量 + 新增键仅 scale_xattn + 共有键逐位相同 + 无 PBA + '
          'rest 组 + 零初始化', flush=True)

    in_ch = next(m for m in n4.encoder.modules()
                 if isinstance(m, torch.nn.Conv2d)).in_channels
    torch.manual_seed(123)
    x = torch.randn(2, in_ch, 256, 256)
    n2.eval(); n4.eval()
    with torch.no_grad():
        assert torch.equal(n2(x)['FLOE'], n4(x)['FLOE']), \
            '初始严格等价被破坏：v4 与 v2 logits 不一致'
    print('[B] PASS：同 seed eval logits 与 v2 逐位一致', flush=True)

    with torch.no_grad():
        base = n4(x)['FLOE']
        torch.nn.init.normal_(xa.out_proj.weight, std=0.05)
        assert not torch.allclose(base, n4(x)['FLOE']), \
            'xattn 置非零后 logits 未变：未接线'
    print('[C] PASS：scale_xattn 接线有效', flush=True)

    n4b = _build(o4)
    n4b.train()
    torch.nn.init.normal_(n4b.heads['FLOE'].scale_xattn.out_proj.weight, std=0.05)
    n4b(x)['FLOE'].square().mean().backward()
    for name, p in n4b.named_parameters():
        if '.scale_xattn.' in name:
            assert p.grad is not None and p.grad.abs().max().item() > 0, \
                f'{name} 梯度为零'
    print('[D] PASS：scale_xattn 全部参数有梯度', flush=True)
    print('\nALL PASS — PGSG-v4 结构自检通过。GPU 端到端门禁：\n'
          '  python -m tools.checks.smoke_pipeline configs/FLOE/pgsg.py',
          flush=True)


if __name__ == '__main__':
    main()
    sys.exit(0)
