"""Fig. 6 + T0-0.3：从训练日志解析组件读数，出图 + CSV。无卡模式安全（纯文本解析）。

为什么需要这张图
----------------
Introduction 把"可判读的组件级归因"列为第三条贡献（所有模块零初始化 + 训练期读数，
使"模块是否被使用"与"模块是否有效"可以分开判定）。但正文目前一处读数都没展示——
贡献列表里写了、正文没兑现，是评审一眼能看穿的空头支票。

支持两种模式
------------
1) **单配置**（给一个 --logs）：3 栏 = 门控预算 / 选择熵+证据贡献 / 跨尺度通信。
2) **双配置对比**（给两个 --logs，推荐）：4 栏 = (b) 门控预算 / (c) 门控预算 /
   两者的选择熵+证据贡献 / (c) 的跨尺度通信。
   ⚠️ (b) 没有尺度轴 transformer ⇒ 第 4 栏只有 (c) 一条线，这是事实不是缺数据。

解析的两类日志行（由 models/sota/segformer.py 打印，触发点是第 1、100、1000、10000 次
前向，之后每 20000 次一次 ⇒ 横轴天然是对数刻度、点很稀疏，这是设计如此）：

    [SegformerHead] gate #N (train): stage-mean g = [a, b, c, d] | 选择熵 E/1.386 nats…
        | |a|=X T=1 | 证据/特征激活贡献比 R（…）| conv1 权重 RMS f … / e …
    [ScaleXAttn] #N (train): ‖resid‖/‖feat‖ = R（…）| 尺度→证据注意力份额 S（均匀基线 0.200）

用法
----
    # 双配置（论文用）
    python -m tools.figures.fig6_readouts \\
        --logs /root/autodl-tmp/logs/floe_evidence_gate_417_s42.log \\
               /root/autodl-tmp/logs/floe_pgsg_417_s42.log \\
        --labels "(b) + evidence gate" "(c) PGSG (full)"
    # 单配置
    python -m tools.figures.fig6_readouts --logs <log>
"""
import argparse
import csv
import glob
import os
import os.path as osp
import re

from tools.figures.common import *          # noqa: F401,F403

GATE_RE = re.compile(
    r'\[SegformerHead\]\s*gate\s*#(\d+)\s*\((train|val)\):\s*stage-mean g\s*=\s*\[([^\]]+)\]')
ENT_RE = re.compile(r'选择熵\s*([0-9.]+)\s*/\s*([0-9.]+)')
EVC_RE = re.compile(r'证据/特征激活贡献比\s*([0-9.]+)')
XAT_RE = re.compile(
    r'\[ScaleXAttn\]\s*#(\d+)\s*\((train|val)\):.*?=\s*([0-9.]+)')
EVS_RE = re.compile(r'注意力份额\s*([0-9.]+).*?基线\s*([0-9.]+)')

SEARCH = ['/root/autodl-tmp/logs/*.log', '/root/autodl-tmp/*.log',
          '/root/code/FLOE-Git/*.log', '/root/*.log']

# 两个配置的固定配色（与论文消融行名对应；(b) 用暖色、(c) 用主色）
CFG_COLORS = ['#C0762B', '#1F4E79']


def find_log():
    cands = []
    for p in SEARCH:
        cands += glob.glob(p)
    if not cands:
        raise SystemExit('找不到日志，请用 --logs 指定路径')
    cands.sort(key=osp.getmtime, reverse=True)
    print('候选日志（按修改时间）：')
    for c in cands[:8]:
        print('   ', c)
    print(f'使用最近的一个：{cands[0]}\n')
    return [cands[0]]


def parse(path):
    gate, xatt = [], []
    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            m = GATE_RE.search(line)
            if m:
                g = [float(x) for x in m.group(3).split(',')]
                ent = ENT_RE.search(line)
                evc = EVC_RE.search(line)
                gate.append({
                    'call': int(m.group(1)), 'phase': m.group(2), 'g': g,
                    'entropy': float(ent.group(1)) if ent else float('nan'),
                    'entropy_max': float(ent.group(2)) if ent else float('nan'),
                    'ev_contrib': float(evc.group(1)) if evc else float('nan'),
                })
                continue
            m = XAT_RE.search(line)
            if m:
                evs = EVS_RE.search(line)
                xatt.append({
                    'call': int(m.group(1)), 'phase': m.group(2),
                    'resid_ratio': float(m.group(3)),
                    'ev_share': float(evs.group(1)) if evs else float('nan'),
                    'ev_share_uniform': float(evs.group(2)) if evs else float('nan'),
                })
    return gate, xatt


def sel(rows, split):
    return [r for r in rows if split == 'both' or r['phase'] == split]


def panel_gate_budget(ax, rows, title):
    call = np.array([r['call'] for r in rows], dtype=float)
    G = np.array([r['g'] for r in rows])
    for s in range(G.shape[1]):
        ax.plot(call, G[:, s], 'o-', ms=3, lw=1,
                color=plt.cm.viridis(s / max(G.shape[1] - 1, 1)),
                label=f'stride {4 * 2 ** s}')
    ax.axhline(1.0, color='#999999', lw=0.6, ls=':')
    ax.set_xscale('log')
    ax.set_xlabel('forward passes')
    ax.set_ylabel('mean gate value $g_s$')
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=5.8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logs', nargs='+', default=None)
    ap.add_argument('--labels', nargs='+', default=None)
    ap.add_argument('--split', default='train', choices=['train', 'val', 'both'])
    ap.add_argument('--out', default='fig6_readouts')
    ap.add_argument('--csv', default=None)
    args = ap.parse_args()

    logs = args.logs or find_log()
    labels = args.labels or [osp.basename(p).replace('.log', '') for p in logs]
    if len(labels) != len(logs):
        raise SystemExit(f'--labels 数量({len(labels)}) 与 --logs({len(logs)}) 不符')

    cfgs = []
    for path, lab in zip(logs, labels):
        gate, xatt = parse(path)
        print(f'{lab}: {osp.basename(path)} -> {len(gate)} gate 行, {len(xatt)} ScaleXAttn 行')
        if not gate:
            raise SystemExit(f'{path} 里没有 gate 读数行——确认这是 PGSG 系的训练日志。')
        g_rows = sel(gate, args.split) or gate
        cfgs.append({'label': lab, 'path': path, 'gate_all': gate, 'xatt_all': xatt,
                     'gate': g_rows, 'xatt': sel(xatt, args.split)})

    # ---------------------------------------------------------------- CSV
    os.makedirs(FIG_DIR, exist_ok=True)
    csv_path = args.csv or osp.join(FIG_DIR, f'{args.out}.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        for c in cfgs:
            w.writerow([f'# {c["label"]}  ({c["path"]})'])
            w.writerow(['call', 'phase', 'g1', 'g2', 'g3', 'g4', 'entropy',
                        'entropy_max', 'evidence_contrib'])
            for r in c['gate_all']:
                w.writerow([r['call'], r['phase']] + [f'{v:.4f}' for v in r['g']] +
                           [f'{r["entropy"]:.4f}', f'{r["entropy_max"]:.4f}',
                            f'{r["ev_contrib"]:.4f}'])
            if c['xatt_all']:
                w.writerow(['call', 'phase', 'resid_over_feat', 'ev_attn_share', 'uniform'])
                for r in c['xatt_all']:
                    w.writerow([r['call'], r['phase'], f'{r["resid_ratio"]:.4f}',
                                f'{r["ev_share"]:.4f}', f'{r["ev_share_uniform"]:.4f}'])
            w.writerow([])
    print(f'  wrote {csv_path}')

    # ---------------------------------------------------------------- 打印首末值
    print('\n=== 首末读数（论文正文/图注可直接引用）===')
    for c in cfgs:
        a, b = c['gate'][0], c['gate'][-1]
        print(f'-- {c["label"]}')
        print(f'   gate stage-mean g : {["%.2f" % v for v in a["g"]]} (call {a["call"]})'
              f'  ->  {["%.2f" % v for v in b["g"]]} (call {b["call"]})')
        print(f'   selection entropy : {a["entropy"]:.3f} -> {b["entropy"]:.3f}'
              f'  (uniform upper bound {b["entropy_max"]:.3f} nats)')
        print(f'   evidence contrib  : {a["ev_contrib"]:.3f} -> {b["ev_contrib"]:.3f}')
        if c['xatt']:
            xa, xb = c['xatt'][0], c['xatt'][-1]
            print(f'   |resid|/|feat|    : {xa["resid_ratio"]:.3f} -> {xb["resid_ratio"]:.3f}')
            print(f'   evidence attn     : {xa["ev_share"]:.3f} -> {xb["ev_share"]:.3f}'
                  f'  (uniform {xb["ev_share_uniform"]:.3f})')
        else:
            print('   |resid|/|feat|    : —（该配置没有尺度轴 transformer）')

    # ---------------------------------------------------------------- 画图
    multi = len(cfgs) > 1
    n_panel = (2 + len(cfgs)) if multi else (2 + (1 if cfgs[0]['xatt'] else 0))
    fig, axes = plt.subplots(1, n_panel, figsize=(1.95 * n_panel + 0.5, 2.45))
    axes = np.atleast_1d(axes)
    letters = 'abcdefgh'
    k = 0

    for c in cfgs:
        panel_gate_budget(axes[k], c['gate'],
                          f'({letters[k]}) gate budget\n{c["label"]}'
                          if multi else f'({letters[k]}) gate budget')
        k += 1

    # 选择熵（左轴）+ 证据贡献比（右轴），多配置时同轴叠放
    ax = axes[k]
    ax2 = ax.twinx()
    for i, c in enumerate(cfgs):
        call = np.array([r['call'] for r in c['gate']], dtype=float)
        col = CFG_COLORS[i % len(CFG_COLORS)] if multi else '#333333'
        ax.plot(call, [r['entropy'] for r in c['gate']], 'o-', ms=3, lw=1,
                color=col, label=c['label'] if multi else 'selection entropy')
        ax2.plot(call, [r['ev_contrib'] for r in c['gate']], 's--', ms=3, lw=1,
                 color=col, alpha=0.65)
    emax = cfgs[0]['gate'][-1]['entropy_max']
    if np.isfinite(emax):
        ax.axhline(emax, color='#999999', lw=0.8, ls='--')
        ax.text(0.97, emax, ' uniform bound', fontsize=5.8, va='bottom', ha='right',
                transform=ax.get_yaxis_transform())
    ax.set_xscale('log')
    ax.set_xlabel('forward passes')
    ax.set_ylabel('selection entropy (nats)')
    ax2.set_ylabel('evidence contribution')
    ax.set_title(f'({letters[k]}) gate: entropy (solid)\nand evidence share (dashed)')
    if multi:
        ax.legend(frameon=False, fontsize=5.8, loc='lower right')
    k += 1

    # 跨尺度通信（只有含 scale_xattn 的配置有）
    ax = axes[k]
    drawn = False
    for i, c in enumerate(cfgs):
        if not c['xatt']:
            continue
        xc = np.array([r['call'] for r in c['xatt']], dtype=float)
        col = CFG_COLORS[i % len(CFG_COLORS)] if multi else '#2c7fb8'
        ax.plot(xc, [r['resid_ratio'] for r in c['xatt']], 'o-', ms=3, lw=1,
                color=col, label=r'$\|$resid$\|/\|$feat$\|$')
        ax.plot(xc, [r['ev_share'] for r in c['xatt']], 's--', ms=3, lw=1,
                color=col, alpha=0.55, label='evidence attention')
        u = c['xatt'][-1]['ev_share_uniform']
        if np.isfinite(u):
            ax.axhline(u, color='#999999', lw=0.6, ls=':')
        drawn = True
    ax.set_xscale('log')
    ax.set_xlabel('forward passes')
    ax.set_title(f'({letters[k]}) cross-scale\ncommunication')
    if drawn:
        ax.legend(frameon=False, fontsize=5.8)
    else:
        ax.text(0.5, 0.5, 'no scale-axis\ntransformer', ha='center', va='center',
                fontsize=7, color='#888888', transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    savefig(fig, args.out)


if __name__ == '__main__':
    main()
