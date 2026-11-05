"""动机 B 的最后一个数：暗景里真中浮冰被判成小浮冰的比例。纯 CPU，只读已有文件。

做法
----
① 从 `docs/_radiometric_profile_417.csv` 取每景的 HH 均值（每景 4 tile 抽样），按它把
   fold0 验证集的 83 景分成"暗端"与"其余"；
② 从辐射探针 `id` 变体的逐景混淆矩阵取每景的 medium 行，算行归一化的 medium→small 比例；
③ 分组汇总（按像元加权，不是按景平均——景的支撑差几个数量级）。

⚠️ 这是**观察性**分组，不是干预：暗景同时还可能更晚、更靠北、传感器不同。论文里要写成
"in the radiometrically darkest scenes …"，**不能**写成"变暗导致了这个比例"——因果那一步
由 Fig. 2 的受控扰动来承担。

用法
----
    python -m tools.analysis.darkscenes
    python -m tools.analysis.darkscenes --quantile 0.25     # 暗端取多少比例的景
"""
import argparse
import csv
import os.path as osp

from tools.figures.common import *          # noqa: F401,F403

MEDIUM, SMALL = 3, 2


def load_profile(path):
    out = {}
    with open(path, encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[osp.splitext(osp.basename(r['scene']))[0]] = float(r['hh_mean'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--npz', default=osp.join(osp.join(CM_DIR, 'radio'),
                                              'baseline_id.npz'))
    ap.add_argument('--profile', default='docs/_radiometric_profile_417.csv')
    ap.add_argument('--quantile', type=float, default=0.25)
    args = ap.parse_args()

    prof = load_profile(args.profile)
    d = np.load(args.npz, allow_pickle=True)
    scenes = [osp.splitext(str(s))[0] for s in d['scenes']]
    cms = d['cm'].astype(np.float64)
    print(f'CM: {args.npz}  ({len(scenes)} scenes)')

    hh = np.array([prof.get(s, np.nan) for s in scenes])
    miss = [s for s, v in zip(scenes, hh) if not np.isfinite(v)]
    if miss:
        print(f'⚠️ {len(miss)} 个场景在侧写 CSV 里找不到，已排除：{miss[:3]}')
    ok = np.isfinite(hh)
    thr = np.quantile(hh[ok], args.quantile)
    print(f'HH 均值：min {hh[ok].min():.2f} / p25 {np.quantile(hh[ok], .25):.2f} / '
          f'median {np.median(hh[ok]):.2f} / max {hh[ok].max():.2f}')
    print(f'暗端阈值（{args.quantile:.0%} 分位）：HH mean < {thr:.2f}\n')

    groups = {'darkest': ok & (hh < thr), 'rest': ok & (hh >= thr)}
    print(f'{"group":10s}{"scenes":>7s}{"medium Mpx":>12s}'
          f'{"->small %":>11s}{"->water %":>11s}{"recall %":>10s}')
    res = {}
    for name, m in groups.items():
        rows = cms[m][:, MEDIUM, :].sum(axis=0)      # 该组所有景的 true-medium 行相加
        tot = rows.sum()
        if tot == 0:
            print(f'{name:10s}{int(m.sum()):7d}   (无 medium 像元)')
            continue
        res[name] = (100 * rows[SMALL] / tot, 100 * rows[0] / tot,
                     100 * rows[MEDIUM] / tot, tot)
        print(f'{name:10s}{int(m.sum()):7d}{tot / 1e6:12.1f}'
              f'{res[name][0]:11.1f}{res[name][1]:11.1f}{res[name][2]:10.1f}')

    if len(res) == 2:
        print(f'\n>>> 论文用数：在辐射最暗的 {args.quantile:.0%} 场景里，'
              f'{res["darkest"][0]:.0f}% 的真中浮冰像元被判成小浮冰；'
              f'其余场景为 {res["rest"][0]:.0f}%。')
        print('    （中浮冰召回：暗端 '
              f'{res["darkest"][2]:.0f}% vs 其余 {res["rest"][2]:.0f}%）')

    print('\n逐景明细（按 HH 均值升序，只列暗端）：')
    order = np.argsort(np.where(ok, hh, np.inf))
    for i in order[:int(ok.sum() * args.quantile)]:
        row = cms[i, MEDIUM, :]
        t = row.sum()
        if t < 1e5:
            continue
        print(f'  HH {hh[i]:+.2f}  medium {t / 1e6:6.2f} Mpx  '
              f'->small {100 * row[SMALL] / t:5.1f}%  '
              f'->water {100 * row[0] / t:5.1f}%  '
              f'recall {100 * row[MEDIUM] / t:5.1f}%   {scenes[i][:44]}')


if __name__ == '__main__':
    main()
