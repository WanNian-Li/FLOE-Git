"""论文绘图/统计脚本的公共设置（无卡模式安全：CPU、单进程、低内存）。

⚠️ 无卡模式只有 0.5 核 / 2GB 内存。本模块及所有 `_paper_*` 脚本遵守：
  - 只用 h5py 切片读，**绝不整景解码**（一景 float32 约 3GB，当场 OOM）；
  - 不开多进程；matplotlib 强制 Agg 后端；
  - 输出同时落 PDF（LaTeX 用矢量图）与 PNG（预览/scp 回本地看）。

用法：各脚本 `from _paper_figs_common import *`。
"""
import os
import os.path as osp

import matplotlib
matplotlib.use('Agg')                      # 无显示环境必须
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------- 路径
DATA_DIR = os.environ.get('FLOE_DATA_DIR', '/root/autodl-tmp/FLOE_datanc')
TILE_INDEX = os.environ.get('FLOE_TILE_INDEX',
                            '/root/autodl-tmp/floe_tile_index_417.npz')
CM_DIR = os.environ.get('FLOE_CM_DIR', '/root/autodl-tmp/cm/single')
FIG_DIR = os.environ.get('FLOE_FIG_DIR', '/root/autodl-tmp/figs')

# ---------------------------------------------------------------- 类别
N_CLASSES = 6
CLASS_NAMES = ['Open water', 'Cake ice', 'Small floe',
               'Medium floe', 'Big floe', 'Vast floe']
CLASS_SHORT = ['Water', 'Cake', 'Small', 'Medium', 'Big', 'Vast']
CLASS_RANGE = ['', '$<$20 m', '20--100 m', '100--500 m', '0.5--2 km', '2--10 km']
# 类别配色：水=深蓝，冰从小到大由浅到深的暖色（对色盲相对友好，打印成灰度也有梯度）
CLASS_COLORS = ['#1f4e79', '#ffe08a', '#f4a259', '#e76f51', '#a4508b', '#5c3c92']
IGNORE_COLOR = '#d9d9d9'

PIXEL_M = 80.0                              # 像元边长（米）

# ---------------------------------------------------------------- 样式
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 8,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
    'axes.grid': False,
    'pdf.fonttype': 42,                     # TrueType，期刊排版友好
    'ps.fonttype': 42,
})


def savefig(fig, name):
    """同时存 PDF + PNG 到 FIG_DIR，并打印路径。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ('pdf', 'png'):
        p = osp.join(FIG_DIR, f'{name}.{ext}')
        fig.savefig(p)
        print(f'  wrote {p}')
    plt.close(fig)


def load_tile_index(path=TILE_INDEX):
    """载入 tile 索引。字段见 scan_tiles.py：
    scene_id / row / col / valid_ratio / label_valid_ratio / hist (n,6) / hist_w / files.
    """
    d = np.load(path, allow_pickle=True)
    return {
        'scene_id': d['scene_id'],
        'row': d['row'],
        'col': d['col'],
        'valid_ratio': d['valid_ratio'],
        'label_valid_ratio': d['label_valid_ratio'],
        'hist': d['hist'].astype(np.int64),
        'files': [str(x) for x in d['files']],
        'tile_size': int(d['tile_size']),
    }


def read_tile(scene_file, row, col, size, variables):
    """h5py 只读一个对齐 tile 的若干变量。返回 dict[var] -> (size, size) 数组。

    ⚠️ 这是无卡模式下唯一允许的读法：每变量一个 chunk，约 1MB。
    """
    import h5py
    out = {}
    with h5py.File(osp.join(DATA_DIR, scene_file), 'r') as f:
        for v in variables:
            out[v] = f[v][row:row + size, col:col + size]
    return out


def label_to_rgb(lab):
    """标签图 -> RGB，>=6 的值（冰山 6 与 mask 255）统一画成 ignore 灰。"""
    from matplotlib.colors import to_rgb
    h, w = lab.shape
    rgb = np.empty((h, w, 3), dtype=np.float32)
    rgb[...] = to_rgb(IGNORE_COLOR)
    for c in range(N_CLASSES):
        m = (lab == c)
        if m.any():
            rgb[m] = to_rgb(CLASS_COLORS[c])
    return rgb


def stretch(img, valid=None, lo=2, hi=98):
    """按百分位拉伸到 [0,1] 作灰度显示。valid 为 True 的像元参与统计。"""
    a = img.astype(np.float32)
    s = a[valid] if valid is not None and valid.any() else a
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.zeros_like(a)
    v0, v1 = np.percentile(s, [lo, hi])
    if v1 <= v0:
        v1 = v0 + 1e-6
    return np.clip((a - v0) / (v1 - v0), 0, 1)


def resolve_ckpt(run_name, explicit=None):
    """按实验名找 best checkpoint。

    项目里两种写法都出现过——CLAUDE.md 记的是 `work_dirs/<名>/best_model_<名>.pth`，
    而 pgsg 系 config 头部的用法示例写的是 `work_dir/<名>`。这里把候选都试一遍，
    免得开卡后因为一个目录名少个 s 而白跑一趟。
    """
    import glob
    if explicit:
        if osp.exists(explicit):
            return explicit
        print(f'  [resolve_ckpt] 指定路径不存在，回退到自动查找：{explicit}')
    pats = [f'work_dirs/{run_name}/best_model_{run_name}.pth',
            f'work_dir/{run_name}/best_model_{run_name}.pth',
            f'work_dirs/{run_name}/best_model_*.pth',
            f'work_dir/{run_name}/best_model_*.pth',
            f'/root/autodl-tmp/work_dirs/{run_name}/best_model_*.pth',
            f'**/best_model_{run_name}.pth']
    for p in pats:
        hits = sorted(glob.glob(p, recursive=True))
        if hits:
            if len(hits) > 1:
                print(f'  [resolve_ckpt] {run_name}: 命中多个，取第一个 -> {hits[0]}')
            return hits[0]
    raise SystemExit(f'找不到 {run_name} 的 checkpoint；用 --ckpt-* 显式指定。'
                     f'\n  试过：{pats}')


def iou_from_cm(cm):
    """cm: (C, C) 行=真实、列=预测。返回逐类 IoU（该类在真实与预测中都不出现时记 0）。"""
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denom = cm.sum(axis=0) + cm.sum(axis=1) - tp
    return np.divide(tp, denom, out=np.zeros_like(tp), where=denom > 0)
