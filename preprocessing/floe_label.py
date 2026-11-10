"""
FLOE 标注共享模块——单一阈值源 + 单一分类逻辑。
预筛(prescreen)、下载过滤(step2)、标注(step4) 都从这里取，保证口径完全一致。

类别口径(2026-07 起)：类 1 = Cake(<20m，SIGRID 码 2 Ice cake + 码 1 Shuga/brash 合并，
见 utils.FLOE_LOOKUP)；Small(2)/Medium(3) 照旧。全量 DMI 扫描与阈值论证见
FLOE\DMI\shuga_cake_scan\REPORT_shuga_cake_scan.md。
"""
import os
import numpy as np
import label_definitions as utils

NF = utils.ICECHART_NOT_FILLED_VALUE  # -9

# ===== 唯一阈值源：为收集 Small/Medium 已放宽 2、3 类(其余保持高纯度) =====
# (min_ct, min_ratio)：总浓度下限(0-10) 与 主导浮冰占比下限。
DYNAMIC_THRESHOLDS_FLOE = {
    1: (6, 0.65),   # Cake(<20m, 码1+码2 合并) —— 与 2/3 同口径(用户 2026-07 定)；DMI 中该档≈658 多边形/210 图-日
    2: (6, 0.65),   # Small floe —— 放宽：CT>=60% 且占比>=0.65
    3: (6, 0.65),   # Medium floe —— 放宽
    4: (9, 0.9),    # Big
    5: (9, 0.9),    # Vast
    6: (9, 0.9),    # Bergs
}


def _safe_int(value, default=NF):
    try:
        import pandas as pd
        if pd.isna(value):
            return default
    except Exception:
        if value is None:
            return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _conv(code, lookup):
    return lookup.get(code, lookup['mask'])


def _get(row, name):
    return _safe_int(row[name]) if name in row.index else NF


def classify_polygon_floe(row, thresholds=None):
    """
    对单个冰图多边形(行)返回 FLOE 最终类别：0(水)/1..6/255(mask)，或 'land'。
    与 make_labels.py 的逐多边形逻辑完全一致。
    """
    if thresholds is None:
        thresholds = DYNAMIC_THRESHOLDS_FLOE

    pt = str(row['POLY_TYPE']).strip().lower() if 'POLY_TYPE' in row.index else ''
    if pt == 'l':
        return 'land'

    ct = _get(row, 'CT'); ca = _get(row, 'CA'); cb = _get(row, 'CB'); cc = _get(row, 'CC')
    fa = _get(row, 'FA'); fb = _get(row, 'FB'); fc = _get(row, 'FC')

    ctc = _conv(ct, utils.SIC_LOOKUP)
    cac = _conv(ca, utils.SIC_LOOKUP) if ca != NF else ca
    cbc = _conv(cb, utils.SIC_LOOKUP) if cb != NF else cb
    ccc = _conv(cc, utils.SIC_LOOKUP) if cc != NF else cc
    fac = _conv(fa, utils.FLOE_LOOKUP) if fa != NF else fa
    fbc = _conv(fb, utils.FLOE_LOOKUP) if fb != NF else fb
    fcc = _conv(fc, utils.FLOE_LOOKUP) if fc != NF else fc

    if ctc > utils.SIC_LOOKUP[0] and cac == NF:
        cac = ctc

    psic = np.array([cac, cbc, ccc], dtype=np.int32)
    pfloe = np.array([fac, fbc, fcc], dtype=np.int32)
    sic_class = utils.SIC_LOOKUP[0] if pt == 'w' else ctc

    weight = {}
    for i in range(3):
        f = int(pfloe[i]); w = int(psic[i])
        if f in (NF, utils.FLOE_LOOKUP['mask']):
            continue
        if w in (NF, utils.SIC_LOOKUP['mask']) or w <= 0:
            continue
        weight[f] = weight.get(f, 0) + w

    floe = utils.FLOE_LOOKUP['invalid']
    if ctc > 0:
        if weight:
            cand, cw = max(weight.items(), key=lambda it: (it[1], -it[0]))
            ratio = np.divide(cw, ctc)
            if np.isfinite(ratio):
                min_ct, min_ratio = thresholds.get(cand, (9, 0.9))
                if ctc >= min_ct and ratio >= min_ratio:
                    floe = cand
        if np.any(pfloe == utils.FLOE_LOOKUP['fastice_class']):
            floe = utils.FLOE_LOOKUP['fastice_class']

    if floe == utils.FLOE_LOOKUP['invalid']:
        floe = utils.FLOE_LOOKUP['mask']
    if sic_class == utils.SIC_LOOKUP[0]:
        floe = utils.FLOE_LOOKUP['water']
    return int(floe)


def load_chart_gdf(shp_path, target_crs=None):
    """读取冰图 shapefile，统一大写列名、去重、设好 geometry；可选重投影。"""
    import geopandas as gpd
    g = gpd.read_file(shp_path)
    g.columns = g.columns.str.strip().str.upper()
    if not g.columns.is_unique:
        g = g.loc[:, ~g.columns.duplicated()]
    gcol = 'GEOMETRY' if 'GEOMETRY' in g.columns else 'geometry'
    g = g.set_geometry(gcol)
    if target_crs is not None and g.crs is not None:
        g = g.to_crs(target_crs)
    return g, gcol


def ice_2_3_union(gdf, geom_col, thresholds=None, classes=(2, 3)):
    """返回 gdf 中 FLOE 判为 classes 之一的多边形并集(几何在 gdf 当前 CRS)；无则 None。

    历史名沿用(默认 2/3)；cake 采集等场景传 classes=(1,) 即可复用。
    """
    import shapely.ops as ops
    geoms = []
    for _, r in gdf.iterrows():
        if classify_polygon_floe(r, thresholds) in classes:
            g = r[geom_col]
            if g is not None and not g.is_empty:
                geoms.append(g)
    if not geoms:
        return None
    u = ops.unary_union(geoms)
    return u if u.is_valid else u.buffer(0)


def _classes_from_env(name, default):
    """从 env 读逗号分隔的类别元组(如 FLOE_KEEP_CLASSES="0,1,2,3,4,5")；未设/非法时用默认。"""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return tuple(sorted({int(x) for x in raw.split(",") if x.strip() != ""}))
    except ValueError:
        print(f"[floe_label] env {name}={raw!r} 非法，回退默认 {default}")
        return default


# 写入 NC 的保留类(其余→255 mask)与"空景检查"的目标类。
# 默认保留 0–5 全六类，目标冰类为 1–5；难类子集可用 env 显式选择：
#   FLOE_KEEP_CLASSES="0,1,2,3"  FLOE_TARGET_CLASSES="1,2,3"
KEEP_CLASSES = _classes_from_env("FLOE_KEEP_CLASSES", (0, 1, 2, 3, 4, 5))
TARGET_CLASSES = _classes_from_env("FLOE_TARGET_CLASSES", (1, 2, 3, 4, 5))
