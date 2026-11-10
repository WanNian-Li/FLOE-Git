"""
预筛脚本（不需要 SAR / 不生成数据）：扫描多年/多区域 CIS 冰图，
只保留"在目标区域 AOI 内、且 Small/Medium(2-3) 面积达标、且落在破碎季"的冰图日期，
输出白名单(floe_whitelist.json / .csv) 供第 2 步只下载这些日期的 SAR。

支持你后续补充近 5 年冰图：把它们的文件夹都放进 CHART_ROOTS 即可（按目录名 CEXPR<区域> 自动识别区域）。
"""
import os
from _settings import DATA_ROOT
import re
import csv
import glob
import json
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import geopandas as gpd
from shapely import wkt as shapely_wkt

import floe_label as fl

# ======================= 配置 =======================
# 冰图根目录（可多个；递归找 *.shp）。补充多年数据时往这里加目录。
CHART_ROOTS = [
    str(DATA_ROOT / "CIS"),
    str(DATA_ROOT / "CIS" / "Ice_chart"),
]

# 只针对这些区域做 2-3 类采集（East Coast 是唯一富集区；如需 Medium 可加 'HB'）
TARGET_REGIONS = ["EC"]

# 每区域 AOI（与第 2 步检索框一致）。East Coast 用方案 B 近岸框。
REGION_AOIS = {
    # East Coast 扩大 AOI（方案C 甜点，与 the external SAR acquisition workflow 的 aoi_wkt 保持一致）
    "EC": "POLYGON((-70 46.5,-51.5 46.5,-51.5 57.7,-70 57.7,-70 46.5))",
    "HB": "POLYGON((-95 55,-77 55,-77 64,-95 64,-95 55))",
}
# 每区域采集季节（月份集合）：EC 破碎期 2-5 月；HB 4-6 月 + 12 月
REGION_SEASONS = {
    "EC": {2, 3, 4, 5},
    "HB": {4, 5, 6, 12},
}

AREA_CRS = "EPSG:6933"
MIN_SUM23_KM2 = 500.0          # AOI 内 2-3 类面积达此阈值才纳入白名单
OUTPUT_JSON = str(DATA_ROOT / "floe_whitelist_cis.json")
OUTPUT_CSV = str(DATA_ROOT / "floe_whitelist_cis.csv")

REGION_NAMES = {"EA": "Eastern Arctic", "WA": "Western Arctic", "HB": "Hudson Bay",
                "EC": "East Coast", "GL": "Gulf/GreatLakes"}


def region_of(path):
    m = re.search(r"CEXPR([A-Z]{2})", path.upper())
    return m.group(1) if m else "UNKNOWN"


def date_of(path):
    m = re.search(r"_(\d{8})_", path)
    return m.group(1) if m else None


def find_charts():
    seen = {}
    for root in CHART_ROOTS:
        for shp in glob.glob(os.path.join(root, "**", "*.shp"), recursive=True):
            reg = region_of(shp)
            d = date_of(shp)
            if d is None:
                continue
            key = (reg, d)
            seen.setdefault(key, shp.replace("\\", "/"))  # 同 区域+日期 去重，保留首个
    return seen


def assess_chart(shp_path, aoi_geom_4326):
    """裁剪到 AOI，按 2-3 类统计面积(km²)。返回 (small_km2, med_km2)。"""
    g, gcol = fl.load_chart_gdf(shp_path, target_crs=4326)
    aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geom_4326], crs=4326)
    clipped = gpd.clip(g, aoi_gdf)
    if len(clipped) == 0:
        return 0.0, 0.0
    clipped = clipped.to_crs(AREA_CRS)
    areas = clipped.geometry.area.values / 1e6
    small = med = 0.0
    for (_, r), a in zip(clipped.iterrows(), areas):
        if not np.isfinite(a) or a <= 0:
            continue
        lab = fl.classify_polygon_floe(r)
        if lab == 2:
            small += float(a)
        elif lab == 3:
            med += float(a)
    return small, med


def main():
    charts = find_charts()
    print(f"发现冰图(区域+日期去重后): {len(charts)}")
    aoi_geoms = {reg: shapely_wkt.loads(REGION_AOIS[reg]) for reg in TARGET_REGIONS if reg in REGION_AOIS}

    whitelist = []
    considered = 0
    for (reg, d), shp in sorted(charts.items()):
        if reg not in TARGET_REGIONS or reg not in aoi_geoms:
            continue
        month = int(d[4:6])
        if reg in REGION_SEASONS and month not in REGION_SEASONS[reg]:
            continue
        considered += 1
        try:
            small, med = assess_chart(shp, aoi_geoms[reg])
        except Exception as e:
            print(f"  [跳过] {os.path.basename(shp)}: {e}")
            continue
        sum23 = small + med
        flag = "✓" if sum23 >= MIN_SUM23_KM2 else "·"
        print(f"  {flag} {reg} {d}  Small={small:8.0f}  Med={med:8.0f}  2-3={sum23:8.0f} km²")
        if sum23 >= MIN_SUM23_KM2:
            whitelist.append({
                "region": reg, "date": d, "shp": shp, "aoi_wkt": REGION_AOIS[reg],
                "small_km2": round(small, 1), "med_km2": round(med, 1), "sum23_km2": round(sum23, 1),
            })

    whitelist.sort(key=lambda x: (x["region"], x["date"]))
    print(f"\n候选(区域+季节内): {considered}；入选(2-3>= {MIN_SUM23_KM2:.0f} km²): {len(whitelist)}")
    # 按区域/年份汇总
    by_ry = {}
    for w in whitelist:
        by_ry.setdefault((w["region"], w["date"][:4]), 0)
        by_ry[(w["region"], w["date"][:4])] += 1
    for (reg, yr), n in sorted(by_ry.items()):
        print(f"  {REGION_NAMES.get(reg, reg)} {yr}: {n} 个合格日期")

    for path in (OUTPUT_JSON, OUTPUT_CSV):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": {str(k): v for k, v in fl.DYNAMIC_THRESHOLDS_FLOE.items()},
            "min_sum23_km2": MIN_SUM23_KM2,
            "region_aois": {r: REGION_AOIS[r] for r in TARGET_REGIONS if r in REGION_AOIS},
            "count": len(whitelist),
            "entries": whitelist,
        }, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "date", "small_km2", "med_km2", "sum23_km2", "shp"])
        for e in whitelist:
            w.writerow([e["region"], e["date"], e["small_km2"], e["med_km2"], e["sum23_km2"], e["shp"]])
    print(f"\n白名单已写出:\n  {OUTPUT_JSON}\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
