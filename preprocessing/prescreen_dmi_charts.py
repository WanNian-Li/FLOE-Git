"""
DMI(格陵兰 RIC)版预筛——CIS 版 prescreen_floe_charts.py 的姊妹脚本。

与 CIS 版的两点关键差异（DMI 是"小范围区域图"）：
  1. 区域识别走 DMI 命名 `<YYYYMMDDHHMM>_<Region>_RIC`（CIS 是 `CEXPR<XX>`）。
  2. **不预设 AOI**：逐图统计整张冰图的 2-3 类面积达标即入白名单，
     再从所有入选图的 2-3 多边形并集**自动推导一个收紧的 bbox AOI**，直接喂给第 2 步。
     （DMI 单图只含一个区域、只有几十个多边形，整图统计即可，无需先裁剪。）

输出（与第 2 步 the external SAR acquisition workflow 完全兼容，可无缝衔接）：
  - floe_whitelist.json / .csv：entries 带 region=<REGION>、date=YYYYMMDD，
    顶层 derived_aoi_wkt 即第 2 步该填进 aoi_wkt 的收紧框。
  - 第 2 步配置：WHITELIST_REGION=<REGION>、ICECHART_ROOT_FOR_FILTER=CHART_ROOT、
    aoi_wkt=本脚本打印的 derived AOI。

按区域单独跑：把 CHART_ROOT / REGION / OUTPUT_* 改到对应东格陵兰区即可复用。
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
from shapely.ops import unary_union

import floe_label as fl

# ======================= 配置 =======================
# 单区域冰图根目录（递归找 *.shp）。可 env 覆盖；指向 DMI\Ice_chart 时靠 REGION 过滤即可，无需复制。
CHART_ROOT = os.environ.get("FLOE_PS_CHART_ROOT",
                            str(DATA_ROOT / "Ice_chart"))
# 只处理该区域的图（与 DMI 文件名 `_<Region>_RIC` 对应）。None=该目录下全部。
REGION = os.environ.get("FLOE_PS_REGION", "CapeFarewell")

# 仅统计这些月份（DMI 东格陵兰 8-9 月空窗）。env FLOE_PS_MONTHS="4,5,6,7"；空=全年。
_mon = os.environ.get("FLOE_PS_MONTHS", "")
SEASON_MONTHS = {int(x) for x in _mon.split(",") if x.strip()} if _mon.strip() else None

# 日期窗口(含边界)，YYYYMMDD 字符串比较。env FLOE_PS_DATE_MIN / FLOE_PS_DATE_MAX；空=不限。
DATE_MIN = os.environ.get("FLOE_PS_DATE_MIN", "").strip()
DATE_MAX = os.environ.get("FLOE_PS_DATE_MAX", "").strip()

# 预筛目标类别：默认 2/3(Small/Medium)；cake 采集批次设 env FLOE_PS_CLASSES="1"。
_ps_cls = os.environ.get("FLOE_PS_CLASSES", "2,3")
CLASSES = tuple(sorted({int(x) for x in _ps_cls.split(",") if x.strip()}))

AREA_CRS = "EPSG:6933"          # 等面积投影，用于面积阈值
# 整图目标类合计面积达此阈值才纳入白名单。FLOE_PS_MIN_AREA 优先；FLOE_PS_MIN_SUM23 为历史名兼容。
MIN_SUM23_KM2 = float(os.environ.get("FLOE_PS_MIN_AREA",
                                     os.environ.get("FLOE_PS_MIN_SUM23", "500")))
AOI_BUFFER_DEG = 0.3            # 推导 AOI 时在 2-3 并集 bbox 外扩的经纬度余量（给 SAR footprint 留边）
COORD_NDIGITS = 3               # 推导 AOI 坐标保留小数位

_out_dir = os.environ.get("FLOE_PS_OUT_DIR",
                          str(DATA_ROOT))
_out_name = os.environ.get("FLOE_PS_OUT_NAME", "floe_whitelist")  # 批2用 floe_whitelist_batch2，避免覆盖第一批
OUTPUT_JSON = os.path.join(_out_dir, f"{_out_name}.json")
OUTPUT_CSV = os.path.join(_out_dir, f"{_out_name}.csv")


def region_of(path):
    """DMI 命名：<YYYYMMDDHHMM>_<Region>_RIC。返回区域名或 UNKNOWN。"""
    m = re.search(r"\d{12}_([A-Za-z]+)_RIC", path)
    return m.group(1) if m else "UNKNOWN"


def date_of(path):
    """取 DMI 12 位时戳的前 8 位 YYYYMMDD。"""
    m = re.search(r"(\d{12})_[A-Za-z]+_RIC", path)
    return m.group(1)[:8] if m else None


def find_charts():
    """{(region, date): shp_path}：按 区域+日期 去重，保留首个（与第 2 步按日期取首图一致）。"""
    seen = {}
    # sorted: 同日多图(不同 HHMM)时确定性取首张(最早时戳)，与 step2/step4 口径一致。
    for shp in sorted(glob.glob(os.path.join(CHART_ROOT, "**", "*.shp"), recursive=True)):
        reg = region_of(shp)
        d = date_of(shp)
        if d is None:
            continue
        if REGION is not None and reg != REGION:
            continue
        seen.setdefault((reg, d), shp.replace("\\", "/"))
    return seen


def assess_chart(shp_path):
    """整图统计 CLASSES 各类面积(km²) 并返回目标多边形并集(EPSG:4326)。

    返回 ({cls: km2}, union4326 | None)。
    """
    g, gcol = fl.load_chart_gdf(shp_path, target_crs=4326)
    items = []  # (lab, geom4326)
    for (_, r), geom in zip(g.iterrows(), g[gcol]):
        if geom is None or geom.is_empty:
            continue
        lab = fl.classify_polygon_floe(r)
        if lab in CLASSES:
            items.append((lab, geom))
    by_cls = {c: 0.0 for c in CLASSES}
    if not items:
        return by_cls, None

    geoms = [gm for _, gm in items]
    areas = gpd.GeoSeries(geoms, crs=4326).to_crs(AREA_CRS).area.values / 1e6
    for (lab, _), a in zip(items, areas):
        if not np.isfinite(a) or a <= 0:
            continue
        by_cls[lab] += float(a)

    union = unary_union(geoms)
    if not union.is_valid:
        union = union.buffer(0)
    return by_cls, union


def bbox_aoi_wkt(union_geom, buffer_deg, ndigits):
    """从 2-3 并集取外包 bbox、外扩 buffer，输出轴对齐矩形 WKT(EPSG:4326)。"""
    minx, miny, maxx, maxy = union_geom.bounds
    minx -= buffer_deg; miny -= buffer_deg
    maxx += buffer_deg; maxy += buffer_deg
    r = lambda v: round(v, ndigits)
    minx, miny, maxx, maxy = r(minx), r(miny), r(maxx), r(maxy)
    return (f"POLYGON(({minx} {miny},{maxx} {miny},{maxx} {maxy},"
            f"{minx} {maxy},{minx} {miny}))")


def main():
    charts = find_charts()
    print(f"发现冰图(区域={REGION or '全部'}，区域+日期去重后): {len(charts)}")

    whitelist = []
    considered = 0
    qualifying_unions = []  # 入选图的 2-3 并集（用于推导 AOI）
    for (reg, d), shp in sorted(charts.items()):
        month = int(d[4:6])
        if SEASON_MONTHS is not None and month not in SEASON_MONTHS:
            continue
        if DATE_MIN and d < DATE_MIN:
            continue
        if DATE_MAX and d > DATE_MAX:
            continue
        considered += 1
        try:
            by_cls, union = assess_chart(shp)
        except Exception as e:
            print(f"  [跳过] {os.path.basename(shp)}: {e}")
            continue
        total = sum(by_cls.values())
        flag = "✓" if total >= MIN_SUM23_KM2 else "·"
        cls_str = "  ".join(f"c{c}={by_cls[c]:8.0f}" for c in CLASSES)
        print(f"  {flag} {reg} {d}  {cls_str}  合计={total:8.0f} km²")
        if total >= MIN_SUM23_KM2:
            entry = {"region": reg, "date": d, "shp": shp, "area_km2": round(total, 1)}
            for c in CLASSES:
                entry[f"c{c}_km2"] = round(by_cls[c], 1)
            if CLASSES == (2, 3):  # 兼容旧字段名(有零散脚本读它们)
                entry["small_km2"] = entry["c2_km2"]
                entry["med_km2"] = entry["c3_km2"]
                entry["sum23_km2"] = entry["area_km2"]
            whitelist.append(entry)
            if union is not None and not union.is_empty:
                qualifying_unions.append(union)

    whitelist.sort(key=lambda x: (x["region"], x["date"]))
    print(f"\n候选(区域/季节内): {considered}；入选(类{CLASSES}合计>= {MIN_SUM23_KM2:.0f} km²): {len(whitelist)}")

    # 按年份汇总
    by_year = {}
    for w in whitelist:
        by_year.setdefault(w["date"][:4], 0)
        by_year[w["date"][:4]] += 1
    for yr, n in sorted(by_year.items()):
        print(f"  {REGION} {yr}: {n} 个合格日期")

    # 从所有入选图的目标类并集推导收紧 AOI
    derived_aoi = None
    if qualifying_unions:
        total_union = unary_union(qualifying_unions)
        if not total_union.is_valid:
            total_union = total_union.buffer(0)
        derived_aoi = bbox_aoi_wkt(total_union, AOI_BUFFER_DEG, COORD_NDIGITS)
        bx = [round(v, COORD_NDIGITS) for v in total_union.bounds]
        print(f"\n类{CLASSES} 并集 bbox(lon {bx[0]}~{bx[2]}, lat {bx[1]}~{bx[3]})；外扩 {AOI_BUFFER_DEG}° 后：")
        print(f"  derived AOI = {derived_aoi}")
        print("  → 把它填进 the external SAR acquisition workflow 的 aoi_wkt。")
    else:
        print("\n[警告] 无入选图，无法推导 AOI（请检查阈值/数据）。")

    # 每个 entry 也带上 derived_aoi（与 CIS 版 entry 带 aoi_wkt 习惯一致）
    for e in whitelist:
        e["aoi_wkt"] = derived_aoi

    for path in (OUTPUT_JSON, OUTPUT_CSV):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "thresholds": {str(k): v for k, v in fl.DYNAMIC_THRESHOLDS_FLOE.items()},
            "classes": list(CLASSES),
            "min_area_km2": MIN_SUM23_KM2,
            "min_sum23_km2": MIN_SUM23_KM2,  # 历史键名兼容
            "region": REGION,
            "chart_root": CHART_ROOT,
            "aoi_buffer_deg": AOI_BUFFER_DEG,
            "derived_aoi_wkt": derived_aoi,
            "count": len(whitelist),
            "entries": whitelist,
        }, f, ensure_ascii=False, indent=2)
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        cls_cols = [f"c{c}_km2" for c in CLASSES]
        w.writerow(["region", "date", *cls_cols, "area_km2", "shp"])
        for e in whitelist:
            w.writerow([e["region"], e["date"], *[e[c] for c in cls_cols], e["area_km2"], e["shp"]])
    print(f"\n白名单已写出:\n  {OUTPUT_JSON}\n  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
