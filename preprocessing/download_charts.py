"""
DMI 格陵兰区域冰图(RIC, SIGRID-3 shapefile)下载器。
来源开放目录: https://download.dmi.dk/public/ICESERVICE/SIGRID3/<年>/<YYYYMMDDHHMM>_<区域>_RIC/

DMI 冰图含完整 FA/FB/FC 浮冰尺寸编码(与 CIS 同 SIGRID-3 字段、EPSG:4326)，可直接喂给现有流水线。
按 区域 + 年份 + 月份 过滤下载；断点续跑(本地已存在 .shp 则跳过)。
"""
import os
from _settings import DATA_ROOT
import re
import time
import warnings

warnings.filterwarnings("ignore")
import requests

# ======================= 配置 =======================
DMI_BASE = "https://download.dmi.dk/public/ICESERVICE/SIGRID3/"
PROXY = None  # requests respects standard HTTP(S)_PROXY environment variables
VERIFY_SSL = True

# 可 env 覆盖(第二批采集用)：FLOE_DMI_YEARS=2022,2023  FLOE_DMI_REGIONS=CapeFarewell,CentralEast,SouthEast
_y = os.environ.get("FLOE_DMI_YEARS", "")
YEARS = [int(x) for x in _y.split(",") if x.strip()] if _y.strip() else [2024, 2025, 2026]
_r = os.environ.get("FLOE_DMI_REGIONS", "")
REGIONS = [x.strip() for x in _r.split(",") if x.strip()] if _r.strip() else ["NorthEast", "CentralEast", "SouthEast", "CapeFarewell"]
_m = os.environ.get("FLOE_DMI_MONTHS", "")
MONTHS = {int(x) for x in _m.split(",") if x.strip()} if _m.strip() else None  # 全年(None)
MAX_CHARTS = 0          # 0 = 不限；>0 时只下前 N 张(快速抽样用)

OUT_DIR = os.environ.get("FLOE_DMI_OUT_DIR", str(DATA_ROOT / "Ice_chart"))
PART_EXTS = (".shp", ".shx", ".dbf", ".prj", ".xml")
TIMEOUT = 120
RETRIES = 6
RETRY_WAIT = 15


def http_get(url):
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            return requests.get(url, proxies=PROXY, timeout=TIMEOUT, verify=VERIFY_SSL)
        except Exception as e:
            last = e
            print(f"    [重试] {attempt}/{RETRIES} {type(e).__name__}，{RETRY_WAIT}s...")
            time.sleep(RETRY_WAIT)
    raise last


def list_links(url):
    r = http_get(url)
    if r.status_code != 200:
        print(f"[警告] 列目录失败 HTTP {r.status_code}: {url}")
        return []
    hrefs = re.findall(r'href="([^"?][^"]*)"', r.text)
    out = []
    for h in hrefs:
        if h in ("../", "/") or h.startswith("/"):
            continue
        out.append(h)
    return sorted(set(out))


FOLDER_RE = re.compile(r"^(\d{12})_([A-Za-z]+)_RIC/?$")


def select_chart_folders(year):
    """返回该年份中符合 区域+月份 的 RIC 文件夹名列表。"""
    entries = list_links(f"{DMI_BASE}{year}/")
    selected = []
    for e in entries:
        m = FOLDER_RE.match(e)
        if not m:
            continue
        stamp, region = m.group(1), m.group(2)
        month = int(stamp[4:6])
        if region not in REGIONS:
            continue
        if MONTHS is not None and month not in MONTHS:
            continue
        selected.append(e.rstrip("/"))
    return selected


def download_chart(year, folder):
    """下载一个 RIC 文件夹的 shapefile 各部件到 OUT_DIR/<folder>/。"""
    local_dir = os.path.join(OUT_DIR, folder)
    shp_local = os.path.join(local_dir, f"{folder}.shp")
    if os.path.isfile(shp_local):
        return "skip"
    folder_url = f"{DMI_BASE}{year}/{folder}/"
    files = [f for f in list_links(folder_url) if f.lower().endswith(PART_EXTS)]
    if not any(f.lower().endswith(".shp") for f in files):
        print(f"  [跳过] 无 .shp: {folder}")
        return "noshp"
    os.makedirs(local_dir, exist_ok=True)
    for f in files:
        r = http_get(folder_url + f)
        if r.status_code == 200:
            with open(os.path.join(local_dir, f), "wb") as fh:
                fh.write(r.content)
    return "ok"


WORKERS = int(os.environ.get("FLOE_DMI_WORKERS", "1"))  # 并行下载线程数(小文件多请求，并行提速明显)


def main():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"DMI 下载器 | 区域={REGIONS} 年份={YEARS} 月份={MONTHS} 并行={WORKERS}", flush=True)
    print(f"输出: {OUT_DIR}", flush=True)
    cnt = {"ok": 0, "skip": 0, "fail": 0}
    lock = threading.Lock()

    def worker(year, folder, i, n):
        try:
            status = download_chart(year, folder)
        except Exception as e:
            with lock:
                cnt["fail"] += 1
            print(f"  [{year} {i}/{n}] 失败: {folder} -> {type(e).__name__}", flush=True)
            return
        with lock:
            if status == "ok":
                cnt["ok"] += 1
                if cnt["ok"] % 20 == 0:
                    print(f"  ... 已新下 {cnt['ok']} (最近 {folder})", flush=True)
            elif status == "skip":
                cnt["skip"] += 1

    for year in YEARS:
        folders = select_chart_folders(year)
        print(f"\n[{year}] 符合条件的 RIC: {len(folders)}", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, folder in enumerate(folders, 1):
                ex.submit(worker, year, folder, i, len(folders))
        print(f"[{year}] 完成 | 累计 新下 {cnt['ok']} 跳过 {cnt['skip']} 失败 {cnt['fail']}", flush=True)
    print(f"\n完成：新下 {cnt['ok']}，已存在跳过 {cnt['skip']}，失败 {cnt['fail']}", flush=True)
    print(f"本地 RIC 文件夹总数: {len([d for d in os.listdir(OUT_DIR) if os.path.isdir(os.path.join(OUT_DIR, d))])}", flush=True)


if __name__ == "__main__":
    main()
