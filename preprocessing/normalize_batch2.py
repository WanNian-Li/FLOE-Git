"""Append a second batch with existing statistics; adapted from _batch2_finalize.py. Output state files live in DST; see README."""
import os, sys, json, glob, time, shutil, warnings
from _settings import DATA_ROOT
warnings.filterwarnings("ignore")

MIN_FREE_GB = float(os.environ.get("FLOE_MIN_FREE_GB", "30"))  # 磁盘守卫:低于此暂停归一化(防撑爆写损坏)

HERE = os.path.dirname(os.path.abspath(__file__))
CF = str(DATA_ROOT)
SRC = os.environ.get("FLOE_APPEND_INPUT", str(DATA_ROOT / "data_nc_batch2"))
DST = os.environ.get("FLOE_APPEND_OUTPUT", str(DATA_ROOT / "data_nc_zscore_batch2"))
STATS_JSON = os.environ.get("FLOE_STATS_JSON", str(DATA_ROOT / "data_nc_zscore" / "zscore_stats.json"))
DONE_COLLECT = os.environ.get("FLOE_COLLECT_DONE", str(DATA_ROOT / ".collect_done.flag"))
DONE_NORM = os.path.join(DST, ".normalize_done.flag")
LOCK = os.path.join(DST, ".normalize.lock")

NODATA = -9999.0
EPS = 1e-6
CLIP = {"nersc_sar_primary": (-35.0, 5.0), "nersc_sar_secondary": (-35.0, 5.0)}
FEATURES = ["nersc_sar_primary", "nersc_sar_secondary", "sar_incidenceangle",
            "glcm_sigma0_hh_contrast", "glcm_sigma0_hh_dissimilarity", "glcm_sigma0_hh_homogeneity"]
CHUNK = (512, 512)
WAIT_SEC = 300
MAX_WAIT_HOURS = 30


def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _pid_alive(pid):
    import subprocess
    try:  # 中文 Windows 的 tasklist 是 GBK，用字节解码(忽略错误)避免 UnicodeDecodeError
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True).stdout
        txt = out.decode("gbk", "ignore") + out.decode("utf-8", "ignore")
        return str(pid) in txt
    except Exception:
        return False  # 查不准时认为已死，避免陈旧锁把新实例卡死


def acquire_lock():
    if os.path.isfile(LOCK):
        try:
            old = int((open(LOCK).read().strip() or "0"))
        except Exception:
            old = 0
        if old and old != os.getpid() and _pid_alive(old):
            print(f"[{ts()}] 已有归一化实例 (PID {old})，退出。", flush=True)
            return False
    open(LOCK, "w").write(str(os.getpid()))
    return True


def release_lock():
    try:
        os.remove(LOCK)
    except Exception:
        pass


def renorm_one(src, stats):
    import numpy as np, xarray as xr
    name = os.path.basename(src)
    out = os.path.join(DST, name); tmp = out + ".tmp"
    if os.path.exists(out):
        return "skip"
    ds = xr.open_dataset(src, engine="h5netcdf")
    dims = ds[FEATURES[0]].dims
    valid = None
    for v in FEATURES:
        a = np.asarray(ds[v].values)
        fv = ds[v].attrs.get("_FillValue", None)
        m = np.isfinite(a) & (a != NODATA)
        if fv is not None and np.isfinite(fv):
            m &= (a != float(fv))
        valid = m if valid is None else (valid & m)
        del a, m
    base = {}
    for v in ds.data_vars:
        if v in FEATURES:
            continue
        base[v] = (ds[v].dims, np.asarray(ds[v].values), dict(ds[v].attrs))
    base["global_valid_mask"] = (dims, valid.astype("uint8"),
                                 {"long_name": "Global valid mask for all input features",
                                  "flag_values": np.array([0, 1], dtype="uint8"), "flag_meanings": "invalid valid"})
    gattrs = dict(ds.attrs)
    gattrs.update({"normalization": "global_zscore", "normalization_nodata": NODATA,
                   "training_safe_output": np.int8(1), "training_safe_mask_layer": "global_valid_mask",
                   "training_safe_invalid_mapped_to": 0.0})
    enc0 = {}
    for v in base:
        e = {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": CHUNK}
        if v == "global_valid_mask":
            e["dtype"] = "uint8"; e["_FillValue"] = np.uint8(0)
        enc0[v] = e
    if os.path.exists(tmp):
        os.remove(tmp)
    xr.Dataset(base, attrs=gattrs).to_netcdf(tmp, mode="w", format="NETCDF4", engine="h5netcdf", encoding=enc0)
    del base
    for v in FEATURES:
        st = stats[v]; mean = float(st["mean"]); std = float(st["std"]); denom = std if std > EPS else EPS
        a = np.asarray(ds[v].values).astype("float32")
        if v in CLIP:
            lo, hi = CLIP[v]; np.clip(a, lo, hi, out=a)
        o = np.zeros_like(a); o[valid] = (a[valid] - mean) / denom
        del a
        attrs = dict(ds[v].attrs); attrs.pop("_FillValue", None)
        attrs.update({"zscore_mean": mean, "zscore_std": std, "zscore_applied": np.int8(1),
                      "invalid_value_mapped_to": 0.0, "training_safe_output": np.int8(1)})
        enc = {v: {"zlib": True, "complevel": 4, "shuffle": True, "chunksizes": CHUNK,
                   "dtype": "float32", "_FillValue": np.float32(NODATA)}}
        xr.Dataset({v: (dims, o, attrs)}).to_netcdf(tmp, mode="a", engine="h5netcdf", encoding=enc)
        del o
    ds.close()
    os.replace(tmp, out)
    return "ok"


def main():
    os.makedirs(DST, exist_ok=True)
    if os.path.isfile(DONE_NORM):
        print(f"[{ts()}] 归一化已完成。", flush=True); return 0
    if not acquire_lock():
        return 0
    os.makedirs(DST, exist_ok=True)
    try:
        # 1) 等采集完成(FLOE_SKIP_WAIT=1 时跳过等待，直接归一化已有的)
        t0 = time.time()
        while os.environ.get("FLOE_SKIP_WAIT", "1") != "1" and not os.path.isfile(DONE_COLLECT):
            if time.time() - t0 > MAX_WAIT_HOURS * 3600:
                print(f"[{ts()}] 等采集完成超时({MAX_WAIT_HOURS}h)，先归一化已有的。", flush=True)
                break
            print(f"[{ts()}] 等采集完成... 当前批2 NC {len(glob.glob(os.path.join(SRC,'*.nc')))} 个", flush=True)
            time.sleep(WAIT_SEC)
        if not os.path.isfile(STATS_JSON):
            print(f"[{ts()}] 缺 zscore_stats.json，无法 apply。", flush=True); return 2
        stats = json.load(open(STATS_JSON, encoding="utf-8"))["stats"]
        # 2) 逐个分块归一化
        files = sorted(glob.glob(os.path.join(SRC, "*.nc")))
        print(f"[{ts()}] 开始归一化 {len(files)} 个批2文件 -> {DST}", flush=True)
        ok = skip = fail = 0
        for i, f in enumerate(files, 1):
            free_gb = shutil.disk_usage(DST).free / 1e9
            if free_gb < MIN_FREE_GB:
                print(f"\n[{ts()}] [磁盘守卫] 可用 {free_gb:.0f}G < {MIN_FREE_GB}G，暂停归一化(已做 {ok})。"
                      f"请清理空间后重跑(断点续,已归一化的跳过);或删批2未归一化以腾空间。", flush=True)
                return 3
            try:
                r = renorm_one(f, stats)
                if r == "ok":
                    ok += 1; print(f"[{ts()}] [{i}/{len(files)}] 归一化 {os.path.basename(f)}", flush=True)
                else:
                    skip += 1
            except Exception as e:
                fail += 1
                print(f"[{ts()}] [{i}/{len(files)}] 失败 {os.path.basename(f)}: {type(e).__name__}: {e}", flush=True)
        print(f"\n[{ts()}] 归一化完成：新做 {ok}，跳过 {skip}，失败 {fail}，共 {len(files)}", flush=True)
        if fail == 0:
            open(DONE_NORM, "w").write(f"{ts()} batch2 normalize done, {ok+skip} files\n")
            print(f"[{ts()}] 已写归一化完成标记。", flush=True)
        return 1 if fail else 0
    finally:
        release_lock()


if __name__ == "__main__":
    raise SystemExit(main())
