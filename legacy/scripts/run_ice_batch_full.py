import os

# 强制无缓冲输出
os.environ.setdefault("PYTHONUNBUFFERED", "1")
# 锁定底层数学库线程，避免隐性多线程超卖 CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import threading
import time
import sys
from pathlib import Path
import pandas as pd
import xarray as xr
# 统一日志初始化（确保 run_id 注入，避免 KeyError: run_id）
try:
    from logging_config import get_logger as _get_logger
    _get_logger(__name__)
except Exception:
    pass

# 确保项目根目录在 sys.path，避免 logging_config 等模块导入失败
try:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
except Exception:
    pass

# 尝试限制 PyTorch 线程（当前进程）。
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# 可选 psutil
try:
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

from ArcticRoute.core.predictors.ice_batch import forecast_block
from ArcticRoute.core.utils.progress import ProgressReporter

def build_parser():
    p = argparse.ArgumentParser(description="Run batch sea-ice forecast with autoscaling.")
    p.add_argument("--src", type=str, default=r"C:\\Users\\sgddsf\\Desktop\\minimum\\ArcticRoute\\data\\raw\\cmems_arc\\cmems_mod_arc_phy_my_topaz4_P1M_1762143591239.nc", help="Input NetCDF path")
    p.add_argument("--save-dir", type=str, default=r"C:\\Users\\sgddsf\\Desktop\\minimum\\ArcticRoute\\data_processed\\ice_forecast", help="Output directory")
    p.add_argument("--var", type=str, default="sic", help="Variable to forecast (sic)")
    p.add_argument("--block", type=str, default="50,50", help="Block size as 'by,bx' (default 50,50)")
    p.add_argument("--horizon", type=int, default=12, help="Forecast horizon in months")
    p.add_argument("--min-len", type=int, default=120, help="Minimum valid length for modeling")
    p.add_argument("--min-valid-frac", type=float, default=0.6, help="Minimum valid fraction")
    p.add_argument("--std-thresh", type=float, default=1e-6, help="Std dev threshold to avoid near-constant series")
    p.add_argument("--per-month-min", type=int, default=2, help="Minimum valid samples per calendar month")
    p.add_argument("--use-lstm", action="store_true", help="Enable LSTM residual modeling")
    p.add_argument("--no-lstm", dest="use_lstm", action="store_false", help="Disable LSTM residual modeling")
    p.set_defaults(use_lstm=True)
    p.add_argument("--max-blocks", type=int, default=5, help="Run first N blocks only (for testing)")
    p.add_argument("--dry-run", action="store_true", help="Do not write final merged file (still writes block cache)")

    # 进度/可视化
    p.add_argument("--label", type=str, default="202412", help="标签，用于输出文件/进度文件命名")
    p.add_argument("--progress", type=str, choices=["off","console","both"], default="console", help="进度显示模式")
    p.add_argument("--progress-interval", type=float, default=2.0, help="健康心跳与刷新间隔（秒）")
    p.add_argument("--progress-pixel-step", type=int, default=200, help="像元级 tick 粒度")

    # Autoscale flags
    p.add_argument("--autoscale", action="store_true", help="Enable autoscaling workers by mem/cpu")
    p.add_argument("--no-autoscale", dest="autoscale", action="store_false", help="Disable autoscaling")
    p.set_defaults(autoscale=True)
    p.add_argument("--min-workers", type=int, default=1, help="Min workers")
    p.add_argument("--max-workers", type=int, default=3, help="Max workers")
    p.add_argument("--mem-low", type=float, default=65.0, help="Mem percent low threshold to +1 worker")
    p.add_argument("--mem-high", type=float, default=80.0, help="Mem percent high threshold to -1 worker")
    p.add_argument("--cpu-cap", type=float, default=90.0, help="Do not increase workers if CPU percent above this")
    return p


def main():
    ap = build_parser()
    args = ap.parse_args()

    path = args.src
    save_dir = args.save_dir

    if not os.path.exists(path):
        print("File not found:", path)
        return
    os.makedirs(save_dir, exist_ok=True)

    ds = xr.open_dataset(path)
    # 变量映射
    if args.var == "sic" and "sic" not in ds:
        for k in ["siconc","ci","ice_conc","sea_ice_concentration"]:
            if k in ds:
                ds = ds.assign(sic=ds[k])
                break
    if "sit" not in ds:
        for k in ["sithick_corr","sithick","sea_ice_thickness","ice_thickness"]:
            if k in ds:
                ds = ds.assign(sit=ds[k])
                break
    # 维度规范（更健壮的重命名：支持 lat/lon/latitude/longitude 以及 t -> time）
    ren = {}
    # 标准名称 -> 可能的候选名
    _dim_cands = {
        "time": ["time", "t", "Time", "TIME"],
        "y": ["y", "latitude", "lat", "Y", "nav_lat"],
        "x": ["x", "longitude", "lon", "X", "nav_lon"],
    }
    for std, cands in _dim_cands.items():
        if std not in ds.dims:
            for c in cands:
                if c in ds.dims:
                    ren[c] = std
                    break
    if ren:
        ds = ds.rename(ren)

    by, bx = map(int, args.block.split(","))

    # 计算 blocks_total
    Ny, Nx = int(ds.sizes.get("y", 0)), int(ds.sizes.get("x", 0))
    blocks_total = (len(range(0, Ny, by)) * len(range(0, Nx, bx)))
    if args.max_blocks is not None:
        blocks_total = min(blocks_total, max(0, int(args.max_blocks)))

    # 创建进度 reporter
    rep = ProgressReporter(mode=args.progress, save_dir=save_dir, label=args.label, blocks_total=blocks_total, interval=args.progress_interval)
    rep.start()
    rep.emit(kind="run_start", label=args.label)

    # 健康心跳线程
    stop_evt = threading.Event()
    def _health_loop():
        while not stop_evt.is_set():
            cpu = mem = None
            if _HAS_PSUTIL:
                try:
                    cpu = psutil.cpu_percent(interval=None)
                    mem = psutil.virtual_memory().percent
                except Exception:
                    cpu = mem = None
            rep.emit(kind="health", label=args.label, cpu=cpu, mem=mem)
            stop_evt.wait(timeout=max(0.5, float(args.progress_interval)))
    hb_th = threading.Thread(target=_health_loop, daemon=True)
    if args.progress in ("console", "both") or True:
        hb_th.start()

    st = time.time()
    try:
        res = forecast_block(
            ds, var=args.var,
            block=(by,bx), horizon=args.horizon,
            min_len=args.min_len, min_valid_frac=args.min_valid_frac,
            std_thresh=args.std_thresh, per_month_min=args.per_month_min,
            use_lstm=args.use_lstm,
            save_dir=save_dir,
            max_blocks=args.max_blocks,
            dry_run=args.dry_run,
            autoscale=args.autoscale,
            min_workers=args.min_workers,
            max_workers=args.max_workers,
            mem_low=args.mem_low,
            mem_high=args.mem_high,
            cpu_cap=args.cpu_cap,
            progress=rep,
            label=args.label,
            progress_pixel_step=args.progress_pixel_step,
            progress_interval=args.progress_interval,
        )
    finally:
        rep.emit(kind="run_done", label=args.label)
        stop_evt.set()
        try:
            hb_th.join(timeout=3)
        except Exception:
            pass
        rep.stop()
    el = time.time()-st

    print("run_seconds:", round(el,2))
    print("out dims:", dict(res.sizes))
    print("vars:", list(res.data_vars))
    print("time coverage:", str(pd.to_datetime(res.time.values[0]))[:10],"->", str(pd.to_datetime(res.time.values[-1]))[:10])
    print("pred flag ratio:", float(res.pred_flag.mean().values))
    fut = res.isel(time=slice(-12,None))
    print("sic_pred stats future:", float(fut.sic_pred.min().values), float(fut.sic_pred.max().values))

if __name__ == "__main__":
    main()
