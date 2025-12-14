from __future__ import annotations

import os
import logging
import warnings
import time
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

from ArcticRoute.io.ice_sarima_lstm import IceForecaster
from ArcticRoute.core.utils.progress import ProgressReporter

# 尝试导入 psutil，若失败则 autoscale 功能不可用
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# 降噪：statsmodels 收敛告警仅提示一次/或忽略特定文案
try:
    from statsmodels.tools.sm_exceptions import ConvergenceWarning  # type: ignore
    warnings.filterwarnings("once", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=r"Maximum Likelihood.*")
except Exception:
    pass

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def monthly_climatology(s: pd.Series) -> np.ndarray:
    if not isinstance(s.index, pd.DatetimeIndex) or len(s) == 0:
        return np.zeros(12, dtype=float)
    s2 = s.sort_index()
    if s2.index.freq is None:
        s2 = s2.asfreq("MS")
    v = s2.values.astype(float)
    idx = s2.index
    mask = np.isfinite(v)
    if not mask.any():
        return np.zeros(12, dtype=float)

    prof = np.full(12, np.nan, dtype=float)
    for m in range(1, 13):
        sel = (idx.month == m) & mask
        if sel.any():
            prof[m - 1] = float(np.mean(v[sel]))
    finite_prof = prof[np.isfinite(prof)]
    fill = float(np.mean(finite_prof)) if finite_prof.size else 0.0
    prof[np.isnan(prof)] = fill
    return prof


def _seasonal_mean_forecast(series: pd.Series, horizon: int) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex) or len(series) == 0:
        idx = pd.date_range(pd.Timestamp("2000-01-01"), periods=horizon, freq="MS")
        return pd.Series(np.zeros(horizon, dtype=float), index=idx)
    s = series.sort_index()
    if s.index.freq is None:
        s = s.asfreq("MS")
    prof = monthly_climatology(s)
    start = s.index[-1] + pd.offsets.MonthBegin(1)
    idx = pd.date_range(start, periods=horizon, freq=s.index.freqstr or "MS")
    m0 = start.month
    y = np.array([prof[(m0 - 1 + i) % 12] for i in range(horizon)], dtype=float)
    y = np.clip(y, 0.0, 1.0)
    return pd.Series(y, index=idx)


def _passes_fence(s: pd.Series, min_len: int, min_valid_frac: float, std_thresh: float, per_month_min: int) -> bool:
    if not isinstance(s.index, pd.DatetimeIndex):
        return False
    s2 = s.sort_index()
    if s2.index.freq is None:
        s2 = s2.asfreq("MS")
    v = s2.values.astype(float)
    valid_mask = np.isfinite(v)
    n_total = len(v)
    n_valid = int(valid_mask.sum())
    if n_valid < min_len:
        return False
    if n_valid / max(1, n_total) < min_valid_frac:
        return False
    if np.nanstd(v) < std_thresh:
        return False
    months = s2.index.month
    for m in range(1, 13):
        m_valid = int(np.isfinite(v[months == m]).sum())
        if m_valid < per_month_min:
            return False
    return True


def _fit_predict_point(
    series: pd.Series, horizon: int, min_len: int, use_lstm: bool = True,
    min_valid_frac: float = 0.6, std_thresh: float = 1e-6, per_month_min: int = 2,
) -> Tuple[pd.Series, bool]:
    s = series.copy().astype(float)
    if not isinstance(s.index, pd.DatetimeIndex):
        return _seasonal_mean_forecast(pd.Series([], dtype=float), horizon), False
    if s.index.freq is None:
        s = s.asfreq("MS")
    s = s.interpolate(limit=2).clip(0.0, 1.0)

    if not _passes_fence(s, min_len=min_len, min_valid_frac=min_valid_frac, std_thresh=std_thresh, per_month_min=per_month_min):
        try:
            return _seasonal_mean_forecast(s, horizon), False
        except Exception:
            start = s.index[-1] + pd.offsets.MonthBegin(1)
            idx = pd.date_range(start, periods=horizon, freq=s.index.freqstr or "MS")
            return pd.Series(np.full(horizon, 0.5, dtype=float), index=idx), False

    try:
        lstm_hidden = 32 if use_lstm else 0
        model = IceForecaster(lstm_hidden=lstm_hidden, epochs=25, lr=1e-3)
        model.fit(s)
        pred = model.predict(horizon)
        return pred, True if (model.lambda_shrink > 0 or lstm_hidden == 0) else True
    except Exception:
        return _seasonal_mean_forecast(s, horizon), False


def _process_block(args) -> Tuple[int, int, xr.Dataset]:
    (
        ds_block, var, has_sit, y0, x0, horizon, min_len, use_lstm,
        bi, total_blocks, min_valid_frac, std_thresh, per_month_min,
        progress_queue, label, progress_pixel_step,
    ) = args

    # Worker 进程内：再次锁定底层数学库线程，防止隐性多线程
    try:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
        import torch  # type: ignore
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
    except Exception:
        pass

    mem_peak = 0.0
    if _HAS_PSUTIL:
        p = psutil.Process()
        mem_peak = p.memory_info().rss / (1024 * 1024)  # MB

    try:
        ds_block = ds_block.load()
    except Exception:
        pass

    time_dim = ds_block.time
    T = time_dim.size
    by, bx = ds_block.sizes["y"], ds_block.sizes["x"]
    last_ts = pd.to_datetime(str(time_dim.values[-1]))
    idx_future = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")

    sic_pred = np.full((T + horizon, by, bx), np.nan, dtype=np.float32)
    pred_flag = np.zeros((by, bx), dtype=np.uint8)
    sit_pred = np.full((T + horizon, by, bx), np.nan, dtype=np.float32) if has_sit else None

    total_pix = by * bx
    ok_cnt, skip_cnt = 0, 0

    # 块开始事件（仅发到队列，不打印/写文件）
    try:
        ProgressReporter.emit_to_queue(progress_queue, kind="block_start", label=label, block_idx=bi, blocks_total=total_blocks)
    except Exception:
        pass

    for iy in range(by):
        for ix in range(bx):
            if _HAS_PSUTIL:
                mem_peak = max(mem_peak, p.memory_info().rss / (1024 * 1024))

            try:
                s_sic = ds_block[var].isel(y=iy, x=ix).to_pandas()
                pred_sic, ok_sic = _fit_predict_point(
                    s_sic, horizon, min_len, use_lstm, min_valid_frac, std_thresh, per_month_min
                )
                sic_pred[T:, iy, ix] = pred_sic.values.astype(np.float32)
                pred_flag[iy, ix] = 1 if ok_sic else 0
            except Exception:
                pred_flag[iy, ix] = 0

            if has_sit:
                try:
                    s_sit = ds_block["sit"].isel(y=iy, x=ix).to_pandas()
                    pred_sit, ok_sit_sit = _fit_predict_point(
                        s_sit, horizon, min_len, use_lstm, min_valid_frac, std_thresh, per_month_min
                    )
                    if sit_pred is not None:
                        sit_pred[T:, iy, ix] = pred_sit.values.astype(np.float32)
                    if not ok_sit_sit:
                        pred_flag[iy, ix] = 0  # 任何一个失败则flag=0
                except Exception:
                    pred_flag[iy, ix] = 0

            # 统计 ok/skip
            if pred_flag[iy, ix] == 1:
                ok_cnt += 1
            else:
                skip_cnt += 1

            done = iy * bx + ix + 1
            # 粒度型像元 tick
            if progress_queue is not None and (done % max(1, int(progress_pixel_step)) == 0):
                cpu = mem = None
                if _HAS_PSUTIL:
                    try:
                        cpu = psutil.cpu_percent(interval=None)
                        mem = psutil.virtual_memory().percent
                    except Exception:
                        cpu = mem = None
                try:
                    ProgressReporter.emit_to_queue(
                        progress_queue,
                        kind="pixel_tick", label=label, block_idx=bi, blocks_total=total_blocks,
                        pixels_done=done, pixels_total=total_pix, ok=ok_cnt, skip=skip_cnt,
                        cpu=cpu, mem=mem,
                    )
                except Exception:
                    pass

    # 块完成事件
    cpu = mem = None
    if _HAS_PSUTIL:
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
        except Exception:
            cpu = mem = None
    try:
        ProgressReporter.emit_to_queue(
            progress_queue,
            kind="block_done", label=label, block_idx=bi, blocks_total=total_blocks,
            ok=ok_cnt, skip=skip_cnt, cpu=cpu, mem=mem,
        )
    except Exception:
        pass

    time_full = np.concatenate([time_dim.values, idx_future.values])
    coords = {"time": ("time", time_full), "y": ds_block["y"], "x": ds_block["x"]}
    data_vars = {"sic_pred": ("time y x".split(), sic_pred), "pred_flag": (("y", "x"), pred_flag)}
    if has_sit and sit_pred is not None:
        data_vars["sit_pred"] = ("time y x".split(), sit_pred)

    out = xr.Dataset(data_vars=data_vars, coords=coords)
    out.attrs["mem_peak_mb"] = mem_peak
    return y0, x0, out


def _decide_next_workers(curr: int, min_w: int, max_w: int, mem_low: float, mem_high: float, cpu_cap: float) -> int:
    if not _HAS_PSUTIL:
        return curr
    mem = psutil.virtual_memory().percent
    cpu = psutil.cpu_percent(interval=0.5)
    if mem > mem_high:
        next_w = max(min_w, curr - 1)
        if next_w < curr: logger.info(f"MEM {mem:.1f}% > {mem_high}%, reducing workers to {next_w}")
        return next_w
    if mem < mem_low and cpu < cpu_cap:
        next_w = min(max_w, curr + 1)
        if next_w > curr: logger.info(f"MEM {mem:.1f}% < {mem_low}% and CPU {cpu:.1f}% < {cpu_cap}%, increasing workers to {next_w}")
        return next_w
    return curr


def forecast_block(
    ds: xr.Dataset, var: str = "sic", block: Tuple[int, int] = (100, 100),
    horizon: int = 12, min_len: int = 120, use_lstm: bool = True,
    save_dir: Optional[str] = None, max_blocks: Optional[int] = None, dry_run: bool = False,
    min_valid_frac: float = 0.6, std_thresh: float = 1e-6, per_month_min: int = 2,
    autoscale: bool = True, min_workers: int = 1, max_workers: Optional[int] = None,
    mem_low: float = 65.0, mem_high: float = 80.0, cpu_cap: float = 90.0,
    progress: Optional["ProgressReporter"] = None, label: Optional[str] = None,
    progress_pixel_step: int = 200, progress_interval: float = 2.0
) -> xr.Dataset:

    assert var in ds, f"变量 {var} 不存在于 ds 中"
    assert all(d in ds.dims for d in ("time", "y", "x")), "ds 需包含 (time,y,x) 维度"

    has_sit = "sit" in ds
    By, Bx = block
    Ny, Nx, T = ds.sizes["y"], ds.sizes["x"], ds.sizes["time"]

    y_starts = list(range(0, Ny, By))
    x_starts = list(range(0, Nx, Bx))

    last_ts = pd.to_datetime(str(ds.time.values[-1]))
    idx_future = pd.date_range(last_ts + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    time_full = np.concatenate([ds.time.values, idx_future.values])

    sic_pred_full = np.full((T + horizon, Ny, Nx), np.nan, dtype=np.float32)
    pred_flag_full = np.zeros((Ny, Nx), dtype=np.uint8)
    sit_pred_full = np.full((T + horizon, Ny, Nx), np.nan, dtype=np.float32) if has_sit else None

    tag = pd.to_datetime(str(ds.time.values[-1])).strftime("%Y%m")
    blocks_dir, out_path, out_part = None, None, None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        blocks_dir = os.path.join(save_dir, "_blocks", tag)
        os.makedirs(blocks_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"ice_forecast_{tag}.nc")
        out_part = out_path + ".part.nc"

    block_indices = [(bi + 1, y0, x0) for bi, (y0, x0) in enumerate(np.array(np.meshgrid(y_starts, x_starts)).T.reshape(-1, 2))]
    total_blocks = len(block_indices)
    logger.info(f"total_blocks={total_blocks}, block={block}, grid=({Ny},{Nx}) horizon={horizon}")

    if max_blocks is not None:
        block_indices = block_indices[: max(0, int(max_blocks))]
        logger.info(f"limit max_blocks={len(block_indices)}")

    def _cached_path(y0, x0): return os.path.join(blocks_dir, f"block_y{y0}_x{x0}.nc") if blocks_dir else None

    compute_tasks = []
    skip_count = 0
    for bi, y0, x0 in block_indices:
        p = _cached_path(y0, x0)
        if p and os.path.exists(p):
            try:
                cached = xr.open_dataset(p)
                by, bx = cached.sizes["y"], cached.sizes["x"]
                sic_pred_full[:, y0:y0+by, x0:x0+bx] = cached["sic_pred"].values
                pred_flag_full[y0:y0+by, x0:x0+bx] = cached["pred_flag"].values
                if has_sit and sit_pred_full is not None and "sit_pred" in cached:
                    sit_pred_full[:, y0:y0+by, x0:x0+bx] = cached["sit_pred"].values
                skip_count += 1
                logger.info(f"[{bi}/{total_blocks}] hit cache for block (y0={y0},x0={x0})")
                # 视为该块完成，发 block_done 事件（无像元细节）
                try:
                    if progress is not None:
                        from ArcticRoute.core.utils.progress import ProgressReporter as _PR
                        cpu = mem = None
                        try:
                            if _HAS_PSUTIL:
                                cpu = psutil.cpu_percent(interval=None)
                                mem = psutil.virtual_memory().percent
                        except Exception:
                            pass
                        progress.emit(kind="block_done", label=(label or ""), block_idx=bi, blocks_total=total_blocks, ok=None, skip=None, cpu=cpu, mem=mem)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Cache read failed for block (y0={y0},x0={x0}): {e}")
                compute_tasks.append((bi, y0, x0))
        else:
            compute_tasks.append((bi, y0, x0))

    completed_blocks = skip_count

    if compute_tasks:
        use_autoscale = autoscale and _HAS_PSUTIL
        if not _HAS_PSUTIL and autoscale:
            logger.warning("psutil not found, autoscale disabled.")
        
        max_w = max_workers if (isinstance(max_workers, int) and max_workers > 0) else (os.cpu_count() or 4)
        min_w = min(min_workers, max_w)
        current_workers = min_w

        task_queue = list(compute_tasks)
        last_health = time.time()
        while task_queue:
            if use_autoscale:
                current_workers = _decide_next_workers(current_workers, min_w, max_w, mem_low, mem_high, cpu_cap)

            # 若内存超过 mem_high+5，则暂停提交，直到恢复
            if _HAS_PSUTIL:
                mem_now = psutil.virtual_memory().percent
                if mem_now > (mem_high + 5):
                    logger.warning("PAUSE submit due to high memory")
                    # 等待直到内存降至 mem_high 以下，期间每15秒打健康日志
                    while True:
                        time.sleep(1.0)
                        mem_now = psutil.virtual_memory().percent
                        if (time.time() - last_health) >= 15.0:
                            cpu_now = psutil.cpu_percent(interval=None)
                            inflight = 0
                            logger.info(f"[HEALTH] cpu={cpu_now:.1f}% mem={mem_now:.1f}% workers={current_workers} inflight={inflight} block={completed_blocks}/{total_blocks} done={skip_count + completed_blocks} skip={skip_count}")
                            last_health = time.time()
                        if mem_now <= mem_high:
                            break
            
            wave_size = current_workers * 2  # 块内限流：在飞任务数
            wave_tasks = task_queue[:wave_size]
            task_queue = task_queue[wave_size:]
            logger.info(f"Starting wave with {len(wave_tasks)} tasks and {current_workers} workers.")

            with ProcessPoolExecutor(max_workers=current_workers) as ex:
                futures: Dict = {}
                for bi, y0, x0 in wave_tasks:
                    y1, x1 = min(Ny, y0 + By), min(Nx, x0 + Bx)
                    sub = ds.isel(y=slice(y0, y1), x=slice(x0, x1))
                    args = (sub, var, has_sit, y0, x0, horizon, min_len, use_lstm, bi, total_blocks, min_valid_frac, std_thresh, per_month_min, (progress.mq if progress is not None else None), (label or ""), int(progress_pixel_step))
                    future = ex.submit(_process_block, args)
                    futures[future] = (y0, x0)

                # 以循环+wait的方式处理完成事件，并周期性打印健康日志
                while futures:
                    done_set, _ = wait(list(futures.keys()), timeout=1.0, return_when=FIRST_COMPLETED)
                    for fu in list(done_set):
                        y0_res, x0_res = futures.pop(fu)
                        try:
                            _, _, out = fu.result()
                            by, bx = out.sizes["y"], out.sizes["x"]
                            sic_pred_full[:, y0_res:y0_res+by, x0_res:x0_res+bx] = out["sic_pred"].values
                            pred_flag_full[y0_res:y0_res+by, x0_res:x0_res+bx] = out["pred_flag"].values
                            if has_sit and sit_pred_full is not None and "sit_pred" in out:
                                sit_pred_full[:, y0_res:y0_res+by, x0_res:x0_res+bx] = out["sit_pred"].values
                            out.attrs["workers_used"] = current_workers
                            if blocks_dir and not dry_run:
                                p = _cached_path(y0_res, x0_res)
                                try:
                                    out.to_netcdf(p, encoding={k: {"zlib": True, "complevel": 2} for k in out.data_vars})
                                    logger.info(f"Cache saved: {p} (peak_mem={out.attrs.get('mem_peak_mb',0):.1f}MB)")
                                except Exception as e:
                                    logger.warning(f"Cache save failed for block (y0={y0_res},x0={x0_res}): {e}")
                        except Exception as e:
                            logger.error(f"Block (y0={y0_res},x0={x0_res}) failed: {e}")
                        finally:
                            completed_blocks += 1

                    if _HAS_PSUTIL and (time.time() - last_health) >= 15.0:
                        cpu_now = psutil.cpu_percent(interval=None)
                        mem_now = psutil.virtual_memory().percent
                        inflight = len(futures)
                        logger.info(f"[HEALTH] cpu={cpu_now:.1f}% mem={mem_now:.1f}% workers={current_workers} inflight={inflight} block={completed_blocks}/{total_blocks} done={skip_count + completed_blocks} skip={skip_count}")
                        last_health = time.time()

    coords = {"time": ("time", time_full), "y": ds["y"], "x": ds["x"]}
    data_vars = {"sic_pred": ("time y x".split(), sic_pred_full), "pred_flag": (("y", "x"), pred_flag_full)}
    if has_sit and sit_pred_full is not None:
        data_vars["sit_pred"] = ("time y x".split(), sit_pred_full)
    ds_out = xr.Dataset(data_vars=data_vars, coords=coords)

    if save_dir and not dry_run:
        enc = {k: {"zlib": True, "complevel": 4} for k in ds_out.data_vars}
        tmp = out_part or os.path.join(save_dir, "_tmp_output.part.nc")
        ds_out.to_netcdf(tmp, encoding=enc)
        final = out_path or os.path.join(save_dir, "ice_forecast.nc")
        os.replace(tmp, final)
        logger.info(f"Saved: {final}")

    return ds_out
