from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio.enums import Resampling
import rioxarray  # type: ignore
import xarray as xr

try:
    from skimage.filters import threshold_otsu
    from skimage.morphology import opening

    try:
        from skimage.morphology import footprint_rectangle as _sk_footprint_rectangle

        def _morph_kernel() -> np.ndarray:
            return _sk_footprint_rectangle(3, 3)

    except ImportError:  # pragma: no cover - compatibility with older versions
        from skimage.morphology import square as _sk_square  # type: ignore

        def _morph_kernel() -> np.ndarray:
            return _sk_square(3)

except ModuleNotFoundError:  # pragma: no cover - optional dependency
    threshold_otsu = None  # type: ignore[assignment]
    opening = None  # type: ignore[assignment]

    def _morph_kernel() -> np.ndarray:  # type: ignore[return-type]
        raise RuntimeError("morphology unavailable")

from ArcticRoute.exceptions import ArcticRouteError
from logging_config import get_logger

from ..interfaces import Predictor
from ...io.stac_ingest import extract_env_grid

LOGGER = get_logger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"


def _parse_env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _parse_env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


DEFAULT_TIME_PAD_DAYS = _parse_env_int("AR_CV_TIME_PAD_DAYS", 14)
DEFAULT_MAX_CLOUD = _parse_env_float("AR_CV_MAX_CLOUD", 40.0)
DEFAULT_MOSAIC = os.getenv("AR_CV_MOSAIC", "median")
MIN_SAMPLES = _parse_env_int("AR_CV_MIN_SAMPLES", 10_000)
MIN_VALID_FRACTION = _parse_env_float("AR_CV_MIN_VALID_FRAC", 0.01)


def compute_otsu_mask(
    values: np.ndarray,
    *,
    method: str = "otsu",
    min_samples: int = MIN_SAMPLES,
    min_fraction: float = MIN_VALID_FRACTION,
) -> tuple[np.ndarray, float, dict]:
    """Return a stable Otsu mask/threshold pair, even for degenerate inputs."""
    arr = np.asarray(values, dtype=np.float32)
    info: dict = {"method": method}
    mask = np.zeros_like(arr, dtype=bool)

    if threshold_otsu is None:
        LOGGER.warning("skimage.filters.threshold_otsu unavailable; returning empty mask.")
        info.update(reason="skimage_missing")
        return mask, float("nan"), info

    total_count = int(arr.size)
    if total_count == 0:
        LOGGER.info("Received empty array for Otsu thresholding; returning empty mask.")
        info.update(reason="empty_candidate", total=0, valid=0, fraction=0.0, nan_fraction=1.0)
        return mask, float("nan"), info

    finite = np.isfinite(arr)
    valid_count = int(np.count_nonzero(finite))
    valid_fraction = float(valid_count / total_count) if total_count else 0.0
    nan_fraction = 1.0 - valid_fraction
    info.update(valid=valid_count, total=total_count, fraction=valid_fraction, nan_fraction=nan_fraction)

    if valid_count == 0:
        LOGGER.warning("Otsu candidate has no finite pixels; returning empty mask.")
        info.update(reason="no_finite_pixels")
        return mask, float("nan"), info

    if nan_fraction > 0.3:
        LOGGER.warning(
            "Otsu candidate has %.1f%% NaN/inf pixels; returning empty mask.", nan_fraction * 100.0
        )
        info.update(reason="too_many_nan")
        return mask, float("nan"), info

    if valid_count < max(1, min_samples) or valid_fraction < min_fraction:
        LOGGER.info(
            "Otsu candidate rejected due to insufficient samples (valid=%s, total=%s, fraction=%.4f).",
            valid_count,
            total_count,
            valid_fraction,
        )
        info.update(
            reason="insufficient_samples",
            min_samples=min_samples,
            min_fraction=min_fraction,
        )
        return mask, float("nan"), info

    finite_values = np.nan_to_num(arr[finite], nan=0.0, posinf=0.0, neginf=0.0, copy=True)
    v_min = float(np.min(finite_values))
    v_max = float(np.max(finite_values))
    if not np.isfinite(v_min) or not np.isfinite(v_max):
        LOGGER.warning("Otsu candidate has invalid range; returning empty mask.")
        info.update(reason="invalid_range")
        return mask, float("nan"), info

    span = v_max - v_min
    if span <= 1e-9:
        LOGGER.info("Otsu candidate is constant (span %.3e); returning empty mask.", span)
        info.update(reason="flat_field")
        return mask, float("nan"), info

    normalised = np.clip((finite_values - v_min) / span, 0.0, 1.0)
    try:
        threshold_normalised = float(threshold_otsu(normalised))
    except ValueError as err:
        LOGGER.warning("Otsu thresholding failed (%s); returning empty mask.", err)
        info.update(reason="otsu_failed")
        return mask, float("nan"), info

    mask_valid = normalised > threshold_normalised
    mask = np.zeros_like(arr, dtype=bool)
    mask[finite] = mask_valid

    threshold_actual = threshold_normalised * span + v_min
    info.update(
        reason="ok",
        threshold=threshold_actual,
        threshold_normalised=threshold_normalised,
        ndwi_mean=float(np.mean(finite_values)),
        ndwi_std=float(np.std(finite_values)),
    )
    info["coverage_pct"] = float(np.mean(mask_valid) * 100.0) if mask_valid.size else 0.0
    return mask, threshold_actual, info


def _run_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Convenience wrapper returning just the mask/threshold pair."""
    mask, threshold, _ = compute_otsu_mask(
        img,
        method="_run_otsu",
        min_samples=1,
        min_fraction=0.0,
    )
    return mask, threshold


def _load_env_defaults() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _resolve_project_path(env_key: str, default: str) -> Path:
    value = os.environ.get(env_key, default)
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _calc_basic_stats(array: np.ndarray) -> dict:
    finite = np.isfinite(array)
    if not np.count_nonzero(finite):
        return {"mean": None, "max": None, "coverage_pct": 0.0, "valid_ratio_pct": 0.0}
    valid = array[finite]
    mean_val = float(np.mean(valid))
    max_val = float(np.max(valid))
    coverage_pct = float(np.mean(valid > 0.5) * 100.0)
    valid_ratio = float((np.count_nonzero(finite) / array.size) * 100.0) if array.size else 0.0
    return {
        "mean": mean_val,
        "max": max_val,
        "coverage_pct": coverage_pct,
        "valid_ratio_pct": valid_ratio,
    }


class SatCVPredictor(Predictor):
    """Satellite predictor that attaches mosaicked Sentinel bands."""

    def __init__(
        self,
        env_path: Path,
        var_name: str,
        *,
        mission: Optional[str] = None,
        max_items: int = 4,
        time_pad_days: Optional[int] = None,
        max_cloud: Optional[float] = None,
        mosaic_mode: Optional[str] = None,
    ) -> None:
        _load_env_defaults()
        self.env_path = Path(env_path)
        self.var_name = var_name
        self.mission = (mission or os.environ.get("DEFAULT_MISSION", "S2")).upper()
        self.stac_source = os.environ.get("DEFAULT_STAC_SOURCE", "MPC").upper()
        self.max_items = int(max_items)
        self.processed_dir = _resolve_project_path("PROCESSED_DIR", "data_processed")
        self.logs_dir = _resolve_project_path("LOG_DIR", "logs")
        self.cv_cache_dir = _resolve_project_path("CV_CACHE_DIR", "data_processed/cv_cache")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.cv_cache_dir.mkdir(parents=True, exist_ok=True)
        env_time_pad = _parse_env_int("AR_CV_TIME_PAD_DAYS", DEFAULT_TIME_PAD_DAYS)
        try:
            resolved_time_pad = env_time_pad if time_pad_days is None else int(time_pad_days)
        except (TypeError, ValueError):
            resolved_time_pad = env_time_pad
        self.time_pad_days = max(0, resolved_time_pad)
        env_cloud = _parse_env_float("AR_CV_MAX_CLOUD", DEFAULT_MAX_CLOUD)
        resolved_cloud = env_cloud if max_cloud is None else float(max_cloud)
        if not np.isfinite(resolved_cloud) or resolved_cloud <= 0.0:
            self.max_cloud = None
        else:
            self.max_cloud = float(min(resolved_cloud, 100.0))
        mosaic_candidate = mosaic_mode or os.getenv("AR_CV_MOSAIC") or (DEFAULT_MOSAIC or "median")
        mosaic_token = str(mosaic_candidate).strip().lower()
        if mosaic_token not in {"median", "mean", "best"}:
            mosaic_token = "median"
        self.mosaic_mode = mosaic_token
        self._dataset: Optional[xr.Dataset] = None
        self._mosaic_path: Optional[Path] = None
        self._last_query_datetime: Optional[str] = None
        self._last_query_window: Optional[tuple[str, str]] = None

    # ---------------------- helpers ---------------------- #
    def _available_mosaics(self) -> List[Path]:
        pattern = f"sat_mosaic_*_{self.mission}.tif"
        return sorted(self.processed_dir.glob(pattern), key=lambda p: p.stat().st_mtime)

    def _compute_ice_stats(self, ice_da: xr.DataArray) -> dict:
        if "time" in ice_da.dims and ice_da.sizes.get("time", 1) > 1:
            ice_slice = ice_da.isel(time=0).drop_vars("time") if "time" in ice_da.coords else ice_da.isel(time=0)
        else:
            ice_slice = ice_da
        values = np.asarray(ice_slice.values, dtype=np.float32)
        return _calc_basic_stats(values)

    def _save_ice_prob_cache(self, ice_da: xr.DataArray, stats: dict) -> None:
        try:
            if "time" in ice_da.dims and ice_da.sizes.get("time", 1) > 1:
                ice_slice = ice_da.isel(time=0).drop_vars("time") if "time" in ice_da.coords else ice_da.isel(time=0)
            else:
                ice_slice = ice_da
            ds = xr.Dataset({"ice_prob": ice_slice.astype("float32")})
            out_nc = self.cv_cache_dir / "ice_prob_latest.nc"
            encoding = {"ice_prob": {"dtype": "float32", "zlib": True, "complevel": 4}}
            ds.to_netcdf(out_nc, encoding=encoding)
            digest = hashlib.sha1(out_nc.read_bytes()).hexdigest()
            summary = {
                "mean": stats.get("mean"),
                "max": stats.get("max"),
                "coverage": (stats.get("coverage_pct") / 100.0) if stats.get("coverage_pct") is not None else None,
                "coverage_pct": stats.get("coverage_pct"),
                "valid_ratio_pct": stats.get("valid_ratio_pct"),
                "sha1": digest,
                "path": str(out_nc),
            }
            out_json = self.cv_cache_dir / "ice_prob_latest.json"
            out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            LOGGER.info("Ice probability cached to %s", out_nc)
        except Exception as err:  # pragma: no cover - IO heavy
            LOGGER.warning("Failed to cache ice probability: %s", err)

    def _find_existing_mosaic(self) -> Optional[Path]:
        candidates = self._available_mosaics()
        return candidates[-1] if candidates else None

    def _time_to_date(self, tidx: int) -> str:
        try:
            with xr.open_dataset(self.env_path) as ds:
                times = ds.coords.get("time")
                if times is None or times.size == 0:
                    raise ValueError("time coordinate missing")
                idx = max(0, min(int(tidx), times.size - 1))
                value = np.datetime64(times.values[idx], "D")
                return str(value)
        except Exception:
            return datetime.utcnow().strftime("%Y-%m-%d")

    def _compute_bbox(self) -> str:
        with xr.open_dataset(self.env_path) as ds:
            lat = ds.coords.get("latitude") or ds.coords.get("lat")
            lon = ds.coords.get("longitude") or ds.coords.get("lon")
            if lat is None or lon is None:
                raise KeyError("latitude/longitude coordinates not found in env dataset")
            north = float(np.max(lat.values))
            south = float(np.min(lat.values))
            west = float(np.min(lon.values))
            east = float(np.max(lon.values))
        return f"{north},{west},{south},{east}"

    def _time_window_bounds(self, date_str: str) -> tuple[str, str]:
        token = (date_str or "").strip()
        if not token:
            token = datetime.utcnow().strftime("%Y-%m-%d")
        try:
            base = np.datetime64(token, "D")
        except ValueError:
            base = np.datetime64(datetime.utcnow().date(), "D")
        pad = np.timedelta64(self.time_pad_days, "D")
        start = base - pad
        end = base + pad
        return str(start), str(end)

    def _build_stac_datetime(self, date_str: str) -> str:
        token = (date_str or "").strip()
        if not token:
            token = datetime.utcnow().strftime("%Y-%m-%d")
        if "/" in token:
            return token
        if self.time_pad_days <= 0:
            return token
        start, end = self._time_window_bounds(token)
        if start == end:
            return start
        return f"{start}/{end}"

    def _run_stac_fetch(self, bbox: str, date_str: str) -> Path:
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "stac_fetch.py"),
            "--bbox",
            bbox,
            "--mission",
            self.mission,
            "--source",
            self.stac_source,
            "--limit",
            str(self.max_items),
            "--lazy",
        ]
        datetime_param = self._build_stac_datetime(date_str)
        cmd.extend(["--date", datetime_param])
        if self.max_cloud is not None and self.mission != "S1":
            cmd.extend(["--max-cloud", f"{self.max_cloud:g}"])
        self._last_query_datetime = datetime_param
        if "/" in datetime_param:
            start_str, end_str = datetime_param.split("/", 1)
            self._last_query_window = (start_str, end_str)
        elif self.time_pad_days > 0:
            self._last_query_window = self._time_window_bounds(date_str)
        else:
            self._last_query_window = (datetime_param, datetime_param)
        log_hint = f"[CV] STAC query datetime={datetime_param}"
        if self.max_cloud is not None and self.mission != "S1":
            log_hint += f", max_cloud={self.max_cloud:g}%"
        if self.time_pad_days > 0:
            log_hint += f", pad_days={self.time_pad_days}"
        log_hint += f", mosaic={self.mosaic_mode}"
        LOGGER.info(log_hint)
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        return self._latest_stac_log(datetime_param)

    def _normalise_env_arg(self, path: Path) -> str:
        try:
            return path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return str(path)

    def _run_mosaic(self, log_path: Path, date_str: str) -> Path:
        date_token = date_str.replace("-", "")
        out_path = self.processed_dir / f"sat_mosaic_{date_token}_{self.mission}.tif"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "cog_mosaic_to_grid.py"),
            "--cogs",
            self._normalise_env_arg(log_path),
            "--env",
            self._normalise_env_arg(self.env_path),
            "--mission",
            self.mission,
            "--out",
            self._normalise_env_arg(out_path),
        ]
        if self.mosaic_mode:
            cmd.extend(["--mosaic", self.mosaic_mode])
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
        if not out_path.exists():
            raise FileNotFoundError(f"Mosaic generation failed; missing {out_path}")
        return out_path

    def _latest_stac_log(self, date_str: str) -> Path:
        tokens = {
            date_str,
            date_str.replace("-", ""),
            date_str.replace("/", "_"),
            date_str.replace("/", ""),
        }
        tokens.add("".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in date_str))
        pattern = f"stac_results_{self.mission.lower()}_*.json"
        candidates = sorted(self.logs_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
        for path in candidates:
            if any(token in path.name for token in tokens):
                return path
        if candidates:
            return candidates[0]
        raise FileNotFoundError("No STAC log available after fetch operation.")

    def _load_mosaic_dataset(self, mosaic_path: Path) -> xr.Dataset:
        lat_values, lon_values, transform = extract_env_grid(self.env_path)
        target_shape = (len(lat_values), len(lon_values))

        da = rioxarray.open_rasterio(mosaic_path)
        aligned = da.rio.reproject(
            dst_crs="EPSG:4326",
            transform=transform,
            shape=target_shape,
            resampling=Resampling.bilinear,
        )
        aligned = aligned.assign_coords(
            {
                "y": ("y", lat_values),
                "x": ("x", lon_values),
            }
        ).rename({"y": "latitude", "x": "longitude"})

        with rasterio.open(mosaic_path) as src:
            band_labels = [
                desc if desc else f"band{idx}"
                for idx, desc in enumerate(src.descriptions, start=1)
            ]
        aligned = aligned.assign_coords(band=band_labels).astype("float32")
        aligned.name = "sat_bands"
        aligned.attrs.update(
            {
                "mission": self.mission,
                "source": self.stac_source,
                "path": str(mosaic_path),
                "mosaic_mode": self.mosaic_mode,
            }
        )
        dataset = xr.Dataset({"sat_bands": aligned})
        window = self._last_query_window
        dataset.attrs["cv_query"] = {
            "datetime": self._last_query_datetime,
            "time_pad_days": self.time_pad_days,
            "max_cloud": self.max_cloud,
            "mosaic_mode": self.mosaic_mode,
            "source": self.stac_source,
            "time_window": {"start": window[0], "end": window[1]} if window else None,
        }

        ice_da, info = self._compute_ice_prob(aligned)
        dataset.attrs["ice_prob_info"] = info
        if ice_da is not None:
            dataset["ice_prob"] = ice_da
            stats = self._compute_ice_stats(ice_da)
            self._save_ice_prob_cache(ice_da, stats)
            method = info.get("method", "unknown")
            ratio_mean = info.get("ratio_mean")
            mean_val = stats.get("mean")
            max_val = stats.get("max")
            coverage_val = stats.get("coverage_pct")
            valid_val = stats.get("valid_ratio_pct")
            ratio_txt = f"{ratio_mean:.4f}" if ratio_mean is not None and np.isfinite(ratio_mean) else "nan"
            mean_txt = f"{mean_val:.4f}" if mean_val is not None else "nan"
            max_txt = f"{max_val:.4f}" if max_val is not None else "nan"
            coverage_txt = f"{coverage_val:.2f}" if coverage_val is not None else "nan"
            valid_txt = f"{valid_val:.2f}" if valid_val is not None else "nan"
            LOGGER.info(
                "Ice probability computed (mission=%s, method=%s, ratio_mean=%s, mean=%s, max=%s, coverage=%s%%, valid=%s%%)",
                self.mission,
                method,
                ratio_txt,
                mean_txt,
                max_txt,
                coverage_txt,
                valid_txt,
            )
        else:
            reason = info.get("reason", "unknown") if isinstance(info, dict) else "unknown"
            LOGGER.warning("Ice probability unavailable (reason=%s).", reason)

        return dataset

    def _select_band(self, sat_bands: xr.DataArray, aliases: Sequence[str]) -> Optional[xr.DataArray]:
        alias_upper = [alias.upper() for alias in aliases]
        for index, name in enumerate(sat_bands.coords["band"].values):
            if str(name).upper() in alias_upper:
                return sat_bands.isel(band=index)
        return None

    def _run_otsu(self, candidate: xr.DataArray, method_desc: str) -> Tuple[xr.DataArray, dict]:
        values = np.asarray(candidate.values, dtype=np.float32)
        mask_bool, threshold, info = compute_otsu_mask(
            values,
            method=method_desc,
            min_samples=MIN_SAMPLES,
            min_fraction=MIN_VALID_FRACTION,
        )

        if info.get("reason") != "ok":
            LOGGER.warning("Otsu fallback (reason=%s, method=%s)", info.get("reason"), method_desc)

        valid = np.isfinite(values)
        if opening is not None and info.get("reason") == "ok":
            try:
                kernel = _morph_kernel()
                opened = opening(mask_bool.astype(np.uint8), kernel).astype(bool)
                mask_bool = opened & valid
            except Exception as err:
                LOGGER.warning("Morphology opening failed (%s); continuing without post-processing.", err)
                mask_bool &= valid
        else:
            mask_bool &= valid

        mask_float = mask_bool.astype(np.float32)
        mask_float[~valid] = np.nan

        coverage_pct = float(np.nanmean(mask_float) * 100.0) if np.count_nonzero(valid) else 0.0
        ratio_mean = float(np.nanmean(values[valid])) if np.count_nonzero(valid) else None
        ndwi_std = float(np.nanstd(values[valid])) if np.count_nonzero(valid) else None

        info.setdefault("threshold", threshold)
        info.setdefault("ndwi_mean", ratio_mean if ratio_mean is not None else float("nan"))
        info.setdefault("ndwi_std", ndwi_std if ndwi_std is not None else float("nan"))
        info.setdefault("coverage", coverage_pct)
        info.setdefault("valid_count", int(np.count_nonzero(valid)))

        ice_da = xr.DataArray(
            mask_float.astype(np.float32),
            coords=candidate.coords,
            dims=candidate.dims,
            name="ice_prob",
        )
        ice_da.attrs.update(info)
        return ice_da, info

    def _compute_s2_ice_prob(self, sat_bands: xr.DataArray) -> Tuple[Optional[xr.DataArray], dict]:
        green = self._select_band(sat_bands, ("B03", "B3", "GREEN", "G"))
        nir = self._select_band(sat_bands, ("B08", "B8", "NIR"))
        if green is None or nir is None:
            LOGGER.warning("S2 mosaic missing required bands for NDWI (B03/B08).")
            return None, {"reason": "missing_bands"}
        denom = green + nir
        denom = denom.where(np.abs(denom) > 1e-6)
        ndwi = (green - nir) / denom
        ndwi = ndwi.where(np.isfinite(ndwi))
        return self._run_otsu(ndwi, "NDWI/Otsu")

    def _compute_s1_ice_prob(self, sat_bands: xr.DataArray) -> Tuple[Optional[xr.DataArray], dict]:
        vv = self._select_band(sat_bands, ("VV", "SIGMA0_VV", "BVV"))
        if vv is None:
            LOGGER.warning("S1 mosaic missing VV band; cannot estimate ice probability.")
            return None, {"reason": "missing_bands"}
        vv_amp = np.log1p(np.abs(vv))
        vh = self._select_band(sat_bands, ("VH", "SIGMA0_VH", "BVH"))
        if vh is not None:
            vh_amp = np.log1p(np.abs(vh))
            ratio = vv_amp / vh_amp.where(vh_amp > 1e-6)
            ratio = ratio.where(np.isfinite(ratio))
            ice_da, info = self._run_otsu(ratio, "SAR/Otsu(VV/VH)")
            if ice_da is not None:
                return ice_da, info
        return self._run_otsu(vv_amp, "SAR/Otsu(VV)")

    def _compute_ice_prob(self, sat_bands: xr.DataArray) -> Tuple[Optional[xr.DataArray], dict]:
        values = np.asarray(sat_bands.values, dtype=np.float32)
        if not np.any(np.isfinite(values)):
            LOGGER.warning("Satellite bands contain no finite values; skipping ice probability.")
            return None, {"reason": "all_nan"}
        try:
            if self.mission == "S2":
                return self._compute_s2_ice_prob(sat_bands)
            if self.mission == "S1":
                return self._compute_s1_ice_prob(sat_bands)
        except Exception as err:  # pragma: no cover - defensive
            LOGGER.warning("Ice probability computation failed: %s", err)
            return None, {"reason": f"exception:{err}"}
        return None, {"reason": "unsupported_mission"}

    # ---------------------- Predictor API ---------------------- #
    def prepare(self, base_time_index: int) -> xr.Dataset:
        try:
            mosaic_path = self._find_existing_mosaic()
            if mosaic_path is None:
                date_str = self._time_to_date(base_time_index)
                bbox = self._compute_bbox()
                LOGGER.info(
                    "Fetching Sentinel assets (mission=%s, source=%s, date=%s)",
                    self.mission,
                    self.stac_source,
                    date_str,
                )
                log_path = self._run_stac_fetch(bbox, date_str)
                LOGGER.info("STAC fetch complete (log=%s)", log_path)
                mosaic_path = self._run_mosaic(log_path, date_str)
            else:
                LOGGER.info("Using cached mosaic: %s", mosaic_path)

            dataset = self._load_mosaic_dataset(mosaic_path)
        except ArcticRouteError:
            raise
        except FileNotFoundError as err:
            LOGGER.exception("Satellite mosaic missing")
            raise ArcticRouteError("ARC-CV-404", "Satellite mosaic missing", detail=str(err)) from err
        except subprocess.CalledProcessError as err:
            LOGGER.exception("STAC ingestion failed")
            raise ArcticRouteError("ARC-CV-102", "STAC ingestion failed", detail=str(err)) from err
        except Exception as err:
            LOGGER.exception("CV predictor failed")
            raise ArcticRouteError("ARC-CV-000", "CV predictor failed", detail=str(err)) from err

        self._dataset = dataset
        self._mosaic_path = mosaic_path
        bands = list(dataset["sat_bands"].coords["band"].values)
        shape = dataset["sat_bands"].shape
        LOGGER.info(
            "Satellite mosaic attached (mission=%s, shape=%s, bands=%s)",
            self.mission,
            shape,
            bands,
        )
        return dataset

    def predict(self, var: Optional[str], t0: int, horizon: Optional[int]) -> Optional[xr.DataArray]:
        return None
