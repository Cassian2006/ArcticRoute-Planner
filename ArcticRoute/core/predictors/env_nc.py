from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import xarray as xr

from ArcticRoute.exceptions import ArcticRouteError
from logging_config import get_logger

from ..interfaces import Predictor, PredictorOutput
from ...io import loaders

logger = get_logger(__name__)


@lru_cache(maxsize=8)
def load_corridor_prob_safe(path: str) -> tuple[Optional[xr.DataArray], Dict[str, Any]]:
    """
    尝试加载 corridor_prob，并进行质量自检：
    - 文件/变量缺失 → 返回 (None, info)
    - attrs['constant_like'] 为 True → 返回 (None, info) 并记录 warn
    - 统计 p5/p95 扩展度过小（spread<pthr）→ 仍返回对象，但 info['low_contrast']=True
    绝不抛异常，调用侧可据此关闭/降权。
    """
    info: Dict[str, Any] = {"ok": False, "path": path}
    pthr = 0.01
    try:
        ds = loaders.load_dataset(Path(path))
        var = None
        for cand in ("corridor_prob", "prob", "mask"):
            if cand in ds:
                var = cand
                break
        if var is None:
            info["error"] = "variable_missing"
            try:
                ds.close()
            except Exception:
                pass
            logger.warning("corridor_prob 安全加载失败：变量缺失 (%s)", path)
            return None, info
        da = ds[var].load()
        info["shape"] = tuple(da.shape)
        const_like = bool(getattr(da, "attrs", {}).get("constant_like", False))
        arr = np.asarray(da.values, dtype=float)
        finite = np.isfinite(arr)
        if not finite.any():
            info["error"] = "all_nan"
            try:
                ds.close()
            except Exception:
                pass
            logger.warning("corridor_prob 安全加载失败：全 NaN (%s)", path)
            return None, info
        vals = arr[finite]
        p5 = float(np.nanpercentile(vals, 5))
        p95 = float(np.nanpercentile(vals, 95))
        spread = float(p95 - p5)
        info.update({"min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)), "mean": float(np.nanmean(vals)), "p5": p5, "p95": p95, "spread": spread, "constant_like": const_like})
        if const_like:
            logger.warning("关闭走廊偏好：corridor_prob.attrs.constant_like=True (%s)", path)
            try:
                ds.close()
            except Exception:
                pass
            return None, info
        info["ok"] = True
        if spread < pthr:
            info["low_contrast"] = True
            logger.warning("corridor_prob 对比度偏低（spread=%.6f<pthr=%.6f），将继续使用但效果有限：%s", spread, pthr, path)
        try:
            ds.close()
        except Exception:
            pass
        # 统一变量名
        if da.name != "corridor_prob":
            da = xr.DataArray(da.values, dims=da.dims, coords=da.coords, name="corridor_prob")
        return da, info
    except FileNotFoundError:
        info["error"] = "file_missing"
        logger.warning("关闭走廊偏好：corridor_prob 文件不存在 (%s)", path)
        return None, info
    except Exception as e:
        info["error"] = f"exception:{e}"
        logger.warning("关闭走廊偏好：corridor_prob 加载异常 (%s): %s", path, e)
        return None, info


def _coord_name(obj: xr.Dataset | xr.DataArray, candidates: Tuple[str, ...]) -> str:
    coords = obj.coords if isinstance(obj, xr.DataArray) else obj.coords
    for name in candidates:
        if name in coords:
            return name
    raise KeyError(f"missing coordinate among {candidates}; available: {list(coords)}")


def _make_bbox_key(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    return tuple(round(float(v), 4) for v in bbox)


@lru_cache(maxsize=8)
def _load_env_base(
    env_path_str: str,
    var_name: str,
    bbox_key: Optional[Tuple[float, float, float, float]],
    coarsen: int,
) -> Tuple[xr.DataArray, np.ndarray, np.ndarray]:
    bbox = tuple(bbox_key) if bbox_key is not None else None
    env_path = Path(env_path_str)

    ds_env = loaders.load_dataset(env_path, bbox=bbox, coarsen=coarsen)
    if var_name not in ds_env:
        raise KeyError(f"risk dataset missing variable {var_name}")
    risk_da = ds_env[var_name].load()

    lat_coord = _coord_name(ds_env, ("latitude", "lat"))
    lon_coord = _coord_name(ds_env, ("longitude", "lon"))
    lat = np.asarray(ds_env[lat_coord].values, dtype="float32")
    lon = np.asarray(ds_env[lon_coord].values, dtype="float32")
    ds_env.close()

    return risk_da, lat, lon


def _build_predictor_output(
    env_path_str: str,
    var_name: str,
    corridor_path_str: Optional[str],
    accident_path_str: Optional[str],
    accident_mode: Optional[str],
    base_time_index: int,
    bbox_key: Optional[Tuple[float, float, float, float]],
    coarsen: int,
) -> PredictorOutput:
    risk_da, lat, lon = _load_env_base(env_path_str, var_name, bbox_key, coarsen)
    bbox = tuple(bbox_key) if bbox_key is not None else None

    corridor_da = None
    if corridor_path_str:
        template_ds = risk_da.to_dataset(name=var_name)
        da_safe, info = load_corridor_prob_safe(corridor_path_str)
        if da_safe is not None:
            try:
                ds_tmp = da_safe.to_dataset(name="corridor_prob")
                corridor_ds = loaders.align_corridor(template_ds, ds_tmp)
                corridor_da = corridor_ds["corridor_prob"].load()
            except Exception:
                # 回退到缓存对齐（完整路径）
                try:
                    corridor_ds = loaders.cached_corridor_aligned(Path(corridor_path_str), template_ds, bbox, coarsen)
                    corridor_da = corridor_ds["corridor_prob"].load()
                except Exception:
                    corridor_da = None
        else:
            logger.warning("关闭走廊偏好（quality check 未通过）：%s; info=%s", corridor_path_str, info)
        if corridor_da is not None:
            max_val = float(np.nanmax(corridor_da.values))
            if max_val <= 0.0:
                complement = 1.0 - np.clip(risk_da.values, 0.0, 1.0)
                comp_min = float(np.nanmin(complement))
                comp_max = float(np.nanmax(complement))
                if comp_max > comp_min:
                    scaled = (complement - comp_min) / (comp_max - comp_min)
                    scaled = np.clip(scaled, 0.0, 1.0) ** 0.25
                else:
                    scaled = np.zeros_like(complement)
                corridor_da = xr.DataArray(
                    scaled,
                    dims=risk_da.dims,
                    coords=risk_da.coords,
                    name="corridor_prob",
                )

    accident_da = None
    incident_lat = None
    incident_lon = None
    incident_time = None
    if accident_path_str:
        accident_da, incident_lat, incident_lon, incident_time = loaders.cached_accident_resample(
            Path(accident_path_str),
            risk_da,
            bbox,
            coarsen,
            accident_mode=accident_mode,
        )

    return PredictorOutput(
        risk=risk_da,
        corridor=corridor_da,
        lat=lat,
        lon=lon,
        base_time_index=base_time_index,
        accident=accident_da,
        incident_lat=incident_lat,
        incident_lon=incident_lon,
        incident_time=incident_time,
        accident_mode=accident_mode,
        accident_source=accident_path_str,
    )


class EnvNCPredictor(Predictor):
    """Load risk and corridor tensors from NetCDF files."""

    def __init__(
        self,
        env_path: Path,
        var_name: str,
        corridor_path: Optional[Path] = None,
        accident_path: Optional[Path] = None,
        accident_mode: Optional[str] = None,
        bbox: Optional[Tuple[float, float, float, float]] = None,
        coarsen: int = 1,
    ):
        self.env_path = env_path
        self.var_name = var_name
        self.corridor_path = corridor_path
        self.accident_path = accident_path
        self.accident_mode = accident_mode.lower() if accident_mode else None
        self.bbox = bbox
        self.coarsen = coarsen

    def prepare(self, base_time_index: int) -> PredictorOutput:
        env_path_str = str(self.env_path.resolve())
        corridor_path_str = str(self.corridor_path.resolve()) if self.corridor_path else None
        accident_path_str = str(self.accident_path.resolve()) if self.accident_path else None
        bbox_key = _make_bbox_key(self.bbox)

        before_hits = _load_env_base.cache_info().hits
        try:
            output = _build_predictor_output(
                env_path_str,
                self.var_name,
                corridor_path_str,
                accident_path_str,
                self.accident_mode,
                base_time_index,
                bbox_key,
                self.coarsen,
            )
        except ArcticRouteError:
            raise
        except FileNotFoundError as err:
            logger.exception("Environment dataset missing: %s", env_path_str)
            raise ArcticRouteError("ARC-ENV-404", "Environment dataset missing", detail=str(err)) from err
        except KeyError as err:
            logger.exception("Environment dataset invalid (variable=%s)", self.var_name)
            raise ArcticRouteError("ARC-ENV-002", "Environment dataset invalid", detail=str(err)) from err
        except Exception as err:
            logger.exception("Failed to prepare environment predictor")
            raise ArcticRouteError("ARC-ENV-000", "Environment predictor failed", detail=str(err)) from err

        after_hits = _load_env_base.cache_info().hits
        if after_hits > before_hits:
            logger.debug("Environment predictor cache hit")
        return output
