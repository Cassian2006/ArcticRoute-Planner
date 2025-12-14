from __future__ import annotations

import hashlib
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

from logging_config import get_logger

logger = get_logger(__name__)


def auto_open_dataset(path: Path) -> xr.Dataset:
    """自动选择可用引擎打开 NetCDF。"""
    candidates = [None, "h5netcdf", "netcdf4", "scipy"]
    available = set(xr.backends.list_engines())
    errors = []
    for engine in candidates:
        if engine and engine not in available:
            continue
        kwargs = {"engine": engine} if engine else {}
        try:
            return xr.open_dataset(path, **kwargs)
        except Exception as err:  # pragma: no cover - 仅在引擎失败时报错
            errors.append((engine or "default", str(err)))
    raise RuntimeError(f"无法打开 {path}，尝试过的 engine: {errors}")


def _select_coord(ds: xr.Dataset, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in ds.coords:
            return name
    raise KeyError(f"数据集中缺少坐标: 尝试 {candidates} 未命中，现有 {list(ds.coords)}")


def subset_bbox(
    ds: xr.Dataset,
    bbox: Optional[Tuple[float, float, float, float]],
) -> xr.Dataset:
    """根据 N,W,S,E 子集化数据。"""
    if not bbox:
        return ds
    north, west, south, east = bbox
    lat_min, lat_max = sorted((south, north))
    lon_min, lon_max = sorted((west, east))

    lat_name = _select_coord(ds, ("latitude", "lat"))
    lon_name = _select_coord(ds, ("longitude", "lon"))

    lat_values = ds[lat_name].values
    lon_values = ds[lon_name].values

    if lat_values[0] > lat_values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    if lon_values[0] > lon_values[-1]:
        lon_slice = slice(lon_max, lon_min)
    else:
        lon_slice = slice(lon_min, lon_max)

    return ds.sel({lat_name: lat_slice, lon_name: lon_slice})


def apply_coarsen(ds: xr.Dataset, factor: int) -> xr.Dataset:
    """对纬度/经度进行降采样。"""
    if factor <= 1:
        return ds
    if factor not in (2, 3):
        raise ValueError("coarsen 仅支持因子 1/2/3")
    lat_name = _select_coord(ds, ("latitude", "lat"))
    lon_name = _select_coord(ds, ("longitude", "lon"))
    return ds.coarsen({lat_name: factor, lon_name: factor}, boundary="trim").mean()


def load_dataset(
    path: Path,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    coarsen: int = 1,
) -> xr.Dataset:
    ds = auto_open_dataset(path)
    ds = subset_bbox(ds, bbox)
    ds = apply_coarsen(ds, coarsen)
    return ds


def parse_bbox(value: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
    if not value:
        return None
    parts = [p.strip() for p in value.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError("bbox 需要四个数值: N W S E")
    north, west, south, east = map(float, parts)
    return north, west, south, east


def hash_paths(paths: Iterable[Optional[Path]]) -> str:
    sha1 = hashlib.sha1()
    for path in paths:
        if not path or not Path(path).exists():
            continue
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(1024 * 1024)
                if not chunk:
                    break
                sha1.update(chunk)
    return sha1.hexdigest()


def align_corridor(env_ds: xr.Dataset, corr_ds: xr.Dataset) -> xr.Dataset:
    """将 corridor 数据对齐到 env 数据集的坐标与时间。"""
    env_lat = _select_coord(env_ds, ("latitude", "lat"))
    env_lon = _select_coord(env_ds, ("longitude", "lon"))
    env_time = _select_coord(env_ds, ("time",))

    corr_lat = _select_coord(corr_ds, ("latitude", "lat"))
    corr_lon = _select_coord(corr_ds, ("longitude", "lon"))
    corr_time = _select_coord(corr_ds, ("time",))

    if (
        np.array_equal(env_ds[env_lat].values, corr_ds[corr_lat].values)
        and np.array_equal(env_ds[env_lon].values, corr_ds[corr_lon].values)
        and np.array_equal(env_ds[env_time].values, corr_ds[corr_time].values)
    ):
        return corr_ds

    aligned = corr_ds.interp(
        {
            corr_lat: env_ds[env_lat].values,
            corr_lon: env_ds[env_lon].values,
            corr_time: env_ds[env_time].values,
        },
        method="nearest",
    )
    return aligned


class _LRUCache:
    def __init__(self, name: str, maxsize: int = 8):
        self.name = name
        self.maxsize = maxsize
        self._store: "OrderedDict[Any, Any]" = OrderedDict()

    def fetch(self, key: Any, label: str) -> Tuple[bool, Any]:
        if key in self._store:
            self._store.move_to_end(key)
            logger.debug("Cache hit (%s): %s", self.name, label)
            return True, self._store[key]
        return False, None

    def store(self, key: Any, value: Any) -> None:
        self._store[key] = value
        self._store.move_to_end(key)
        if len(self._store) > self.maxsize:
            self._store.popitem(last=False)


def _normalize_bbox(bbox: Optional[Tuple[float, float, float, float]]) -> Optional[Tuple[float, float, float, float]]:
    if bbox is None:
        return None
    return tuple(round(float(v), 4) for v in bbox)


def _coords_signature(ds: xr.Dataset) -> str:
    lat_name = _select_coord(ds, ("latitude", "lat"))
    lon_name = _select_coord(ds, ("longitude", "lon"))
    try:
        time_name = _select_coord(ds, ("time",))
    except KeyError:
        time_name = None

    lat_vals = np.asarray(ds[lat_name].values)
    lon_vals = np.asarray(ds[lon_name].values)
    sha = hashlib.sha1()
    sha.update(lat_vals.tobytes())
    sha.update(lon_vals.tobytes())
    if time_name:
        time_vals = np.asarray(ds[time_name].values)
        if np.issubdtype(time_vals.dtype, np.datetime64):
            sha.update(time_vals.view("int64").tobytes())
        else:
            sha.update(time_vals.tobytes())
    return sha.hexdigest()


_CORRIDOR_CACHE = _LRUCache("corridor", maxsize=6)
_ACCIDENT_CACHE = _LRUCache("accident", maxsize=6)


def cached_corridor_aligned(
    path: Path,
    env_template: xr.Dataset,
    bbox: Optional[Tuple[float, float, float, float]],
    coarsen: int,
) -> xr.Dataset:
    bbox_key = _normalize_bbox(bbox)
    signature = _coords_signature(env_template)
    key = (str(Path(path).resolve()), bbox_key, int(coarsen), signature)
    hit, cached = _CORRIDOR_CACHE.fetch(key, Path(path).name)
    if hit:
        return cached
    ds_corr = load_dataset(path, bbox=bbox, coarsen=coarsen)
    aligned = align_corridor(env_template, ds_corr).load()
    ds_corr.close()
    _CORRIDOR_CACHE.store(key, aligned)
    return aligned


def cached_accident_resample(
    path: Path,
    env_template: xr.DataArray,
    bbox: Optional[Tuple[float, float, float, float]],
    coarsen: int,
    accident_mode: Optional[str] = None,
) -> Tuple[xr.DataArray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    bbox_key = _normalize_bbox(bbox)
    template_ds = env_template.to_dataset(name="template_var")
    signature = _coords_signature(template_ds)
    mode_key = (accident_mode or "auto").lower()
    key = (str(Path(path).resolve()), bbox_key, int(coarsen), signature, mode_key)
    hit, cached = _ACCIDENT_CACHE.fetch(key, Path(path).name)
    if hit:
        return cached

    ds_acc = load_dataset(path, bbox=bbox, coarsen=coarsen)
    var_candidates = []
    if mode_key == "static":
        var_candidates = ["acc_density_static", "accident_density_static", "accident_density"]
    elif mode_key == "time":
        var_candidates = ["acc_density_time", "accident_density_time", "accident_density"]
    else:
        var_candidates = [
            "acc_density_time",
            "acc_density_static",
            "accident_density_time",
            "accident_density_static",
            "accident_density",
        ]

    target_var = None
    for name in var_candidates:
        if name in ds_acc:
            target_var = name
            break
    if target_var is None:
        raise KeyError("accident density variable missing in accident file")

    accident_da = ds_acc[target_var].load()
    accident_attrs = {**ds_acc.attrs, **accident_da.attrs}

    env_coords = env_template.coords
    coord_map = {coord: coord for coord in accident_da.dims}
    # Ensure longitude/latitude naming consistency
    if "lat" in coord_map and "latitude" in env_coords:
        accident_da = accident_da.rename({"lat": "latitude"})
    if "lon" in accident_da.dims and "longitude" in env_coords:
        accident_da = accident_da.rename({"lon": "longitude"})
    if "Latitude" in accident_da.dims:
        accident_da = accident_da.rename({"Latitude": "latitude"})
    if "Longitude" in accident_da.dims:
        accident_da = accident_da.rename({"Longitude": "longitude"})

    if "time" in accident_da.dims:
        accident_aligned = accident_da.interp_like(env_template, method="nearest")
    else:
        # Align spatially then broadcast along time dimension
        spatial_template = env_template.isel(time=0)
        spatial_aligned = accident_da.interp_like(spatial_template, method="nearest")
        spatial_aligned = spatial_aligned.fillna(0.0)
        data = np.broadcast_to(
            spatial_aligned.values.astype("float32"),
            env_template.shape,
        )
        accident_aligned = xr.DataArray(
            data,
            coords=env_template.coords,
            dims=env_template.dims,
            name="accident_density",
        )

    accident_aligned = accident_aligned.transpose(*env_template.dims)
    accident_aligned = accident_aligned.fillna(0.0).astype("float32")
    accident_aligned.name = "accident_density"

    max_acc = float(np.nanmax(accident_aligned.values))
    if max_acc > 0.0:
        accident_aligned = (accident_aligned / max_acc).clip(min=0.0, max=1.0)
    else:
        accident_aligned = xr.zeros_like(env_template)

    accident_aligned.attrs.update(accident_attrs)
    if "accident_mode" not in accident_aligned.attrs and accident_mode:
        accident_aligned.attrs["accident_mode"] = accident_mode

    incident_lat = np.asarray(ds_acc["incident_lat"].values, dtype="float32") if "incident_lat" in ds_acc else None
    incident_lon = np.asarray(ds_acc["incident_lon"].values, dtype="float32") if "incident_lon" in ds_acc else None
    incident_time = np.asarray(ds_acc["incident_time"].values) if "incident_time" in ds_acc else None
    ds_acc.close()

    result = (accident_aligned, incident_lat, incident_lon, incident_time)
    _ACCIDENT_CACHE.store(key, result)
    return result
