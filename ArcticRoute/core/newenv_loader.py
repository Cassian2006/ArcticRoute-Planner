# ArcticRoute/core/newenv_loader.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import xarray as xr
from functools import lru_cache


def get_project_root() -> Path:
    # 该文件位于 minimum/ArcticRoute/core/ 下
    # parents[0] -> .../ArcticRoute/core
    # parents[1] -> .../ArcticRoute
    # parents[2] -> .../minimum
    return Path(__file__).resolve().parents[2]


@dataclass
class NewEnvLayer:
    """简单封装一个新环境图层及其基础元信息，用于可视化层."""
    name: str
    da: xr.DataArray
    lat_name: str
    lon_name: str


NEWENV_DIR = get_project_root() / "ArcticRoute" / "data_processed" / "newenv"


def _detect_lat_lon_names(ds: xr.Dataset) -> Tuple[str, str]:
    cand_lat = [n for n in ds.coords if "lat" in n.lower()]
    cand_lon = [n for n in ds.coords if "lon" in n.lower()]
    if not cand_lat or not cand_lon:
        # 尝试从 data_vars 的 dims 中猜
        for da in ds.data_vars.values():
            dims_lower = [d.lower() for d in da.dims]
            lat = next((d for d in da.dims if "lat" in d.lower()), None)
            lon = next((d for d in da.dims if "lon" in d.lower()), None)
            if lat and lon:
                return lat, lon
        raise ValueError(f"Cannot detect lat/lon coords in {list(ds.coords)} / vars.")
    return cand_lat[0], cand_lon[0]


essential_vars = {
    "sic": ["sic"],
    "sithick": ["sithick", "si_thickness", "ice_thickness"],
    "wave_swh": ["wave_swh", "swh", "VHM0"],
    "elevation": ["elevation", "z", "depth"],
}


def _load_single_da(path: Path, var_name: str) -> Optional[NewEnvLayer]:
    if not path.exists():
        return None
    ds = xr.open_dataset(path)
    try:
        lat_name, lon_name = _detect_lat_lon_names(ds)
        var_name_eff = var_name
        if var_name not in ds.data_vars:
            # 兼容只有一个变量的情况
            if len(ds.data_vars) == 1:
                only = next(iter(ds.data_vars))
                print(f"[NEWENV] {path.name}: var {var_name!r} 不存在，改用唯一变量 {only!r}")
                var_name_eff = only
            else:
                # 如果 essential_vars 中存在别名则尝试一次
                aliases = essential_vars.get(var_name, [])
                alt = next((v for v in aliases if v in ds.data_vars), None)
                if alt is None:
                    print(f"[NEWENV] {path.name}: 未找到 {var_name!r}，data_vars={list(ds.data_vars)}")
                    return None
                var_name_eff = alt

        da = ds[var_name_eff]

        # 丢弃 time 等多余维度，只保留 2D(lat,lon)
        while da.ndim > 2:
            time_dim = next((d for d in da.dims if d.lower().startswith("time")), None)
            if time_dim is None:
                da = da.isel({da.dims[0]: 0})
            else:
                da = da.isel({time_dim: 0})

        # 统一经度到 [-180, 180]，便于和底图对齐
        lon = da[lon_name]
        lon_wrapped = ((lon + 180.0) % 360.0) - 180.0
        da = da.assign_coords({lon_name: lon_wrapped})

        # 转为 float32，避免太大
        da = da.astype("float32")

        return NewEnvLayer(name=var_name, da=da, lat_name=lat_name, lon_name=lon_name)
    finally:
        ds.close()


@lru_cache(maxsize=1)
def load_newenv_layers_for_viz() -> dict[str, NewEnvLayer]: 
    """
    加载 newenv 下可用于可视化的图层：
    - ice_copernicus_sic.nc -> 'sic'
    - wave_swh.nc -> 'wave_swh'
    - gebco_bathy_clip.nc -> 'elevation'
    仅用于前端热力图显示，不参与 cost 计算.
    """
    layers: dict[str, NewEnvLayer] = {}

    sic_path = NEWENV_DIR / "ice_copernicus_sic.nc"
    sic_layer = _load_single_da(sic_path, "sic")
    if sic_layer is not None:
        layers["sic"] = sic_layer
        try:
            print(
                f"[NEWENV] sic: {sic_path.name} "
                f"range={float(np.nanmin(sic_layer.da)):.3f}..{float(np.nanmax(sic_layer.da)):.3f}"
            )
        except Exception:
            pass

    sith_path = NEWENV_DIR / "ice_copernicus_sithick.nc"
    sith_layer = _load_single_da(sith_path, "sithick")
    if sith_layer is not None:
        layers["sithick"] = sith_layer

    swh_path = NEWENV_DIR / "wave_swh.nc"
    swh_layer = _load_single_da(swh_path, "wave_swh")
    if swh_layer is not None:
        layers["wave_swh"] = swh_layer
        try:
            print(
                f"[NEWENV] wave_swh: {swh_path.name} "
                f"range={float(np.nanmin(swh_layer.da)):.3f}..{float(np.nanmax(swh_layer.da)):.3f}"
            )
        except Exception:
            pass

    bathy_path = NEWENV_DIR / "gebco_bathy_clip.nc"
    bathy_layer = _load_single_da(bathy_path, "elevation")
    if bathy_layer is not None:
        layers["bathy"] = bathy_layer
        try:
            print(
                f"[NEWENV] bathy: {bathy_path.name} "
                f"range={float(np.nanmin(bathy_layer.da)):.1f}..{float(np.nanmax(bathy_layer.da)):.1f}"
            )
        except Exception:
            pass

    return layers


def load_newenv_for_cost() -> dict[str, NewEnvLayer]:
    """
    给成本构建用的简化入口：返回可用于 cost 的若干物理场（已在 newenv 目录下预处理好的二维场）。
    目前关注:
      - 'sic' (0..1)
      - 'wave_swh' (m)
    若文件不存在则对应键缺失.
    """
    layers = load_newenv_layers_for_viz()
    out: dict[str, NewEnvLayer] = {}
    if "sic" in layers:
        out["sic"] = layers["sic"]
    if "wave_swh" in layers:
        out["wave_swh"] = layers["wave_swh"]
    return out


def load_newenv_for_eco(ym: str, env_lat: np.ndarray, env_lon: np.ndarray) -> Dict[str, xr.DataArray]:
    """
    返回用于 Eco 模型的 newenv 场，并插值/对齐到给定的环境网格。
    输出键：{"sic": 2D(y,x), "sithick": 2D(y,x), "wave_swh": 2D(y,x)}（若缺失则不含该键）。
    - env_lat/env_lon 可为 1D 或 2D；优先按 1D rectilinear 插值，2D 则尝试直接对齐/近邻。
    """
    layers = load_newenv_layers_for_viz()
    out: Dict[str, xr.DataArray] = {}

    def _interp_layer(layer_key: str) -> Optional[xr.DataArray]:
        lyr = layers.get(layer_key)
        if lyr is None or getattr(lyr, "da", None) is None:
            return None
        da = lyr.da
        latn, lonn = lyr.lat_name, lyr.lon_name
        try:
            if env_lat.ndim == 1 and env_lon.ndim == 1:
                da2 = da.interp({latn: xr.DataArray(env_lat, dims=(latn,)), lonn: xr.DataArray(env_lon, dims=(lonn,))})
                # 统一命名 (y,x)
                try:
                    da2 = da2.rename({latn: "y", lonn: "x"})
                except Exception:
                    pass
                return da2.astype("float32")
            else:
                # 2D 网格：尽量直接对齐（使用 2D DataArray 作为 coords）
                try:
                    da2 = da.interp({latn: xr.DataArray(env_lat, dims=("y", "x")), lonn: xr.DataArray(env_lon, dims=("y", "x"))})
                except Exception:
                    # 兜底：近邻采样
                    arr = np.asarray(da.values)
                    H, W = int(env_lat.shape[-2]), int(env_lat.shape[-1])
                    si, sj = arr.shape[-2], arr.shape[-1]
                    yi = (np.linspace(0, si - 1, H)).astype(int)
                    xj = (np.linspace(0, sj - 1, W)).astype(int)
                    da2 = xr.DataArray(arr[yi[:, None], xj[None, :]].astype("float32"), dims=("y", "x"))
                # 确保维度 (y,x)
                if da2.dims[-2:] != ("y", "x"):
                    try:
                        da2 = da2.rename({da2.dims[-2]: "y", da2.dims[-1]: "x"})
                    except Exception:
                        pass
                return da2.astype("float32")
        except Exception:
            return None

    for k in ("sic", "sithick", "wave_swh"):
        da = _interp_layer(k)
        if da is not None:
            out[k] = da
    return out
