from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlparse

import numpy as np
import requests
import xarray as xr
from requests.auth import HTTPBasicAuth
from requests.exceptions import HTTPError, RequestException

from logging_config import get_logger

try:
    import planetary_computer as pc
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    pc = None  # type: ignore[assignment]

try:  # optional dependency for catalogue access
    from pystac import Item
    from pystac_client import Client
    from pystac_client.exceptions import APIError
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    Client = None  # type: ignore[assignment]
    Item = None  # type: ignore[assignment]
    APIError = None  # type: ignore[assignment]

try:
    import rasterio
    from rasterio.transform import Affine, from_origin
    from rasterio.warp import reproject, Resampling
except ModuleNotFoundError:  # pragma: no cover - optional
    rasterio = None
    from_origin = None  # type: ignore[assignment]
    Affine = None  # type: ignore[assignment]
    Resampling = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
CDSE_NEW_STAC_URL = "https://stac.dataspace.copernicus.eu/v1"
CDSE_OLD_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"
PROTECTED_HOST_KEYWORDS = ("dataspace.copernicus.eu",)
DEFAULT_MPC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
MISSION_TO_COLLECTION = {
    "S2": "sentinel-2-l2a",
    "S1": "sentinel-1-grd",
}
MISSION_BAND_ORDER: Dict[str, Tuple[str, ...]] = {
    "S2": ("B02", "B03", "B04", "B08"),
    "S1": ("VV", "VH"),
}
S2_BAND_ALIASES: Dict[str, str] = {
    "B2": "B02",
    "B02": "B02",
    "BLUE": "B02",
    "B3": "B03",
    "B03": "B03",
    "GREEN": "B03",
    "B4": "B04",
    "B04": "B04",
    "RED": "B04",
    "B8": "B08",
    "B08": "B08",
    "NIR": "B08",
}
S1_BAND_ALIASES: Dict[str, str] = {
    "VV": "VV",
    "VH": "VH",
    "IWVV": "VV",
    "IWVH": "VH",
    "GRD_VV": "VV",
    "GRD_VH": "VH",
}
_ENV_LOADED = False

logger = get_logger(__name__)


class STACAuthError(RuntimeError):
    """Raised when authentication fails while accessing protected STAC resources."""

# ... all previous helper functions unchanged ...


def mosaic_assets_to_env(env_path: Path, hrefs: Sequence[str], output_path: Path, *, band_index: int = 1, resampling: str = "average", reduction: str = "average", href_weights: Optional[Mapping[str, float]] = None) -> Path:
    """将多个 COG 资产重投影/拼接到 env_clean.nc 的经纬网格。

    - env_path: 参考 NetCDF（提供 lat/lon 1D 坐标）
    - hrefs: COG 资产列表（支持 http/https，本地路径）
    - output_path: 输出 GeoTIFF（EPSG:4326，与 env 同行列）
    - resampling: 重采样（nearest/bilinear/average/...）
    - reduction: 多资产聚合（average|max|median|weighted）
    - href_weights: 可选 {href: weight}，当 reduction=weighted 时使用

    认证：
    - 若目标 href 需要 Authorization/BasicAuth，会尝试用 requests 先下载到临时文件再打开。
    - 环境变量见 .env：CDSE_TOKEN 或 CDSE_USERNAME/CDSE_PASSWORD。

    降级：
    - 缺少 rasterio 时，抛出 ModuleNotFoundError；上层可回退到 stub_mosaic_to_grid。
    """
    if rasterio is None:
        raise ModuleNotFoundError("rasterio is required for mosaic")
    hrefs = [h for h in hrefs if isinstance(h, str) and h]
    if not hrefs:
        raise ValueError("no hrefs provided")

    # 目标格网
    _, _, transform = extract_env_grid(env_path)
    with xr.open_dataset(env_path) as ds:
        lat = ds.coords.get("latitude") or ds.coords.get("lat")
        lon = ds.coords.get("longitude") or ds.coords.get("lon")
        height = int((lat.size - 0))  # rows
        width = int((lon.size - 0))   # cols

    dst_crs = "EPSG:4326"

    # 聚合器
    reduction = (reduction or "average").lower()
    use_median = reduction == "median"
    use_max = reduction == "max"
    use_weighted = reduction == "weighted"

    arrays_for_median: List[np.ndarray] = []
    accum = np.zeros((height, width), dtype=np.float32)
    weight_sum = np.zeros((height, width), dtype=np.float32) if use_weighted else None
    counts = np.zeros((height, width), dtype=np.uint16)
    max_arr = np.full((height, width), -np.inf, dtype=np.float32)

    import tempfile

    def _open_src(href: str):
        headers, auth = build_asset_access_params(href)
        if not headers and auth is None:
            return rasterio.open(href)
        with requests.get(href, stream=True, headers=headers or None, auth=auth, timeout=60) as resp:
            resp.raise_for_status()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".tif")
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    tmp.write(chunk)
            tmp.flush(); tmp.close()
            return rasterio.open(tmp.name)

    for href in hrefs:
        try:
            with _open_src(href) as src:
                dst = np.zeros((height, width), dtype=np.float32)
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=getattr(Resampling, resampling, Resampling.average) if Resampling else 0,
                    num_threads=2,
                )
                mask = np.isfinite(dst)
                if use_median:
                    arrays_for_median.append(np.where(mask, dst, np.nan))
                elif use_max:
                    np.maximum(max_arr, np.where(mask, dst, -np.inf), out=max_arr)
                elif use_weighted:
                    w = 1.0
                    if href_weights and href in href_weights:
                        try:
                            w = float(href_weights[href])
                        except Exception:
                            w = 1.0
                    if weight_sum is not None:
                        accum[mask] += dst[mask] * w
                        weight_sum[mask] += w
                else:
                    accum[mask] += dst[mask]
                    counts[mask] = counts[mask] + 1
        except Exception as e:
            logger.warning("mosaic reproject failed for %s: %s", href, e)
            continue

    if use_median:
        if not arrays_for_median:
            raise RuntimeError("no valid pixels reprojected from assets")
        stack = np.stack(arrays_for_median, axis=0)
        out = np.nanmedian(stack, axis=0).astype(np.float32)
    elif use_max:
        if not np.isfinite(max_arr).any():
            raise RuntimeError("no valid pixels reprojected from assets")
        out = np.where(np.isfinite(max_arr), max_arr, np.nan).astype(np.float32)
    elif use_weighted:
        if weight_sum is None or not np.any(weight_sum > 0):
            raise RuntimeError("no valid pixels reprojected from assets (weighted)")
        out = np.zeros_like(accum)
        mask = weight_sum > 0
        out[mask] = accum[mask] / weight_sum[mask]
    else:
        valid = counts > 0
        if not np.any(valid):
            raise RuntimeError("no valid pixels reprojected from assets")
        out = np.zeros_like(accum)
        out[valid] = accum[valid] / counts[valid].astype(np.float32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype="float32",
        transform=transform,
        crs=dst_crs,
    ) as dst_ds:
        dst_ds.write(out, 1)
    return output_path

# ---- Minimal stubs to ensure ingest.nrt.pull works even without full STAC stack ----

def get_stac_client(url: Optional[str] = None):  # REUSE minimal
    return None


def stac_search_sat(bbox: Sequence[float], date: Optional[str], mission: str = "S2", source: Optional[str] = None, limit: int = 10):  # REUSE minimal
    # 返回空列表（上层将走 stub_mosaic_to_grid 占位路径）
    return []


def build_asset_access_params(href: str):  # REUSE minimal
    # 无鉴权
    return {}, None


def download_asset_preview(href: str) -> bytes:  # REUSE minimal
    # 返回空字节（仅用于连通性测试）
    return b""


def write_stac_results(mission: str, date: str, payload: Mapping[str, Any]):  # REUSE minimal
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    p = LOG_DIR / f"stac_results_{mission}_{(date or 'unknown').replace(':','_').replace('/','-')}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return p


def extract_env_grid(env_path: Path):  # REUSE minimal
    # 从 env_clean.nc 读取 lat/lon 1D 或 2D，构造仿射
    with xr.open_dataset(env_path) as ds:
        lat = ds.coords.get("latitude") or ds.coords.get("lat")
        lon = ds.coords.get("longitude") or ds.coords.get("lon")
        if lat is None or lon is None:
            # 退化：以像素索引为坐标
            height = int(ds.dims.get("y", 256))
            width = int(ds.dims.get("x", 256))
            tr = from_origin(0.0, float(height), 1.0, 1.0) if from_origin else Affine(1,0,0,0,-1,float(height))
            return height, width, tr
        if lat.ndim == 1 and lon.ndim == 1:
            dy = float(abs(lat.values[1] - lat.values[0])) if lat.size > 1 else 1.0
            dx = float(abs(lon.values[1] - lon.values[0])) if lon.size > 1 else 1.0
            height = int(lat.size); width = int(lon.size)
            y0 = float(lat.values.max()); x0 = float(lon.values.min())
            tr = from_origin(x0, y0, dx, dy) if from_origin else Affine(dx,0,x0,0,-dy,y0)
            return height, width, tr
        # 2D 情况：取第一行列差近似
        height, width = int(lat.shape[0]), int(lat.shape[1])
        dy = float(abs(lat.values[1,0] - lat.values[0,0])) if height > 1 else 1.0
        dx = float(abs(lon.values[0,1] - lon.values[0,0])) if width > 1 else 1.0
        y0 = float(lat.values.min() + dy * height)  # 近似
        x0 = float(lon.values.min())
        tr = from_origin(x0, y0, dx, dy) if from_origin else Affine(dx,0,x0,0,-dy,y0)
        return height, width, tr


def stub_mosaic_to_grid(env_path: Path, output_path: Path):  # REUSE minimal
    # 无真实影像时生成占位 GeoTIFF（全零），尺寸与 env_clean 一致
    try:
        height, width, transform = extract_env_grid(env_path)
    except Exception:
        height, width = 256, 256
        transform = from_origin(0, float(height), 1, 1) if from_origin else Affine(1,0,0,0,-1,float(height))
    arr = np.zeros((height, width), dtype=np.float32)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rasterio is None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("placeholder: no rasterio, zero grid\n")
        return output_path
    with rasterio.open(
        output_path, "w", driver="GTiff", height=height, width=width, count=1, dtype="float32", transform=transform, crs="EPSG:4326"
    ) as dst:
        dst.write(arr, 1)
    return output_path


def collect_cog_hrefs(items: Sequence[Mapping[str, Any]]) -> List[str]:  # REUSE minimal
    hrefs: List[str] = []
    for it in items:
        assets = (it.get("assets") or {}) if isinstance(it, dict) else {}
        for name, aset in assets.items():
            href = aset.get("href") if isinstance(aset, dict) else None
            if isinstance(href, str) and href:
                hrefs.append(href)
    return hrefs

__all__ = [
    "stac_search_sat",
    "get_stac_client",
    "build_asset_access_params",
    "download_asset_preview",
    "write_stac_results",
    "collect_cog_hrefs",
    "extract_env_grid",
    "stub_mosaic_to_grid",
    "mosaic_assets_to_env",
    "STACAuthError",
]
