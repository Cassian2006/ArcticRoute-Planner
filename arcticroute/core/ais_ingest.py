from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import xarray as xr


@dataclass
class AISDensityBuildResult:
    da: xr.DataArray
    num_points: int
    num_binned: int
    frac_binned: float


def _nearest_index_2d(lat2d: np.ndarray, lon2d: np.ndarray, lat: float, lon: float) -> Tuple[int, int]:
    """
    在规则网格上用最近邻找到 (lat, lon) 的索引（最小化欧氏距离，简化处理）。
    超界时采用边界截断，保证不报错（测试要求“不会崩溃”）。
    """
    # 简单做法：分别在行/列方向上找最近的 1D 索引（假设规则网格且随行列单调）
    # 若网格不是规则单调，退化到整体距离最小。
    try:
        # 优先尝试按行列独立最近（更快）
        lat1d = lat2d[:, 0]
        lon1d = lon2d[0, :]
        i = int(np.clip(np.searchsorted(lat1d, lat) - 1, 0, len(lat1d) - 1))
        # 比较相邻两个点哪个更近
        if i + 1 < len(lat1d) and abs(lat1d[i + 1] - lat) < abs(lat1d[i] - lat):
            i += 1
        j = int(np.clip(np.searchsorted(lon1d, lon) - 1, 0, len(lon1d) - 1))
        if j + 1 < len(lon1d) and abs(lon1d[j + 1] - lon) < abs(lon1d[j] - lon):
            j += 1
        return i, j
    except Exception:
        # 退化到全局最近邻（O(N)），仅在小网格/异常时触发
        lat_diff = lat2d - float(lat)
        lon_diff = lon2d - float(lon)
        d2 = lat_diff * lat_diff + lon_diff * lon_diff
        i, j = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
        return int(i), int(j)


def rasterize_ais_density_to_grid(
    lat_points: np.ndarray,
    lon_points: np.ndarray,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    normalize: bool = False,
) -> xr.DataArray:
    """
    将点集(radarized)栅格化到给定网格（最近邻计数）。
    - 空点集返回全零
    - 越界点将被吸附到最近格点（保证不崩溃）
    - normalize=True 时按最大值归一化到 [0,1]
    返回 xarray.DataArray，name='ais_density'，shape=(ny,nx)
    """
    ny, nx = lat2d.shape
    acc = np.zeros((ny, nx), dtype=float)

    if lat_points is None or lon_points is None or len(lat_points) == 0 or len(lon_points) == 0:
        da = xr.DataArray(acc, dims=("y", "x"), name="ais_density")
        return da

    n = int(min(len(lat_points), len(lon_points)))
    for k in range(n):
        lat = float(lat_points[k])
        lon = float(lon_points[k])
        i, j = _nearest_index_2d(lat2d, lon2d, lat, lon)
        # 保护边界
        i = int(np.clip(i, 0, ny - 1))
        j = int(np.clip(j, 0, nx - 1))
        acc[i, j] += 1.0

    if normalize and acc.size > 0:
        m = float(np.nanmax(acc))
        if m > 0:
            acc = acc / m
        acc = np.clip(acc, 0.0, 1.0)

    da = xr.DataArray(acc, dims=("y", "x"), name="ais_density")
    return da


def _read_ais_csv_points(csv_path: Path, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    读取 CSV，返回 (lat_array, lon_array, total_rows)。
    兼容大小写列名，常见列包含：lat, latitude, lon, longitude。
    若文件不存在/读取失败，返回空数组和 0。
    """
    try:
        import csv
        p = Path(csv_path)
        if not p.exists():
            return np.array([], dtype=float), np.array([], dtype=float), 0
        total = 0
        lats: list[float] = []
        lons: list[float] = []
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            # 归一列名到小写
            field_map = {name.lower(): name for name in (reader.fieldnames or [])}
            lat_key = field_map.get("lat") or field_map.get("latitude")
            lon_key = field_map.get("lon") or field_map.get("longitude")
            if not lat_key or not lon_key:
                # 尝试原始列名集合里包含 lat/lon 子串的
                lat_key = next((c for c in (reader.fieldnames or []) if "lat" in c.lower()), None)
                lon_key = next((c for c in (reader.fieldnames or []) if "lon" in c.lower()), None)
            if not lat_key or not lon_key:
                return np.array([], dtype=float), np.array([], dtype=float), 0
            for row in reader:
                total += 1
                try:
                    lat = float(row[lat_key])
                    lon = float(row[lon_key])
                    lats.append(lat)
                    lons.append(lon)
                except Exception:
                    # 跳过坏行
                    pass
                if max_rows is not None and len(lats) >= max_rows:
                    break
        return np.asarray(lats, dtype=float), np.asarray(lons, dtype=float), int(total)
    except Exception:
        return np.array([], dtype=float), np.array([], dtype=float), 0


def build_ais_density_for_grid(
    csv_path: str | Path,
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    max_rows: Optional[int] = None,
) -> AISDensityBuildResult:
    """
    从 AIS CSV 构建密度场（最近邻计数）。
    - 文件不存在时返回全零密度与 0 统计
    - num_points 表示读取到的点数（受 max_rows 限制）
    - num_binned 为落入网格的点数（此实现对所有点进行最近邻吸附，故与 num_points 相同）
    - frac_binned = num_binned / max(1, num_points)
    """
    p = Path(csv_path)
    lat_pts, lon_pts, total_rows = _read_ais_csv_points(p, max_rows=max_rows)
    num_points = int(len(lat_pts))

    da = rasterize_ais_density_to_grid(lat_pts, lon_pts, lat2d, lon2d, normalize=False)
    num_binned = int(num_points)  # 最近邻吸附，全部计入
    frac_binned = float(num_binned) / float(num_points if num_points > 0 else 1)

    return AISDensityBuildResult(
        da=da,
        num_points=num_points,
        num_binned=num_binned,
        frac_binned=frac_binned,
    )
