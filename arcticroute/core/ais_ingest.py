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
    """在规则网格上用最近邻找到 (lat, lon) 的索引（最小化欧氏距离，简化处理）。
    超界时采用边界截断，保证不报错（测试要求\"不会崩溃\"）。"""
    try:
        # 优先尝试按行列独立最近（更快，假设经纬度单调）
        lat1d = lat2d[:, 0]
        lon1d = lon2d[0, :]
        i = int(np.clip(np.searchsorted(lat1d, lat) - 1, 0, len(lat1d) - 1))
        if i + 1 < len(lat1d) and abs(lat1d[i + 1] - lat) < abs(lat1d[i] - lat):
            i += 1
        j = int(np.clip(np.searchsorted(lon1d, lon) - 1, 0, len(lon1d) - 1))
        if j + 1 < len(lon1d) and abs(lon1d[j + 1] - lon) < abs(lon1d[j] - lon):
            j += 1
        return i, j
    except Exception:
        # 回退：全局最近邻
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
    """将点集栅格化到给定网格（最近邻计数）。
    - 空点集返回全零
    - 越界点吸附到最近格点（不崩溃）
    - normalize=True 时按最大值归一化到 [0,1]
    返回 DataArray(name='ais_density', dims=('y','x'))."""
    ny, nx = lat2d.shape
    acc = np.zeros((ny, nx), dtype=float)

    if lat_points is None or lon_points is None or len(lat_points) == 0 or len(lon_points) == 0:
        return xr.DataArray(acc, dims=("y", "x"), name="ais_density")

    n = int(min(len(lat_points), len(lon_points)))
    for k in range(n):
        lat = float(lat_points[k])
        lon = float(lon_points[k])
        i, j = _nearest_index_2d(lat2d, lon2d, lat, lon)
        i = int(np.clip(i, 0, ny - 1))
        j = int(np.clip(j, 0, nx - 1))
        acc[i, j] += 1.0

    if normalize and acc.size > 0:
        m = float(np.nanmax(acc))
        if m > 0:
            acc = acc / m
        acc = np.clip(acc, 0.0, 1.0)

    return xr.DataArray(acc, dims=("y", "x"), name="ais_density")


def _read_ais_csv_points(csv_path: Path, max_rows: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, int]:
    """读取 CSV，返回 (lat_array, lon_array, total_rows)。
    兼容大小写列名：lat/latitude, lon/longitude。文件不存在时返回空数组。"""
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
            fieldnames = reader.fieldnames or []
            field_map = {name.lower(): name for name in fieldnames}
            lat_key = field_map.get("lat") or field_map.get("latitude")
            lon_key = field_map.get("lon") or field_map.get("longitude")
            if not lat_key or not lon_key:
                lat_key = next((c for c in fieldnames if "lat" in c.lower()), None)
                lon_key = next((c for c in fieldnames if "lon" in c.lower()), None)
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
    """从 AIS CSV 构建密度场（最近邻计数，最小实现）。"""
    p = Path(csv_path)
    lat_pts, lon_pts, _ = _read_ais_csv_points(p, max_rows=max_rows)
    num_points = int(len(lat_pts))
    da = rasterize_ais_density_to_grid(lat_pts, lon_pts, lat2d, lon2d, normalize=False)
    num_binned = int(num_points)  # 最近邻吸附，全部计入
    frac_binned = float(num_binned) / float(num_points if num_points > 0 else 1)
    return AISDensityBuildResult(da=da, num_points=num_points, num_binned=num_binned, frac_binned=frac_binned)


def build_ais_density_da_for_demo_grid(
    raw_dir: str | Path,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> xr.DataArray:
    """从目录中的所有 CSV 合并构建密度场。attrs['source']='real_ais'。"""
    raw_dir = Path(raw_dir)
    acc = np.zeros_like(grid_lat, dtype=float)
    if raw_dir.exists():
        for csv_file in sorted(raw_dir.glob("*.csv")):
            try:
                lat_pts, lon_pts, _ = _read_ais_csv_points(csv_file)
                if len(lat_pts) > 0:
                    da_single = rasterize_ais_density_to_grid(lat_pts, lon_pts, grid_lat, grid_lon, normalize=False)
                    acc += np.asarray(da_single.values, dtype=float)
            except Exception:
                pass
    result = xr.DataArray(
        acc,
        dims=("y", "x"),
        coords={"lat": (("y", "x"), grid_lat), "lon": (("y", "x"), grid_lon)},
        name="ais_density",
        attrs={"source": "real_ais"},
    )
    return result


@dataclass
class AISSummary:
    """AIS CSV 检查摘要。"""
    path: str
    num_rows: int
    columns: list[str]
    has_mmsi: bool
    has_lat: bool
    has_lon: bool
    has_timestamp: bool
    lat_min: Optional[float] = None
    lat_max: Optional[float] = None
    lon_min: Optional[float] = None
    lon_max: Optional[float] = None
    time_min: Optional[str] = None
    time_max: Optional[str] = None


def inspect_ais_csv(csv_path: str | Path, sample_n: Optional[int] = None) -> AISSummary:
    """
    检查 AIS CSV 文件的基本信息。
    
    Args:
        csv_path: CSV 文件路径
        sample_n: 最多读取的行数（None 表示全部）
    
    Returns:
        AISSummary 对象
    """
    import csv
    from datetime import datetime
    
    p = Path(csv_path)
    
    if not p.exists():
        return AISSummary(
            path=str(csv_path),
            num_rows=0,
            columns=[],
            has_mmsi=False,
            has_lat=False,
            has_lon=False,
            has_timestamp=False,
        )
    
    try:
        with p.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            columns = list(fieldnames)
            
            # 检查必需列（大小写不敏感）
            field_map = {name.lower(): name for name in fieldnames}
            has_mmsi = "mmsi" in field_map
            has_lat = "lat" in field_map or "latitude" in field_map
            has_lon = "lon" in field_map or "longitude" in field_map
            has_timestamp = "timestamp" in field_map or "time" in field_map
            
            # 获取实际列名
            mmsi_key = field_map.get("mmsi")
            lat_key = field_map.get("lat") or field_map.get("latitude")
            lon_key = field_map.get("lon") or field_map.get("longitude")
            time_key = field_map.get("timestamp") or field_map.get("time")
            
            num_rows = 0
            lat_values = []
            lon_values = []
            time_values = []
            
            for row in reader:
                num_rows += 1
                
                if has_lat and lat_key:
                    try:
                        lat_values.append(float(row[lat_key]))
                    except (ValueError, KeyError):
                        pass
                
                if has_lon and lon_key:
                    try:
                        lon_values.append(float(row[lon_key]))
                    except (ValueError, KeyError):
                        pass
                
                if has_timestamp and time_key:
                    try:
                        time_values.append(row[time_key])
                    except (ValueError, KeyError):
                        pass
                
                if sample_n is not None and num_rows >= sample_n:
                    break
            
            # 计算范围
            lat_min = float(np.min(lat_values)) if lat_values else None
            lat_max = float(np.max(lat_values)) if lat_values else None
            lon_min = float(np.min(lon_values)) if lon_values else None
            lon_max = float(np.max(lon_values)) if lon_values else None
            time_min = min(time_values) if time_values else None
            time_max = max(time_values) if time_values else None
            
            return AISSummary(
                path=str(csv_path),
                num_rows=num_rows,
                columns=columns,
                has_mmsi=has_mmsi,
                has_lat=has_lat,
                has_lon=has_lon,
                has_timestamp=has_timestamp,
                lat_min=lat_min,
                lat_max=lat_max,
                lon_min=lon_min,
                lon_max=lon_max,
                time_min=time_min,
                time_max=time_max,
            )
    except Exception:
        return AISSummary(
            path=str(csv_path),
            num_rows=0,
            columns=[],
            has_mmsi=False,
            has_lat=False,
            has_lon=False,
            has_timestamp=False,
        )


def load_ais_from_raw_dir(
    raw_dir: str | Path,
    time_min: Optional[object] = None,
    time_max: Optional[object] = None,
    prefer_json: bool = False,
) -> "pd.DataFrame":
    """
    从目录加载 AIS 数据（CSV 或 JSON）。
    
    Args:
        raw_dir: 数据目录
        time_min: 最小时间（可选）
        time_max: 最大时间（可选）
        prefer_json: 优先加载 JSON 文件
    
    Returns:
        合并后的 DataFrame
    """
    import pandas as pd
    import json
    
    raw_dir = Path(raw_dir)
    dfs = []
    
    if prefer_json:
        # 优先加载 JSON，如果有 JSON 文件就不加载 CSV
        for json_file in sorted(raw_dir.glob("*.json")):
            try:
                with json_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # 处理嵌套结构：如果是列表且每个元素有 "data" 字段
                records = []
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "data" in item:
                            records.extend(item["data"])
                        else:
                            records.append(item)
                else:
                    records = [data]
                
                if records:
                    df = pd.DataFrame(records)
                    dfs.append(df)
            except Exception:
                pass
        
        # 如果找到了 JSON 文件，就不加载 CSV
        if dfs:
            pass  # 已经加载了 JSON
        else:
            # 没有 JSON 文件，降级到 CSV
            for csv_file in sorted(raw_dir.glob("*.csv")):
                try:
                    df = pd.read_csv(csv_file)
                    dfs.append(df)
                except Exception:
                    pass
    else:
        # 加载 CSV
        for csv_file in sorted(raw_dir.glob("*.csv")):
            try:
                df = pd.read_csv(csv_file)
                dfs.append(df)
            except Exception:
                pass
    
    if not dfs:
        return pd.DataFrame()
    
    result = pd.concat(dfs, ignore_index=True)
    
    # 标准化列名（大小写不敏感）
    # 定义列名映射规则
    col_mapping_rules = {
        "mmsi": ["mmsi", "MMSI", "Mmsi"],
        "lat": ["lat", "LAT", "Lat", "latitude", "LATITUDE", "Latitude"],
        "lon": ["lon", "LON", "Lon", "longitude", "LONGITUDE", "Longitude", "lng", "LNG", "Lng"],
        "timestamp": ["timestamp", "TIMESTAMP", "Timestamp", "time", "TIME", "Time", "datetime", "DATETIME", "DateTime", "basedatetime", "BASEDATETIME", "BaseDateTime", "basedatetimeutc"],
        "sog": ["sog", "SOG", "speed", "SPEED", "Speed"],
        "cog": ["cog", "COG", "Cog"],
        "nav_status": ["nav_status", "NAV_STATUS", "Nav_Status"],
    }
    
    # 创建映射：对于每个标准列名，找到第一个存在的变体
    col_map = {}
    for std_col, variants in col_mapping_rules.items():
        for variant in variants:
            if variant in result.columns and variant not in col_map:
                col_map[variant] = std_col
    
    result = result.rename(columns=col_map)
    
    # 处理重复的标准列名：合并它们
    standard_cols = ["mmsi", "lat", "lon", "timestamp", "sog", "cog", "nav_status"]
    for std_col in standard_cols:
        # 找出所有名为 std_col 的列的索引
        col_indices = [i for i, col in enumerate(result.columns) if col == std_col]
        if len(col_indices) > 1:
            # 合并这些列：使用 fillna 来填充缺失值
            # 获取第一列的 Series
            merged_series = result.iloc[:, col_indices[0]].copy()
            # 用后续列的值填充缺失值
            for idx in col_indices[1:]:
                merged_series = merged_series.fillna(result.iloc[:, idx])
            
            # 删除旧列（从后向前删除以保持索引正确）
            for idx in sorted(col_indices, reverse=True):
                result = result.iloc[:, [i for i in range(len(result.columns)) if i != idx]]
            
            # 添加合并后的列
            result[std_col] = merged_series
    
    # 无论是否过滤，都尝试将 timestamp 解析为 datetime（UTC）
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
    
    # 时间过滤
    if time_min is not None or time_max is not None:
        if "timestamp" in result.columns:
            if time_min is not None:
                time_min_dt = pd.to_datetime(time_min, utc=True)
                result = result[result["timestamp"] >= time_min_dt]
            
            if time_max is not None:
                time_max_dt = pd.to_datetime(time_max, utc=True)
                result = result[result["timestamp"] <= time_max_dt]
    
    # 数据清理：移除越界的纬度/经度
    if "lat" in result.columns:
        result = result[(result["lat"] >= -90) & (result["lat"] <= 90)]
    if "lon" in result.columns:
        result = result[(result["lon"] >= -180) & (result["lon"] <= 180)]
    
    return result.reset_index(drop=True)





