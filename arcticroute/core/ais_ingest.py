"""
AIS 数据摄取与处理模块。

提供 AIS CSV / JSON 数据的 schema 探测、栅格化等功能。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr

# ============================================================================
# AIS 数据路径常量
# ============================================================================

AIS_RAW_DIR = Path(__file__).resolve().parents[2] / "data_real" / "ais" / "raw"


# ============================================================================
# Step 0: AIS 多文件读取与清洗
# ============================================================================

STANDARD_COLS = ["mmsi", "timestamp", "lat", "lon", "sog", "cog", "nav_status"]
REQUIRED_COLS = ["mmsi", "timestamp", "lat", "lon"]


def has_raw_ais_files(raw_dir: Path | str | None = None) -> bool:
    """
    检查目录中是否存在可识别的 AIS 文件（.json, .jsonl, .geojson, .csv）。
    
    Args:
        raw_dir: AIS 原始数据目录路径；若为 None，使用默认的 AIS_RAW_DIR
    
    Returns:
        True 如果目录存在且包含至少一个 AIS 文件
    """
    if raw_dir is None:
        raw_dir = AIS_RAW_DIR
    
    raw_dir = Path(raw_dir)
    
    if not raw_dir.is_dir():
        return False
    
    for ext in (".json", ".jsonl", ".geojson", ".csv"):
        if any(raw_dir.glob(f"*{ext}")):
            return True
    
    return False

COLUMN_ALIASES: Dict[str, List[str]] = {
    "mmsi": ["mmsi", "MMSI"],
    "timestamp": [
        "timestamp",
        "time",
        "datetime",
        "basedatetime",
        "basedatetimeutc",
        "utc",
        "ts",
        "postime",
        "BaseDateTime",
        "DateTime",
    ],
    "lat": ["lat", "latitude", "Lat", "LAT", "Latitude"],
    "lon": ["lon", "longitude", "long", "lng", "Lon", "LON", "Longitude"],
    "sog": ["sog", "speed", "speed_knots", "speedoverground"],
    "cog": ["cog", "course", "heading", "hdg"],
    "nav_status": ["nav_status", "navstatus", "status"],
}


def _detect_column_mapping(columns: List[str], schema_hint: Dict[str, str] | None = None) -> Dict[str, str]:
    """将可能的原始列名映射成标准列名。"""
    lower_to_original = {c.lower(): c for c in columns}
    mapping: Dict[str, str] = {}

    if schema_hint:
        for std, raw in schema_hint.items():
            if isinstance(raw, str) and raw in columns:
                mapping[raw] = std

    for std, candidates in COLUMN_ALIASES.items():
        if std in mapping.values():
            continue
        for cand in candidates:
            if cand.lower() in lower_to_original:
                mapping[lower_to_original[cand.lower()]] = std
                break

    return mapping


def _load_schema_hint(schema_hint_path: Path | str | None) -> Dict[str, str] | None:
    """加载 schema hint（格式 std: raw）。"""
    if schema_hint_path is None:
        return None

    try:
        hint_text = Path(schema_hint_path).read_text(encoding="utf-8")
    except Exception:
        return None

    schema_hint: Dict[str, str] = {}
    for line in hint_text.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        schema_hint[k.strip()] = v.strip()
    return schema_hint or None


def _normalize_ais_columns(
    df: pd.DataFrame,
    standard_cols: List[str],
    schema_hint: Dict[str, str] | None = None,
    required_cols: Iterable[str] = (),
    strict_required: bool = False,
) -> pd.DataFrame:
    """根据 schema hint / 别名，整理成标准列，并补齐缺失列。"""
    lower_to_original = {c.lower(): c for c in df.columns}
    missing_required: list[str] = []
    normalized_cols: dict[str, pd.Series] = {}

    for col in standard_cols:
        candidates = []
        if schema_hint and schema_hint.get(col):
            candidates.append(schema_hint[col])
        candidates.extend(COLUMN_ALIASES.get(col, []))

        merged_series: pd.Series | None = None
        for cand in candidates:
            matching_columns = [c for c in df.columns if c.lower() == cand.lower()]
            for raw_name in matching_columns:
                col_series = df[raw_name]
                if merged_series is None:
                    merged_series = col_series
                else:
                    merged_series = merged_series.where(merged_series.notna(), col_series)

        if merged_series is None:
            normalized_cols[col] = pd.Series([np.nan] * len(df))
            if col in required_cols:
                missing_required.append(col)
        else:
            normalized_cols[col] = merged_series

    if missing_required and strict_required:
        raise ValueError(f"[AIS] Missing required columns: {missing_required}")

    return pd.DataFrame(normalized_cols)


def _clean_ais_dataframe(
    df: pd.DataFrame,
    *,
    schema_hint: Dict[str, str] | None = None,
    dt_min: pd.Timestamp | None = None,
    dt_max: pd.Timestamp | None = None,
    strict_required: bool = False,
) -> pd.DataFrame:
    """标准化 AIS DataFrame，包含列映射、类型转换、空间/时间过滤。"""
    df = _normalize_ais_columns(
        df,
        standard_cols=STANDARD_COLS,
        schema_hint=schema_hint,
        required_cols=REQUIRED_COLS,
        strict_required=strict_required,
    )

    # 数值化
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["sog"] = pd.to_numeric(df["sog"], errors="coerce")
    df["cog"] = pd.to_numeric(df["cog"], errors="coerce")

    # 地理范围检查
    df = df.dropna(subset=["lat", "lon"])
    df = df[(df["lat"] >= -90.0) & (df["lat"] <= 90.0)]
    df = df[(df["lon"] >= -180.0) & (df["lon"] <= 180.0)]

    # 时间转换
    ts_series = df["timestamp"].astype(str).str.strip().str.replace("Z", "+00:00", regex=False)
    df["timestamp"] = ts_series.apply(lambda x: pd.to_datetime(x, errors="coerce", utc=True))
    df = df.dropna(subset=["timestamp"])

    year = df["timestamp"].dt.year
    df = df[(year >= 2018) & (year <= 2030)]

    if dt_min is not None:
        df = df[df["timestamp"] >= dt_min]
    if dt_max is not None:
        df = df[df["timestamp"] <= dt_max]

    return df.reset_index(drop=True)


def _read_json_lines(path: Path, batch_size: int = 100_000, max_records: int | None = None) -> pd.DataFrame | None:
    """尝试使用 pandas 分块读 JSON Lines 格式。"""
    try:
        reader = pd.read_json(path, lines=True, chunksize=batch_size)
    except ValueError:
        return None

    frames: list[pd.DataFrame] = []
    total = 0
    try:
        for chunk in reader:
            if max_records is not None and total >= max_records:
                break
            if max_records is not None:
                remaining = max_records - total
                chunk = chunk.head(max(0, remaining))
            total += len(chunk)
            if len(chunk) > 0:
                frames.append(chunk)
    except ValueError:
        # 不是 JSONL 格式
        return None

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _stream_json_array_objects(path: Path):
    """在不加载全部文件到内存的情况下，流式解析 JSON 数组对象。"""
    decoder = json.JSONDecoder()
    buffer = ""
    in_array = False
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), ""):
            if not chunk:
                break
            buffer += chunk
            while True:
                buffer = buffer.lstrip()
                if not buffer:
                    break
                if not in_array and buffer.startswith("["):
                    in_array = True
                    buffer = buffer[1:]
                    continue
                if in_array and buffer.startswith("]"):
                    in_array = False
                    buffer = buffer[1:]
                    continue
                try:
                    obj, offset = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    break
                yield obj
                buffer = buffer[offset:]
                if in_array and buffer.startswith(","):
                    buffer = buffer[1:]


def _extract_records_from_obj(obj) -> list[dict]:
    """从任意对象提取 AIS 记录行。"""
    records: list[dict] = []
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            for item in obj["data"]:
                if isinstance(item, dict):
                    records.append(item)
        else:
            records.append(obj)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                records.append(item)
    return records


def _records_to_dataframe(records_iter, batch_size: int = 50_000, max_records: int | None = None) -> pd.DataFrame:
    """将生成器记录合并为 DataFrame，分批落盘避免一次性占用内存。"""
    frames: list[pd.DataFrame] = []
    batch: list[dict] = []
    total = 0
    for rec in records_iter:
        if max_records is not None and total >= max_records:
            break
        batch.append(rec)
        total += 1
        if len(batch) >= batch_size:
            frames.append(pd.DataFrame(batch))
            batch = []

    if batch:
        frames.append(pd.DataFrame(batch))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_ais_json_file(
    path: Path,
    *,
    schema_hint: Dict[str, str] | None = None,
    dt_min: pd.Timestamp | None = None,
    dt_max: pd.Timestamp | None = None,
    max_records_per_file: int | None = None,
) -> pd.DataFrame:
    """加载 JSON / JSONL AIS 文件，至少支持 JSON Lines 或包含 data 数组的文件。"""
    # 1) 尝试 JSON Lines
    df = _read_json_lines(path, max_records=max_records_per_file)
    if df is not None:
        return _clean_ais_dataframe(
            df,
            schema_hint=schema_hint,
            dt_min=dt_min,
            dt_max=dt_max,
            strict_required=True,
        )

    # 2) 尝试小文件直接 json.load
    try:
        if path.stat().st_size < 50 * 1024 * 1024:
            obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
            records: list[dict] = []
            if isinstance(obj, list):
                for item in obj:
                    records.extend(_extract_records_from_obj(item))
            else:
                records.extend(_extract_records_from_obj(obj))

            if max_records_per_file is not None:
                records = records[:max_records_per_file]
            df = pd.DataFrame(records)
            return _clean_ais_dataframe(
                df,
                schema_hint=schema_hint,
                dt_min=dt_min,
                dt_max=dt_max,
                strict_required=True,
            )
    except Exception:
        pass

    # 3) 流式解析顶层数组
    records_iter = (rec for obj in _stream_json_array_objects(path) for rec in _extract_records_from_obj(obj))
    df = _records_to_dataframe(records_iter, max_records=max_records_per_file)
    return _clean_ais_dataframe(
        df,
        schema_hint=schema_hint,
        dt_min=dt_min,
        dt_max=dt_max,
        strict_required=True,
    )


def _load_ais_csv_file(
    path: Path,
    *,
    schema_hint: Dict[str, str] | None = None,
    dt_min: pd.Timestamp | None = None,
    dt_max: pd.Timestamp | None = None,
    max_records_per_file: int | None = None,
) -> pd.DataFrame:
    """加载 AIS CSV 文件并清洗。"""
    df = pd.read_csv(path, nrows=max_records_per_file)
    return _clean_ais_dataframe(
        df,
        schema_hint=schema_hint,
        dt_min=dt_min,
        dt_max=dt_max,
        strict_required=False,
    )


def load_ais_from_raw_dir(
    raw_dir: Path | str = AIS_RAW_DIR,
    schema_hint_path: Path | str | None = None,
    time_min: datetime | None = None,
    time_max: datetime | None = None,
    prefer_json: bool = True,
    max_records_per_file: int | None = None,
) -> pd.DataFrame:
    """
    从 raw_dir 目录收集 AIS 文件，优先 JSON / JSONL / GEOJSON，统一字段为
    [mmsi, timestamp, lat, lon, sog, cog, nav_status]。
    
    Args:
        raw_dir: AIS 原始数据目录路径（默认为 data_real/ais/raw）
        schema_hint_path: 可选的 schema hint 文件路径
        time_min: 时间范围下界（UTC）
        time_max: 时间范围上界（UTC）
        prefer_json: 优先读取 JSON/JSONL 格式（若存在）
        max_records_per_file: 每个文件最多读取的记录数
    
    Returns:
        标准化的 AIS DataFrame，列为 [mmsi, timestamp, lat, lon, sog, cog, nav_status]
    """
    raw_dir = Path(raw_dir)
    schema_hint = _load_schema_hint(schema_hint_path)
    dt_min = pd.to_datetime(time_min, utc=True) if time_min is not None else None
    dt_max = pd.to_datetime(time_max, utc=True) if time_max is not None else None

    json_suffixes = {".json", ".jsonl", ".geojson"}
    csv_suffixes = {".csv", ".txt"}

    json_files = sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in json_suffixes])
    csv_files = sorted([p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in csv_suffixes])

    use_json = prefer_json and len(json_files) > 0
    target_files = json_files if use_json else csv_files
    loader = _load_ais_json_file if use_json else _load_ais_csv_file

    if not target_files:
        print(f"[AIS] 原始 AIS 目录为空或不存在: {raw_dir}, AIS 数据未加载")
        return pd.DataFrame(columns=STANDARD_COLS)

    dfs: list[pd.DataFrame] = []
    for path in target_files:
        try:
            df = loader(
                path,
                schema_hint=schema_hint,
                dt_min=dt_min,
                dt_max=dt_max,
                max_records_per_file=max_records_per_file,
            )
        except Exception as e:
            print(f"[AIS] Skip file {path.name}: {e}")
            continue

        if not df.empty:
            dfs.append(df)
            print(f"[AIS] loaded {len(df)} rows from {path.name}")

    if not dfs:
        print(f"[AIS] 原始 AIS 目录为空或不存在: {raw_dir}, AIS 数据未加载")
        return pd.DataFrame(columns=STANDARD_COLS)

    out = pd.concat(dfs, ignore_index=True)
    source_label = "JSON" if use_json else "CSV"
    print(f"[AIS] load_ais_from_raw_dir: {source_label} files={len(dfs)} rows={len(out)}")
    return out


# ============================================================================
# Step 1: AIS Schema 探测
# ============================================================================

@dataclass
class AISSchemaSummary:
    """AIS CSV 的schema 摘要信息。"""

    path: str
    num_rows: int
    columns: List[str]
    has_mmsi: bool
    has_lat: bool
    has_lon: bool
    has_timestamp: bool
    time_min: Optional[str]
    time_max: Optional[str]
    lat_min: Optional[float]
    lat_max: Optional[float]
    lon_min: Optional[float]
    lon_max: Optional[float]


def inspect_ais_csv(path: str, sample_n: int = 5000) -> AISSchemaSummary:
    """
    读取前 sample_n 行 AIS CSV，推断基本 schema 和范围。

    要求至少不要崩溃，列不存在时返回 has_xxx=False, *_min/max=None。

    Args:
        path: CSV 文件路径
        sample_n: 采样行数（默认5000）

    Returns:
        AISSchemaSummary 对象
    """
    try:
        df = pd.read_csv(path, nrows=sample_n)
    except Exception as e:
        print(f"[AIS_INGEST] Failed to read CSV {path}: {e}")
        return AISSchemaSummary(
            path=path,
            num_rows=0,
            columns=[],
            has_mmsi=False,
            has_lat=False,
            has_lon=False,
            has_timestamp=False,
            time_min=None,
            time_max=None,
            lat_min=None,
            lat_max=None,
            lon_min=None,
            lon_max=None,
        )

    columns = list(df.columns)
    num_rows = len(df)

    # 检查必需列
    has_mmsi = "mmsi" in columns
    has_lat = "lat" in columns
    has_lon = "lon" in columns
    has_timestamp = "timestamp" in columns

    # 提取范围信息
    time_min = None
    time_max = None
    if has_timestamp:
        try:
            time_min = str(df["timestamp"].min())
            time_max = str(df["timestamp"].max())
        except Exception:
            pass

    lat_min = None
    lat_max = None
    if has_lat:
        try:
            lat_min = float(df["lat"].min())
            lat_max = float(df["lat"].max())
        except Exception:
            pass

    lon_min = None
    lon_max = None
    if has_lon:
        try:
            lon_min = float(df["lon"].min())
            lon_max = float(df["lon"].max())
        except Exception:
            pass

    return AISSchemaSummary(
        path=path,
        num_rows=num_rows,
        columns=columns,
        has_mmsi=has_mmsi,
        has_lat=has_lat,
        has_lon=has_lon,
        has_timestamp=has_timestamp,
        time_min=time_min,
        time_max=time_max,
        lat_min=lat_min,
        lat_max=lat_max,
        lon_min=lon_min,
        lon_max=lon_max,
    )


# ============================================================================
# Step 2: AIS 栅格化为密度场
# ============================================================================

def rasterize_ais_density_to_grid(
    lat_points: np.ndarray,
    lon_points: np.ndarray,
    grid_lat2d: np.ndarray,
    grid_lon2d: np.ndarray,
    *,
    normalize: bool = True,
) -> xr.DataArray:
    """
    将一组 AIS 经纬度点栅格化到给定网格。

    对每个点找到最近网格 (i,j)，bin 记一次。

    Args:
        lat_points: 1D 数组，AIS 点的纬度
        lon_points: 1D 数组，AIS 点的经度
        grid_lat2d: 2D 数组，网格纬度（形状 (H, W)）
        grid_lon2d: 2D 数组，网格经度（形状 (H, W)）
        normalize: 是否归一化到 [0, 1]（默认True）

    Returns:
        xr.DataArray，dims=("y","x"), name="ais_density"
    """
    ny, nx = grid_lat2d.shape

    # 初始化密度网格
    density = np.zeros((ny, nx), dtype=float)

    # 对每个 AIS 点找到最近的网格
    for lat, lon in zip(lat_points, lon_points):
        dist_sq = (grid_lat2d - lat) ** 2 + (grid_lon2d - lon) ** 2
        i, j = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        density[i, j] += 1.0

    # 归一化
    if normalize:
        max_density = np.max(density)
        if max_density > 0:
            density = density / max_density

    da = xr.DataArray(
        density,
        dims=["y", "x"],
        name="ais_density",
        attrs={"long_name": "AIS density (normalized)"},
    )

    return da


@dataclass
class AISDensityResult:
    """AIS 密度场生成结果。"""

    da: xr.DataArray
    num_points: int
    num_binned: int
    frac_binned: float


def _ensure_grid_2d(grid_lat: np.ndarray, grid_lon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """将 lat/lon 转成 2D 网格（支持1D输入）。"""
    if grid_lat.ndim == 1 and grid_lon.ndim == 1:
        lon2d, lat2d = np.meshgrid(grid_lon, grid_lat)
    else:
        lat2d, lon2d = np.broadcast_arrays(grid_lat, grid_lon)
    return lat2d, lon2d


def build_ais_density_for_grid(
    csv_path: str,
    grid_lat2d: np.ndarray,
    grid_lon2d: np.ndarray,
    max_rows: int = 50000,
) -> AISDensityResult:
    """
    从 AIS CSV 里取前 max_rows 行，过滤掉 lat/lon 缺失的点，
    调用 rasterize_ais_density_to_grid 得到 ais_density。

    Args:
        csv_path: AIS CSV 文件路径
        grid_lat2d: 网格纬度（形状 (H, W)）
        grid_lon2d: 网格经度（形状 (H, W)）
        max_rows: 最多读取的行数（默认50000）

    Returns:
        AISDensityResult 对象
    """
    try:
        df = pd.read_csv(csv_path, nrows=max_rows)
    except Exception as e:
        print(f"[AIS_INGEST] Failed to read CSV {csv_path}: {e}")
        ny, nx = grid_lat2d.shape
        empty_da = xr.DataArray(
            np.zeros((ny, nx), dtype=float),
            dims=["y", "x"],
            name="ais_density",
        )
        return AISDensityResult(
            da=empty_da,
            num_points=0,
            num_binned=0,
            frac_binned=0.0,
        )

    # 检查必需列
    if "lat" not in df.columns or "lon" not in df.columns:
        print(f"[AIS_INGEST] CSV missing lat/lon columns")
        ny, nx = grid_lat2d.shape
        empty_da = xr.DataArray(
            np.zeros((ny, nx), dtype=float),
            dims=["y", "x"],
            name="ais_density",
        )
        return AISDensityResult(
            da=empty_da,
            num_points=len(df),
            num_binned=0,
            frac_binned=0.0,
        )

    lat_points = pd.to_numeric(df["lat"], errors="coerce")
    lon_points = pd.to_numeric(df["lon"], errors="coerce")

    valid_mask = lat_points.notna() & lon_points.notna()
    lat_valid = lat_points[valid_mask].values
    lon_valid = lon_points[valid_mask].values

    num_points = len(df)
    num_binned = len(lat_valid)
    frac_binned = num_binned / num_points if num_points > 0 else 0.0

    print(
        f"[AIS_INGEST] Loaded {num_points} points, {num_binned} valid ({frac_binned*100:.1f}%)"
    )

    da = rasterize_ais_density_to_grid(
        lat_valid, lon_valid, grid_lat2d, grid_lon2d, normalize=True
    )

    return AISDensityResult(
        da=da,
        num_points=num_points,
        num_binned=num_binned,
        frac_binned=frac_binned,
    )


def build_ais_density_da_for_demo_grid(
    raw_dir: Path | str,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    time_min: datetime | None = None,
    time_max: datetime | None = None,
) -> xr.DataArray:
    """
    从真实 AIS 原始目录构建 demo 网格上的 AIS 密度 DataArray。
    """
    lat2d, lon2d = _ensure_grid_2d(grid_lat, grid_lon)
    ny, nx = lat2d.shape

    df = load_ais_from_raw_dir(raw_dir, time_min=time_min, time_max=time_max)
    if df.empty:
        density = np.zeros((ny, nx), dtype=float)
    else:
        lat_points = df["lat"].to_numpy()
        lon_points = df["lon"].to_numpy()
        da = rasterize_ais_density_to_grid(lat_points, lon_points, lat2d, lon2d, normalize=True)
        density = da.values

    da_out = xr.DataArray(
        density,
        dims=("y", "x"),
        coords={
            "y": np.arange(ny),
            "x": np.arange(nx),
            "lat": (("y", "x"), lat2d),
            "lon": (("y", "x"), lon2d),
        },
        name="ais_density",
        attrs={"source": "real_ais", "norm": "0-1"},
    )
    return da_out

