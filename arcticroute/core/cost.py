"""
成本构建逻辑模块。

提供成本网格构建、融合等功能。
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

import numpy as np
import xarray as xr

from .grid import Grid2D
from .env_real import RealEnvLayers, load_real_env_for_grid
# from .eco.vessel_profiles import VesselProfile  # 暂时注释，VesselProfile 类未定义
# 使用 Any 作为类型提示的替代

# 可选依赖：miles-guess 后端检测
try:
    from .edl_backend_miles import has_miles_guess
except Exception:
    has_miles_guess = lambda: False

# 可选依赖：场景配置
try:
    from ..config.scenarios import get_scenario_by_name
except Exception:
    get_scenario_by_name = lambda name: None


@dataclass
class CostField:
    """成本场数据类。"""

    grid: Grid2D
    cost: np.ndarray  # float32/float64, shape = grid.shape()
    land_mask: np.ndarray  # bool, shape = grid.shape()
    components: Dict[str, np.ndarray] = field(default_factory=dict)  # 可选的成本组件分解
    edl_uncertainty: Optional[np.ndarray] = None  # 可选的 EDL 不确定性，shape = grid.shape()
    meta: Dict[str, any] = field(default_factory=dict)  # 元数据，包括 edl_source 等


# ============================================================================
# AIS 密度数据路径常量与搜索配置
# ============================================================================

AIS_DENSITY_PATH_DEMO = Path(__file__).resolve().parents[2] / "data_real" / "ais" / "derived" / "ais_density_2024_demo.nc"
AIS_DENSITY_PATH_REAL = Path(__file__).resolve().parents[2] / "data_real" / "ais" / "derived" / "ais_density_2024_real.nc"

# 向后兼容别名
AIS_DENSITY_PATH = AIS_DENSITY_PATH_DEMO

# AIS 密度搜索目录与文件模式
AIS_DENSITY_SEARCH_DIRS = [
    Path("data_real/ais/density"),
    Path("data_real/ais/derived"),
]

AIS_DENSITY_PATTERNS = [
    "ais_density*.nc",
    "*.nc",
]

AIS_WARNED_ONCE = False

logger = logging.getLogger(__name__)


# ============================================================================
# 指数参数管理
# ============================================================================

def load_fitted_exponents(ym: str | None = None) -> Tuple[float, float, str]:
    """
    尝试从拟合结果文件读取指数参数。
    
    优先级：
    1. 如果 ym 提供，查找 reports/fitted_speed_exponents_{ym}.json
    2. 否则查找最新的拟合结果文件
    3. 如果找不到，返回 (None, None, "not_found")
    
    Args:
        ym: 目标月份 (YYYYMM)，可选
    
    Returns:
        (p_sic, q_wave, source)
        其中 source 为 "fitted" 或 "not_found"
    """
    reports_dir = Path(__file__).resolve().parents[2] / "reports"
    
    if ym:
        # 查找指定月份的拟合结果
        fit_file = reports_dir / f"fitted_speed_exponents_{ym}.json"
        if fit_file.exists():
            try:
                with open(fit_file, 'r') as f:
                    data = json.load(f)
                    p_sic = float(data.get('p_sic'))
                    q_wave = float(data.get('q_wave'))
                    logger.info(f"从 {fit_file.name} 读取拟合指数: p={p_sic:.4f}, q={q_wave:.4f}")
                    return p_sic, q_wave, "fitted"
            except Exception as e:
                logger.warning(f"读取拟合结果失败 {fit_file}: {e}")
    else:
        # 查找最新的拟合结果文件
        if reports_dir.exists():
            fit_files = sorted(reports_dir.glob("fitted_speed_exponents_*.json"), reverse=True)
            if fit_files:
                try:
                    with open(fit_files[0], 'r') as f:
                        data = json.load(f)
                        p_sic = float(data.get('p_sic'))
                        q_wave = float(data.get('q_wave'))
                        logger.info(f"从 {fit_files[0].name} 读取拟合指数: p={p_sic:.4f}, q={q_wave:.4f}")
                        return p_sic, q_wave, "fitted"
                except Exception as e:
                    logger.warning(f"读取拟合结果失败: {e}")
    
    return None, None, "not_found"


def get_default_exponents(scenario_name: str | None = None, ym: str | None = None) -> Tuple[float, float]:
    """
    获取指数参数 (sic_exp, wave_exp)。
    
    优先级：
    1. 如果 ym 提供，尝试从拟合结果文件读取
    2. 如果 scenario_name 提供，从场景配置读取
    3. 否则使用全局默认值 (1.5, 2.0)
    
    Args:
        scenario_name: 场景名称（可选）
        ym: 目标月份 (YYYYMM)，用于查找拟合结果（可选）
    
    Returns:
        (sic_exp, wave_exp) 元组
    """
    # 尝试读取拟合结果
    if ym:
        p_sic, q_wave, source = load_fitted_exponents(ym)
        if source == "fitted":
            return p_sic, q_wave
    
    # 尝试从场景配置读取
    default_sic_exp = 1.5
    default_wave_exp = 2.0
    
    if scenario_name is not None:
        try:
            scenario = get_scenario_by_name(scenario_name)
            if scenario is not None:
                return scenario.sic_exp, scenario.wave_exp
        except Exception as e:
            logger.warning(f"Failed to get exponents from scenario {scenario_name}: {e}")
    
    return default_sic_exp, default_wave_exp


# ============================================================================
# 任务 A：Grid Signature 定义与 AIS 密度匹配
# ============================================================================

def compute_grid_signature(grid: Grid2D) -> str:
    """
    计算网格签名，用于 AIS 密度文件的匹配和缓存。
    
    签名格式：{ny}x{nx}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}
    
    例如：101x1440_60.0000_85.0000_-180.0000_179.7500
    
    Args:
        grid: Grid2D 对象
    
    Returns:
        网格签名字符串
    """
    ny, nx = grid.shape()
    lat_min = float(np.nanmin(grid.lat2d))
    lat_max = float(np.nanmax(grid.lat2d))
    lon_min = float(np.nanmin(grid.lon2d))
    lon_max = float(np.nanmax(grid.lon2d))
    
    signature = f"{ny}x{nx}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}"
    return signature


def discover_ais_density_candidates(grid_signature: str | None = None) -> List[Dict[str, str]]:
    """
    扫描常见目录，返回可用的 AIS density NetCDF 文件列表。
    
    优先级：
    1. 如果 grid_signature 提供，优先返回 attrs["grid_signature"] 匹配的文件
    2. 其次返回带 _demo 的文件（通用/演示文件）
    3. 最后返回其他文件

    每个元素:
        {
            "path": "data_real/ais/density/ais_density_2024_demo.nc",
            "label": "ais_density_2024_demo.nc (density)",
            "grid_signature": "101x1440_60.0000_85.0000_-180.0000_179.7500",  # 可选
            "match_type": "exact" | "demo" | "generic",
        }
    
    不会返回带 training 字样的文件，避免误选训练集。
    
    Args:
        grid_signature: 可选的网格签名，用于优先匹配
    
    Returns:
        候选文件列表，按优先级排序
    """
    candidates_exact: List[Dict[str, str]] = []
    candidates_demo: List[Dict[str, str]] = []
    candidates_generic: List[Dict[str, str]] = []
    
    for d in AIS_DENSITY_SEARCH_DIRS:
        # 如果是相对路径，转换为绝对路径
        search_dir = d if d.is_absolute() else Path.cwd() / d
        if not search_dir.exists():
            continue
        for pattern in AIS_DENSITY_PATTERNS:
            for p in sorted(search_dir.glob(pattern)):
                if not p.is_file():
                    continue
                # 避免把训练数据当成密度场
                name_lower = p.name.lower()
                if "train" in name_lower or "training" in name_lower:
                    continue
                
                # 返回相对于当前工作目录的路径字符串
                try:
                    rel = p.relative_to(Path.cwd()).as_posix()
                except ValueError:
                    # 如果无法相对化，使用绝对路径
                    rel = p.as_posix()
                
                # 尝试读取文件的 grid_signature 属性
                file_grid_sig = None
                try:
                    with xr.open_dataset(p) as ds:
                        if "grid_signature" in ds.attrs:
                            file_grid_sig = ds.attrs["grid_signature"]
                except Exception:
                    pass
                
                label = f"{p.name} ({d.name})"
                candidate = {
                    "path": rel,
                    "label": label,
                    "grid_signature": file_grid_sig,
                }
                
                # 按优先级分类
                if grid_signature and file_grid_sig == grid_signature:
                    candidate["match_type"] = "exact"
                    candidates_exact.append(candidate)
                elif "demo" in name_lower:
                    candidate["match_type"] = "demo"
                    candidates_demo.append(candidate)
                else:
                    candidate["match_type"] = "generic"
                    candidates_generic.append(candidate)
    
    # 按优先级合并：精确匹配 > demo > 通用
    return candidates_exact + candidates_demo + candidates_generic


def _warn_ais_once(message: str) -> None:
    """Emit the AIS missing-data warning only once to avoid log spam."""
    global AIS_WARNED_ONCE
    if not AIS_WARNED_ONCE:
        try:
            logger.warning(message)
        except Exception:
            print(f"[WARN] {message}")
        AIS_WARNED_ONCE = True


def _resolve_data_root(data_root: Path | None = None) -> Path:
    """
    Infer the data root from parameter, env, or project default.
    
    优先级：
    1. data_root 参数
    2. ARCTICROUTE_DATA_ROOT 环境变量
    3. 项目根目录 / data_real
    
    如果环境变量指向备份目录（不含 ais），则回退到项目 data_real。
    """
    if data_root is not None:
        return Path(data_root)
    
    env_root = os.getenv("ARCTICROUTE_DATA_ROOT")
    if env_root:
        env_path = Path(env_root)
        # 检查是否是备份目录（通常不含 ais/density）
        # 如果有 ais/density，则使用它
        if (env_path / "ais" / "density").exists() or (env_path / "ais" / "derived").exists():
            return env_path
        # 否则检查是否是备份目录的父目录
        if env_path.name == "ArcticRoute_data_backup":
            # 尝试从备份目录找 data_real
            project_root = Path(__file__).resolve().parents[2]
            if (project_root / "data_real").exists():
                return project_root / "data_real"
    
    # 默认返回项目 data_real
    return Path(__file__).resolve().parents[2] / "data_real"


def list_available_ais_density_files(data_root: Path | None = None) -> Dict[str, Path]:
    """
    扫描 data_real/ais/derived、data_real/ais/density 和 data_real/ais 目录下的 *.nc 文件，
    返回 {label: path} 映射，用于 UI 选择。

    label 示例：
      "demo density (40x80) 2024"
      "real density (500x5333) 2024"

    只匹配文件名里带 'density' 或 'ais_density' 的 nc 文件，忽略原始 json/csv/parquet。
    """
    root = _resolve_data_root(data_root)
    # 支持多个搜索目录
    search_dirs = [
        root / "ais" / "derived",
        root / "ais" / "density",
        root / "ais",
    ]
    results: Dict[str, Path] = {}
    seen: set[Path] = set()

    def _build_label(path: Path, shape: Tuple[int, ...] | None) -> str:
        name_lower = path.name.lower()
        if "demo" in name_lower:
            prefix = "demo density"
        elif "real" in name_lower:
            prefix = "real density"
        else:
            prefix = "ais density"

        dims = ""
        if shape and len(shape) >= 2:
            dims = f" ({shape[0]}x{shape[1]})"

        year = ""
        match = re.search(r"20\\d{2}", path.name)
        if match:
            year = f" {match.group(0)}"

        label = f"{prefix}{dims}{year}".strip()
        if label in results:
            label = f"{label} ({path.stem})"
        return label

    for folder in search_dirs:
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.nc")):
            name_lower = path.name.lower()
            if "density" not in name_lower and "ais_density" not in name_lower:
                continue
            if path in seen:
                continue
            seen.add(path)

            data_shape: Tuple[int, ...] | None = None
            try:
                with xr.open_dataset(path) as ds:
                    da = None
                    for key in ["ais_density", "density", "ais"]:
                        if key in ds:
                            da = ds[key]
                            break
                    if da is None and ds.data_vars:
                        da = list(ds.data_vars.values())[0]
                    if da is not None and hasattr(da, "shape"):
                        data_shape = tuple(int(s) for s in da.shape)
            except Exception:
                data_shape = None

            label = _build_label(path, data_shape)
            results[label] = path

    return dict(sorted(results.items(), key=lambda kv: kv[0]))


def _normalize_ais_density_array(arr: np.ndarray) -> np.ndarray:
    """Clip/scale AIS density into [0, 1] for cost usage."""
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    max_val = float(np.nanmax(arr)) if arr.size > 0 else 0.0
    if max_val > 0:
        arr = arr / max_val
    return np.clip(arr, 0.0, 1.0)


def _resolve_ais_weights(
    w_ais: float | None,
    w_ais_corridor: float,
    w_ais_congestion: float,
    ais_weight: float | None = None,
    *,
    map_legacy_to_corridor: bool = False,
) -> tuple[float, float, float, bool]:
    """
    规范化 AIS 权重输入。
    
    兼容旧版：当仅提供 legacy w_ais/ais_weight 且未提供新权重时，
    将 legacy 值映射到 w_ais_corridor，以满足旧用例/测试预期。
    
    Returns:
        (w_corridor, w_congestion, legacy_w_ais, mapped_from_legacy)
    """
    legacy_w = w_ais if w_ais is not None else (ais_weight if ais_weight is not None else 0.0)
    w_corridor = float(w_ais_corridor or 0.0)
    w_congestion = float(w_ais_congestion or 0.0)
    mapped = False

    if map_legacy_to_corridor and legacy_w > 0 and w_corridor <= 0 and w_congestion <= 0:
        # 仅当明确要求映射且未提供新权重时，才将 legacy 值映射到 corridor
        w_corridor = legacy_w
        legacy_w = 0.0
        mapped = True
    elif (w_corridor > 0 or w_congestion > 0) and legacy_w > 0:
        # 新权重已提供时忽略 legacy，避免重复计入
        legacy_w = 0.0

    return w_corridor, w_congestion, legacy_w, mapped


def _load_normalized_ais_density(
    grid: Grid2D,
    density_source: Optional[np.ndarray | xr.DataArray],
    ais_density_path: Path | str | None,
    *,
    prefer_real: bool,
    warn_if_missing: bool = True,
    cache_resampled: bool = False,
) -> Optional[np.ndarray]:
    """
    Load + align + normalize AIS density for a given grid.
    """
    source_path = ais_density_path
    density = density_source
    if density is None:
        density = load_ais_density_for_grid(
            grid=grid,
            prefer_real=prefer_real,
            explicit_path=ais_density_path,
        )
        if density is None:
            if warn_if_missing:
                _warn_ais_once("[AIS] no density file selected or found; AIS cost disabled.")
            return None

    aligned = None
    regridded = False
    if isinstance(density, xr.DataArray):
        aligned = _regrid_ais_density_to_grid(density, grid)
        if aligned is not None and density.shape != grid.shape():
            regridded = True
            if cache_resampled:
                try:
                    _save_resampled_ais_density(
                        aligned,
                        grid,
                        str(source_path) if source_path else "unknown",
                    )
                except Exception as e:
                    print(f"[AIS] warning: failed to cache resampled density: {e}")
    else:
        try:
            if hasattr(density, "shape") and density.shape == grid.shape():
                aligned = np.asarray(density, dtype=float)
        except Exception:
            aligned = None

    if aligned is None:
        if warn_if_missing:
            _warn_ais_once("[AIS] density shape mismatch -> auto-resample failed -> skipped AIS cost")
        return None

    if regridded:
        print(f"[AIS] detected density shape mismatch -> auto-resampled -> cached -> AIS enabled")

    return _normalize_ais_density_array(aligned)


def _save_resampled_ais_density(
    resampled_data: np.ndarray,
    grid: Grid2D,
    source_file: str,
    data_root: Path | None = None,
) -> Path | None:
    """
    将重采样后的 AIS 密度保存到 data_real/ais/density/derived/ 目录。
    
    文件名格式：ais_density_2024_{grid_signature}.nc
    
    保存的 NetCDF 文件包含：
    - 数据变量：ais_density（shape = grid.shape()）
    - 属性：grid_signature, source_file, generated_at
    
    Args:
        resampled_data: 重采样后的密度数组，shape = grid.shape()
        grid: Grid2D 对象
        source_file: 源文件路径（用于记录）
        data_root: 数据根目录（可选）
    
    Returns:
        保存的文件路径，或 None 如果保存失败
    """
    try:
        root = _resolve_data_root(data_root)
        derived_dir = root / "ais" / "density" / "derived"
        derived_dir.mkdir(parents=True, exist_ok=True)
        
        # 计算网格签名
        grid_sig = compute_grid_signature(grid)
        
        # 构造输出文件名
        output_filename = f"ais_density_2024_{grid_sig}.nc"
        output_path = derived_dir / output_filename
        
        # 创建 DataArray
        ais_da = xr.DataArray(
            resampled_data,
            dims=("y", "x"),
            coords={
                "lat": (("y", "x"), grid.lat2d),
                "lon": (("y", "x"), grid.lon2d),
            },
            name="ais_density",
        )
        
        # 创建 Dataset 并添加属性
        ds = xr.Dataset(
            {"ais_density": ais_da},
            attrs={
                "grid_signature": grid_sig,
                "source_file": str(source_file),
                "generated_at": str(datetime.now().isoformat()),
            }
        )
        
        # 保存到 NetCDF
        ds.to_netcdf(output_path)
        print(f"[AIS] saved resampled density to {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"[AIS] warning: failed to save resampled density: {e}")
        return None


def _nearest_neighbor_resample_no_scipy(
    src_lat2d: np.ndarray,
    src_lon2d: np.ndarray,
    src_data: np.ndarray,
    tgt_lat2d: np.ndarray,
    tgt_lon2d: np.ndarray,
) -> np.ndarray:
    """
    不依赖 scipy 的最近邻重采样实现。
    
    对于目标网格中的每个点 (lat_tgt, lon_tgt)，找到源网格中最近的点，
    并将其数据值复制到目标点。
    
    Args:
        src_lat2d: 源网格纬度，shape (ny_src, nx_src)
        src_lon2d: 源网格经度，shape (ny_src, nx_src)
        src_data: 源数据，shape (ny_src, nx_src)
        tgt_lat2d: 目标网格纬度，shape (ny_tgt, nx_tgt)
        tgt_lon2d: 目标网格经度，shape (ny_tgt, nx_tgt)
    
    Returns:
        重采样后的数据，shape (ny_tgt, nx_tgt)
    """
    ny_tgt, nx_tgt = tgt_lat2d.shape
    ny_src, nx_src = src_lat2d.shape
    
    # 初始化输出数组
    resampled = np.zeros((ny_tgt, nx_tgt), dtype=float)
    
    # 对每个目标点找最近邻
    for i_tgt in range(ny_tgt):
        for j_tgt in range(nx_tgt):
            lat_tgt = tgt_lat2d[i_tgt, j_tgt]
            lon_tgt = tgt_lon2d[i_tgt, j_tgt]
            
            # 计算与所有源点的距离（使用欧氏距离）
            lat_diff = src_lat2d - lat_tgt
            lon_diff = src_lon2d - lon_tgt
            distances = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # 找最近的源点
            i_src, j_src = np.unravel_index(np.argmin(distances), distances.shape)
            
            # 复制数据
            resampled[i_tgt, j_tgt] = src_data[i_src, j_src]
    
    return resampled


def _validate_ais_density_for_grid(ais_da: xr.DataArray, grid: Grid2D) -> Tuple[bool, str]:
    """
    任务 C3：验证 AIS 密度数据是否可以用于当前网格。
    
    规则：
    1. 如果 AIS DataArray 带有 latitude/longitude 坐标：允许重采样
    2. 如果只有 (y,x) 且无坐标：拒绝，给出清晰提示
    
    Args:
        ais_da: AIS 密度 DataArray
        grid: 目标网格
    
    Returns:
        (is_valid, message) 元组
    """
    # 检查是否已经匹配
    if ais_da.shape == grid.shape():
        return True, "形状已匹配，无需重采样"
    
    # 检查是否有坐标信息
    has_lat = 'latitude' in ais_da.coords or 'lat' in ais_da.coords
    has_lon = 'longitude' in ais_da.coords or 'lon' in ais_da.coords
    
    if has_lat and has_lon:
        return True, "有坐标信息，可以进行重采样"
    
    # 检查文件属性中的网格信息
    if hasattr(ais_da, 'attrs'):
        grid_shape_attr = ais_da.attrs.get('grid_shape', None)
        grid_source_attr = ais_da.attrs.get('grid_source', None)
        
        if grid_shape_attr:
            return False, (
                f"该密度文件为 {grid_source_attr or 'demo'} 网格产物（{grid_shape_attr}），"
                f"请生成 {grid.shape()[0]}×{grid.shape()[1]} 版本。"
            )
    
    # 默认拒绝
    return False, (
        f"AIS 密度文件（{ais_da.shape}）与当前网格（{grid.shape()}）维度不匹配，"
        f"且文件缺少坐标信息，无法进行重采样。"
    )


def _regrid_ais_density_to_grid(ais_da: xr.DataArray, grid: Grid2D) -> Optional[np.ndarray]:
    """
    Align AIS density DataArray to the target grid via nearest-neighbor interpolation.
    
    任务 C3：允许有坐标的密度场做重采样；没坐标的直接拒绝
    
    尝试多种对齐策略：
    1. 如果形状已匹配，直接返回
    2. 如果有 lat/lon 坐标，使用 xarray.interp 重采样
    3. 如果是 demo 网格大小，赋予 demo 网格坐标后重采样
    4. 使用纯 numpy 的最近邻重采样（不依赖 scipy）
    
    Returns:
        重采样后的密度数组，或 None 如果对齐失败
    """
    try:
        if ais_da.shape == grid.shape():
            return np.asarray(ais_da.values if hasattr(ais_da, "values") else ais_da, dtype=float)

        # 策略 1: 如果有 lat/lon 坐标，优先尝试 xarray.interp；失败则使用自带经纬度做最近邻
        if {"lat", "lon"}.issubset(set(ais_da.coords)):
            try:
                target = ais_da.interp(
                    lat=(("y", "x"), grid.lat2d),
                    lon=(("y", "x"), grid.lon2d),
                    method="nearest",
                )
                print(f"[AIS] resampled density using xarray.interp: {ais_da.shape} -> {grid.shape()}")
                return np.asarray(target.values, dtype=float)
            except Exception as e:
                print(f"[AIS] xarray.interp failed: {e}, trying NN with source coords...")
                try:
                    src_lat = np.asarray(ais_da.coords["lat"])
                    src_lon = np.asarray(ais_da.coords["lon"])
                    # 若为 1D，先网格化
                    if src_lat.ndim == 1 and src_lon.ndim == 1:
                        src_lon2d, src_lat2d = np.meshgrid(src_lon, src_lat)
                    else:
                        src_lat2d, src_lon2d = np.asarray(src_lat), np.asarray(src_lon)
                    resampled = _nearest_neighbor_resample_no_scipy(
                        src_lat2d, src_lon2d, np.asarray(ais_da), grid.lat2d, grid.lon2d
                    )
                    print(f"[AIS] resampled density using source lat/lon NN: {ais_da.shape} -> {grid.shape()}")
                    return resampled
                except Exception as e2:
                    print(f"[AIS] NN with source coords failed: {e2}, trying demo-based fallback...")
        
        # 策略 2: 尝试将 demo 网格密度赋予 lat/lon 后再重采样
        try:
            from .grid import make_demo_grid

            demo_grid, _ = make_demo_grid()
            if ais_da.shape == demo_grid.shape():
                lat1d = np.asarray(demo_grid.lat2d[:, 0])
                lon1d = np.asarray(demo_grid.lon2d[0, :])
                ais_da_with_coords = xr.DataArray(
                    np.asarray(ais_da),
                    dims=("lat", "lon"),
                    coords={"lat": lat1d, "lon": lon1d},
                    name=getattr(ais_da, "name", "ais_density"),
                )
                target = ais_da_with_coords.interp(
                    lat=(("y", "x"), grid.lat2d),
                    lon=(("y", "x"), grid.lon2d),
                    method="nearest",
                )
                print(f"[AIS] resampled demo density using xarray.interp: {ais_da.shape} -> {grid.shape()}")
                return np.asarray(target.values, dtype=float)
        except Exception as e:
            print(f"[AIS] demo grid resampling failed: {e}, trying pure numpy...")
        
        # 策略 3: 使用纯 numpy 的最近邻重采样（不依赖 scipy）
        try:
            # 获取原始密度的坐标（假设是 demo 网格或有规则的纬度/经度）
            ais_shape = ais_da.shape
            if len(ais_shape) == 2:
                # 假设是 (lat, lon) 或 (y, x) 形状
                # 尝试从 demo 网格推断坐标
                from .grid import make_demo_grid
                
                demo_grid, _ = make_demo_grid()
                if ais_shape == demo_grid.shape():
                    old_lat2d = demo_grid.lat2d
                    old_lon2d = demo_grid.lon2d
                    ais_array = np.asarray(ais_da)
                    
                    # 使用纯 numpy 的最近邻重采样
                    resampled = _nearest_neighbor_resample_no_scipy(
                        old_lat2d, old_lon2d, ais_array,
                        grid.lat2d, grid.lon2d
                    )
                    print(f"[AIS] resampled density using pure numpy NN: {ais_da.shape} -> {grid.shape()}")
                    return resampled
        except Exception as e:
            print(f"[AIS] pure numpy resampling failed: {e}")
    
    except Exception as e:
        print(f"[AIS] warning: failed to align AIS density: {e}")
    
    return None


def _add_ais_cost_component(
    base_cost: np.ndarray,
    components: Dict[str, np.ndarray],
    ais_density: Optional[np.ndarray | xr.DataArray],
    weight_ais: float,
    grid: Grid2D,
    *,
    ais_density_path: Path | str | None = None,
    prefer_real: bool = True,
) -> None:
    """
    规范化 AIS 密度，添加加权成本，并记录组件。
    
    如果加载的密度维度不匹配网格，自动进行重采样并保存到缓存。
    
    Args:
        base_cost: 基础成本数组（会被原地修改）
        components: 成本组件字典（会被更新）
        ais_density: 可选的 AIS 密度数据（numpy 数组或 xr.DataArray）
        weight_ais: AIS 拥挤度权重
        grid: Grid2D 对象
        ais_density_path: 显式指定的 AIS 密度 NC 文件路径
        prefer_real: 是否优先加载真实分辨率的 NC 文件
    """
    if weight_ais <= 0:
        return

    normalized = _load_normalized_ais_density(
        grid=grid,
        density_source=ais_density,
        ais_density_path=ais_density_path,
        prefer_real=prefer_real,
        warn_if_missing=True,
        cache_resampled=True,
    )
    if normalized is None:
        return

    cost_increment = weight_ais * normalized
    base_cost += cost_increment
    components["ais_density"] = cost_increment


def load_ais_density_for_demo_grid(path: Path | None = None) -> xr.DataArray | None:
    """
    加载 demo 网格的 AIS 密度 DataArray。
    
    Args:
        path: 可选的 AIS 密度 NC 文件路径；若为 None，使用默认的 AIS_DENSITY_PATH_DEMO
    
    Returns:
        AIS 密度 DataArray，或在文件缺失/加载失败时返回 None
    """
    target = Path(path) if path is not None else AIS_DENSITY_PATH_DEMO
    try:
        if not target.exists():
            _warn_ais_once("[AIS] no density file selected or found; AIS cost disabled.")
            return None
        ds = xr.load_dataset(target)
        if "ais_density" not in ds:
            _warn_ais_once(f"[AIS] 密度变量未找到: {target}")
            return None
        return ds["ais_density"]
    except Exception as e:
        _warn_ais_once(f"[AIS] 加载密度文件失败 {target}: {e}")
        return None


def load_ais_density_for_grid(
    grid: Grid2D | None = None,
    prefer_real: bool = True,
    explicit_path: Optional[Path | str] = None,
) -> xr.DataArray | None:
    """
    加载与给定 grid 对齐的 AIS 拥挤度字段。

    优先级:
        1) explicit_path 指定的 .nc
        2) 按 grid_signature 匹配的文件（精确匹配 > demo > 通用）
        3) 原有的 AIS_DENSITY_PATH_REAL / AIS_DENSITY_PATH_DEMO 回退逻辑
    
    Args:
        grid: Grid2D 对象（可选，用于计算 grid_signature 以优先匹配）
        prefer_real: 是否优先加载真实分辨率的 NC 文件
        explicit_path: 显式指定的 AIS 密度 NC 文件路径
    
    Returns:
        AIS 密度 DataArray，或在文件缺失/加载失败时返回 None
    """
    # 1) 显式路径优先
    if explicit_path is not None:
        p = Path(explicit_path)
        # 确保路径是绝对路径或相对于当前工作目录的有效路径
        if not p.is_absolute():
            p = Path.cwd() / p
        if Path(p).exists():
            try:
                ds = xr.load_dataset(p)
                if "ais_density" in ds:
                    return ds["ais_density"]
                _warn_ais_once(f"[AIS] 密度变量未找到: {p}")
            except Exception as e:
                _warn_ais_once(f"[AIS] 加载密度文件失败 {p}: {e}")
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
            return None
    
    # 2) 按 grid_signature 自动发现候选文件
    grid_signature = None
    if grid is not None:
        try:
            grid_signature = compute_grid_signature(grid)
        except Exception:
            grid_signature = None
    
    for cand in discover_ais_density_candidates(grid_signature=grid_signature):
        p = Path(cand["path"])
        # 确保路径是绝对路径或相对于当前工作目录的有效路径
        if not p.is_absolute():
            p = Path.cwd() / p
        if Path(p).exists():
            try:
                ds = xr.load_dataset(p)
                if "ais_density" in ds:
                    match_type = cand.get("match_type", "unknown")
                    print(f"[AIS] loaded density from {p.name} (match_type={match_type})")
                    return ds["ais_density"]
                _warn_ais_once(f"[AIS] 密度变量未找到: {p}")
            except Exception as e:
                _warn_ais_once(f"[AIS] 加载密度文件失败 {p}: {e}")
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
    
    # 3) 保留旧的常量路径回退
    candidates = []
    if prefer_real:
        candidates.append(AIS_DENSITY_PATH_REAL)
    # 兼容旧用法：优先尝试别名路径（可被测试用 monkeypatch 覆盖）
    try:
        candidates.append(AIS_DENSITY_PATH)
    except Exception:
        pass
    candidates.append(AIS_DENSITY_PATH_DEMO)

    # 先尝试别名路径（便于测试通过 monkeypatch 设置）
    alias = AIS_DENSITY_PATH
    for path in [alias] + candidates:
        path = Path(path)
        if path.exists():
            try:
                ds = xr.load_dataset(path)
                if "ais_density" in ds:
                    print(f"[AIS] loaded density from fallback path {path.name}")
                    return ds["ais_density"]
                _warn_ais_once(f"[AIS] 密度变量未找到: {path}")
            except Exception as e:
                _warn_ais_once(f"[AIS] 加载密度文件失败 {path}: {e}")
            finally:
                try:
                    ds.close()
                except Exception:
                    pass
    
    _warn_ais_once("[AIS] no density file selected or found; AIS cost disabled.")
    return None


def has_ais_density_data(grid: Grid2D | None = None, prefer_real: bool = True) -> bool:
    """
    最大努力检查当前是否有 AIS 密度文件可用，不抛异常。
    
    若存在可读的 ais_density 变量则返回 True。
    
    Args:
        grid: Grid2D 对象（可选，当前未使用）
        prefer_real: 是否优先检查真实分辨率的 NC 文件
    
    Returns:
        True 如果至少有一个可用的 AIS 密度文件
    """
    candidates = []
    if prefer_real:
        candidates.append(AIS_DENSITY_PATH_REAL)
    candidates.append(AIS_DENSITY_PATH_DEMO)

    for path in candidates:
        if not path.exists():
            continue
        try:
            with xr.open_dataset(path) as ds:
                if "ais_density" in ds:
                    return True
        except Exception:
            continue
    return False


def build_demo_cost(
    grid: Grid2D,
    land_mask: np.ndarray,
    ice_penalty: float = 4.0,
    ice_lat_threshold: float = 75.0,
    w_ais: float = 0.0,
    ais_density: Optional[np.ndarray | xr.DataArray] = None,
    ais_density_path: Path | str | None = None,
    w_ais_corridor: float = 0.0,
    w_ais_congestion: float = 0.0,
) -> CostField:
    """
    ? demo grid ???????????

    ?????
      - ???: ???? 1.0
      - ?? (lat >= ice_lat_threshold) ?????"??"??: cost += ice_penalty
      - ???: ?? = np.inf??????

    ?????
      - base_distance: ????? 1.0???? np.inf
      - ice_risk: ? lat >= ice_lat_threshold ????? ice_penalty???? 0.0

    Args:
        grid: Grid2D ??
        land_mask: bool ???True = ??
        ice_penalty: ??????? 4.0????????
        ice_lat_threshold: ????????? 75.0?

    Returns:
        CostField ????? components ??
    """
    ny, nx = grid.shape()
    
    # ?? base_distance ????? 1.0??? inf
    base_distance = np.ones((ny, nx), dtype=float)
    base_distance = np.where(land_mask, np.inf, base_distance)
    
    # ?? ice_risk ???????? ice_penalty???? 0.0
    lat2d = grid.lat2d
    ice_band = lat2d >= ice_lat_threshold
    ice_risk = np.zeros((ny, nx), dtype=float)
    ice_risk[ice_band & ~land_mask] = ice_penalty
    
    # ??? = base_distance + ice_risk
    cost = base_distance + ice_risk

    components = {
        "base_distance": base_distance,
        "ice_risk": ice_risk,
    }

    w_corridor, w_congestion, legacy_w_ais, _ = _resolve_ais_weights(
        w_ais=w_ais,
        w_ais_corridor=w_ais_corridor,
        w_ais_congestion=w_ais_congestion,
        ais_weight=None,
        map_legacy_to_corridor=True,
    )
    need_ais = any(weight > 0 for weight in (w_corridor, w_congestion, legacy_w_ais))
    ais_norm = None
    if need_ais:
        ais_norm = _load_normalized_ais_density(
            grid=grid,
            density_source=ais_density,
            ais_density_path=ais_density_path,
            prefer_real=False,
            warn_if_missing=True,
            cache_resampled=True,
        )

    if ais_norm is not None:
        # 始终记录原始 AIS 密度到 components（用于诊断和验证）
        components["ais_density"] = ais_norm
        
        if w_corridor > 0:
            corridor_cost = 1.0 - np.sqrt(np.clip(ais_norm, 0.0, 1.0))
            corridor_cost = np.clip(corridor_cost, 0.0, 1.0)
            corridor_cost = w_corridor * np.where(land_mask, np.inf, corridor_cost)
            cost = cost + corridor_cost
            components["ais_corridor"] = corridor_cost

        if w_congestion > 0:
            ocean_mask = (~land_mask) & np.isfinite(ais_norm)
            if np.any(ocean_mask):
                p90 = float(np.nanquantile(ais_norm[ocean_mask], 0.90))
            else:
                p90 = 1.0
            denom = max(1e-6, 1.0 - p90)
            x = np.maximum(0.0, ais_norm - p90) / denom
            penalty = np.power(x, 2)
            penalty_cost = w_congestion * np.where(land_mask, np.inf, penalty)
            cost = cost + penalty_cost
            components["ais_congestion"] = penalty_cost

        if legacy_w_ais > 0:
            legacy_cost = legacy_w_ais * ais_norm
            cost = cost + legacy_cost
            # 注意：legacy_w_ais 情况下，components["ais_density"] 已在上面设置

    return CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask.astype(bool),
        components=components,
    )


def build_cost_from_real_env(
    grid: Grid2D,
    land_mask: np.ndarray,
    env: RealEnvLayers,
    ice_penalty: float = 4.0,
    wave_penalty: float = 0.0,
    vessel_profile: Any | None = None,  # VesselProfile 类型暂未定义
    ice_class_soft_weight: float = 3.0,
    ym: str | None = None,
    *,
    w_edl: float = 0.0,
    use_edl: bool = False,
    use_edl_uncertainty: bool = False,
    edl_uncertainty_weight: float = 0.0,
    ais_density: np.ndarray | xr.DataArray | None = None,
    ais_weight: float = 0.0,
    ais_density_path: Path | str | None = None,
    ais_density_da=None,
    w_ais_corridor: float = 0.0,
    w_ais_congestion: float = 0.0,
    w_ais: float | None = None,
    sic_exp: float | None = None,
    wave_exp: float | None = None,
    scenario_name: str | None = None,
) -> CostField:
    """
    使用真实环境场（sic + wave_swh + ice_thickness_m）构建成本场。

    成本规则：
      - base_distance: 海洋格 1.0，陆地格 np.inf
      - ice_risk: 若有 sic，则 ice_penalty * sic^1.5；否则 0
      - wave_risk: 若有 wave_swh 且 wave_penalty > 0，则 wave_penalty * (wave_norm^1.5)；否则 0
      - ice_class_soft: 若有 ice_thickness_m 和 vessel_profile，则对超出安全范围的冰厚施加软惩罚
      - ice_class_hard: 若有 ice_thickness_m 和 vessel_profile，则对超出硬限制的冰厚设置 inf
      - edl_risk: 若 use_edl=True 且 w_edl > 0，则通过 EDL 模块计算的风险分数
      - edl_uncertainty_penalty: 若 use_edl_uncertainty=True 且 edl_uncertainty_weight > 0，则基于 EDL 不确定性的额外惩罚

    其中 wave_norm = wave_swh / max_wave，max_wave = 6.0（米）。

    冰级约束逻辑（仅当 ice_thickness_m 和 vessel_profile 都存在时启用）：
      - 安全区: T <= 0.7 * T_max_effective -> 无额外成本
      - 软风险区: 0.7*T_max_effective < T <= T_max_effective -> 二次惩罚
      - 硬禁区: T > T_max_effective -> 成本设为 inf

    EDL 风险逻辑（仅当 use_edl=True 且 w_edl > 0 时启用）：
      - 从 env 与 grid 构造特征立方体：[sic_norm, wave_swh_norm, ice_thickness_norm, lat_norm, lon_norm]
      - 调用 run_edl_on_features(...) 得到 risk_mean（期望风险分数，0..1）
      - edl_cost = w_edl * risk_mean
      - 若 PyTorch 不可用，使用 fallback（打印日志但不报错）

    EDL 不确定性逻辑（仅当 use_edl_uncertainty=True 且 edl_uncertainty_weight > 0 时启用）：
      - 需要先计算 EDL 输出（包含 uncertainty）
      - unc_cost = edl_uncertainty_weight * uncertainty（clipped to [0, 1]）
      - 累加进总成本
      - 记录到 components["edl_uncertainty_penalty"]

    AIS 逻辑（仅当提供密度且相关权重 > 0 时启用）：
      - 对 ais_density 做 safe 归一化到 [0,1]（clip + max 归一化）
      - corridor（主航线偏好）：cost = w_ais_corridor * (1 - sqrt(d_norm))，高密度更便宜
      - congestion（拥挤惩罚）：计算海域 p90，penalty = ((max(0, d - p90) / max(1e-6, 1-p90))^2)
      - legacy w_ais 会被映射为 corridor 权重（向后兼容），仅当未单独提供新权重时启用

    返回的 components 至少包含：
      - "base_distance"
      - "ice_risk"
    如有 wave 分量且 wave_penalty > 0，则再包含：
      - "wave_risk"
    如有冰级约束，则再包含：
      - "ice_class_soft"
      - "ice_class_hard"
    如有 EDL 风险且 use_edl=True 且 w_edl > 0，则再包含：
      - "edl_risk"
    如有 EDL 不确定性且 use_edl_uncertainty=True 且 edl_uncertainty_weight > 0，则再包含：
      - "edl_uncertainty_penalty"
    如有 AIS 密度且 ais_density is not None 且 ais_weight > 0，则再包含：
      - "ais_density"

    Args:
        grid: Grid2D 对象
        land_mask: bool 数组，True = 陆地
        env: RealEnvLayers 对象，包含 sic、wave_swh 和/或 ice_thickness_m 数据
        ice_penalty: 冰风险权重（默认 4.0）
        wave_penalty: 波浪风险权重（默认 0.0，即不考虑波浪）
        vessel_profile: VesselProfile 对象，用于冰级约束；若为 None，则不应用冰级约束
        ice_class_soft_weight: 冰级软约束的权重系数（默认 3.0）
        w_edl: EDL 风险权重（默认 0.0，即不启用 EDL）
        use_edl: 是否启用 EDL 风险头（默认 False）
        use_edl_uncertainty: 是否启用 EDL 不确定性进成本（默认 False）
        edl_uncertainty_weight: EDL 不确定性权重（默认 0.0，即不考虑不确定性）
        ais_density: 可选的 AIS 密度场（np.ndarray 或 xr.DataArray，形状与网格相同）
        ais_weight: AIS 旧版权重（默认 0.0，若 w_ais 提供则被覆盖，向后兼容用途）
        ais_density_da: 可选的 AIS 主航道密度 DataArray（形状与网格相同，若 ais_density 缺失时作为候选）
        w_ais_corridor: AIS 主航线偏好权重（默认 0.0，即不使用走廊成本）
        w_ais_congestion: AIS 拥挤惩罚权重（默认 0.0）
        w_ais: 旧版 AIS 拥挤度权重（优先于 ais_weight，若新权重均为 0 时映射为 corridor）
        sic_exp: 海冰浓度指数（可选，覆盖默认值 1.5）
        wave_exp: 波浪高度指数（可选，覆盖默认值 1.5）
        scenario_name: 场景名称（可选，用于从配置读取默认指数参数）

    Returns:
        CostField 对象，包含 components 分解
    """
    if env is None and ym is not None:
        env = load_real_env_for_grid(ym=ym)

    # 获取指数参数：优先使用显式参数，否则从拟合结果读取，再从场景配置读取，最后使用默认值
    if sic_exp is None or wave_exp is None:
        default_sic_exp, default_wave_exp = get_default_exponents(scenario_name, ym=ym)
        if sic_exp is None:
            sic_exp = default_sic_exp
        if wave_exp is None:
            wave_exp = default_wave_exp
    
    logger.info(f"[COST] using exponents: sic_exp={sic_exp:.2f}, wave_exp={wave_exp:.2f}")

    w_corridor, w_congestion, legacy_w_ais, _ = _resolve_ais_weights(
        w_ais=w_ais,
        w_ais_corridor=w_ais_corridor,
        w_ais_congestion=w_ais_congestion,
        ais_weight=ais_weight,
    )
    density_source = ais_density if ais_density is not None else ais_density_da
    effective_grid = env.grid if env and env.grid is not None else grid
    effective_land_mask = env.land_mask if env and env.land_mask is not None else land_mask

    if env is None or effective_grid is None or env.sic is None or effective_land_mask is None:
        print(f"[COST] real env unavailable for ym={ym}, falling back to demo cost")
        return build_demo_cost(
            effective_grid if effective_grid is not None else grid,
            effective_land_mask if effective_land_mask is not None else land_mask,
            ice_penalty=ice_penalty,
            ice_lat_threshold=75.0,
            w_ais=legacy_w_ais,
            ais_density=density_source,
            ais_density_path=ais_density_path,
            w_ais_corridor=w_corridor,
            w_ais_congestion=w_congestion,
        )

    grid = effective_grid
    land_mask = effective_land_mask
    ny, nx = grid.shape()

    if (env.sic is not None and env.sic.shape != (ny, nx)) or (
        env.wave_swh is not None and env.wave_swh.shape != (ny, nx)
    ):
        print(
            f"[COST] real env shape mismatch (sic={None if env.sic is None else env.sic.shape}, "
            f"wave={None if env.wave_swh is None else env.wave_swh.shape}) vs grid {(ny, nx)}, using demo cost"
        )
        return build_demo_cost(
            grid,
            land_mask,
            ice_penalty=ice_penalty,
            ice_lat_threshold=75.0,
            w_ais=legacy_w_ais,
            ais_density=density_source,
            ais_density_path=ais_density_path,
            w_ais_corridor=w_corridor,
            w_ais_congestion=w_congestion,
        )

    # 构建 base_distance 组件：海洋 1.0，陆地 inf
    base_distance = np.ones((ny, nx), dtype=float)
    base_distance = np.where(land_mask, np.inf, base_distance)

    # 构建 ice_risk 组件：基于 sic 的非线性放大
    ice_risk = np.zeros((ny, nx), dtype=float)
    if env.sic is not None:
        sic = env.sic
        # 确保 sic 形状与网格一致
        if sic.shape == (ny, nx):
            # 确保 sic 在 0..1 范围内
            sic = np.clip(sic, 0.0, 1.0)
            # ice_risk = ice_penalty * sic^sic_exp
            ice_risk = ice_penalty * np.power(sic, sic_exp)
        else:
            print(
                f"[COST] warning: sic shape {sic.shape} != grid shape ({ny}, {nx}), "
                f"using zero ice_risk"
            )

    # 构建 wave_risk 组件：基于 wave_swh 的非线性放大
    wave_risk = np.zeros((ny, nx), dtype=float)
    if env.wave_swh is not None and wave_penalty > 0:
        wave = env.wave_swh
        # 确保 wave 形状与网格一致
        if wave.shape == (ny, nx):
            # 确保 wave 在 0..10 范围内
            wave = np.clip(wave, 0.0, 10.0)
            # 归一化：max_wave = 6.0 米
            max_wave = 6.0
            wave_norm = wave / max_wave
            # wave_risk = wave_penalty * wave_norm^wave_exp
            wave_risk = wave_penalty * np.power(wave_norm, wave_exp)
        else:
            print(
                f"[COST] warning: wave shape {wave.shape} != grid shape ({ny}, {nx}), "
                f"using zero wave_risk"
            )

    # 总成本 = base_distance + ice_risk + wave_risk
    cost = base_distance + ice_risk + wave_risk
    # 确保陆地格点为 inf
    cost = np.where(land_mask, np.inf, cost)

    # 构建 components 字典
    components = {
        "base_distance": base_distance,
        "ice_risk": ice_risk,
    }
    
    # 如果有 wave 分量且 wave_penalty > 0，加入 wave_risk
    if env.wave_swh is not None and wave_penalty > 0:
        components["wave_risk"] = wave_risk

    # 应用冰级约束（仅当 ice_thickness_m 和 vessel_profile 都存在时）
    if env.ice_thickness_m is not None and vessel_profile is not None:
        thickness = env.ice_thickness_m
        
        # 确保 thickness 形状与网格一致
        if thickness.shape == (ny, nx):
            # 获取有效的最大冰厚（考虑安全裕度）
            T_max_effective = vessel_profile.get_effective_max_ice_thickness()
            
            # 定义海洋掩码（非陆地且成本有限）
            ocean_mask = (~land_mask) & np.isfinite(cost)
            
            # 仅在 finite 区域生效
            finite = np.isfinite(thickness)
            ocean_finite = ocean_mask & finite
            
            # 三个区域的掩码（仅针对 finite 区域）
            safe_threshold = 0.7 * T_max_effective
            safe_mask = ocean_finite & (thickness <= safe_threshold)
            soft_mask = ocean_finite & (thickness > safe_threshold) & (thickness <= T_max_effective)
            hard_mask = ocean_finite & (thickness > T_max_effective)
            
            # 对软风险区施加二次惩罚；非 finite 区域保持 0，不改 cost
            ice_class_soft = np.zeros((ny, nx), dtype=float)
            if np.any(soft_mask):
                soft_thickness = thickness[soft_mask]
                ratio = (soft_thickness - safe_threshold) / (T_max_effective - safe_threshold)
                ratio = np.clip(ratio, 0.0, 1.0)
                soft_penalty = ice_class_soft_weight * (ratio ** 2)
                ice_class_soft[soft_mask] = soft_penalty
                cost[soft_mask] += soft_penalty
            components["ice_class_soft"] = ice_class_soft
            
            # 对硬禁区设置 inf；非 finite 区域保持 0，不改 cost
            ice_class_hard = np.zeros((ny, nx), dtype=float)
            if np.any(hard_mask):
                cost[hard_mask] = np.inf
                ice_class_hard[hard_mask] = 1.0
            components["ice_class_hard"] = ice_class_hard
            
            finite_frac = float(np.isfinite(thickness).mean())
            print(
                f"[COST] ice_thickness finite_frac={finite_frac:.6f}, "
                f"soft_cells={int(np.sum(soft_mask))}, hard_cells={int(np.sum(hard_mask))}"
            )
        else:
            print(
                f"[COST] warning: ice_thickness shape {thickness.shape} != grid shape ({ny}, {nx}), "
                f"skipping ice_class constraints"
            )

    # ========================================================================
    # Phase EDL-CORE Step 3: 应用 EDL 风险头（优先使用 miles-guess）
    # ========================================================================
    if use_edl and w_edl > 0:
        try:
            # 首先尝试使用 miles-guess 后端
            edl_output = None
            edl_source = None
            
            try:
                from .edl_backend_miles import run_miles_edl_on_grid
                
                # 准备输入数据
                sic = env.sic if env.sic is not None else np.zeros((ny, nx), dtype=float)
                swh = env.wave_swh if env.wave_swh is not None else None
                ice_thickness = env.ice_thickness_m if env.ice_thickness_m is not None else None
                
                # 调用 miles-guess 后端
                edl_output = run_miles_edl_on_grid(
                    sic=sic,
                    swh=swh,
                    ice_thickness=ice_thickness,
                    grid_lat=grid.lat2d,
                    grid_lon=grid.lon2d,
                    model_name="default",
                    device="cpu",
                )
                
                # 检查是否成功
                if edl_output.meta.get("source") == "miles-guess":
                    edl_source = "miles-guess"
                    print(f"[COST] using miles-guess EDL backend")
                else:
                    # miles-guess 不可用或失败，回退到 PyTorch 实现
                    edl_output = None
                    edl_source = None
                    
            except Exception as e:
                print(f"[COST] miles-guess backend failed: {e}, falling back to PyTorch")
                edl_output = None
                edl_source = None
            
            # 如果 miles-guess 失败，回退到 PyTorch 实现
            if edl_output is None:
                try:
                    from ..ml.edl_core import run_edl_on_features, EDLConfig, TORCH_AVAILABLE

                    # 构造特征立方体
                    # 特征顺序：[sic_norm, wave_swh_norm, ice_thickness_norm, lat_norm, lon_norm]
                    features_list = []

                    # 1) sic_norm (0..1)
                    if env.sic is not None:
                        sic_norm = np.clip(env.sic, 0.0, 1.0)
                    else:
                        sic_norm = np.zeros((ny, nx), dtype=float)
                    features_list.append(sic_norm)

                    # 2) wave_swh_norm (0..1，归一化到 0..10m)
                    if env.wave_swh is not None:
                        wave_norm = np.clip(env.wave_swh / 10.0, 0.0, 1.0)
                    else:
                        wave_norm = np.zeros((ny, nx), dtype=float)
                    features_list.append(wave_norm)

                    # 3) ice_thickness_norm (0..1，归一化到 0..2m)
                    if env.ice_thickness_m is not None:
                        ice_thickness_norm = np.clip(env.ice_thickness_m / 2.0, 0.0, 1.0)
                    else:
                        ice_thickness_norm = np.zeros((ny, nx), dtype=float)
                    features_list.append(ice_thickness_norm)

                    # 4) lat_norm (0..1，线性缩放到 60..85N)
                    lat_min, lat_max = 60.0, 85.0
                    lat_norm = np.clip((grid.lat2d - lat_min) / (lat_max - lat_min), 0.0, 1.0)
                    features_list.append(lat_norm)

                    # 5) lon_norm (0..1，线性缩放到 -180..180)
                    lon_min, lon_max = -180.0, 180.0
                    lon_norm = np.clip((grid.lon2d - lon_min) / (lon_max - lon_min), 0.0, 1.0)
                    features_list.append(lon_norm)

                    # 堆叠成 (H, W, F) 的特征立方体
                    features = np.stack(features_list, axis=-1)  # shape (ny, nx, 5)

                    # 调用 EDL 推理
                    edl_config = EDLConfig(num_classes=3)
                    from ..ml.edl_core import EDLGridOutput
                    
                    edl_pytorch_output = run_edl_on_features(features, config=edl_config)

                    # 如果 PyTorch 不可用，打印日志
                    if not TORCH_AVAILABLE:
                        print("[EDL] torch not available; using fallback constant risk.")

                    # 转换为统一的 EDLGridOutput 格式
                    edl_output = type('EDLGridOutput', (), {
                        'risk': edl_pytorch_output.risk_mean,
                        'uncertainty': edl_pytorch_output.uncertainty,
                        'meta': {'source': 'pytorch_edl_core'}
                    })()
                    edl_source = "pytorch"
                    
                except Exception as e:
                    print(f"[COST] warning: EDL risk computation failed: {e}")
                    edl_output = None
                    edl_source = None

            # 如果成功获得 EDL 输出，应用到成本
            if edl_output is not None:
                # 使用 risk 字段（兼容 miles-guess 和 PyTorch 实现）
                risk_field = edl_output.risk if hasattr(edl_output, 'risk') else edl_output.risk_mean
                
                # 将 risk 映射为成本组件
                edl_cost = w_edl * risk_field

                # 加入总成本
                cost = cost + edl_cost

                # 记录到 components
                components["edl_risk"] = edl_cost

                print(
                    f"[COST] EDL risk applied ({edl_source}): "
                    f"w_edl={w_edl:.3f}, "
                    f"edl_risk_range=[{edl_cost.min():.3f}, {edl_cost.max():.3f}]"
                )
            else:
                print(f"[COST] warning: EDL risk not applied (no valid backend)")

        except Exception as e:
            print(f"[COST] warning: EDL risk computation failed: {e}")

    # 构建完成后输出 cost 的有限比例
    try:
        cost_finite_frac = float(np.isfinite(cost).mean())
        print(f"[COST] cost finite_frac={cost_finite_frac:.6f}")
    except Exception:
        pass
            # 不报错，继续使用不含 EDL 的成本

    # ========================================================================
    # Step 2: EDL 不确定性处理（新增）
    # ========================================================================
    # 初始化 edl_uncertainty（可选）
    edl_uncertainty = None
    if (use_edl and w_edl > 0) or (use_edl_uncertainty and edl_uncertainty_weight > 0):
        try:
            from ..ml.edl_core import run_edl_on_features, EDLConfig, TORCH_AVAILABLE

            # 如果前面已经计算过 edl_output，直接使用其 uncertainty
            # 否则重新计算一次
            if 'edl_output' in locals() and edl_output is not None:
                edl_uncertainty = np.asarray(edl_output.uncertainty, dtype=float)
                edl_uncertainty = np.clip(edl_uncertainty, 0.0, 1.0)
            else:
                # 需要重新构造特征并计算
                features_list = []

                # 1) sic_norm (0..1)
                if env.sic is not None:
                    sic_norm = np.clip(env.sic, 0.0, 1.0)
                else:
                    sic_norm = np.zeros((ny, nx), dtype=float)
                features_list.append(sic_norm)

                # 2) wave_swh_norm (0..1，归一化到 0..10m)
                if env.wave_swh is not None:
                    wave_norm = np.clip(env.wave_swh / 10.0, 0.0, 1.0)
                else:
                    wave_norm = np.zeros((ny, nx), dtype=float)
                features_list.append(wave_norm)

                # 3) ice_thickness_norm (0..1，归一化到 0..2m)
                if env.ice_thickness_m is not None:
                    ice_thickness_norm = np.clip(env.ice_thickness_m / 2.0, 0.0, 1.0)
                else:
                    ice_thickness_norm = np.zeros((ny, nx), dtype=float)
                features_list.append(ice_thickness_norm)

                # 4) lat_norm (0..1，线性缩放到 60..85N)
                lat_min, lat_max = 60.0, 85.0
                lat_norm = np.clip((grid.lat2d - lat_min) / (lat_max - lat_min), 0.0, 1.0)
                features_list.append(lat_norm)

                # 5) lon_norm (0..1，线性缩放到 -180..180)
                lon_min, lon_max = -180.0, 180.0
                lon_norm = np.clip((grid.lon2d - lon_min) / (lon_max - lon_min), 0.0, 1.0)
                features_list.append(lon_norm)

                # 堆叠成 (H, W, F) 的特征立方体
                features = np.stack(features_list, axis=-1)  # shape (ny, nx, 5)

                # 调用 EDL 推理
                edl_config = EDLConfig(num_classes=3)
                edl_output = run_edl_on_features(features, config=edl_config)
                edl_uncertainty = np.asarray(edl_output.uncertainty, dtype=float)
                edl_uncertainty = np.clip(edl_uncertainty, 0.0, 1.0)

        except Exception as e:
            print(f"[COST] warning: EDL uncertainty extraction failed: {e}")
            edl_uncertainty = None

    # 应用 EDL 不确定性进成本（仅当启用且权重 > 0）
    if use_edl_uncertainty and edl_uncertainty_weight > 0 and edl_uncertainty is not None:
        try:
            # 构造不确定性成本：edl_uncertainty_weight * uncertainty
            unc_cost = edl_uncertainty_weight * edl_uncertainty
            
            # 累加进总成本
            cost = cost + unc_cost
            
            # 记录到 components
            components["edl_uncertainty_penalty"] = unc_cost
            
            print(
                f"[COST] EDL uncertainty penalty applied: "
                f"edl_uncertainty_weight={edl_uncertainty_weight:.3f}, "
                f"unc_cost_range=[{unc_cost.min():.3f}, {unc_cost.max():.3f}]"
            )
        except Exception as e:
            print(f"[COST] warning: EDL uncertainty penalty computation failed: {e}")
            # 不报错，继续使用不含不确定性的成本

    # ========================================================================
    # Step 3: AIS 拥挤/走廊处理（拆分为 corridor + congestion）
    # ========================================================================
    need_ais = any(weight > 0 for weight in (w_corridor, w_congestion, legacy_w_ais))
    ais_norm = None
    if need_ais:
        ais_norm = _load_normalized_ais_density(
            grid=grid,
            density_source=density_source,
            ais_density_path=ais_density_path,
            prefer_real=True,
            warn_if_missing=True,
            cache_resampled=True,
        )

    if ais_norm is not None:
        # 始终记录原始 AIS 密度到 components（用于诊断和验证）
        components["ais_density"] = ais_norm
        
        if legacy_w_ais > 0:
            ais_cost = legacy_w_ais * ais_norm
            cost = cost + ais_cost
            print(
                f"[COST] AIS legacy cost applied: "
                f"w_ais={legacy_w_ais:.3f}, "
                f"ais_cost_range=[{ais_cost.min():.3f}, {ais_cost.max():.3f}]"
            )

        if w_corridor > 0:
            corridor_cost = 1.0 - np.sqrt(np.clip(ais_norm, 0.0, 1.0))
            corridor_cost = np.clip(corridor_cost, 0.0, 1.0)
            corridor_cost = w_corridor * np.where(land_mask, np.inf, corridor_cost)
            cost = cost + corridor_cost
            components["ais_corridor"] = corridor_cost
            print(
                f"[COST] AIS corridor bonus applied: "
                f"w_ais_corridor={w_corridor:.3f}, "
                f"corridor_range=[{corridor_cost.min():.3f}, {corridor_cost.max():.3f}]"
            )

        if w_congestion > 0:
            ocean_mask = (~land_mask) & np.isfinite(ais_norm)
            if np.any(ocean_mask):
                p90 = float(np.nanquantile(ais_norm[ocean_mask], 0.90))
            else:
                p90 = 1.0
            denom = max(1e-6, 1.0 - p90)
            x = np.maximum(0.0, ais_norm - p90) / denom
            penalty = np.power(x, 2)
            congestion_cost = w_congestion * np.where(land_mask, np.inf, penalty)
            cost = cost + congestion_cost
            components["ais_congestion"] = congestion_cost
            print(
                f"[COST] AIS congestion penalty applied: "
                f"w_ais_congestion={w_congestion:.3f}, "
                f"p90={p90:.3f}, "
                f"penalty_range=[{congestion_cost.min():.3f}, {congestion_cost.max():.3f}]"
            )

    # 构造元数据
    # 记录指数来源
    sic_power_source = "fitted" if ym and sic_exp is not None else "default"
    wave_power_source = "fitted" if ym and wave_exp is not None else "default"
    
    meta = {
        "edl_source": edl_source if 'edl_source' in locals() else None,
        "sic_power_effective": sic_exp,
        "wave_power_effective": wave_exp,
        "sic_power_source": sic_power_source,
        "wave_power_source": wave_power_source,
    }
    
    return CostField(
        grid=grid,
        cost=cost,
        land_mask=land_mask.astype(bool),
        components=components,
        edl_uncertainty=edl_uncertainty,
        meta=meta,
    )


def build_cost_from_sic(
    grid: Grid2D,
    land_mask: np.ndarray,
    env: RealEnvLayers,
    ice_penalty: float = 4.0,
) -> CostField:
    """
    使用真实 sic（海冰浓度）构建成本场。

    这是 build_cost_from_real_env() 的简化 wrapper，仅考虑冰风险，不考虑波浪。

    成本规则：
      - 海洋格: 基础成本 1.0
      - 冰风险组件: 基于 sic 的非线性放大，ice_penalty * sic^1.5
      - 陆地格: 成本 = np.inf（不可通行）

    成本分解：
      - base_distance: 全部海域为 1.0，陆地为 np.inf
      - ice_risk: 基于 sic 的冰风险，ice_penalty * sic^1.5

    Args:
        grid: Grid2D 对象
        land_mask: bool 数组，True = 陆地
        env: RealEnvLayers 对象，包含 sic 数据
        ice_penalty: 冰风险权重（默认 4.0）

    Returns:
        CostField 对象，包含 components 分解
    """
    return build_cost_from_real_env(
        grid=grid,
        land_mask=land_mask,
        env=env,
        ice_penalty=ice_penalty,
        wave_penalty=0.0,
    )
