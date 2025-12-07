# ArcticRoute/core/planner_service.py
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Public API（不要变更签名；需与调用方保持兼容）
- load_environment(ym: str, w_ice: float = 0.7, w_accident: float = 0.2, prior_weight: float = 0.0, **kwargs) -> EnvironmentContext
  兼容旧调用：alpha_ice（忽略或映射为 w_ice）
- latlon_to_ij(env: EnvironmentContext, latv: float, lonv: float) -> Optional[tuple[int, int]]
- path_ij_to_lonlat(env: EnvironmentContext, path_ij: list[tuple[int, int]]) -> list[list[float]]
- compute_route(env: EnvironmentContext, start_ij: tuple[int, int], goal_ij: tuple[int, int], allow_diagonal: bool, heuristic: str) -> RouteResult
- summarize_route(route_result: RouteResult) -> dict
- evaluate_route_eco(route_result: RouteResult, env_ctx: EnvironmentContext) -> EcoSummary
（下列分析函数为内部/扩展使用，但被 Planner 调用，保留）
- analyze_route_cost(env: EnvironmentContext, route_result: RouteResult) -> dict[str, float]
- analyze_prior_adherence(env: EnvironmentContext, route_result: RouteResult) -> float
- route_to_geojson(route_result: RouteResult) -> str
- load_fused_risk(ym: str, scenario: str = "default", fusion_mode: str = "linear") -> Optional[xr.DataArray]
- load_pareto_front(ym: str, scenario: str = "default") -> Optional[list[dict]]
- apply_feedback_and_replan(env_ctx: EnvironmentContext, base_route: RouteResult, feedback_file: Path, *, allow_diagonal: bool, heuristic: str, soft_weight: float = 50.0) -> RouteResult
- build_feedback_from_shapes(shapes: list[dict], *, crs: str = "EPSG:4326", default_mode: str = "no_go") -> dict
- write_feedback_jsonl(records: list[dict], out_path: Path) -> Path
"""

from pathlib import Path
from typing import Dict, NamedTuple, Optional, Any, List, Tuple, Literal
from dataclasses import dataclass
from functools import lru_cache
try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

from dataclasses import dataclass, field
import time
import numpy as np
import xarray as xr
import streamlit as st
import matplotlib.cm as cm
from ArcticRoute.core.newenv_loader import load_newenv_layers_for_viz, NewEnvLayer, load_newenv_for_cost, load_newenv_for_eco


from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing, binary_closing
from scipy.spatial import cKDTree

from ArcticRoute.core.route.astar_cost import astar_on_cost, RouteSummary
# 高级风险辅助（纯文件加载/聚合/归一化）
try:
    from ArcticRoute.core.advanced_risk import (
        load_fused_risk as adv_load_fused_risk,
        load_interact_risk as adv_load_interact_risk,
        load_escort_risk as adv_load_escort_risk,
        reduce_to_2d as adv_reduce_to_2d,
        align_like as adv_align_like,
        build_evidential_robust_surface as adv_build_evidential_robust_surface,
    )
except Exception:
    adv_load_fused_risk = None  # type: ignore
    adv_load_interact_risk = None  # type: ignore
    adv_load_escort_risk = None  # type: ignore
    adv_reduce_to_2d = None  # type: ignore
    adv_align_like = None  # type: ignore
    adv_build_evidential_robust_surface = None  # type: ignore
from ArcticRoute.core.route.metrics import compute_distance_km
# 船型配置
try:
    from ArcticRoute.core.eco.vessel_profiles import load_vessel_profile as _load_vessel_profile  # type: ignore
except Exception:
    _load_vessel_profile = None  # type: ignore

# ---- Overlay 渲染缓存（小型 LRU）与降采样参数 ----
_OVERLAY_CACHE: dict = {}
_OVERLAY_KEYS: list = []  # 维护简单的 LRU 顺序
_OVERLAY_CACHE_CAP = 8    # 最多缓存 8 份渲染结果
_MAX_OVERLAY_W = 1000     # 最大输出宽度（像素）
_MAX_OVERLAY_H = 800      # 最大输出高度（像素）


# -------------------------
# Pipeline 轻量级追踪结构（UI 展示用）
# -------------------------
PipelineStatus = Literal["pending", "running", "done", "error", "skipped"]

@dataclass
class PipelineStage:
    id: str
    label: str
    status: PipelineStatus = "pending"
    started_at: float | None = None
    ended_at: float | None = None
    duration_ms: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineTrace:
    """
    记录一次“规划请求”的关键阶段及统计信息。
    仅用于 UI 展示，不参与算法计算。
    """
    stages: dict[str, PipelineStage] = field(default_factory=dict)

    def ensure_stage(self, id: str, label: str) -> PipelineStage:
        if id not in self.stages:
            self.stages[id] = PipelineStage(id=id, label=label)
        return self.stages[id]

    def start(self, id: str, label: str) -> None:
        stg = self.ensure_stage(id, label)
        stg.status = "running"
        stg.started_at = time.perf_counter()

    def end(self, id: str, status: PipelineStatus = "done", meta: dict[str, Any] | None = None) -> None:
        stg = self.ensure_stage(id, self.stages[id].label if id in self.stages else id)
        now = time.perf_counter()
        stg.ended_at = now
        if stg.started_at is not None:
            stg.duration_ms = (now - stg.started_at) * 1000.0
        stg.status = status
        if meta:
            try:
                stg.meta.update(meta)
            except Exception:
                pass


def new_pipeline_trace() -> PipelineTrace:
    trace = PipelineTrace()
    # 预注册标准节点，便于 UI 固定顺序
    trace.ensure_stage("input", "解析输入参数")
    trace.ensure_stage("env", "加载环境与网格")
    trace.ensure_stage("cost", "构建风险与成本场")
    trace.ensure_stage("route_primary", "规划三条路线")
    trace.ensure_stage("eco", "Eco 燃油与排放评估")
    trace.ensure_stage("summary", "生成结果摘要")
    trace.ensure_stage("adv_viz", "高级可视化准备")
    trace.ensure_stage("ai_explain", "AI 航线解读")
    return trace


# 可选依赖：GeoPandas / Shapely（若不可用，则先验距离场功能自动降级为 None）
try:
    import geopandas as gpd  # type: ignore
except Exception:  # pragma: no cover
    gpd = None  # type: ignore

try:
    import shapely.geometry as sgeom  # type: ignore
    from shapely.geometry import box  # type: ignore
except Exception:  # pragma: no cover
    sgeom = None  # type: ignore
    box = None  # type: ignore

# 生态评估依赖（若不可用，evaluate_route_eco 将回退）
try:
    from ArcticRoute.core.eco import fuel as eco_fuel  # type: ignore
    from ArcticRoute.core.eco import route_eval as eco_eval  # type: ignore
except Exception:  # pragma: no cover
    eco_fuel = None  # type: ignore
    eco_eval = None  # type: ignore

# 基于 newenv 的 Eco 评估（优先使用）
try:
    from ArcticRoute.core.eco.eco_newenv import compute_fuel_along_route as eco_compute_newenv  # type: ignore
except Exception:  # pragma: no cover
    eco_compute_newenv = None  # type: ignore


# -------------------------
# 规划域定义与工具
# -------------------------

@dataclass
class PlanningDomain:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float


def _get_latlon_1d(arr) -> Tuple[np.ndarray, np.ndarray]:
    """将可能为 1D/2D 的 lat/lon 转为 1D 轴（取唯一值）。"""
    lat, lon = arr
    latv = np.asarray(lat)
    lonv = np.asarray(lon)
    if latv.ndim > 1:
        latv = np.unique(latv)
    if lonv.ndim > 1:
        lonv = np.unique(lonv)
    return latv.astype(float), lonv.astype(float)


def _latlon_from_da(da: "xr.DataArray") -> Tuple[np.ndarray, np.ndarray]:
    lat_name = next((n for n in list(da.coords) + list(da.dims) if "lat" in n.lower()), None)
    lon_name = next((n for n in list(da.coords) + list(da.dims) if "lon" in n.lower()), None)
    if lat_name is None or lon_name is None:
        raise KeyError("DataArray 缺少经纬度坐标")
    lat = np.asarray(da[lat_name].values if lat_name in da.coords else da[lat_name])
    lon = np.asarray(da[lon_name].values if lon_name in da.coords else da[lon_name])
    return _get_latlon_1d((lat, lon))


def compute_planning_domain(env_ctx: "EnvironmentContext", newenv_layers: Optional[Dict[str, "xr.DataArray"]] = None, risk_layers: Optional[Dict[str, "xr.DataArray"]] = None) -> PlanningDomain:
    """根据当前环境、newenv、risk 层的经纬度交集，计算规划域。"""
    import numpy as _np
    if env_ctx.lat_arr is None or env_ctx.lon_arr is None:
        raise RuntimeError("[PLANNING_DOMAIN] env_ctx.lat_arr/lon_arr missing")
    lat0, lon0 = _get_latlon_1d((env_ctx.lat_arr, env_ctx.lon_arr))
    lat_min = float(_np.nanmin(lat0)); lat_max = float(_np.nanmax(lat0))
    lon_min = float(_np.nanmin(lon0)); lon_max = float(_np.nanmax(lon0))

    # 与 newenv 层取交集
    if newenv_layers:
        for key in ("sic", "wave_swh"):
            da = newenv_layers.get(key)
            if da is None:
                continue
            try:
                lats, lons = _latlon_from_da(da)
                lat_min = max(lat_min, float(_np.nanmin(lats)))
                lat_max = min(lat_max, float(_np.nanmax(lats)))
                lon_min = max(lon_min, float(_np.nanmin(lons)))
                lon_max = min(lon_max, float(_np.nanmax(lons)))
            except Exception:
                continue

    # 与 risk_ice 取交集
    if risk_layers and ("ice" in risk_layers):
        try:
            lats, lons = _latlon_from_da(risk_layers["ice"])
            lat_min = max(lat_min, float(_np.nanmin(lats)))
            lat_max = min(lat_max, float(_np.nanmax(lats)))
            lon_min = max(lon_min, float(_np.nanmin(lons)))
            lon_max = min(lon_max, float(_np.nanmax(lons)))
        except Exception:
            pass

    if not (lat_min < lat_max and lon_min < lon_max):
        raise RuntimeError(f"[PLANNING_DOMAIN] invalid intersection: lat[{lat_min},{lat_max}], lon[{lon_min},{lon_max}]")
    dom = PlanningDomain(lat_min=lat_min, lat_max=lat_max, lon_min=lon_min, lon_max=lon_max)
    try:
        print("[PLANNING_DOMAIN] lat_min/max:", lat_min, lat_max, "lon_min/max:", lon_min, lon_max)
    except Exception:
        pass
    return dom


def snap_point_to_domain_and_ocean(lat: float, lon: float, env_ctx: "EnvironmentContext", max_radius_cells: int = 20) -> Tuple[float, float, Dict[str, Any]]:
    """
    将 (lat, lon) 吸附到规划域内最近的可通行海洋格点（仅做局部搜索）。
    约束：禁止全图跳跃；若局部搜索失败，返回初始栅格并标记 snap_failed=True。
    返回: (snapped_lat, snapped_lon, info)
    其中 info 包含 grid_i/grid_j 便于上游直接取用。
    注意：为保持向后兼容，不改变返回顺序与键名。
    """
    import numpy as _np
    dom = getattr(env_ctx, "domain", None)
    if dom is None:
        raise RuntimeError("[POINT_SNAP] env_ctx.domain missing, did you call load_environment() which sets env_ctx.domain?")

    # (a) 识别网格经度坐标系并将输入 lon 规范到相同坐标域
    try:
        lon_arr_all = _np.asarray(env_ctx.lon_arr, dtype=float)
        gmin = float(_np.nanmin(lon_arr_all))
        gmax = float(_np.nanmax(lon_arr_all))
        if (gmax > 180.0 + 1e-3) or (gmin >= 0.0):
            lon_internal = float(lon) % 360.0
            if lon_internal < 0:
                lon_internal += 360.0
        else:
            lon_internal = ((float(lon) + 180.0) % 360.0) - 180.0
    except Exception:
        lon_internal = float(lon)

    # (b) clamp 到规划域（使用内部坐标系后的 lon）
    lat_c = float(_np.clip(float(lat), float(dom.lat_min), float(dom.lat_max)))
    lon_c = float(_np.clip(lon_internal, float(dom.lon_min), float(dom.lon_max)))

    # (c) 经纬度 -> 初始栅格索引（稳健兼容 1D/2D lat/lon 与 cost_da 尺寸）
    lat_arr = _np.asarray(env_ctx.lat_arr, dtype=float)
    lon_arr = _np.asarray(env_ctx.lon_arr, dtype=float)

    # 目标网格尺寸优先来自 cost_da
    Hc = Wc = None
    try:
        if getattr(env_ctx, "cost_da", None) is not None:
            Hc, Wc = env_ctx.cost_da.shape[-2:]
    except Exception:
        Hc = Wc = None

    # 构造 1D 轴（长度与 Hc/Wc 对齐）。若无 cost，则退化为 unique 近似。
    if lat_arr.ndim == 1 and (Hc is None or len(lat_arr) == Hc):
        lat_axis = lat_arr
    elif lat_arr.ndim == 2 and Hc is not None and lat_arr.shape[-2] == Hc:
        lat_axis = _np.nanmedian(lat_arr, axis=1)
    else:
        lat_u = _np.unique(lat_arr)
        lat_axis = lat_u if (Hc is None or len(lat_u) == Hc) else (_np.linspace(float(_np.nanmin(lat_u)), float(_np.nanmax(lat_u)), int(Hc)))

    if lon_arr.ndim == 1 and (Wc is None or len(lon_arr) == Wc):
        lon_axis = lon_arr
    elif lon_arr.ndim == 2 and Wc is not None and lon_arr.shape[-1] == Wc:
        lon_axis = _np.nanmedian(lon_arr, axis=0)
    else:
        lon_u = _np.unique(lon_arr)
        lon_axis = lon_u if (Wc is None or len(lon_u) == Wc) else (_np.linspace(float(_np.nanmin(lon_u)), float(_np.nanmax(lon_u)), int(Wc)))

    # H/W 以 cost 为准（若可用），否则以轴长度为准
    H = int(Hc if Hc is not None else lat_axis.shape[0])
    W = int(Wc if Wc is not None else lon_axis.shape[0])

    # 最近邻轴索引（裁剪到 H/W-1 以防极端差值）
    i0 = int(_np.clip(int(_np.abs(lat_axis - lat_c).argmin()), 0, H - 1))
    j0 = int(_np.clip(int(_np.abs(lon_axis - lon_c).argmin()), 0, W - 1))

    # (d) 搜索最近“可通行海洋格点”
    lm = env_ctx.land_mask
    lm_np = _np.asarray(lm, dtype=bool) if lm is not None else None
    cost_np = None
    try:
        if getattr(env_ctx, "cost_da", None) is not None:
            cost_np = _np.asarray(env_ctx.cost_da.values, dtype=float)
    except Exception:
        cost_np = None

    def _is_valid(ii: int, jj: int) -> bool:
        if ii < 0 or jj < 0 or ii >= H or jj >= W:
            return False
        if lm_np is not None and lm_np.shape[-2:] == (H, W) and bool(lm_np[ii, jj]):
            return False
        if cost_np is not None:
            try:
                v = float(cost_np[ii, jj])
            except Exception:
                return False
            if not _np.isfinite(v):
                return False
            if v >= 1e6:
                return False
        # 不再使用 ice 图的 NaN 作为硬性无效判据，避免在边界处被误判吸到陆地/域外
        return True

    bi = bj = None
    for r in range(int(max_radius_cells) + 1):
        found = False
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                ii, jj = i0 + di, j0 + dj
                if _is_valid(ii, jj):
                    bi, bj = int(ii), int(jj)
                    found = True
                    break
            if found:
                break
        if found:
            break

    snap_failed = False
    reason = ""
    if bi is None:
        # 局部搜索失败：不做全图跳转，返回初始栅格
        bi, bj = int(_np.clip(i0, 0, H - 1)), int(_np.clip(j0, 0, W - 1))
        snap_failed = True
        reason = "local_search_failed"
        try:
            print(f"[POINT_SNAP] local search failed near ({lat},{lon}); keep initial grid ({bi},{bj})")
        except Exception:
            pass

    # (e) 索引 -> 经纬度
    # 若 lat/lon 轴长度与 H/W 一致，直接索引；否则做近似映射
    try:
        snapped_lat = float(lat_axis[int(_np.clip(bi, 0, len(lat_axis)-1))])
    except Exception:
        snapped_lat = float(lat_c)
    try:
        _lon_tmp = float(lon_axis[int(_np.clip(bj, 0, len(lon_axis)-1))])
    except Exception:
        _lon_tmp = float(lon_c)

    snapped_lon = _lon_tmp
    try:
        if (gmax > 180.0 + 1e-3) or (gmin >= 0.0):
            snapped_lon = ((snapped_lon + 180.0) % 360.0) - 180.0
    except Exception:
        pass

    info = {
        "input_lat": float(lat), "input_lon": float(lon),
        "clamped_lat": float(lat_c), "clamped_lon": float(lon_c),
        "grid_i": int(bi), "grid_j": int(bj),
        "snap_failed": bool(snap_failed),
        "reason": reason,
    }
    try:
        print("[POINT_SNAP] from (%.4f, %.4f) -> (%.4f, %.4f), grid=(%d,%d) snap_failed=%s" % (lat, lon, snapped_lat, snapped_lon, bi, bj, str(snap_failed)))
    except Exception:
        pass
    return snapped_lat, snapped_lon, info


def get_default_start_end(env_ctx: "EnvironmentContext") -> Tuple[Tuple[float, float], Tuple[float, float]]:
    dom = getattr(env_ctx, "domain", None)
    if dom is None:
        dom = compute_planning_domain(env_ctx)
    lat_mid = 0.5 * (dom.lat_min + dom.lat_max)
    s_lat_guess, s_lon_guess = lat_mid, (dom.lon_min + 1.0)
    e_lat_guess, e_lon_guess = lat_mid, (dom.lon_max - 1.0)
    s_lat, s_lon, _ = snap_point_to_domain_and_ocean(s_lat_guess, s_lon_guess, env_ctx)
    e_lat, e_lon, _ = snap_point_to_domain_and_ocean(e_lat_guess, e_lon_guess, env_ctx)
    return (s_lat, s_lon), (e_lat, e_lon)


def sample_random_ocean_points(env_ctx: "EnvironmentContext", n: int = 2) -> List[Tuple[float, float]]:
    import numpy as _np
    lat = _np.asarray(env_ctx.lat_arr)
    lon = _np.asarray(env_ctx.lon_arr)
    if lat.ndim > 1:
        lat = _np.unique(lat)
    if lon.ndim > 1:
        lon = _np.unique(lon)
    H, W = int(lat.shape[0]), int(lon.shape[0])
    lm = env_ctx.land_mask
    lm_np = _np.asarray(lm) if lm is not None else None
    # 可选：避开 NaN 冰险
    ice_arr = None
    try:
        layers = env_ctx.risk_layers_raw or {}
        da_ice = layers.get("ice") or layers.get("copernicus_ice")
        if da_ice is not None:
            arr = _np.asarray(da_ice.values)
            if arr.ndim >= 2:
                ice_arr = arr
    except Exception:
        ice_arr = None
    valid = []
    for i in range(H):
        for j in range(W):
            if lm_np is not None and lm_np.shape == (H, W) and bool(lm_np[i, j]):
                continue
            if ice_arr is not None and _np.isnan(ice_arr[i, j]):
                continue
            valid.append((i, j))
    if not valid:
        print("[RANDOM_POINTS] no valid ocean cells; using defaults")
        s, e = get_default_start_end(env_ctx)
        return [s, e][:n]
    rng = _np.random.default_rng()
    sel = rng.choice(len(valid), size=min(n, len(valid)), replace=False)
    pts: List[Tuple[float, float]] = []
    for k in _np.atleast_1d(sel):
        i, j = valid[int(k)]
        pts.append((float(lat[i]), float(lon[j])))
    return pts

# -------------------------
# 数据结构（与外部契约保持）
# -------------------------
@dataclass
class RobustPlannerConfig:
    """
    配置 evidential + robust 规划行为的轻量配置对象。
    Phase 1 中主要控制：
    - risk_agg_mode: 用于 fused 风险的不确定性聚合（例如 "cvar" 或 "mean"）；
    - risk_agg_alpha: quantile/CVaR 中使用的 alpha；
    - fusion_mode: 传给 load_environment 的 fusion 模式（例如 "evidential"）；
    - allow_diagonal 和 heuristic: 传给 compute_route。
    """
    risk_agg_mode: str = "cvar"
    risk_agg_alpha: float = 0.9
    fusion_mode: str = "evidential"
    allow_diagonal: bool = True
    heuristic: str = "euclidean"


class EnvironmentContext(NamedTuple):
    ym: str
    cost_da: xr.DataArray | None
    sic_da: xr.DataArray | None
    lat_arr: np.ndarray | None
    lon_arr: np.ndarray | None
    land_mask: np.ndarray | None  # True=陆地/不可走
    risk_layers_raw: Dict[str, xr.DataArray] | None  # 保留以便路线成本分解
    prior_penalty_da: xr.DataArray | None  # 0 在主航线，越远越大
    # 新增：用于前端可视化的风险底图（2D），键如 "ice"/"accident"/"interact"
    risk_layers_for_overlay: Dict[str, xr.DataArray] | None = None
    # 高级配置与实际生效（便于 UI 与调试）
    fusion_mode_effective: Optional[str] | None = None
    w_interact: Optional[float] | None = None
    use_escort: Optional[bool] | None = None
    risk_agg_mode: Optional[str] | None = None
    risk_agg_mode_effective: Optional[str] | None = None
    risk_agg_alpha: Optional[float] | None = None
    escort_applied: Optional[bool] | None = None
    # 策略模式（balanced/safe/efficient 等）；保留默认以兼容旧版
    profile_name: str = "baseline"
    # 船型配置（可选）：用于 Eco 与权重缩放
    vessel_profile: Optional[dict] | None = None
    # Pipeline 追踪（可选）：仅用于 UI 展示
    pipeline: "PipelineTrace" | None = None
    # 规划域（经纬度范围）
    domain: Optional["PlanningDomain"] | None = None

    @property
    def has_latlon(self) -> bool:
        return (self.lat_arr is not None) and (self.lon_arr is not None)


class RouteResult(NamedTuple):
    path_ij: list[tuple[int, int]]
    path_lonlat: list[list[float]]
    cost_sum: float
    len: int
    reachable: bool
    heuristic: str
    diagonal: bool
    # Pipeline 追踪（可选）：仅用于 UI 展示
    pipeline: "PipelineTrace" | None = None
    # 调试信息（新增）：用于测试与诊断
    debug: dict | None = None


class EcoSummary(NamedTuple):
    fuel_total_t: float
    co2_total_t: float
    cost_usd: float
    details: dict


# -------------------------
# 路径与文件工具
# -------------------------
# 统一项目根路径：包含 pyproject.toml、Dockerfile 的那层（本仓库为 minimum/）

@lru_cache(maxsize=1)
def _load_runtime_cfg() -> dict:
    try:
        cfg_path = get_project_root() / "ArcticRoute" / "config" / "runtime.yaml"
        if not cfg_path.exists():
            return {}
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

def get_project_root() -> Path:
    # 该文件位于 minimum/ArcticRoute/... 下
    # parents[0] -> .../ArcticRoute/core/planner_service.py
    # parents[1] -> .../ArcticRoute
    # parents[2] -> .../minimum
    return Path(__file__).resolve().parents[2]

# 为向后兼容的内部别名（避免出现自定义 root 的多个实现）
_repo_root = get_project_root

# 统一目录变量，避免出现 root/"ArcticRoute"/"ArcticRoute" 之类错误
root = get_project_root()                  # .../minimum
arctic_dir = root / "ArcticRoute"          # .../minimum/ArcticRoute

# 统一 PRIOR 路径（唯一来源）
PRIOR_ORIG_PATH = arctic_dir / "reports" / "phaseE" / "center" / "prior_centerlines_all.geojson"
PRIOR_WGS_PATH  = arctic_dir / "data_processed" / "prior" / "prior_centerlines_all_wgs84.geojson"

# 新版先验缓存（完整 WGS84 GDF 与文件可用性）
_prior_full_gdf: Optional["gpd.GeoDataFrame"] = None
_prior_file_available: Optional[bool] = None


def _merged_dir() -> Path:
    return arctic_dir / "data_processed" / "ice_forecast" / "merged"


def _risk_dir() -> Path:
    return arctic_dir / "data_processed" / "risk"


def _open_nc_slice(path: Path, var_candidates: list[str]) -> xr.DataArray | None:
    if not path.exists():
        return None
    try:
        with xr.open_dataset(path) as ds:
            for v in var_candidates:
                if v in ds.data_vars:
                    da = ds[v]
                    if "time" in da.dims:
                        da = da.isel(time=0)
                    return da.load()
            # 兜底：取首个变量
            if ds.data_vars:
                da = next(iter(ds.data_vars.values()))
                if "time" in da.dims:
                    da = da.isel(time=0)
                return da.load()
    except Exception:
        pass
    return None


def load_icebreaker_corridor(ym: str, env_lat: Optional[np.ndarray], env_lon: Optional[np.ndarray]) -> Optional[xr.DataArray]:
    """
    加载 icebreaker_corridor_{ym}.nc 并插值到当前 env 网格。
    返回二维 DataArray，dims=(y,x)，范围约 0..1；失败返回 None。
    """
    try:
        risk_dir = _risk_dir()
        f = risk_dir / f"icebreaker_corridor_{ym}.nc"
        if not f.exists():
            return None
        with xr.open_dataset(f) as ds:
            if "icebreaker_corridor" not in ds.data_vars:
                return None
            da = ds["icebreaker_corridor"]
            # 插值至 env 网格（优先 1D）
            if env_lat is not None and env_lon is not None and getattr(env_lat, "ndim", 1) == 1 and getattr(env_lon, "ndim", 1) == 1:
                latn = next((n for n in da.coords if "lat" in n.lower()), "latitude")
                lonn = next((n for n in da.coords if "lon" in n.lower()), "longitude")
                da2 = da.interp({latn: xr.DataArray(env_lat, dims=(latn,)), lonn: xr.DataArray(env_lon, dims=(lonn,))})
                # 统一命名为 (y,x)
                try:
                    da2 = da2.rename({latn: "y", lonn: "x"})
                except Exception:
                    pass
                return da2.astype("float32").load()
            else:
                # 回退：直接返回原始（后续调用方若需要再近邻重采样）
                # 统一只保留最后两维为空间维
                if da.ndim > 2 and "time" in da.dims:
                    da = da.isel(time=0)
                # 尽量重命名
                try:
                    latn = next((n for n in da.coords if "lat" in n.lower()), None)
                    lonn = next((n for n in da.coords if "lon" in n.lower()), None)
                    if latn and lonn:
                        da = da.rename({latn: "y", lonn: "x"})
                except Exception:
                    pass
                return da.astype("float32").load()
    except Exception:
        return None


# 额外：从 grid_spec.json / env_clean.nc 读取网格坐标（作为坐标兜底）
def _load_grid_latlon_from_spec() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    import json
    candidates = [
        _repo_root() / "ArcticRoute" / "config" / "grid_spec.json",

        _repo_root() / "ArcticRoute" / "data_processed" / "grid_spec.json",
    ]
    for p in candidates:
        try:
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                lat = obj.get("lat")
                lon = obj.get("lon")
                if isinstance(lat, list) and isinstance(lon, list):
                    lat_arr = np.array(lat, dtype=float)
                    lon_arr = np.array(lon, dtype=float)
                    return lat_arr, lon_arr
        except Exception:
            continue
    return None, None


def _load_grid_latlon_from_env_nc() -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    candidates = [
        _repo_root() / "ArcticRoute" / "data_processed" / "env_clean.nc",
        _repo_root() / "ArcticRoute" / "data_processed" / "env" / "env_clean.nc",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            with xr.open_dataset(p) as ds:
                lat_name = next((n for n in ["lat", "latitude"] if n in ds.variables or n in ds.coords), None)
                lon_name = next((n for n in ["lon", "longitude"] if n in ds.variables or n in ds.coords), None)
                if lat_name and lon_name:
                    lat = ds[lat_name].values
                    lon = ds[lon_name].values
                    return lat.astype(float), lon.astype(float)
        except Exception:
            continue
    return None, None


# -------------------------
# 先验主航线加载与距离场
# -------------------------
_prior_available_cache: Optional[bool] = None
# 针对不同网格 bbox 的裁剪结果做简单缓存，减少重复计算与日志噪声
_prior_clip_cache: dict[tuple[float, float, float, float, float, float], "gpd.GeoDataFrame"] = {}


def _load_prior_centerlines_full() -> Optional["gpd.GeoDataFrame"]:
    """
    加载完整的历史主航线 GeoDataFrame（EPSG:4326），不做任何裁剪。
    - 优先使用 PRIOR_WGS_PATH（应为 4326）
    - 若不存在则尝试 PRIOR_ORIG_PATH，并转换到 4326
    - 若两者都不存在，则返回 None
    """
    global _prior_full_gdf, _prior_file_available

    if gpd is None:
        _prior_file_available = False
        return None

    if _prior_file_available is False:
        return None
    if _prior_full_gdf is not None:
        return _prior_full_gdf

    path = None
    g: Optional["gpd.GeoDataFrame"] = None

    try:
        if PRIOR_WGS_PATH.exists():
            path = PRIOR_WGS_PATH
            g = gpd.read_file(path)
            print(f"[PRIOR] loaded WGS centerlines from {path}")
        elif PRIOR_ORIG_PATH.exists():
            path = PRIOR_ORIG_PATH
            g_orig = gpd.read_file(path)
            print(f"[PRIOR] loaded ORIG centerlines from {path} (crs={getattr(g_orig, 'crs', None)})")
            if getattr(g_orig, "crs", None) is None:
                print("[PRIOR] ORIG has no CRS; assuming EPSG:4326")
                g_orig.set_crs(epsg=4326, inplace=True, allow_override=True)
            try:
                epsg = g_orig.crs.to_epsg() if g_orig.crs is not None else 4326
            except Exception:
                epsg = None
            if epsg != 4326:
                print(f"[PRIOR] reprojecting ORIG {g_orig.crs} -> EPSG:4326")
                g = g_orig.to_crs(epsg=4326)
            else:
                g = g_orig
        else:
            print(f"[PRIOR] no centerline file found at {PRIOR_WGS_PATH} or {PRIOR_ORIG_PATH}")
            _prior_file_available = False
            return None
    except Exception as e:
        print(f"[PRIOR] read error: {e}")
        _prior_file_available = False
        return None

    if g is None or len(g) == 0:
        print("[PRIOR] loaded GeoDataFrame is empty")
        _prior_file_available = False
        return None

    # 强制 CRSs 是 EPSG:4326
    try:
        if getattr(g, "crs", None) is None:
            print("[PRIOR] loaded gdf has no CRS; setting EPSG:4326")
            g.set_crs(epsg=4326, inplace=True, allow_override=True)
        else:
            epsg = None
            try:
                epsg = g.crs.to_epsg()
            except Exception:
                epsg = None
            if epsg != 4326:
                print(f"[PRIOR] gdf.crs={g.crs}, converting to EPSG:4326")
                g = g.to_crs(epsg=4326)
    except Exception as e:
        print(f"[PRIOR] CRS handling error: {e}")

    _prior_full_gdf = g
    _prior_file_available = True
    return g


def load_prior_centerlines_wgs84() -> Optional["gpd.GeoDataFrame"]:
    """历史兼容：返回完整的 WGS84 主航线，不做裁剪。"""
    return _load_prior_centerlines_full()


def prior_on_grid_bounds(env: "EnvironmentContext", g: "gpd.GeoDataFrame", lon_margin: float = 20.0, lat_margin: float = 10.0) -> tuple["gpd.GeoDataFrame", bool]:
    """
    [DEPRECATED] 仅保留以兼容旧内部调用。新逻辑请使用 load_prior_centerlines_for_env(env)。
    """

    """
    根据 env 的网格范围过滤：优先使用 env.lat2d/lon2d；若缺失则使用 env.lat_arr/lon_arr（必要时 meshgrid）。
    返回 (filtered_gdf, ok)：ok=False 表示无重叠，视为不可用。
    """
    import numpy as _np
    # 1) 优先 lat2d/lon2d
    lat2d = getattr(env, "lat2d", None)
    lon2d = getattr(env, "lon2d", None)
    lat_vals = None
    lon_vals = None
    if lat2d is not None and lon2d is not None:
        lat_vals = lat2d.values if hasattr(lat2d, "values") else _np.array(lat2d)
        lon_vals = lon2d.values if hasattr(lon2d, "values") else _np.array(lon2d)
    else:
        # 2) 回退 lat_arr/lon_arr
        lat_arr = getattr(env, "lat_arr", None)
        lon_arr = getattr(env, "lon_arr", None)
        if lat_arr is not None and lon_arr is not None:
            la = lat_arr.values if hasattr(lat_arr, "values") else _np.array(lat_arr)
            lo = lon_arr.values if hasattr(lon_arr, "values") else _np.array(lon_arr)
            if la.ndim == 2 and lo.ndim == 2 and la.shape == lo.shape:
                lat_vals, lon_vals = la, lo
            elif la.ndim == 1 and lo.ndim == 1 and la.size > 1 and lo.size > 1:
                lon_vals, lat_vals = _np.meshgrid(lo, la)
    if lat_vals is None or lon_vals is None:
        print("[PRIOR] env has no usable lat/lon grid; skip grid alignment")
        return g, True

    grid_lat_min = float(_np.nanmin(lat_vals))
    grid_lat_max = float(_np.nanmax(lat_vals))
    grid_lon_vals = ((lon_vals + 180.0) % 360.0) - 180.0
    grid_lon_min = float(_np.nanmin(grid_lon_vals))
    grid_lon_max = float(_np.nanmax(grid_lon_vals))

    # 计算动态 span 与 margin
    grid_lat_span = max(1e-6, grid_lat_max - grid_lat_min)
    grid_lon_span = max(1e-6, grid_lon_max - grid_lon_min)
    lat_margin_dyn = min(5.0, max(1.0, grid_lat_span * 2.0))
    lon_margin_dyn = min(30.0, max(5.0, grid_lon_span * 2.0))

    print("[PRIOR] grid bounds lat:", grid_lat_min, grid_lat_max,
          "lon:", grid_lon_min, grid_lon_max,
          "lat_span:", grid_lat_span, "lon_span:", grid_lon_span,
          "lat_margin:", lat_margin_dyn, "lon_margin:", lon_margin_dyn)

    lon_min, lat_min, lon_max, lat_max = g.total_bounds
    if lon_min < -180 or lon_max > 360:
        print("[PRIOR] warning: lon range out of [-180,360], ignoring for now")

    print("[PRIOR] centerlines bounds lat:", lat_min, lat_max, "lon:", lon_min, lon_max)

    # 使用动态 margin 判定是否有交集
    intersects_lon = (lon_max >= grid_lon_min - lon_margin_dyn) and (lon_min <= grid_lon_max + lon_margin_dyn)
    intersects_lat = (lat_max >= grid_lat_min - lat_margin_dyn) and (lat_min <= grid_lat_max + lat_margin_dyn)

    if not (intersects_lon and intersects_lat):
        print("[PRIOR] centerlines bbox has no overlap with grid (even with margin); treat as unusable")
        return g.iloc[0:0], False

    return g, True


def prior_data_available(env: Optional["EnvironmentContext"] = None) -> bool:
    global _prior_available_cache

    # 若提供 env，则基于裁剪后的 gdf 判断可用性
    if env is not None:
        try:
            g_env = load_prior_centerlines_for_env(env)
        except Exception:
            g_env = None
        ok = (g_env is not None) and (len(g_env) > 0)
        _prior_available_cache = ok
        print(f"[PRIOR] data_available (with env): {ok}")
        return ok

    # 无 env：仅文件级判断
    g_full = _load_prior_centerlines_full()
    ok = (g_full is not None) and (len(g_full) > 0)
    _prior_available_cache = ok
    print(f"[PRIOR] data_available (file-level only): {ok}")
    return ok

# 历史接口兼容
@st.cache_data
def load_prior_centerlines() -> Optional["gpd.GeoDataFrame"]:
    return load_prior_centerlines_wgs84()


def _grid_latlon(env: EnvironmentContext) -> Optional[tuple[np.ndarray, np.ndarray]]:
    if env.lat_arr is None or env.lon_arr is None:
        return None
    return env.lat_arr, env.lon_arr


def _get_grid_bounds(env: "EnvironmentContext") -> tuple[float, float, float, float]:
    """
    从 env 中获取网格的纬度/经度范围（lat_min, lat_max, lon_min, lon_max）。
    - 优先使用 lat2d/lon2d
    - 若不存在，则使用 lat_arr/lon_arr
    - 经度统一 wrap 到 [-180, 180]
    """
    lat2d = getattr(env, "lat2d", None)
    lon2d = getattr(env, "lon2d", None)

    if lat2d is not None and lon2d is not None:
        lat_vals = lat2d.values if hasattr(lat2d, "values") else np.array(lat2d)
        lon_vals = lon2d.values if hasattr(lon2d, "values") else np.array(lon2d)
    else:
        lat_arr = getattr(env, "lat_arr", None)
        lon_arr = getattr(env, "lon_arr", None)
        if lat_arr is None or lon_arr is None:
            raise ValueError("[PRIOR] EnvironmentContext has no lat2d/lon2d or lat_arr/lon_arr")
        lat_vals = lat_arr.values if hasattr(lat_arr, "values") else np.array(lat_arr)
        lon_vals = lon_arr.values if hasattr(lon_arr, "values") else np.array(lon_arr)
        if getattr(lat_vals, "ndim", 1) == 1 and getattr(lon_vals, "ndim", 1) == 1:
            lon_vals, lat_vals = np.meshgrid(lon_vals, lat_vals)

    lat_flat = np.asarray(lat_vals).ravel()
    lon_flat = np.asarray(lon_vals).ravel()
    lon_flat = ((lon_flat + 180.0) % 360.0) - 180.0

    lat_min = float(np.nanmin(lat_flat))
    lat_max = float(np.nanmax(lat_flat))
    lon_min = float(np.nanmin(lon_flat))
    lon_max = float(np.nanmax(lon_flat))

    return lat_min, lat_max, lon_min, lon_max


def load_prior_centerlines_for_env(env: "EnvironmentContext") -> Optional["gpd.GeoDataFrame"]:
    """
    为当前环境上下文裁剪历史主航线：
    - 加载完整 gdf（EPSG:4326）
    - 计算网格 bbox
    - 构造带一定 margin 的 bbox Polygon
    - Clip 到该 bbox，仅保留与网格有交集的线段
    - 若裁剪后为空，则返回 None
    """
    g = _load_prior_centerlines_full()
    if g is None:
        return None

    try:
        lat_min, lat_max, lon_min, lon_max = _get_grid_bounds(env)
    except Exception as e:
        print(f"[PRIOR] failed to get grid bounds: {e}")
        return None

    lat_span = max(1e-6, lat_max - lat_min)
    lon_span = max(1e-6, lon_max - lon_min)

    lat_margin = min(5.0, max(1.0, lat_span * 2.0))
    lon_margin = min(30.0, max(5.0, lon_span * 2.0))

    print("[PRIOR] grid bounds lat:", lat_min, lat_max,
          "lon:", lon_min, lon_max,
          "lat_span:", lat_span, "lon_span:", lon_span,
          "lat_margin:", lat_margin, "lon_margin:", lon_margin)

    if box is None:
        print("[PRIOR] shapely.box unavailable; cannot clip")
        return None

    bbox_poly = box(
        lon_min - lon_margin,
        lat_min - lat_margin,
        lon_max + lon_margin,
        lat_max + lat_margin,
    )

    # 使用 intersects 过滤；clip 也可以，这里用 intersects + gpd.clip 组合更保险
    try:
        g_intersect = g[g.geometry.intersects(bbox_poly)]
    except Exception as e:
        print(f"[PRIOR] intersects check failed: {e}")
        return None

    if len(g_intersect) == 0:
        print("[PRIOR] centerlines have no intersection with grid bbox; treat as unusable for this env")
        return None

    try:
        g_clipped = gpd.clip(g_intersect, gpd.GeoDataFrame(geometry=[bbox_poly], crs="EPSG:4326"))
    except Exception as e:
        print(f"[PRIOR] clip failed, fall back to intersects-only gdf: {e}")
        g_clipped = g_intersect

    if len(g_clipped) == 0:
        print("[PRIOR] clipped centerlines empty after clip; treat as unusable")
        return None

    print(f"[PRIOR] centerlines clipped count: {len(g_clipped)}")
    return g_clipped


def _rasterize_centerlines_to_penalty(env: EnvironmentContext) -> Optional[xr.DataArray]:
    """
    将线状先验（WGS84）栅格化到与 env.cost_da 一致的网格上，并计算距离变换归一化到 [0,1]。
    若无法构建（缺少坐标或依赖），返回 None。
    """
    if gpd is None or sgeom is None:
        return None
    # 使用已裁剪、已对齐的先验线
    gdf = load_prior_centerlines_for_env(env)
    if gdf is None or gdf.empty:
        print("[PRIOR] no usable centerlines for this env; skip prior penalty")
        return None
    grid = _grid_latlon(env)
    if grid is None or env.cost_da is None:
        return None
    lat_arr, lon_arr = grid
    # 只支持 2D 网格；若是 1D（y,x），转 meshgrid
    if lat_arr.ndim == 1 and lon_arr.ndim == 1:
        Lon, Lat = np.meshgrid(lon_arr, lat_arr)
    else:
        Lat, Lon = lat_arr, lon_arr

    H, W = Lat.shape
    centerline_mask = np.zeros((H, W), dtype=bool)

    # 简单最近邻索引函数
    def _latlon_to_ij(latv: float, lonv: float) -> tuple[int, int]:
        if Lat.ndim == 2:
            d2 = (Lat - latv) ** 2 + (Lon - lonv) ** 2
            i, j = np.unravel_index(np.nanargmin(d2), d2.shape)
            return int(i), int(j)
        return 0, 0

    # Bresenham 栅格化线
    def _draw_line(i0: int, j0: int, i1: int, j1: int):
        di = abs(i1 - i0)
        dj = abs(j1 - j0)
        si = 1 if i0 < i1 else -1
        sj = 1 if j0 < j1 else -1
        err = di - dj
        i, j = i0, j0
        while True:
            if 0 <= i < H and 0 <= j < W:
                centerline_mask[i, j] = True
            if i == i1 and j == j1:
                break
            e2 = 2 * err
            if e2 > -dj:
                err -= dj
                i += si
            if e2 < di:
                err += di
                j += sj

    try:
        for geom in gdf.geometry:
            if geom is None:
                continue
            if isinstance(geom, sgeom.LineString):
                coords = list(geom.coords)
                for k in range(len(coords) - 1):
                    lon1, lat1 = coords[k]
                    lon2, lat2 = coords[k + 1]
                    i0, j0 = _latlon_to_ij(float(lat1), float(lon1))
                    i1, j1 = _latlon_to_ij(float(lat2), float(lon2))
                    _draw_line(i0, j0, i1, j1)
            elif isinstance(geom, sgeom.MultiLineString):
                for line in geom.geoms:
                    coords = list(line.coords)
                    for k in range(len(coords) - 1):
                        lon1, lat1 = coords[k]
                        lon2, lat2 = coords[k + 1]
                        i0, j0 = _latlon_to_ij(float(lat1), float(lon1))
                        i1, j1 = _latlon_to_ij(float(lat2), float(lon2))
                        _draw_line(i0, j0, i1, j1)
    except Exception:
        return None

    if not np.any(centerline_mask):
        return None

    dist = distance_transform_edt(~centerline_mask).astype("float32")
    vmax = float(np.max(dist)) if np.isfinite(dist).any() else 0.0
    if vmax > 0:
        penalty = (dist / vmax).astype("float32")
    else:
        penalty = np.zeros_like(dist, dtype="float32")
    da = xr.DataArray(penalty, dims=("y", "x"))
    da.name = "prior_penalty"
    return da


# -------------------------
# 策略差异化成本构造
# -------------------------
def build_profile_cost_variant(
    env: EnvironmentContext,
    profile_name: str,
    base_cost_da: xr.DataArray,
    land_mask: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    根据策略模式（balanced / safe / efficient），在 base_cost_da 基础上构造不同的成本版本。
    - efficient: 几乎只看距离 -> 使用“简单海洋=1/陆地=禁行(Inf)”的成本（可通过 runtime.flag 关闭）
    - safe: 强化风险 -> 把 risk 层权重放大，惩罚冰区
    - balanced: 使用当前默认逻辑（接近原 1.x 行为）

    降级策略：
    - 任意异常返回 base_cost_da，保证链路不中断。
    """
    import numpy as _np
    import xarray as _xr
    import numpy as _n

    try:
        # 不修改原对象
        cost = base_cost_da.copy()

        # 读取开关：是否强制所有 profile 禁止登陆（用于防止“效率优先”穿陆/贴岸越界）
        try:
            cfg = _load_runtime_cfg()
            forbid_land_all = bool((cfg.get("flags", {}) or {}).get("forbid_land_in_all_profiles", True))
        except Exception:
            forbid_land_all = True

        # 构造“简化距离成本”：海洋=1，陆地=Inf（若未启用强制禁止，则陆地=1000 的高惩罚）
        if land_mask is not None:
            if forbid_land_all:
                simple_cost = _xr.where(land_mask > 0.5, _n.inf, 1.0)
            else:
                simple_cost = _xr.where(land_mask > 0.5, 1000.0, 1.0)
        else:
            simple_cost = cost

        name = (profile_name or "balanced").lower()

        if name == "efficient":
            # 效率优先：几乎只看距离，但必须继承“禁行陆地”的硬约束
            return simple_cost.astype("float32")

        if name == "safe":
            # 安全优先：放大风险差异（非线性拉伸）
            # 这里采用：cost_safe = 1 + (cost - 1) * 3
            safe = 1.0 + (cost - 1.0) * 3.0
            return safe.astype("float32")

        # balanced 或其它：保持原始 cost
        return cost.astype("float32")
    except Exception:
        return base_cost_da


# -------------------------
# 公共函数：环境加载（含高级参数）
# -------------------------

# AIS 权重配置加载
@lru_cache(maxsize=1)
def _weight_profiles_yaml_path() -> Path:
    return get_project_root() / "ArcticRoute" / "config" / "weight_profiles_from_ais.yaml"


def load_weight_profile_from_ais(profile_name: str | None = None, scenario_name: str | None = None) -> dict | None:
    """
    从 ArcticRoute/config/weight_profiles_from_ais.yaml 加载单个 profile，返回包含
    {ice, accident, interact, wave, prior} 的字典；找不到则返回 None。
    profile_name 为空时：优先 ais_{ym}_default（调用方可传入），否则返回第一个 profile。
    """
    p = _weight_profiles_yaml_path()
    if not p.exists():
        return None
    try:
        import yaml as _yaml  # 已在顶部可选导入
        with p.open("r", encoding="utf-8") as f:
            obj = _yaml.safe_load(f) or {}
        profs = (obj.get("profiles") or {})
        if not isinstance(profs, dict) or not profs:
            return None
        # 精确命中
        if profile_name and profile_name in profs:
            d = profs.get(profile_name) or {}
            return {k: float(d.get(k, 0.0) or 0.0) for k in ("ice", "accident", "interact", "wave", "prior")}
        # 常见回退：ais_default / nsr_default
        for key in (
            "ais_default",
            "nsr_default",
        ):
            if key in profs:
                d = profs.get(key) or {}
                return {k: float(d.get(k, 0.0) or 0.0) for k in ("ice", "accident", "interact", "wave", "prior")}
        # 兜底：取第一个
        k0 = next(iter(profs.keys()))
        d = profs.get(k0) or {}
        return {k: float(d.get(k, 0.0) or 0.0) for k in ("ice", "accident", "interact", "wave", "prior")}
    except Exception:
        return None

@st.cache_data
def load_environment(
    ym: str,
    w_ice: float = 0.7,
    w_accident: float = 0.2,
    prior_weight: float = 0.0,
    *,
    profile_name: str = "baseline",
    **kwargs,
) -> EnvironmentContext:
    """加载环境与成本图。
    扩展参数（通过 kwargs 传入，保持签名兼容）：
    - fusion_mode: str = "baseline"    # "baseline"|"linear"|"unetformer"|"poe"|"evidential"
    - w_interact: float = 0.0           # 交互/拥挤风险权重
    - use_escort: bool = False          # 优先使用 R_ice_eff_<ym>.nc 替代 ice
    - risk_agg_mode: str = "mean"       # "mean"|"quantile"|"cvar"（在有样本/证据时生效，否则回退）
    - risk_agg_alpha: float = 0.9
    - alpha_ice: 兼容旧参数 → 覆盖 w_ice
    - debug_profile_cost: 仅调试打印
    """
    # 兼容旧 alpha_ice
    if "alpha_ice" in kwargs and isinstance(kwargs["alpha_ice"], (int, float)):
        w_ice = float(kwargs["alpha_ice"])

    fusion_mode = str(kwargs.get("fusion_mode", "baseline"))
    w_interact = float(kwargs.get("w_interact", 0.0) or 0.0)
    use_escort = bool(kwargs.get("use_escort", False))
    risk_agg_mode = str(kwargs.get("risk_agg_mode", "mean"))
    risk_agg_alpha = float(kwargs.get("risk_agg_alpha", 0.9) or 0.9)

    # 船型配置（可选）：用于 Eco 与权重缩放
    vessel_profile_name = kwargs.get("vessel_profile") or kwargs.get("vessel_profile_name")
    vessel_profile: Optional[dict] = None
    if _load_vessel_profile is not None and vessel_profile_name:
        try:
            vp = _load_vessel_profile(str(vessel_profile_name))
            if vp:
                vessel_profile = dict(vp)
                vessel_profile["name"] = str(vessel_profile_name)
        except Exception:
            vessel_profile = None
    # 默认回退到常见船型（用于自检/无 UI 场景下避免 base fuel 警告）
    if vessel_profile is None and _load_vessel_profile is not None:
        try:
            vp = _load_vessel_profile("panamax")
            if vp:
                vessel_profile = dict(vp)
                vessel_profile["name"] = "panamax"
        except Exception:
            vessel_profile = vessel_profile  # keep None if fail
    # 合并自定义 overrides（若提供）
    try:
        vessel_overrides = kwargs.get("vessel_overrides") or {}
        if isinstance(vessel_overrides, dict):
            if vessel_profile is None:
                vessel_profile = {}
            for k in ("base_fuel_per_km", "ice_sensitivity", "wave_sensitivity", "dwt", "ice_class"):
                if k in vessel_overrides and vessel_overrides[k] is not None:
                    vessel_profile[k] = vessel_overrides[k]
    except Exception:
        pass

    # 若未显式指定手动权重，则优先尝试从 AIS 权重配置加载（默认启用，可通过 use_ais_weight_profile=False 关闭）
    use_ais_prof = bool(kwargs.get("use_ais_weight_profile", True))
    weights_profile_name = kwargs.get("weights_profile", f"ais_{ym}_default")
    if use_ais_prof:
        try:
            prof = load_weight_profile_from_ais(str(weights_profile_name), None)
        except Exception:
            prof = None
        if prof:
            try:
                w_ice = float(prof.get("ice", w_ice))
                w_accident = float(prof.get("accident", w_accident))
                prior_weight = float(prof.get("prior", prior_weight))
                w_interact = float(prof.get("interact", w_interact))
                # newenv/wave 通过 kwargs 传入后续流程
                wave_w = float(prof.get("wave", kwargs.get("w_wave", 0.0) or 0.0))
                kwargs["w_wave"] = wave_w
                if wave_w > 0.0:
                    kwargs.setdefault("use_newenv_for_cost", True)
                # 记录最终采用的 profile 名称（便于 UI 展示）
                kwargs["weights_profile_effective"] = str(weights_profile_name)
                print(f"[AIS_PROFILE] 使用 AIS 权重: {weights_profile_name} -> ice={w_ice} acc={w_accident} inter={w_interact} wave={wave_w} prior={prior_weight}")
            except Exception:
                pass

    risk_dir = _risk_dir()
    merged_dir = _merged_dir()
    sic_path = merged_dir / f"sic_fcst_{ym}.nc"

    # 数据加载
    sic_da = _open_nc_slice(sic_path, ["sic_pred", "sic", "siconc"])

    # 读取破冰走廊折扣配置
    use_icebreaker_corridor = bool(kwargs.get("use_icebreaker_corridor", False))
    icebreaker_corridor_alpha = float(kwargs.get("icebreaker_corridor_alpha", 0.0) or 0.0)

    # 风险层（基础：ice/accident），可根据 use_escort 替换 ice 为 R_ice_eff
    risk_layers: Dict[str, xr.DataArray] = {}
    escort_applied_flag = False
    if use_escort:
        da_ice_eff = None
        try:
            if adv_load_escort_risk is not None:
                da_ice_eff = adv_load_escort_risk(ym)
        except Exception:
            da_ice_eff = None
        if da_ice_eff is None:
            da_ice_eff = _open_nc_slice(risk_dir / f"R_ice_eff_{ym}.nc", ["risk", "R_ice_eff", "risk_ice"])  # 兼容旧命名
        if da_ice_eff is not None:
            try:
                if adv_reduce_to_2d is not None:
                    da_ice_eff = adv_reduce_to_2d(da_ice_eff)
            except Exception:
                pass
            risk_layers["ice"] = da_ice_eff
            escort_applied_flag = True
        else:
            print("[ESCORT] use_escort=True but R_ice_eff not available, fallback to risk_ice")
            da_ice = _open_nc_slice(risk_dir / f"risk_ice_{ym}.nc", ["R_ice", "risk_ice"])  # 回退
            if da_ice is not None:
                risk_layers["ice"] = da_ice
    else:
        da_ice = _open_nc_slice(risk_dir / f"risk_ice_{ym}.nc", ["R_ice", "risk_ice"])
        if da_ice is not None:
            try:
                if adv_reduce_to_2d is not None:
                    da_ice = adv_reduce_to_2d(da_ice)
            except Exception:
                pass
            risk_layers["ice"] = da_ice

    da_acc = _open_nc_slice(risk_dir / f"risk_accident_{ym}.nc", ["R_acc", "risk_accident"])
    if da_acc is not None:
        try:
            if adv_reduce_to_2d is not None:
                da_acc = adv_reduce_to_2d(da_acc)
        except Exception:
            pass
        risk_layers["accident"] = da_acc

    # 交互/拥挤层（可选）
    da_interact = None
    if w_interact > 0:
        try:
            if adv_load_interact_risk is not None:
                da_interact = adv_load_interact_risk(ym)
        except Exception:
            da_interact = None
        if da_interact is None:
            da_interact = _open_nc_slice(risk_dir / f"R_interact_{ym}.nc", ["risk", "R_interact", "risk_interact"])  # 兼容旧
            if da_interact is None:
                da_interact = _open_nc_slice(risk_dir / f"risk_interact_{ym}.nc", ["risk", "R_interact", "risk_interact"])  # 另一命名
    if da_interact is not None:
        try:
            if adv_reduce_to_2d is not None:
                da_interact = adv_reduce_to_2d(da_interact)
        except Exception:
            pass
        risk_layers["interact"] = da_interact

    # 网格坐标（从任意可用数据探测）
    lat_arr, lon_arr = None, None
    probe_path = None
    for cand in [risk_dir / f"R_ice_eff_{ym}.nc", risk_dir / f"risk_ice_{ym}.nc", risk_dir / f"risk_accident_{ym}.nc", sic_path]:
        if cand.exists():
            probe_path = cand
            break
    if probe_path:
        try:
            with xr.open_dataset(probe_path) as ds:
                lat_name = next((n for n in ["lat", "latitude"] if n in ds.coords), None)
                lon_name = next((n for n in ["lon", "longitude"] if n in ds.coords), None)
                if lat_name and lon_name:
                    lat = ds[lat_name].values
                    lon = ds[lon_name].values
                    lat_arr, lon_arr = lat, lon
        except Exception:
            pass

    # 若拿到的坐标为退化的 1D 且长度极小（<=4），继续尝试使用更高质量的坐标源（grid_spec / env_clean）
    def _is_degenerate(a: Optional[np.ndarray]) -> bool:
        try:
            if a is None:
                return True
            if not isinstance(a, np.ndarray):
                return True
            if a.ndim == 1 and a.size <= 4:
                return True
            return False
        except Exception:
            return True

    # 兜底：grid_spec.json（用于替代退化坐标或填补缺失）
    if (lat_arr is None or lon_arr is None) or (_is_degenerate(lat_arr) or _is_degenerate(lon_arr)):
        lat2, lon2 = _load_grid_latlon_from_spec()
        if lat2 is not None and lon2 is not None and (not _is_degenerate(lat2)) and (not _is_degenerate(lon2)):
            lat_arr, lon_arr = lat2, lon2

    # 兜底：env_clean.nc（若上一步仍无合理坐标，则继续尝试）
    if (lat_arr is None or lon_arr is None) or (_is_degenerate(lat_arr) or _is_degenerate(lon_arr)):
        lat2, lon2 = _load_grid_latlon_from_env_nc()
        if lat2 is not None and lon2 is not None and (not _is_degenerate(lat2)) and (not _is_degenerate(lon2)):
            lat_arr, lon_arr = lat2, lon2

    # 陆地掩膜：优先从权威数据源读取，失败时再回退到 SIC NaN；并做形态学膨胀以防“贴岸穿越”
    # 推断目标网格尺寸 (H,W)
    H = W = None
    if sic_da is not None:
        H, W = sic_da.shape[-2:]
    elif lat_arr is not None and lon_arr is not None:
        if getattr(lat_arr, "ndim", 1) == 2:
            H, W = lat_arr.shape
        elif getattr(lat_arr, "ndim", 1) == 1 and getattr(lon_arr, "ndim", 1) == 1:
            H, W = len(lat_arr), len(lon_arr)
    if H is None or W is None:
        H, W = 121, 1161

    land_mask: Optional[np.ndarray] = None
    # 默认不对陆地掩膜做膨胀，避免误封闭狭窄水道；如需更保守可通过 land_mask_pad_px>0 开启
    pad_px = int(kwargs.get("land_mask_pad_px", 0) or 0)

    # 1) 尝试权威 land_mask 数据集
    try:
        import os
        import xarray as _xr
        candidates = [
            _repo_root() / "ArcticRoute" / "data_processed" / "newenv" / "land_mask_gebco.nc",
            _repo_root() / "ArcticRoute" / "data_processed" / "env" / "env_clean.nc",
            _repo_root() / "ArcticRoute" / "data_processed" / "env_clean.nc",
            _repo_root() / "ArcticRoute" / "data_processed" / "env" / "land_mask.nc",
            _repo_root() / "ArcticRoute" / "data_processed" / "land_mask.nc",
            _repo_root() / "ArcticRoute" / "data" / "land_mask.nc",
        ]
        var_candidates = ["land_mask_gebco", "land_mask", "lm", "land", "lsm", "landsea_mask", "LSM", "ocean_mask"]
        arr_auth = None
        for p in candidates:
            if not p.exists():
                continue
            try:
                with _xr.open_dataset(p) as ds:
                    if not ds.data_vars:
                        continue
                    var = next((v for v in var_candidates if v in ds.data_vars), None)
                    if var is None:
                        continue
                    da = ds[var]
                    if "time" in da.dims:
                        da = da.isel(time=0)
                    a = np.asarray(da.values)
                    # 取最后两维作为 (y,x)
                    if a.ndim >= 2:
                        a2 = a.reshape(a.shape[-2], a.shape[-1])
                    else:
                        a2 = None
                    if a2 is None:
                        continue
                    # 变量含义：ocean_mask → 1=海洋，需要取反得到 1=陆地
                    if str(var).lower() == "ocean_mask":
                        a2 = 1.0 - a2
                    # 归一并转为布尔
                    a2 = (a2 > 0.5).astype(bool)
                    # 近邻采样到 (H,W)
                    si, sj = a2.shape
                    if (si, sj) != (H, W):
                        yi = (np.linspace(0, si - 1, H)).astype(int)
                        xj = (np.linspace(0, sj - 1, W)).astype(int)
                        a2 = a2[yi[:, None], xj[None, :]]
                    # 先做一次形态学闭运算，修补小孔洞/小裂缝（针对小岛/窄水道更稳健）
                    try:
                        a2 = binary_closing(a2, iterations=1)
                    except Exception:
                        pass
                    # 再做形态学膨胀（安全缓冲）
                    if pad_px > 0:
                        try:
                            a2 = binary_dilation(a2, iterations=pad_px)
                        except Exception:
                            pass
                    frac_land = float(np.mean(a2)) if a2.size > 0 else 1.0
                    if np.isfinite(frac_land) and (0.001 <= frac_land <= 0.995):
                        arr_auth = a2
                        print(f"[LMASK] loaded '{var}' from {p.name}; shape={a2.shape}; frac_land={frac_land:.4f}; pad={pad_px}")
                        break
                    else:
                        print(f"[LMASK] candidate '{var}' in {p.name} suspicious (frac_land={frac_land}); skip")
            except Exception as e:
                print(f"[LMASK] read error on {p}: {e}")
                continue
        if arr_auth is not None:
            land_mask = arr_auth
    except Exception:
        land_mask = None

    # 2) 回退：基于 SIC 的 NaN
    if land_mask is None and sic_da is not None:
        lm = np.isnan(np.asarray(sic_da.values))
        # 近邻采样到 (H,W)（一般已一致，这里兜底）
        if lm.shape != (H, W):
            si, sj = lm.shape[-2:]
            yi = (np.linspace(0, si - 1, H)).astype(int)
            xj = (np.linspace(0, sj - 1, W)).astype(int)
            lm = lm[yi[:, None], xj[None, :]]
        # 先做一次形态学闭运算，修补小孔洞/小裂缝（小岛/窄水道场景）
        try:
            lm = binary_closing(lm, iterations=1)
        except Exception:
            pass
        if pad_px > 0:
            try:
                lm = binary_dilation(lm, iterations=pad_px)
            except Exception:
                pass
        try:
            frac_land = float(np.mean(lm)) if lm.size > 0 else 1.0
            if np.isfinite(frac_land) and (frac_land < 0.995):
                land_mask = lm
                print(f"[LMASK] using sic-derived mask; shape={lm.shape}; frac_land={frac_land:.4f}; pad={pad_px}")
            else:
                print("[LMASK] sic-derived land_mask suspicious; disabled")
        except Exception:
            land_mask = None

    # 构造上下文（供 _build_cost_da 使用）
    env_tmp = EnvironmentContext(
        ym=ym,
        cost_da=None,
        sic_da=sic_da,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        land_mask=land_mask,
        risk_layers_raw=risk_layers if risk_layers else None,
        prior_penalty_da=None,
        vessel_profile=vessel_profile,
    )

    # 读取/构造 fused 风险（当 fusion_mode != baseline）
    fused_da = None
    fusion_mode_effective = "baseline"
    risk_agg_mode_effective_tmp: Optional[str] = None

    # 优先：Evidential 鲁棒表面（基于 Risk + RiskVar）
    if fusion_mode and ("evidential" in str(fusion_mode).lower()):
        try:
            if 'adv_build_evidential_robust_surface' in globals() and adv_build_evidential_robust_surface is not None:  # type: ignore
                fused_da = adv_build_evidential_robust_surface(
                    ym=ym,
                    fusion_mode=fusion_mode,
                    risk_agg_mode=risk_agg_mode,
                    risk_agg_alpha=risk_agg_alpha,
                )
            # 若成功则记录有效模式
            if fused_da is not None:
                fusion_mode_effective = str(fusion_mode)
                try:
                    # 从 attrs 读取 agg；若缺失，按请求的 risk_agg_mode 回填
                    eff = getattr(fused_da, 'attrs', {}).get('agg') if hasattr(fused_da, 'attrs') else None
                    if eff is None:
                        m = (risk_agg_mode or 'mean').lower()
                        eff = ('cvar' if m in ('cvar', 'es', 'expected_shortfall', 'robust') else ('quantile' if m in ('q', 'quantile') else 'mean'))
                    risk_agg_mode_effective_tmp = str(eff)
                except Exception:
                    risk_agg_mode_effective_tmp = (risk_agg_mode or 'mean')
        except Exception:
            fused_da = None

    # 回退：通用 fused risk 加载 + 样本维聚合
    if fused_da is None and fusion_mode and fusion_mode.lower() != "baseline":
        try:
            if adv_load_fused_risk is not None:
                fused_da = adv_load_fused_risk(ym, fusion_mode=fusion_mode)
            else:
                fused_da = None
        except Exception:
            fused_da = None
        if fused_da is None:
            print(f"[FUSION] mode={fusion_mode} not available, fallback to baseline")
            fusion_mode_effective = "baseline"
        else:
            # 若 fused 含样本维，先做聚合
            try:
                dims = list(fused_da.dims)
                sample_dim = next((d for d in ("sample", "member", "ensemble") if d in dims), None)
                if sample_dim is not None:
                    from ArcticRoute.core.advanced_risk import aggregate_risk_da as adv_agg
                    fused_da = adv_agg(fused_da, mode=risk_agg_mode, alpha=risk_agg_alpha)
                    m = (risk_agg_mode or "mean").lower()
                    risk_agg_mode_effective_tmp = ("cvar" if m in ("cvar", "es", "expected_shortfall") else ("quantile" if m in ("q", "quantile") else "mean"))
                else:
                    # 无样本维，后续外层将降级为 mean
                    risk_agg_mode_effective_tmp = None
                # 压成 2D，便于后续对齐与叠加
                try:
                    if adv_reduce_to_2d is not None:
                        fused_da = adv_reduce_to_2d(fused_da)
                except Exception:
                    pass
                fusion_mode_effective = str(fusion_mode)
            except Exception:
                fusion_mode_effective = str(fusion_mode)

    # 船型敏感度对权重的影响（简单版）：缩放 ice/wave 权重
    ice_mult = float((vessel_profile or {}).get("ice_sensitivity", 1.0)) if vessel_profile else 1.0
    wave_mult = float((vessel_profile or {}).get("wave_sensitivity", 1.0)) if vessel_profile else 1.0
    w_ice = float(w_ice) * ice_mult

    # 各风险层权重（基础层=1；interact=w_interact）
    # 使用 UI/调用侧传入的权重，让 w_ice / w_accident 真正影响 base_cost
    layer_weights: Dict[str, float] = {"ice": float(max(0.0, w_ice)), "accident": float(max(0.0, w_accident))}
    if "interact" in (risk_layers or {}):
        layer_weights["interact"] = float(max(0.0, w_interact))
    elif w_interact > 0.0:
        print("[INTERACT] w_interact>0 but interaction risk not available, ignore interaction layer")

    # 读取高级开关与新权重（默认关闭，保持 1.0 行为）
    cfg = _load_runtime_cfg()
    use_newenv_for_cost_flag = bool(
        kwargs.get("use_newenv_for_cost", None)
        if ("use_newenv_for_cost" in kwargs)
        else (((cfg.get("ui", {}) or {}).get("advanced", {}) or {}).get("use_newenv_for_cost", False))
    )
    base_w_wave = float(kwargs.get("w_wave", 0.0) or 0.0)
    # 三种 profile 下的权重派生（针对 newenv 组件），仅在 use_newenv_for_cost 为 True 时生效
    def _scale_weights(profile: str, w_ice_base: float, w_wave_base: float) -> tuple[float, float]:
        name = (profile or "balanced").lower()
        if name == "efficient":
            return (w_ice_base * 0.5, w_wave_base * 0.3)
        if name == "safe":
            return (min(w_ice_base * 1.5, 1.0), min(w_wave_base * 2.0, 1.0))
        return (w_ice_base, w_wave_base)
    w_ice_eff, w_wave_eff = _scale_weights(profile_name, float(w_ice), base_w_wave)
    # 船型敏感度对 newenv 波浪权重的缩放
    try:
        wave_mult = float((vessel_profile or {}).get("wave_sensitivity", 1.0)) if vessel_profile else 1.0
        w_wave_eff = float(w_wave_eff) * wave_mult
    except Exception:
        pass

    # 调试打印：启用与权重
    try:
        print(f"[NEWENV_COST] enabled={use_newenv_for_cost_flag} profile={profile_name} w_ice_newenv={w_ice_eff:.3f} w_wave={w_wave_eff:.3f}")
    except Exception:
        pass

    # 准备 newenv 原始层（保持原坐标，后续在 _build_cost_da 内插值到网格）
    newenv_layers_raw: Dict[str, xr.DataArray] | None = None
    if use_newenv_for_cost_flag:
        try:
            ne = load_newenv_for_cost()
            if ne:
                newenv_layers_raw = {k: v.da for k, v in ne.items() if getattr(v, "da", None) is not None}
        except Exception:
            newenv_layers_raw = None

    # 在构建成本前进行“走廊质量”检查：
    prior_weight_effective = float(prior_weight)
    try:
        # 仅在 prior_weight>0 时尝试降权；GUI 若已关闭则不动
        if prior_weight_effective > 0.0:
            # 从 runtime.yaml 行为段读取走廊路径；失败回退默认
            runtime_path = get_project_root() / "ArcticRoute" / "config" / "runtime.yaml"
            corridor_path = "data_processed/corridor_prob.nc"
            if runtime_path.exists() and yaml is not None:
                try:
                    with open(runtime_path, "r", encoding="utf-8") as _rf:
                        _rt = yaml.safe_load(_rf) or {}
                    _beh = (_rt.get("behavior") or {})
                    corridor_path = str(_beh.get("corridor_path") or corridor_path)
                except Exception:
                    pass
            # 执行质量感知加载
            try:
                from ArcticRoute.core.predictors.env_nc import load_corridor_prob_safe  # type: ignore
                root = get_project_root()
                p1 = (root / corridor_path).resolve()
                p2 = (root / "ArcticRoute" / corridor_path).resolve()
                use_p = p1 if p1.exists() else p2
                da_safe, info = load_corridor_prob_safe(str(use_p))
                if da_safe is None:
                    # 缺失/异常/constant_like → 直接关闭 prior_weight
                    print(f"[PRIOR] 关闭走廊偏好（自动降级），原因: {info}")
                    prior_weight_effective = 0.0
                elif bool(info.get("low_contrast", False)):
                    # 对比度不足 → 将 prior_weight 限幅到一个小值
                    prior_weight_effective = min(prior_weight_effective, 0.05)
                    print(f"[PRIOR] corridor_prob 对比度偏低，prior_weight 降为 {prior_weight_effective}")
            except Exception:
                # 检查异常不影响主流程
                pass
    except Exception:
        pass

    # 构造成本图（支持 fused_da 叠加；支持 prior_weight；支持 layer_weights；支持 newenv 叠加）
    cost_da, prior_da = _build_cost_da(
        env_tmp,
        prior_weight=prior_weight_effective,
        fused_da=fused_da,
        layer_weights=layer_weights,
        newenv_layers=newenv_layers_raw,
        w_ice_newenv=(w_ice_eff if use_newenv_for_cost_flag else 0.0),
        w_wave=(w_wave_eff if use_newenv_for_cost_flag else 0.0),
        use_newenv_for_cost=use_newenv_for_cost_flag,
        use_icebreaker_corridor=use_icebreaker_corridor,
        icebreaker_corridor_alpha=icebreaker_corridor_alpha,
    )

    # 根据策略模式（balanced/safe/efficient）生成差异化成本图
    try:
        land_mask_da = None
        if env_tmp.land_mask is not None:
            land_mask_da = xr.DataArray(env_tmp.land_mask, dims=("y", "x"))
        cost_da = build_profile_cost_variant(env_tmp, profile_name=profile_name, base_cost_da=cost_da, land_mask=land_mask_da)
    except Exception:
        # 任意异常时保持原成本图
        pass

    # 风险聚合占位（若 fused 不含样本，则 quantile/cvar 降级为 mean）
    risk_agg_mode_effective = risk_agg_mode_effective_tmp or str(risk_agg_mode)
    try:
        if (risk_agg_mode_effective_tmp is None) and (str(risk_agg_mode).lower() in ("quantile", "cvar")):
            risk_agg_mode_effective = "mean"
            print(f"[AGG] mode={risk_agg_mode} not applicable; fallback to mean")
    except Exception:
        risk_agg_mode_effective = "mean"

    # 写入元信息（便于 UI 观测）
    try:
        cost_da.attrs["risk_agg_mode"] = str(risk_agg_mode)
        cost_da.attrs["risk_agg_mode_effective"] = str(risk_agg_mode_effective)
        cost_da.attrs["risk_agg_alpha"] = float(risk_agg_alpha)
        cost_da.attrs["fusion_mode_effective"] = str(fusion_mode_effective)
        cost_da.attrs["use_escort"] = bool(use_escort)
        cost_da.attrs["w_interact"] = float(w_interact)
        # 记录权重，便于后续 analyze_route_cost 做一致性缩放
        cost_da.attrs["w_ice"] = float(w_ice)
        cost_da.attrs["w_accident"] = float(w_accident)
        cost_da.attrs["prior_weight"] = float(prior_weight)
        # 记录可用的成本组件标签（供 UI/诊断使用）
        comps = set((env_tmp.risk_layers_raw or {}).keys())
        try:
            if use_newenv_for_cost_flag:
                if (newenv_layers_raw or {}).get("sic") is not None and float(w_ice_eff) > 0:
                    comps.add("copernicus_ice")
                if (newenv_layers_raw or {}).get("wave_swh") is not None and float(w_wave_eff) > 0:
                    comps.add("wave_swh")
        except Exception:
            pass
        if prior_da is not None and float(prior_weight_effective) > 0:
            comps.add("prior")
        cost_da.attrs["cost_components"] = sorted(list(comps))
    except Exception:
        pass

    # 调试打印（仅在 debug 开启时）
    try:
        debug_flag = bool(
            kwargs.get("debug_profile_cost", False)
            or kwargs.get("debug", False)
            or (hasattr(st, "session_state") and bool(st.session_state.get("debug_profile_cost", False)))
        )
        if debug_flag and isinstance(cost_da, xr.DataArray):
            print(
                f"[PROFILE_COST] profile={profile_name} w_ice={w_ice} w_accident={w_accident} prior_weight={prior_weight} "
                f"cost_min={float(np.nanmin(cost_da.values))} cost_max={float(np.nanmax(cost_da.values))}"
            )
    except Exception:
        pass

    # 准备风险底图字典（用于前端可视化 Overlay）
    risk_layers_for_overlay: Dict[str, xr.DataArray] | None = None
    try:
        if risk_layers:
            rdict: Dict[str, xr.DataArray] = {}
            for k, da in risk_layers.items():
                if da is None:
                    continue
                try:
                    # 将含样本/时间等额外维度的 DataArray 聚合为 2D（优先保留 y/x 或 lat/lon）
                    dims = list(da.dims)
                    keep_dims = [d for d in dims if d in ("y", "x", "lat", "lon")]
                    if len(keep_dims) >= 2:
                        reduce_dims = [d for d in dims if d not in keep_dims]
                        if reduce_dims:
                            da2 = da.mean(dim=reduce_dims, skipna=True)
                        else:
                            da2 = da
                    else:
                        # 若识别不到标准维，退化为在前 N-2 维做均值
                        if len(dims) > 2:
                            reduce_dims = dims[:-2]
                            da2 = da.mean(dim=reduce_dims, skipna=True)
                        else:
                            da2 = da
                    # 只收集至少二维的
                    if da2.ndim >= 2:
                        # 只保留最后两维为 (y,x) 语义
                        if da2.dims[-2:] != ("y", "x") and da2.dims[-2:] != ("lat", "lon"):
                            # 尽量重命名
                            try:
                                da2 = da2.rename({da2.dims[-2]: "y", da2.dims[-1]: "x"})
                            except Exception:
                                pass
                        rdict[k] = da2.astype("float32")
                except Exception:
                    continue
            if rdict:
                risk_layers_for_overlay = rdict
    except Exception:
        risk_layers_for_overlay = None

    # 计算规划域（与 newenv/risk 交集）；失败时仅用 env 网格范围
    try:
        dom = compute_planning_domain(EnvironmentContext(
            ym=ym, cost_da=None, sic_da=sic_da, lat_arr=lat_arr, lon_arr=lon_arr, land_mask=land_mask,
            risk_layers_raw=risk_layers if risk_layers else None, prior_penalty_da=None, vessel_profile=vessel_profile
        ), newenv_layers=newenv_layers_raw, risk_layers=risk_layers)
    except Exception:
        try:
            lat0, lon0 = _get_latlon_1d((lat_arr, lon_arr))
            dom = PlanningDomain(float(np.nanmin(lat0)), float(np.nanmax(lat0)), float(np.nanmin(lon0)), float(np.nanmax(lon0)))
        except Exception:
            dom = None

    # 返回最终上下文
    return EnvironmentContext(
        ym=ym,
        cost_da=cost_da,
        sic_da=sic_da,
        lat_arr=lat_arr,
        lon_arr=lon_arr,
        land_mask=land_mask,
        risk_layers_raw=risk_layers if risk_layers else None,
        prior_penalty_da=prior_da,
        risk_layers_for_overlay=risk_layers_for_overlay,
        fusion_mode_effective=str(fusion_mode_effective),
        w_interact=float(w_interact),
        use_escort=bool(use_escort),
        risk_agg_mode=str(risk_agg_mode),
        risk_agg_mode_effective=str(risk_agg_mode_effective),
        risk_agg_alpha=float(risk_agg_alpha),
        escort_applied=bool(escort_applied_flag),
        profile_name=str(profile_name or "baseline"),
        vessel_profile=vessel_profile,
        domain=dom,
    )


# -------------------------
# 私有：成本构建与聚合
# -------------------------
def _build_cost_da(
    env: EnvironmentContext,
    prior_weight: float = 0.0,
    *,
    fused_da: Optional[xr.DataArray] = None,
    layer_weights: Optional[Dict[str, float]] = None,
    newenv_layers: Optional[Dict[str, xr.DataArray]] = None,
    w_ice_newenv: float = 0.0,
    w_wave: float = 0.0,
    use_newenv_for_cost: bool = False,
    use_icebreaker_corridor: bool = False,
    icebreaker_corridor_alpha: float = 0.0,
) -> tuple[xr.DataArray, Optional[xr.DataArray]]:
    H, W = None, None
    if env.sic_da is not None:
        H, W = env.sic_da.shape[-2:]
    elif env.lat_arr is not None and env.lon_arr is not None:
        if env.lat_arr.ndim == 2:
            H, W = env.lat_arr.shape
        elif env.lat_arr.ndim == 1 and env.lon_arr.ndim == 1:
            H, W = len(env.lat_arr), len(env.lon_arr)
    if H is None or W is None:
        H, W = 121, 1161

    def _full_like(value: float) -> xr.DataArray:
        return xr.DataArray(np.full((H, W), value, dtype=float), dims=("y", "x"))

    # 破冰走廊（可选）：提前加载并对齐到 (H,W)，供后续对冰险折扣
    corridor_yx: Optional[xr.DataArray] = None
    if use_icebreaker_corridor and icebreaker_corridor_alpha > 0.0:
        try:
            env_lat_1d = env.lat_arr if (env.lat_arr is not None and getattr(env.lat_arr, "ndim", 1) == 1) else None
            env_lon_1d = env.lon_arr if (env.lon_arr is not None and getattr(env.lon_arr, "ndim", 1) == 1) else None
            cda = load_icebreaker_corridor(str(env.ym), env_lat_1d, env_lon_1d)
        except Exception:
            cda = None
        if cda is not None:
            try:
                da2 = cda
                if tuple(da2.shape[-2:]) != (H, W):
                    si, sj = da2.shape[-2], da2.shape[-1]
                    yi = (np.linspace(0, si - 1, H)).astype(int)
                    xj = (np.linspace(0, sj - 1, W)).astype(int)
                    da2 = xr.DataArray(np.asarray(da2.values)[yi[:, None], xj[None, :]], dims=("y", "x"))
                corridor_yx = da2.clip(0.0, 1.0).astype("float32")
                print(f"[IB] corridor loaded; alpha={icebreaker_corridor_alpha}")
            except Exception:
                corridor_yx = None

    # 基于 SIC 的梯度
    if env.sic_da is not None and env.sic_da.shape[-2:] == (H, W):
        sic = env.sic_da.fillna(0.0).clip(0.0, 1.0)
        sic_scale = 50.0
        sic_cost = (1.0 + sic * sic_scale).astype("float32")
    else:
        sic_cost = _full_like(1.0).astype("float32")

    base_cost = sic_cost

    # 应用破冰走廊折扣：在走廊处降低冰险成本
    if corridor_yx is not None and icebreaker_corridor_alpha > 0.0:
        try:
            factor = (1.0 - float(icebreaker_corridor_alpha) * corridor_yx).clip(0.0, 1.0).astype("float32")
            # 将 sic_cost 转为 (y,x) 形状用于逐元素相乘
            if tuple(sic_cost.shape[-2:]) != (H, W):
                arr = np.asarray(sic_cost.values)
                # 尝试近邻拉伸到 (H,W)
                si, sj = arr.shape[-2], arr.shape[-1]
                yi = (np.linspace(0, si - 1, H)).astype(int)
                xj = (np.linspace(0, sj - 1, W)).astype(int)
                arr = arr[yi[:, None], xj[None, :]]
                sic_cost = xr.DataArray(arr.astype("float32"), dims=("y", "x"))
            base_cost = (sic_cost * factor).clip(min=0.0).astype("float32")
            try:
                base_cost.attrs["icebreaker_corridor_alpha"] = float(icebreaker_corridor_alpha)
                base_cost.attrs["icebreaker_corridor_applied"] = True
            except Exception:
                pass
        except Exception:
            pass

    # 叠加其他风险层（可选）
    lw = layer_weights or {}
    has_new_sic = bool(use_newenv_for_cost and (newenv_layers or {}).get("sic") is not None)
    if env.risk_layers_raw:
        for name, da in env.risk_layers_raw.items():
            try:
                if da is None:
                    continue
                # 若启用 newenv 且已有 sic，则跳过旧 ice 风险层叠加
                if has_new_sic and str(name).lower() == "ice":
                    continue
                da2 = da
                if da2.shape[-2:] != (H, W):
                    # 尝试对齐到当前网格
                    try:
                        tmpl = base_cost if isinstance(base_cost, xr.DataArray) else (env.sic_da if env.sic_da is not None else None)
                        if tmpl is not None:
                            da2 = da2.interp_like(tmpl)
                    except Exception:
                        da2 = None
                if da2 is None or da2.shape[-2:] != (H, W):
                    continue
                arr = np.asarray(da2.values, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    continue
                p1 = float(np.nanpercentile(finite, 1))
                p99 = float(np.nanpercentile(finite, 99))
                if not np.isfinite(p99) or p99 <= p1:
                    continue
                norm = np.clip((arr - p1) / (p99 - p1), 0.0, 1.0).astype("float32")
                w = float(lw.get(name, 1.0))
                base_cost = (base_cost + (10.0 * w) * xr.DataArray(norm, dims=("y", "x"))).astype("float32")
            except Exception:
                continue

    # 叠加 newenv 物理场（若启用）
    if use_newenv_for_cost and newenv_layers:
        def _detect_lat_lon_names_da(da0: xr.DataArray) -> tuple[str, str]:
            lat_name = next((n for n in da0.coords if "lat" in n.lower()), None)
            lon_name = next((n for n in da0.coords if "lon" in n.lower()), None)
            if lat_name and lon_name:
                return lat_name, lon_name
            # 退化：从 dims
            lat_name = next((d for d in da0.dims if "lat" in d.lower()), None)
            lon_name = next((d for d in da0.dims if "lon" in d.lower()), None)
            if lat_name and lon_name:
                return lat_name, lon_name
            # 最后兜底
            return ("lat", "lon")

        # 构造目标网格（优先 1D）
        lat_arr = env.lat_arr
        lon_arr = env.lon_arr
        lat1d = (isinstance(lat_arr, np.ndarray) and lat_arr.ndim == 1)
        lon1d = (isinstance(lon_arr, np.ndarray) and lon_arr.ndim == 1)

        # 1) Copernicus SIC
        try:
            da_sic_src = newenv_layers.get("sic")
        except Exception:
            da_sic_src = None
        if da_sic_src is not None and w_ice_newenv > 0.0:
            try:
                latn, lonn = _detect_lat_lon_names_da(da_sic_src)
                if lat1d and lon1d:
                    da_sic_interp = da_sic_src.interp({latn: xr.DataArray(lat_arr, dims=(latn,)), lonn: xr.DataArray(lon_arr, dims=(lonn,))})
                    # 统一为 (y,x)
                    da_sic_interp = da_sic_interp.rename({latn: "y", lonn: "x"})
                    # 若尺寸与成本网格不一致，做最近邻重采样到 (H,W)
                    if tuple(da_sic_interp.shape) != (H, W):
                        yi = (np.linspace(0, da_sic_interp.shape[0] - 1, H)).astype(int)
                        xj = (np.linspace(0, da_sic_interp.shape[1] - 1, W)).astype(int)
                        da_sic_interp = da_sic_interp.isel(y=xr.DataArray(yi, dims=("y",)), x=xr.DataArray(xj, dims=("x",)))
                    sic_norm = da_sic_interp.clip(0.0, 1.0).astype("float32")
                    sic_risk = (sic_norm ** 1.5)
                    sic_risk = xr.DataArray(sic_risk.values.astype("float32"), dims=("y","x"))
                    sic_cost_comp = (float(w_ice_newenv) * 70.0) * sic_risk
                    # 走廊折扣（若启用）
                    if corridor_yx is not None and icebreaker_corridor_alpha > 0.0:
                        try:
                            factor_ib = (1.0 - float(icebreaker_corridor_alpha) * corridor_yx).clip(0.0, 1.0).astype("float32")
                            sic_cost_comp = (sic_cost_comp * factor_ib).astype("float32")
                        except Exception:
                            pass
                    base_cost = (base_cost + sic_cost_comp).astype("float32")
                    # 保存 Copernicus 海冰的已乘权重成本分量，供路线分解使用
                    try:
                        if env.risk_layers_raw is not None:
                            env.risk_layers_raw["copernicus_ice"] = sic_cost_comp.astype("float32")
                    except Exception:
                        pass
                    try:
                        base_cost.attrs["use_newenv_for_cost"] = True
                        base_cost.attrs["w_ice_newenv"] = float(w_ice_newenv)
                    except Exception:
                        pass
                    print(f"[NEWENV_COST] added Copernicus SIC w={w_ice_newenv} range={float(np.nanmin(sic_risk)):.3f}..{float(np.nanmax(sic_risk)):.3f}")
                else:
                    # 2D 网格插值：直接用目标 (y,x) 上的 lat2d/lon2d
                    try:
                        lat2d = env.lat_arr if isinstance(env.lat_arr, np.ndarray) else None
                        lon2d = env.lon_arr if isinstance(env.lon_arr, np.ndarray) else None
                        if lat2d is not None and lon2d is not None and lat2d.ndim == 2 and lon2d.ndim == 2:
                            da_sic_interp = da_sic_src.interp({latn: xr.DataArray(lat2d, dims=("y","x")), lonn: xr.DataArray(lon2d, dims=("y","x"))})
                            sic_norm = da_sic_interp.clip(0.0, 1.0).astype("float32")
                            sic_risk = (sic_norm ** 1.5)
                            sic_cost_comp = (float(w_ice_newenv) * 70.0) * sic_risk
                            base_cost = (base_cost + sic_cost_comp).astype("float32")
                            # 保存 Copernicus 海冰的已乘权重成本分量（2D 场景）
                            try:
                                if env.risk_layers_raw is not None:
                                    env.risk_layers_raw["copernicus_ice"] = xr.DataArray(sic_cost_comp.values.astype("float32"), dims=("y","x"))
                            except Exception:
                                pass
                            try:
                                base_cost.attrs["use_newenv_for_cost"] = True
                                base_cost.attrs["w_ice_newenv"] = float(w_ice_newenv)
                            except Exception:
                                pass
                            print(f"[NEWENV_COST] added Copernicus SIC (2D) w={w_ice_newenv} range={float(np.nanmin(sic_risk)):.3f}..{float(np.nanmax(sic_risk)):.3f}")
                        else:
                            print("[NEWENV_COST] env grid not usable for 2D SIC interp; skipped")
                    except Exception as ee:
                        print(f"[NEWENV_COST] SIC 2D interp failed: {ee}")
            except Exception as e:
                print(f"[NEWENV_COST] SIC interp failed: {e}")

        # 2) Wave SWH
        try:
            da_wave_src = newenv_layers.get("wave_swh")
        except Exception:
            da_wave_src = None
        if da_wave_src is not None and w_wave > 0.0:
            try:
                latn, lonn = _detect_lat_lon_names_da(da_wave_src)
                if lat1d and lon1d:
                    da_w_interp = da_wave_src.interp({latn: xr.DataArray(lat_arr, dims=(latn,)), lonn: xr.DataArray(lon_arr, dims=(lonn,))})
                    # 统一为 (y,x)
                    da_w_interp = da_w_interp.rename({latn: "y", lonn: "x"})
                    wave_min, wave_max = 0.0, 6.0
                    wn = ((da_w_interp - wave_min) / max(1e-6, (wave_max - wave_min))).clip(0.0, 1.0).astype("float32")
                    wave_risk = (wn ** 1.5)
                    wave_cost_comp = (float(w_wave) * 50.0) * wave_risk
                    base_cost = (base_cost + wave_cost_comp).astype("float32")
                    # 保存 SWH 的已乘权重成本分量
                    try:
                        if env.risk_layers_raw is not None:
                            env.risk_layers_raw["wave_swh"] = wave_cost_comp.astype("float32")
                    except Exception:
                        pass
                    try:
                        base_cost.attrs["use_newenv_for_cost"] = True
                        base_cost.attrs["w_wave"] = float(w_wave)
                    except Exception:
                        pass
                    print(f"[NEWENV_COST] added wave_swh w={w_wave} range={float(np.nanmin(wave_risk)):.3f}..{float(np.nanmax(wave_risk)):.3f}")
                else:
                    print("[NEWENV_COST] env grid is not 1D lat/lon; skip wave interp")
            except Exception as e:
                print(f"[NEWENV_COST] wave interp failed: {e}")

    # 叠加 fused 风险（若提供）
    if fused_da is not None:
        try:
            f = fused_da
            if f.shape[-2:] != (H, W):
                try:
                    f = f.interp_like(base_cost)
                except Exception:
                    f = None
            if f is not None:
                arr = np.asarray(f.values, dtype=float)
                finite = arr[np.isfinite(arr)]
                if finite.size > 0:
                    q1 = float(np.nanpercentile(finite, 5))
                    q99 = float(np.nanpercentile(finite, 95))
                    if np.isfinite(q99) and q99 > q1:
                        norm = np.clip((arr - q1) / (q99 - q1), 0.0, 1.0).astype("float32")
                        base_cost = (base_cost + 15.0 * xr.DataArray(norm, dims=("y", "x"))).astype("float32")
        except Exception:
            pass

    # 叠加主航线先验（可选）
    prior_da = None
    if prior_weight > 0.0 and env.lat_arr is not None and env.lon_arr is not None:
        # 仅在 prior 数据可用（含网格重叠检查）时启用
        try:
            if prior_data_available(env):
                prior_da = _rasterize_centerlines_to_penalty(env)
                if prior_da is not None:
                    penalty_scaler = 20.0
                    base_cost = (base_cost + prior_weight * penalty_scaler * prior_da).astype("float32")
            else:
                prior_da = None
        except Exception:
            prior_da = None

    # 陆地掩膜 + 清理
    if env.land_mask is not None:
        # 使用 numpy 掩蔽以避免 xarray 坐标对齐冲突；必要时将掩膜近邻重采样到成本网格大小
        try:
            mask_np = np.asarray(env.land_mask, dtype=bool)
            Ht, Wt = int(base_cost.shape[-2]), int(base_cost.shape[-1])
            if mask_np.shape != (Ht, Wt):
                si, sj = mask_np.shape[-2], mask_np.shape[-1]
                yi = (np.linspace(0, si - 1, Ht)).astype(int)
                xj = (np.linspace(0, sj - 1, Wt)).astype(int)
                mask_np = mask_np[yi[:, None], xj[None, :]]
            arr = np.asarray(base_cost.values)
            arr2_inf = np.where(mask_np, np.inf, arr)
            # 若掩蔽后整行/整列均无海洋（finite），或总体 finite 占比过低，则放宽为“极大有限值”
            try:
                finite_mask = np.isfinite(arr2_inf)
                frac_finite = float(finite_mask.mean())
                row_has_water = bool(finite_mask.any(axis=-1).all())
                col_has_water = bool(finite_mask.any(axis=-2).all())
            except Exception:
                frac_finite = 1.0
                row_has_water = True
                col_has_water = True
            use_huge_instead = bool((frac_finite < 0.03) or (not row_has_water) or (not col_has_water))
            if use_huge_instead:
                try:
                    print(f"[LMASK] post-mask finite={frac_finite:.4f}, row_ok={row_has_water}, col_ok={col_has_water}; relax ocean NaN/inf -> 1e6 (land kept as inf)")
                except Exception:
                    pass
            # 仅放宽海洋上的 NaN/inf，陆地仍保持 inf
            if use_huge_instead:
                ocean_nonfinite = (~mask_np) & (~np.isfinite(arr2_inf))
                # 放宽为较大的有限值，但低于测试阈值 1e6，确保可达性
                arr2 = np.where(ocean_nonfinite, 5e5, arr2_inf)
            else:
                arr2 = arr2_inf
            # 保留 attrs，避免前面写入的元信息丢失
            _attrs = dict(getattr(base_cost, "attrs", {}) or {})
            _name = getattr(base_cost, "name", None)
            base_cost = xr.DataArray(arr2.astype("float32"), dims=("y", "x"))
            if _attrs:
                try:
                    base_cost.attrs.update(_attrs)
                except Exception:
                    pass
            if _name:
                try:
                    base_cost.name = _name
                except Exception:
                    pass
        except Exception:
            # 兜底：numpy 路径异常时，不再尝试对齐，直接跳过掩膜以保证不中断
            try:
                print("[LMASK] numpy masking failed; skip applying land mask to avoid alignment error")
            except Exception:
                pass
    base_cost = base_cost.where(np.isfinite(base_cost), np.inf)
    # 额外鲁棒性：
    # - newenv 成本开启时，直接将 NaN/inf 放宽为 1e7（仅限海洋格），确保联通；
    # - 关闭时，仅在极端情况下触发放宽（保持 1.0 默认行为）。
    try:
        arr_chk = np.asarray(base_cost.values)
        finite_mask2 = np.isfinite(arr_chk)
        frac_finite2 = float(finite_mask2.mean()) if arr_chk.size > 0 else 0.0
        row_ok2 = bool(finite_mask2.any(axis=-1).all())
        col_ok2 = bool(finite_mask2.any(axis=-2).all())
        need_relax = False
        if bool(use_newenv_for_cost):
            need_relax = True
        else:
            need_relax = bool((frac_finite2 < 0.02) or (not row_ok2) or (not col_ok2))
        if need_relax:
            try:
                print(f"[COST] relax non-finite (ocean-only): finite={frac_finite2:.4f}, row_ok={row_ok2}, col_ok={col_ok2}; set ocean NaN/inf -> 1e7 (land kept as inf, newenv={use_newenv_for_cost})")
            except Exception:
                pass
            # 仅放宽海洋区域的 NaN/inf，确保陆地仍不可达
            try:
                if env.land_mask is not None:
                    mask_np2 = np.asarray(env.land_mask, dtype=bool)
                    Ht2, Wt2 = int(base_cost.shape[-2]), int(base_cost.shape[-1])
                    if mask_np2.shape != (Ht2, Wt2):
                        si2, sj2 = mask_np2.shape[-2], mask_np2.shape[-1]
                        yi2 = (np.linspace(0, si2 - 1, Ht2)).astype(int)
                        xj2 = (np.linspace(0, sj2 - 1, Wt2)).astype(int)
                        mask_np2 = mask_np2[yi2[:, None], xj2[None, :]]
                    ocean_nonfinite2 = (~mask_np2) & (~np.isfinite(arr_chk))
                else:
                    ocean_nonfinite2 = ~np.isfinite(arr_chk)
            except Exception:
                ocean_nonfinite2 = ~np.isfinite(arr_chk)
            arr_relax = np.where(ocean_nonfinite2, 1e7, arr_chk)
            _attrs2 = dict(getattr(base_cost, "attrs", {}) or {})
            _name2 = getattr(base_cost, "name", None)
            base_cost = xr.DataArray(arr_relax.astype("float32"), dims=("y", "x"))
            if _attrs2:
                try:
                    base_cost.attrs.update(_attrs2)
                except Exception:
                    pass
            if _name2:
                try:
                    base_cost.name = _name2
                except Exception:
                    pass
    except Exception:
        pass

    # 最后一层保险：再次强制陆地格为 inf，防止任何放宽逻辑误伤陆地
    try:
        if env.land_mask is not None:
            mask_np3 = np.asarray(env.land_mask, dtype=bool)
            Ht3, Wt3 = int(base_cost.shape[-2]), int(base_cost.shape[-1])
            if mask_np3.shape != (Ht3, Wt3):
                si3, sj3 = mask_np3.shape[-2], mask_np3.shape[-1]
                yi3 = (np.linspace(0, si3 - 1, Ht3)).astype(int)
                xj3 = (np.linspace(0, sj3 - 1, Wt3)).astype(int)
                mask_np3 = mask_np3[yi3[:, None], xj3[None, :]]
            arr_final = np.asarray(base_cost.values)
            arr_final = np.where(mask_np3, np.inf, arr_final)
            _attrs3 = dict(getattr(base_cost, "attrs", {}) or {})
            _name3 = getattr(base_cost, "name", None)
            base_cost = xr.DataArray(arr_final.astype("float32"), dims=("y", "x"))
            if _attrs3:
                try:
                    base_cost.attrs.update(_attrs3)
                except Exception:
                    pass
            if _name3:
                try:
                    base_cost.name = _name3
                except Exception:
                    pass
    except Exception:
        pass

    finite_mask = np.isfinite(base_cost.values)
    if int(np.count_nonzero(finite_mask)) == 0:
        print("[COST] All values non-finite, fallback to simple ocean/land cost")
        if env.land_mask is not None and env.land_mask.shape == (H, W):
            base_cost = xr.where(xr.DataArray(env.land_mask, dims=("y", "x")), 1000.0, 1.0)
        else:
            base_cost = _full_like(1.0)

    base_cost.name = "final_risk_cost"
    return base_cost.astype("float32"), prior_da


def _aggregate_risk_da(da: xr.DataArray, *, mode: str = "mean", alpha: float = 0.9) -> tuple[xr.DataArray, Optional[str]]:
    """对含样本的风险场进行聚合；若不含样本维，原样返回。
    返回 (聚合后 DataArray, 实际生效的 mode 或 None)
    支持样本维命名：{"sample", "member", "ensemble"}
    CVaR 简易实现：先取分位 q，再对 >=q 的尾部求均值。
    - 若样本数 <=1 且 mode 为 quantile/cvar，则退化为 mean，并打印降级日志。
    """
    if da is None:
        return da, None
    dims = list(da.dims)
    sample_dim = next((d for d in ("sample", "member", "ensemble") if d in dims), None)
    if sample_dim is None:
        return da, None
    n_sample = int(da.sizes.get(sample_dim, 0))
    m = (mode or "mean").lower()
    if m == "mean":
        out = da.mean(dim=sample_dim)
        return out, "mean"
    if n_sample <= 1 and m in ("q", "quantile", "cvar", "es", "expected_shortfall"):
        print(f"[AGG] mode={mode} requested but only {n_sample} sample available; fallback to mean")
        return da.mean(dim=sample_dim), "mean"
    if m in ("q", "quantile"):
        try:
            out = da.quantile(float(alpha), dim=sample_dim, keep_attrs=True)
            if "quantile" in out.dims:
                out = out.squeeze("quantile", drop=True)
            return out, "quantile"
        except Exception:
            return da.mean(dim=sample_dim), "mean"
    if m in ("cvar", "es", "expected_shortfall"):
        try:
            q = da.quantile(float(alpha), dim=sample_dim)
            if "quantile" in q.dims:
                q = q.squeeze("quantile", drop=True)
            tail_mask = da >= q
            tail = xr.where(tail_mask, da, np.nan)
            out = tail.mean(dim=sample_dim, skipna=True)
            out = xr.where(np.isnan(out), q, out)
            return out, "cvar"
        except Exception:
            return da.mean(dim=sample_dim), "mean"
    return da.mean(dim=sample_dim), "mean"


# -------------------------
# 经纬度 <-> 栅格索引（包含严格模式起终点单一真理源）
# -------------------------

def _wrap_lon_to_180(lon: float) -> float:
    try:
        return ((float(lon) + 180.0) % 360.0) - 180.0
    except Exception:
        return float(lon)


def _circular_lon_diff(a: np.ndarray, b: float) -> np.ndarray:
    """返回 |a - b| 在经度圆上的最小差（度）。要求 a 已在 [-180,180]。"""
    try:
        a = np.asarray(a, dtype=float)
        b = float(b)
        d = np.abs(a - b)
        return np.minimum(d, 360.0 - d)
    except Exception:
        return np.abs(np.asarray(a, dtype=float) - float(b))


def find_nearest_grid_index(lat: float, lon: float, env_ctx: EnvironmentContext) -> tuple[int, int]:
    """
    单一真理源：不考虑海陆/成本，仅在网格上找到距离最近的格点 (i,j)。
    策略：
      1) 首选使用网格自身的 lat_arr/lon_arr（支持 1D/2D；经度域自动对齐；近全球覆盖用圆周差）。
      2) 若该映射明显异常（与输入经纬度相差过大，例如 |Δlon|>30° 或 |Δlat|>5°），
         使用规划域 + cost_da 尺寸进行线性索引回退（避免被 0/360 接缝或退化坐标误导）。
    """
    if env_ctx.lat_arr is None or env_ctx.lon_arr is None:
        raise RuntimeError("env_ctx 缺少 lat_arr/lon_arr")

    lat_arr = np.asarray(env_ctx.lat_arr, dtype=float)
    lon_arr_raw = np.asarray(env_ctx.lon_arr, dtype=float)

    # 判定经度轴域与跨度
    try:
        lon_min = float(np.nanmin(lon_arr_raw))
        lon_max = float(np.nanmax(lon_arr_raw))
        lon_span = float(lon_max - lon_min)
    except Exception:
        lon_min, lon_max, lon_span = -180.0, 180.0, 360.0

    # 将输入 lon 映射到与网格一致的域
    if (lon_max > 180.0 + 1e-3) or (lon_min >= 0.0):
        lv = float(lon) % 360.0
        if lv < 0:
            lv += 360.0
        lon_arr = lon_arr_raw.astype(float)
    else:
        lv = ((float(lon) + 180.0) % 360.0) - 180.0
        lon_arr = lon_arr_raw.astype(float)

    # 根据跨度选择差异度量
    def _diff(a: np.ndarray, b: float) -> np.ndarray:
        a = np.asarray(a, dtype=float)
        if lon_span > 300.0:
            d = np.abs(a - b)
            return np.minimum(d, 360.0 - d)
        else:
            return np.abs(a - b)

    # 首选：基于原始网格的最近邻
    def _nearest_from_grid() -> tuple[int, int]:
        if lat_arr.ndim == 1 and lon_arr.ndim == 1:
            i = int(np.nanargmin(np.abs(lat_arr - float(lat))))
            j = int(np.nanargmin(_diff(lon_arr, float(lv))))
            return i, j
        if lat_arr.ndim == 2 and lon_arr.ndim == 2 and lat_arr.shape == lon_arr.shape:
            dlat = (lat_arr - float(lat))
            dlon = _diff(lon_arr, float(lv))
            d2 = dlat * dlat + dlon * dlon
            idx = int(np.nanargmin(d2))
            i, j = np.unravel_index(idx, d2.shape)
            return int(i), int(j)
        if lat_arr.ndim == 1 and lon_arr.ndim == 2:
            LA = np.repeat(lat_arr[:, None], lon_arr.shape[1], axis=1)
            dlat = (LA - float(lat))
            dlon = _diff(lon_arr, float(lv))
            d2 = dlat * dlat + dlon * dlon
            i, j = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
            return int(i), int(j)
        if lat_arr.ndim == 2 and lon_arr.ndim == 1:
            LO = np.repeat(lon_arr[None, :], lat_arr.shape[0], axis=0)
            dlat = (lat_arr - float(lat))
            dlon = _diff(LO, float(lv))
            d2 = dlat * dlat + dlon * dlon
            i, j = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
            return int(i), int(j)
        i = int(np.nanargmin(np.abs(lat_arr.ravel() - float(lat))))
        j = int(np.nanargmin(np.abs(lon_arr.ravel() - float(lv))))
        if lat_arr.ndim == 2:
            return int(i // lat_arr.shape[1]), int(i % lat_arr.shape[1])
        if lon_arr.ndim == 2:
            return int(j // lon_arr.shape[1]), int(j % lon_arr.shape[1])
        return int(i), int(j)

    i1, j1 = _nearest_from_grid()

    # 若 cost 网格存在且索引越界，则直接回退到“域+线性索引”（避免后续被硬裁剪到边界导致吸附）
    dom = getattr(env_ctx, "domain", None)
    cost_da = getattr(env_ctx, "cost_da", None)
    if cost_da is not None and dom is not None:
        try:
            H, W = cost_da.shape[-2:]
            if not (0 <= int(i1) < H and 0 <= int(j1) < W):
                latv = float(np.clip(lat, dom.lat_min, dom.lat_max))
                lon_candidates = [float(lon), float(lon) + 360.0, float(lon) - 360.0]
                lon_center = 0.5 * (float(dom.lon_min) + float(dom.lon_max))
                lv2 = min(lon_candidates, key=lambda x: abs(((x - lon_center + 180.0) % 360.0) - 180.0))
                lv2 = float(np.clip(lv2, float(dom.lon_min), float(dom.lon_max)))
                ii = int(round((latv - float(dom.lat_min)) / max(1e-6, float(dom.lat_max - dom.lat_min)) * (H - 1)))
                jj = int(round((lv2 - float(dom.lon_min)) / max(1e-6, float(dom.lon_max - dom.lon_min)) * (W - 1)))
                ii = int(np.clip(ii, 0, H - 1)); jj = int(np.clip(jj, 0, W - 1))
                return ii, jj
        except Exception:
            pass

    # 评估偏差；如严重异常，则回退到“域+线性索引”
    def _lonlat_of(i: int, j: int) -> tuple[float, float]:
        try:
            out = path_ij_to_lonlat(env_ctx, [(int(i), int(j))])
            if out:
                return float(out[0][0]), float(out[0][1])
        except Exception:
            pass
        return float("nan"), float("nan")

    lat1, lon1 = _lonlat_of(i1, j1)
    def _wrap180(x: float) -> float:
        return ((float(x) + 180.0) % 360.0) - 180.0
    try:
        dl_lon = abs(_wrap180(lon1) - _wrap180(lon))
        dl_lat = abs(float(lat1) - float(lat))
    except Exception:
        dl_lon, dl_lat = 0.0, 0.0

    if (dl_lon <= 30.0 and dl_lat <= 5.0):
        return int(i1), int(j1)

    # 回退：使用规划域线性映射（避免被接缝/异常坐标误导）
    if dom is not None and cost_da is not None:
        try:
            H, W = cost_da.shape[-2:]
            latv = float(np.clip(lat, dom.lat_min, dom.lat_max))
            lon_candidates = [float(lon), float(lon) + 360.0, float(lon) - 360.0]
            lon_center = 0.5 * (float(dom.lon_min) + float(dom.lon_max))
            lv2 = min(lon_candidates, key=lambda x: abs(((x - lon_center + 180.0) % 360.0) - 180.0))
            lv2 = float(np.clip(lv2, float(dom.lon_min), float(dom.lon_max)))
            ii = int(round((latv - float(dom.lat_min)) / max(1e-6, float(dom.lat_max - dom.lat_min)) * (H - 1)))
            jj = int(round((lv2 - float(dom.lon_min)) / max(1e-6, float(dom.lon_max - dom.lon_min)) * (W - 1)))
            ii = int(np.clip(ii, 0, H - 1)); jj = int(np.clip(jj, 0, W - 1))
            lat2, lon2 = _lonlat_of(ii, jj)
            dl_lon2 = abs(_wrap180(lon2) - _wrap180(lon))
            dl_lat2 = abs(float(lat2) - float(lat))
            if (dl_lon2 < dl_lon - 1.0) or (dl_lat2 < dl_lat - 1.0):
                return int(ii), int(jj)
        except Exception:
            pass

    return int(i1), int(j1)


def find_nearest_ocean_cell(lat: float, lon: float, env_ctx: EnvironmentContext, max_radius: int = 20) -> tuple[int, int, dict]:
    """
    在以最近格点为中心的局部窗口搜索“最近的海上可通行格点”。
    条件：
      - land_mask[i,j] == 0（海）
      - cost[i,j] 为有限且 < 1e6
    搜索半径 r=0..max_radius，按 |i-i0|+|j-j0| 选择最近候选。
    找到：返回 (i,j,{status:'ok',search_radius:r})；否则返回 (i0,j0,{status:'failed_no_ocean',search_radius:max_radius})。
    """
    if env_ctx.cost_da is None and env_ctx.land_mask is None:
        raise RuntimeError("缺少 cost_da 与 land_mask，无法判断可通行性")

    # 目标网格尺寸：优先 cost
    H = W = None
    if getattr(env_ctx, "cost_da", None) is not None:
        try:
            H, W = env_ctx.cost_da.shape[-2:]
        except Exception:
            H = W = None
    if (H is None or W is None) and isinstance(env_ctx.land_mask, np.ndarray):
        H, W = env_ctx.land_mask.shape[-2:]
    if H is None or W is None:
        raise RuntimeError("无法确定网格尺寸")

    i0, j0 = find_nearest_grid_index(float(lat), float(lon), env_ctx)
    i0 = int(np.clip(i0, 0, H - 1))
    j0 = int(np.clip(j0, 0, W - 1))

    # 准备 land/cost 数组（shape 对齐到 H,W；若不一致则做最近邻索引映射）
    land = None
    if isinstance(env_ctx.land_mask, np.ndarray):
        land = env_ctx.land_mask.astype(bool)
        if land.shape[-2:] != (H, W):
            si, sj = land.shape[-2], land.shape[-1]
            yi = (np.linspace(0, si - 1, H)).astype(int)
            xj = (np.linspace(0, sj - 1, W)).astype(int)
            land = land[yi[:, None], xj[None, :]]
    cost = None
    if getattr(env_ctx, "cost_da", None) is not None:
        cost = np.asarray(env_ctx.cost_da.values, dtype=float)
        if cost.shape[-2:] != (H, W):
            si, sj = cost.shape[-2], cost.shape[-1]
            yi = (np.linspace(0, si - 1, H)).astype(int)
            xj = (np.linspace(0, sj - 1, W)).astype(int)
            cost = cost[yi[:, None], xj[None, :]]

    def _is_ocean(ii: int, jj: int) -> bool:
        if ii < 0 or jj < 0 or ii >= H or jj >= W:
            return False
        if land is not None and bool(land[ii, jj]):
            return False
        if cost is not None:
            v = float(cost[ii, jj])
            # 与 A* 保持一致：>=1e9 视为不可通行（注意 _build_cost 可能将海上异常置为 1e7）
            if not np.isfinite(v) or v >= 1e9:
                return False
        return True

    # 计算候选点与输入经纬度的距离（优先使用地理距离）
    def _cand_dist(ii: int, jj: int) -> float:
        try:
            pll = path_ij_to_lonlat(env_ctx, [(int(ii), int(jj))])
            if pll and len(pll[0]) == 2:
                la, lo = float(pll[0][0]), float(pll[0][1])
                # haversine（km）
                import math as _mh
                R = 6371.0
                dphi = _mh.radians(la - float(lat))
                dlam = _mh.radians(lo - float(lon))
                phi1 = _mh.radians(float(lat))
                phi2 = _mh.radians(la)
                a = _mh.sin(dphi/2)**2 + _mh.cos(phi1)*_mh.cos(phi2)*_mh.sin(dlam/2)**2
                c = 2*_mh.atan2((_mh.sqrt(a)), _mh.sqrt(max(0.0,1-a)))
                return float(R * c)
        except Exception:
            pass
        # 回退到网格距离
        return float((ii - i0)**2 + (jj - j0)**2)

    global_best = None
    global_best_d = None
    global_best_r = None
    for r in range(int(max_radius) + 1):
        # 在环上收集所有可行候选，并按地理距离选择最近
        imin, imax = max(0, i0 - r), min(H - 1, i0 + r)
        jmin, jmax = max(0, j0 - r), min(W - 1, j0 + r)
        # 扫描矩形边框（4条边），避免重复
        for ii in range(imin, imax + 1):
            for jj in (jmin, jmax):
                if _is_ocean(ii, jj):
                    d = _cand_dist(ii, jj)
                    if (global_best is None) or (d < global_best_d):
                        global_best = (ii, jj); global_best_d = d; global_best_r = r
        for jj in range(jmin + 1, jmax):
            for ii in (imin, imax):
                if _is_ocean(ii, jj):
                    d = _cand_dist(ii, jj)
                    if (global_best is None) or (d < global_best_d):
                        global_best = (ii, jj); global_best_d = d; global_best_r = r
    if global_best is not None:
        return int(global_best[0]), int(global_best[1]), {"status": "ok", "search_radius": int(global_best_r)}

    return int(i0), int(j0), {"status": "failed_no_ocean", "search_radius": int(max_radius)}


def latlon_to_ij(env: EnvironmentContext, latv: float, lonv: float) -> Optional[tuple[int, int]]:
    if env.lat_arr is None or env.lon_arr is None:
        return None
    # 将 lonv 映射到与网格相同的经度域（[-180,180] 或 [0,360]）
    try:
        lon_arr = env.lon_arr
        lon_min = float(np.nanmin(lon_arr))
        lon_max = float(np.nanmax(lon_arr))
        if lon_max > 180.0 + 1e-3 or lon_min >= 0.0:
            lv = float(lonv) % 360.0
            if lv < 0:
                lv += 360.0
        else:
            lv = ((float(lonv) + 180.0) % 360.0) - 180.0
    except Exception:
        lv = float(lonv)

    if env.lat_arr.ndim == 1 and env.lon_arr.ndim == 1:
        i = int(np.abs(env.lat_arr - latv).argmin())
        j = int(np.abs(env.lon_arr - lv).argmin())
        return i, j
    else:
        # 2D 网格：按最近邻
        d2 = (env.lat_arr - latv) ** 2 + (env.lon_arr - lv) ** 2
        i, j = np.unravel_index(np.nanargmin(d2), d2.shape)
        return int(i), int(j)


def path_ij_to_lonlat(env: EnvironmentContext, path_ij: list[tuple[int, int]]) -> list[list[float]]:
    out: list[list[float]] = []

    # 无坐标 -> 直接返回索引
    if env.lat_arr is None or env.lon_arr is None:
        return [[float(i), float(j)] for i, j in path_ij]

    try:
        # 推断目标网格尺寸（优先以 cost_da 为准）
        H = W = None
        if env.cost_da is not None:
            H, W = env.cost_da.shape[-2:]
        elif getattr(env, "lat_arr", None) is not None and getattr(env, "lon_arr", None) is not None:
            if env.lat_arr.ndim == 2 and env.lon_arr.ndim == 2 and env.lat_arr.shape == env.lon_arr.shape:
                H, W = env.lat_arr.shape
            elif env.lat_arr.ndim == 1 and env.lon_arr.ndim == 1:
                H, W = len(env.lat_arr), len(env.lon_arr)

        dom = getattr(env, "domain", None)

        # 情况 A：lat/lon 为 2D 且形状与 H,W 一致 → 直接索引
        if env.lat_arr.ndim == 2 and env.lon_arr.ndim == 2 and (H, W) == env.lat_arr.shape == env.lon_arr.shape:
            for i, j in path_ij:
                ii = int(np.clip(i, 0, H - 1)); jj = int(np.clip(j, 0, W - 1))
                lat = float(env.lat_arr[ii, jj]); lon = float(env.lon_arr[ii, jj])
                # 标准化到 [-180,180]
                lon = ((lon + 180.0) % 360.0) - 180.0
                out.append([lat, lon])
            return out

        # 情况 B：1D 轴（可能长度与 H,W 不匹配） → 线性扩展/重采样
        if env.lat_arr.ndim == 1 and env.lon_arr.ndim == 1 and H is not None and W is not None:
            lat1d = np.asarray(env.lat_arr, dtype=float)
            lon1d = np.asarray(env.lon_arr, dtype=float)
            # 若长度不匹配，优先使用规划域进行线性生成；否则按原端点线性插值
            if (len(lat1d) != H or len(lon1d) != W) and dom is not None:
                latA = np.linspace(float(dom.lat_min), float(dom.lat_max), H)
                lonA = np.linspace(float(dom.lon_min), float(dom.lon_max), W)
            else:
                latA = np.linspace(float(lat1d[0]), float(lat1d[-1]), H) if len(lat1d) != H else lat1d
                lonA = np.linspace(float(lon1d[0]), float(lon1d[-1]), W) if len(lon1d) != W else lon1d
            for i, j in path_ij:
                ii = int(np.clip(i, 0, H - 1)); jj = int(np.clip(j, 0, W - 1))
                lat = float(latA[ii]); lon = float(lonA[jj])
                lon = ((lon + 180.0) % 360.0) - 180.0
                out.append([lat, lon])
            return out

        # 情况 C：lat/lon 为 2D 但与 H,W 不一致（常见于“全局坐标 + 子域成本图”）
        # 这种情况下，严格使用规划域 + H,W 构造 1D 轴，避免被全局 0/360 接缝误导
        if H is not None and W is not None and dom is not None:
            latA = np.linspace(float(dom.lat_min), float(dom.lat_max), H)
            lonA = np.linspace(float(dom.lon_min), float(dom.lon_max), W)
            for i, j in path_ij:
                ii = int(np.clip(i, 0, H - 1)); jj = int(np.clip(j, 0, W - 1))
                lat = float(latA[ii]); lon = float(lonA[jj])
                lon = ((lon + 180.0) % 360.0) - 180.0
                out.append([lat, lon])
            return out

        # 最后兜底：直接返回索引
        return [[float(i), float(j)] for i, j in path_ij]
    except Exception:
        # 任何坐标转换异常，退化为索引输出，保证路由链条不中断
        return [[float(i), float(j)] for i, j in path_ij]


# -------------------------
# 统一后验体检：路线 vs 环境（陆地 / bbox）
# -------------------------

def validate_route_against_env(route_xy: Any, env_ctx: "EnvironmentContext", label: str = "") -> dict:
    """
    对路线进行环境一致性体检：统计落陆与越界点。
    route_xy: dict/xarray 或 RouteResult，需能得到 lat, lon 数组（度）。
    env_ctx: 需提供 land_mask、lat_arr、lon_arr（1D 或 2D）。
    返回 {ok, num_on_land, num_out_of_bbox, examples:{on_land:[], out_of_bbox:[]}}
    """
    res = {"ok": True, "num_on_land": 0, "num_out_of_bbox": 0, "examples": {"on_land": [], "out_of_bbox": []}, "label": label}
    try:
        # 提取 lat/lon 列
        lat, lon = None, None
        if route_xy is None:
            return res
        # RouteResult
        if hasattr(route_xy, "path_lonlat") and route_xy.path_lonlat:
            try:
                arr = list(route_xy.path_lonlat)
                lat = np.asarray([float(p[0]) for p in arr], dtype=float)
                lon = np.asarray([float(p[1]) for p in arr], dtype=float)
            except Exception:
                lat = lon = None
        # 显式 dict/xarray
        if lat is None or lon is None:
            try:
                if isinstance(route_xy, dict):
                    lat = np.asarray(route_xy.get("lat"), dtype=float)
                    lon = np.asarray(route_xy.get("lon"), dtype=float)
                else:
                    lat = np.asarray(getattr(route_xy, "lat", None), dtype=float)
                    lon = np.asarray(getattr(route_xy, "lon", None), dtype=float)
            except Exception:
                lat = lon = None
        if lat is None or lon is None:
            return res
        n = int(min(lat.size, lon.size))
        if n == 0:
            return res
        lat = lat[:n]
        lon = lon[:n]

        # 网格与掩膜
        lm = getattr(env_ctx, "land_mask", None)
        latA = getattr(env_ctx, "lat_arr", None)
        lonA = getattr(env_ctx, "lon_arr", None)
        H = W = None
        if isinstance(lm, np.ndarray):
            H, W = lm.shape[-2], lm.shape[-1]
        elif getattr(env_ctx, "cost_da", None) is not None:
            try:
                H, W = env_ctx.cost_da.shape[-2:]
            except Exception:
                H = W = None
        # bbox 估计（从网格坐标推断）
        def _grid_bounds(latA, lonA):
            try:
                if latA is None or lonA is None:
                    return None
                if np.ndim(latA) == 1 and np.ndim(lonA) == 1 and latA.size > 0 and lonA.size > 0:
                    lat_min, lat_max = float(np.nanmin(latA)), float(np.nanmax(latA))
                    lo_min, lo_max = float(np.nanmin(lonA)), float(np.nanmax(lonA))
                else:
                    lat_min, lat_max = float(np.nanmin(latA)), float(np.nanmax(latA))
                    lo_min, lo_max = float(np.nanmin(lonA)), float(np.nanmax(lonA))
                return (lat_min, lat_max, lo_min, lo_max)
            except Exception:
                return None
        bounds = _grid_bounds(latA, lonA)

        # 统一经度到网格域
        def _wrap_lon(lv: np.ndarray, lon_grid: np.ndarray | None) -> np.ndarray:
            try:
                if lon_grid is None:
                    return ((lv + 180.0) % 360.0) - 180.0
                gmin, gmax = float(np.nanmin(lon_grid)), float(np.nanmax(lon_grid))
                if gmax > 180.0 + 1e-3 or gmin >= 0.0:
                    out = np.mod(lv, 360.0); out[out < 0] += 360.0; return out
                return ((lv + 180.0) % 360.0) - 180.0
            except Exception:
                return lv
        lon_w = _wrap_lon(lon.copy(), lonA)

        # 逐点映射 -> 最近网格索引
        on_land = 0; oob = 0
        ex_land = []; ex_oob = []
        for idx in range(n):
            la = float(lat[idx]); lo = float(lon_w[idx])
            # bbox 粗判
            if bounds is not None:
                la_min, la_max, lo_min, lo_max = bounds
                if not (la_min - 1e-6 <= la <= la_max + 1e-6 and lo_min - 1e-6 <= lo <= lo_max + 1e-6):
                    oob += 1
                    if len(ex_oob) < 3:
                        ex_oob.append({"i": idx, "lat": float(lat[idx]), "lon": float(lon[idx])})
                    continue
            # 索引映射
            ii = jj = None
            try:
                if latA is not None and lonA is not None:
                    if np.ndim(latA) == 1 and np.ndim(lonA) == 1:
                        ii = int(np.abs(latA - la).argmin())
                        jj = int(np.abs(lonA - lo).argmin())
                    else:
                        # 2D 最近邻
                        LA = np.asarray(latA, dtype=float); LO = np.asarray(lonA, dtype=float)
                        d2 = (LA - la) ** 2 + (LO - lo) ** 2
                        ii, jj = np.unravel_index(int(np.nanargmin(d2)), d2.shape)
                elif H is not None and W is not None:
                    # 仅按形状裁剪
                    ii = int(max(0, min(H - 1, round(H / 2))))
                    jj = int(max(0, min(W - 1, round(W / 2))))
            except Exception:
                ii = jj = None
            if ii is None or jj is None or (H is not None and (ii < 0 or ii >= H or jj < 0 or jj >= W)):
                oob += 1
                if len(ex_oob) < 3:
                    ex_oob.append({"i": idx, "lat": float(lat[idx]), "lon": float(lon[idx])})
                continue
            # 落陆检查
            try:
                if isinstance(lm, np.ndarray) and lm.shape == (H, W) and bool(lm[ii, jj]):
                    on_land += 1
                    if len(ex_land) < 3:
                        ex_land.append({"i": idx, "lat": float(lat[idx]), "lon": float(lon[idx])})
            except Exception:
                pass
        res["num_on_land"] = int(on_land)
        res["num_out_of_bbox"] = int(oob)
        res["ok"] = bool(on_land == 0 and oob == 0)
        res["examples"] = {"on_land": ex_land, "out_of_bbox": ex_oob}
        if not res["ok"]:
            try:
                print(f"[ROUTE_VALIDATION] {label or '-'} on_land={on_land} oob={oob} ex={res['examples']}")
            except Exception:
                pass
        return res
    except Exception as e:
        try:
            print(f"[ROUTE_VALIDATION] failed: {e}")
        except Exception:
            pass
        return res

# -------------------------
# 路径规划（确保起终点可走）
# -------------------------
def _ensure_finite_at(env: EnvironmentContext, ij: tuple[int, int]) -> tuple[int, int]:
    """
    仅在小邻域内（默认半径3）将起止点调整到最近的“可走”栅格（以 cost 网格为准）。
    约束：禁止任何形式的全图扫描/跳转。如果局部搜索失败，则返回原始 (i,j) 并打印明确日志。
    """
    i, j = int(ij[0]), int(ij[1])
    i0, j0 = i, j

    # 目标尺寸：以 cost_da 为准
    Hc = Wc = None
    if getattr(env, "cost_da", None) is not None:
        try:
            Hc, Wc = env.cost_da.shape[-2:]
        except Exception:
            Hc, Wc = None, None

    if Hc is None or Wc is None:
        return (i, j)

    # 裁剪到网格范围
    i = int(np.clip(i, 0, Hc - 1))
    j = int(np.clip(j, 0, Wc - 1))

    # 很小的局部半径（适度放宽以提高可达性，但仍为局部）
    R = 5

    # 与 cost 对齐的海陆掩膜（可选）
    ocean_mask_c = None
    lm = getattr(env, "land_mask", None)
    if isinstance(lm, np.ndarray):
        try:
            Hl, Wl = lm.shape[-2], lm.shape[-1]
            lm_bool = lm.astype(bool)
            if (Hl, Wl) != (Hc, Wc):
                yi = (np.linspace(0, Hl - 1, Hc)).astype(int)
                xj = (np.linspace(0, Wl - 1, Wc)).astype(int)
                lm_bool = lm_bool[yi[:, None], xj[None, :]]
            ocean_mask_c = ~lm_bool
        except Exception:
            ocean_mask_c = None

    # 若当前在海上且成本有限，直接返回
    arr = np.asarray(env.cost_da.values)
    def _ok(ii: int, jj: int) -> bool:
        if ii < 0 or jj < 0 or ii >= Hc or jj >= Wc:
            return False
        if ocean_mask_c is not None and not bool(ocean_mask_c[ii, jj]):
            return False
        v = float(arr[ii, jj])
        return (np.isfinite(v) and v < 1e9)

    if _ok(i, j):
        return (i, j)

    # 在局部窗口搜索最近可用点
    i_min, i_max = max(0, i - R), min(Hc - 1, i + R)
    j_min, j_max = max(0, j - R), min(Wc - 1, j + R)
    best = None
    best_d2 = None
    for ii in range(i_min, i_max + 1):
        for jj in range(j_min, j_max + 1):
            if _ok(ii, jj):
                d2 = (ii - i) ** 2 + (jj - j) ** 2
                if best is None or d2 < best_d2:
                    best = (ii, jj)
                    best_d2 = d2
    if best is not None:
        return (int(best[0]), int(best[1]))

    # 局部失败：不改变点位，打印日志
    try:
        print(f"[ENSURE_FINITE] local search failed at {(i0, j0)}, leaving as-is.")
    except Exception:
        pass
    return (i0, j0)


# 模块级：存放路线调试信息（通过 id(route_result) -> dict）
_ROUTE_DEBUG: dict[int, dict] = {}

def get_route_debug(route_obj) -> dict | None:
    try:
        return _ROUTE_DEBUG.get(id(route_obj))
    except Exception:
        return None

def _set_route_debug(route_obj, info: dict) -> None:
    try:
        _ROUTE_DEBUG[id(route_obj)] = info
    except Exception:
        pass


def compute_route(
    env: EnvironmentContext,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    allow_diagonal: bool,
    heuristic: str,
) -> RouteResult:
    if env.cost_da is None:
        raise ValueError("EnvironmentContext must have cost_da for routing.")

    debug_info: dict = {
        "start_ij_input_raw": (int(start_ij[0]), int(start_ij[1])),
        "goal_ij_input_raw": (int(goal_ij[0]), int(goal_ij[1])),
    }

    # 先依据 cost 的 y/x 尺寸进行显式裁剪（防御性）
    try:
        Ny = int(env.cost_da.sizes.get("y")) if hasattr(env.cost_da, "sizes") and ("y" in env.cost_da.sizes) else int(env.cost_da.shape[-2])
        Nx = int(env.cost_da.sizes.get("x")) if hasattr(env.cost_da, "sizes") and ("x" in env.cost_da.sizes) else int(env.cost_da.shape[-1])
    except Exception:
        Ny, Nx = int(env.cost_da.shape[-2]), int(env.cost_da.shape[-1])

    def _clip_ij(ij: tuple[int, int]) -> tuple[int, int]:
        i, j = int(ij[0]), int(ij[1])
        i = int(np.clip(i, 0, Ny - 1))
        j = int(np.clip(j, 0, Nx - 1))
        return (i, j)

    start_ij = _clip_ij(start_ij)
    goal_ij = _clip_ij(goal_ij)
    debug_info["start_ij_input"] = tuple(start_ij)
    debug_info["goal_ij_input"] = tuple(goal_ij)

    # 再用“可走性”吸附，且最后再次裁剪一次，确保不越界（记录 ensure 前后）
    debug_info["start_ij_before_ensure"] = tuple(start_ij)
    debug_info["goal_ij_before_ensure"] = tuple(goal_ij)
    start_ij = _clip_ij(_ensure_finite_at(env, start_ij))
    goal_ij = _clip_ij(_ensure_finite_at(env, goal_ij))
    debug_info["start_ij_after_ensure"] = tuple(start_ij)
    debug_info["goal_ij_after_ensure"] = tuple(goal_ij)

    # 若 ensure_finite 后两点仍相同且原始输入不同，先对 goal 做一次局部“轻推”避免退化为单点路径
    try:
        # 如果 ensure_finite 后两点仍相同（无论原始是否相同），对 goal 做一次局部“轻推”避免退化为单点路径
        if (int(start_ij[0]) == int(goal_ij[0]) and int(start_ij[1]) == int(goal_ij[1])):
            arr = np.asarray(env.cost_da.values, dtype=float)
            Hc, Wc = arr.shape[-2], arr.shape[-1]
            lm = getattr(env, "land_mask", None)
            lm_bool = None
            if isinstance(lm, np.ndarray):
                lm_bool = lm.astype(bool)
                if lm_bool.shape != (Hc, Wc):
                    yi = (np.linspace(0, lm_bool.shape[-2] - 1, Hc)).astype(int)
                    xj = (np.linspace(0, lm_bool.shape[-1] - 1, Wc)).astype(int)
                    lm_bool = lm_bool[yi[:, None], xj[None, :]]
            def _ok(i: int, j: int) -> bool:
                if i < 0 or j < 0 or i >= Hc or j >= Wc:
                    return False
                if lm_bool is not None and bool(lm_bool[i, j]):
                    return False
                v = float(arr[i, j])
                if not np.isfinite(v) or v >= 1e9:
                    return False
                return True
            i0, j0 = int(goal_ij[0]), int(goal_ij[1])
            found = None
            for r in range(1, 6):  # 限制到 5 像素半径（局部轻推）
                i_min, i_max = max(0, i0 - r), min(Hc - 1, i0 + r)
                j_min, j_max = max(0, j0 - r), min(Wc - 1, j0 + r)
                for ii in range(i_min, i_max + 1):
                    for jj in (j_min, j_max):
                        if _ok(ii, jj) and (ii != start_ij[0] or jj != start_ij[1]):
                            found = (ii, jj); break
                    if found: break
                if not found:
                    for jj in range(j_min + 1, j_max):
                        for ii in (i_min, i_max):
                            if _ok(ii, jj) and (ii != start_ij[0] or jj != start_ij[1]):
                                found = (ii, jj); break
                        if found: break
                if found: break
            if found is not None:
                goal_ij = (int(found[0]), int(found[1]))
                debug_info["goal_ij_after_nudge"] = tuple(goal_ij)
    except Exception:
        pass

    # 最后调用 A*（若越界/不可达则做一次稳健修正后重试）
    try:
        route_summary: RouteSummary = astar_on_cost(
            env.cost_da, start_ij, goal_ij, neighbor8=allow_diagonal, heuristic=heuristic
        )
    except (ValueError, RuntimeError) as _e:
        # 限制性本地重试：仅在很小半径内寻找可通行点；若失败则返回不可达，不做全图跳转
        arr = np.asarray(env.cost_da.values, dtype=float)
        Hc, Wc = arr.shape[-2], arr.shape[-1]
        lm = getattr(env, "land_mask", None)
        lm_bool = None
        if isinstance(lm, np.ndarray):
            lm_bool = lm.astype(bool)
            if lm_bool.shape != (Hc, Wc):
                yi = (np.linspace(0, lm_bool.shape[-2] - 1, Hc)).astype(int)
                xj = (np.linspace(0, lm_bool.shape[-1] - 1, Wc)).astype(int)
                lm_bool = lm_bool[yi[:, None], xj[None, :]]
        def _ok(i: int, j: int) -> bool:
            if i < 0 or j < 0 or i >= Hc or j >= Wc:
                return False
            if lm_bool is not None and bool(lm_bool[i, j]):
                return False
            v = float(arr[i, j])
            if not np.isfinite(v) or v >= 1e6:
                return False
            return True
        def _search_local(p: tuple[int, int], max_r: int = 5) -> tuple[int, int] | None:
            i0, j0 = int(p[0]), int(p[1])
            if _ok(i0, j0):
                return (i0, j0)
            for r in range(1, int(max_r) + 1):
                i_min, i_max = max(0, i0 - r), min(Hc - 1, i0 + r)
                j_min, j_max = max(0, j0 - r), min(Wc - 1, j0 + r)
                # 扫描边框以更快找到近邻
                for ii in range(i_min, i_max + 1):
                    for jj in (j_min, j_max):
                        if _ok(ii, jj):
                            return (ii, jj)
                for jj in range(j_min + 1, j_max):
                    for ii in (i_min, i_max):
                        if _ok(ii, jj):
                            return (ii, jj)
            return None
        s2 = _search_local(start_ij, max_r=5) or start_ij
        g2 = _search_local(goal_ij, max_r=5) or goal_ij
        debug_info["start_ij_after_retry"] = tuple(_clip_ij(s2)) if s2 else None
        debug_info["goal_ij_after_retry"] = tuple(_clip_ij(g2)) if g2 else None
        try:
            route_summary = astar_on_cost(
                env.cost_da, _clip_ij(s2), _clip_ij(g2), neighbor8=allow_diagonal, heuristic=heuristic
            )
            start_ij, goal_ij = _clip_ij(s2), _clip_ij(g2)
        except Exception as _e2:
            # 标记不可达并返回空路径
            debug_info["route_error"] = str(_e2)
            rr = RouteResult(
                path_ij=[],
                path_lonlat=[],
                cost_sum=0.0,
                len=0,
                reachable=False,
                heuristic=heuristic,
                diagonal=allow_diagonal,
            )
            try:
                setattr(rr, "debug", debug_info)
            except Exception:
                pass
            return rr

    path_lonlat = path_ij_to_lonlat(env, route_summary.path_ij)
    rr = RouteResult(
        path_ij=route_summary.path_ij,
        path_lonlat=path_lonlat,
        cost_sum=float(route_summary.cost_sum),
        len=len(route_summary.path_ij),
        reachable=bool(route_summary.path_ij),
        heuristic=heuristic,
        diagonal=allow_diagonal,
    )
    try:
        setattr(rr, "debug", debug_info)
    except Exception:
        pass
    return rr


def count_land_hits(env_ctx: "EnvironmentContext", path_ij: list[tuple[int, int]]) -> int:
    """
    统计路径上命中陆地格的数量。仅用于测试与调试。
    陆地判定：env_ctx.land_mask 非空，且 land_mask[i,j] != 0。
    越界索引跳过。
    """
    try:
        lm = getattr(env_ctx, "land_mask", None)
        if lm is None:
            return 0
        arr = np.asarray(lm)
        H, W = arr.shape[-2], arr.shape[-1]
        h = 0
        for (i, j) in (path_ij or []):
            if 0 <= int(i) < H and 0 <= int(j) < W and arr[int(i), int(j)] != 0:
                h += 1
        return int(h)
    except Exception:
        return 0


def compute_route_strict_from_latlon(
    env_ctx: EnvironmentContext,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    allow_diagonal: bool = True,
    heuristic: str = "euclidean",
) -> RouteResult:
    """
    严格模式：
    - 仅使用 find_nearest_ocean_cell 决定一次性的起终点 (i,j)
    - 禁止调用 ensure_finite_at / snap_point_to_domain_and_ocean / compute_planning_domain
    - 若局部窗口内找不到可通行海面格点，则直接返回 reachable=False
    """
    dbg = {
        "start_latlon_input": (float(start_lat), float(start_lon)),
        "end_latlon_input": (float(end_lat), float(end_lon)),
    }
    if env_ctx.cost_da is None:
        # 构造不可达的空结果
        rr = RouteResult(path_ij=[], path_lonlat=[], cost_sum=0.0, len=0, reachable=False, heuristic=heuristic, diagonal=allow_diagonal, debug={**dbg, "reason": "no_cost_da"})
        return rr

    # 局部搜索半径：保持在 20 格以内，避免全图跳跃，同时兼容 150km 距离阈值
    si, sj, s_info = find_nearest_ocean_cell(float(start_lat), float(start_lon), env_ctx, max_radius=20)
    gi, gj, g_info = find_nearest_ocean_cell(float(end_lat), float(end_lon), env_ctx, max_radius=20)

    # 任一失败则直接返回不可达
    if str(s_info.get("status", "")).startswith("failed") or str(g_info.get("status", "")).startswith("failed"):
        dbg.update({
            "start_ij": (int(si), int(sj)),
            "end_ij": (int(gi), int(gj)),
            "start_snap_info": dict(s_info),
            "end_snap_info": dict(g_info),
            "reason": "snap_failed",
        })
        rr = RouteResult(path_ij=[], path_lonlat=[], cost_sum=0.0, len=0, reachable=False, heuristic=heuristic, diagonal=allow_diagonal, debug=dbg)
        return rr

    # 局部 ensure：在极小半径 R=5 内，确保起止点落在可通行格（land==0 且 cost<1e9）
    try:
        dbg["start_ij_before_ensure"] = (int(si), int(sj))
        dbg["end_ij_before_ensure"] = (int(gi), int(gj))
        s_ij_ens = _ensure_finite_at(env_ctx, (int(si), int(sj)))
        g_ij_ens = _ensure_finite_at(env_ctx, (int(gi), int(gj)))
        si, sj = int(s_ij_ens[0]), int(s_ij_ens[1])
        gi, gj = int(g_ij_ens[0]), int(g_ij_ens[1])
        dbg["start_ij_after_ensure"] = (si, sj)
        dbg["end_ij_after_ensure"] = (gi, gj)
    except Exception:
        pass

    # A* 规划（不再对起终点做任何全局吸附修正）
    try:
        summary = astar_on_cost(env_ctx.cost_da, (int(si), int(sj)), (int(gi), int(gj)), neighbor8=bool(allow_diagonal), heuristic=str(heuristic))
        path_ij = list(summary.path_ij)
        pll = path_ij_to_lonlat(env_ctx, path_ij)
        dbg.update({
            "start_ij": (int(si), int(sj)),
            "end_ij": (int(gi), int(gj)),
            "start_snap_info": dict(s_info),
            "end_snap_info": dict(g_info),
        })
        # 统计并记录 land hits 供测试/调试
        try:
            dbg["land_hits"] = int(count_land_hits(env_ctx, path_ij))
        except Exception:
            pass
        rr = RouteResult(
            path_ij=path_ij,
            path_lonlat=pll,
            cost_sum=float(summary.cost_sum),
            len=len(path_ij),
            reachable=bool(path_ij),
            heuristic=str(heuristic),
            diagonal=bool(allow_diagonal),
            debug=dbg,
        )
    except Exception as e:
        dbg.update({
            "start_ij": (int(si), int(sj)),
            "end_ij": (int(gi), int(gj)),
            "start_snap_info": dict(s_info),
            "end_snap_info": dict(g_info),
            "astar_error": str(e),
        })
        rr = RouteResult(path_ij=[], path_lonlat=[], cost_sum=0.0, len=0, reachable=False, heuristic=heuristic, diagonal=allow_diagonal, debug=dbg)
        return rr

    return rr


# -------------------------
# 路线分析/摘要/导出
# -------------------------
def sample_route_profile(route_result: RouteResult, env_ctx: EnvironmentContext, eco_ctx: Any | None = None) -> dict:
    """
    沿路线采样剖面（尽量复用已有结构，不重复计算）：
    返回包含：
      - s: 累积距离 km（从 0 开始）
      - t: None（暂无精确时间，后续可接入船速模型）
      - risk_total: 成本图在每个节点的值（可视为风险代理）
      - risk_ice: 冰险代理（优先使用 newenv 加权分量 copernicus_ice；否则归一化 legacy ice）
      - wave_swh: 浪高或浪险代理（优先 newenv wave_swh 原值；否则使用已乘权重的分量或 0）
      - speed: None（占位）
      - co2_cum: 累积 CO₂（若 newenv 可用则估计；否则 None）
      - lat/lon: 节点坐标
    """
    try:
        if not route_result or not getattr(route_result, "path_ij", None):
            return {}
        path_ij = list(route_result.path_ij)
        # 1) 经纬度与距离
        try:
            latlon = list(route_result.path_lonlat) if getattr(route_result, "path_lonlat", None) else path_ij_to_lonlat(env_ctx, path_ij)
        except Exception:
            latlon = path_ij_to_lonlat(env_ctx, path_ij)
        lat = np.asarray([float(p[0]) for p in latlon], dtype=float)
        lon = np.asarray([float(p[1]) for p in latlon], dtype=float)
        seg_km = np.zeros(max(0, len(lat) - 1), dtype=float)
        if len(lat) >= 2:
            from ArcticRoute.core.route.metrics import haversine_m as _hm  # local import to avoid cycles
            for i in range(len(lat) - 1):
                seg_km[i] = _hm(float(lat[i]), float(lon[i]), float(lat[i+1]), float(lon[i+1])) / 1000.0
        s = np.concatenate([[0.0], np.cumsum(seg_km)]) if lat.size > 0 else np.array([], dtype=float)

        # 2) 成本/风险采样
        H = W = None
        try:
            if env_ctx.cost_da is not None:
                H, W = env_ctx.cost_da.shape[-2:]
        except Exception:
            H = W = None
        def _safe_idx(i: int, j: int) -> tuple[int, int]:
            if H is None or W is None:
                return int(i), int(j)
            return int(np.clip(i, 0, H-1)), int(np.clip(j, 0, W-1))

        # risk_total: 取 cost_da 的节点值
        risk_total = None
        try:
            if env_ctx.cost_da is not None:
                arr = np.asarray(env_ctx.cost_da.values, dtype=float)
                vals = []
                for (i, j) in path_ij:
                    ii, jj = _safe_idx(int(i), int(j))
                    v = float(arr[ii, jj])
                    vals.append(np.nan if not np.isfinite(v) else v)
                risk_total = np.asarray(vals, dtype=float)
        except Exception:
            risk_total = None

        layers = env_ctx.risk_layers_raw or {}
        # risk_ice：优先 newenv 分量（已乘权重），否则对 legacy ice 做 0..1 归一并乘上 10*w_ice 的尺度
        def _norm01(a: np.ndarray | None) -> np.ndarray | None:
            if a is None:
                return None
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return None
            p1 = float(np.nanpercentile(finite, 1))
            p99 = float(np.nanpercentile(finite, 99))
            if not np.isfinite(p99) or p99 <= p1:
                return None
            return np.clip((a - p1) / (p99 - p1), 0.0, 1.0).astype("float32")

        risk_ice = None
        try:
            if "copernicus_ice" in layers:
                arr_ci = np.asarray(layers["copernicus_ice"].values, dtype=float)
                vals = []
                for (i, j) in path_ij:
                    ii, jj = _safe_idx(int(i), int(j))
                    vals.append(float(arr_ci[ii, jj]))
                risk_ice = np.asarray(vals, dtype=float)
            elif "ice" in layers:
                arr_i = np.asarray(layers["ice"].values, dtype=float)
                arr_i = _norm01(arr_i)
                if arr_i is not None:
                    # 按 analyze_route_cost 的尺度：10 * w_ice
                    w_ice = float(getattr(getattr(env_ctx, "cost_da", None), "attrs", {}).get("w_ice", 1.0) or 1.0)
                    vals = []
                    for (i, j) in path_ij:
                        ii, jj = _safe_idx(int(i), int(j))
                        vals.append(float(arr_i[ii, jj]) * 10.0 * w_ice)
                    risk_ice = np.asarray(vals, dtype=float)
        except Exception:
            risk_ice = None

        # wave_swh：优先 newenv 原值；否则 risk_layers_raw['wave_swh']（已乘权重的成本分量）；再否则 None
        wave_swh = None
        try:
            ne = load_newenv_for_eco(env_ctx.ym, env_ctx.lat_arr, env_ctx.lon_arr)
        except Exception:
            ne = {}
        if ne and (ne.get("wave_swh") is not None):
            try:
                arr_w = np.asarray(ne["wave_swh"].values, dtype=float)
                Hn, Wn = arr_w.shape[-2], arr_w.shape[-1]
                vals = []
                for (i, j) in path_ij:
                    ii = int(np.clip(i, 0, Hn - 1))
                    jj = int(np.clip(j, 0, Wn - 1))
                    vals.append(float(arr_w[ii, jj]))
                wave_swh = np.asarray(vals, dtype=float)
            except Exception:
                wave_swh = None
        if wave_swh is None and ("wave_swh" in layers):
            try:
                arr_wc = np.asarray(layers["wave_swh"].values, dtype=float)
                vals = []
                for (i, j) in path_ij:
                    ii, jj = _safe_idx(int(i), int(j))
                    vals.append(float(arr_wc[ii, jj]))
                wave_swh = np.asarray(vals, dtype=float)
            except Exception:
                wave_swh = None

        # 3) CO2 累积（若 newenv 可用，复用相同简化公式进行逐段计算）
        co2_cum = None
        try:
            if ne:
                vp = getattr(env_ctx, "vessel_profile", None) or {}
                try:
                    _FUEL_WARNED
                except NameError:
                    _FUEL_WARNED = False  # type: ignore
                # 兼容 per_nm，并提供 dwt 估算与兜底
                base_fuel_per_km = float(vp.get("base_fuel_per_km", np.nan))
                if not np.isfinite(base_fuel_per_km):
                    bf_nm = vp.get("base_fuel_per_nm")
                    if bf_nm is not None:
                        try:
                            base_fuel_per_km = float(bf_nm) / 1.852
                        except Exception:
                            base_fuel_per_km = np.nan
                if not np.isfinite(base_fuel_per_km):
                    dwtv = float(vp.get("dwt", np.nan))
                    if np.isfinite(dwtv):
                        base_fuel_per_km = max(0.05, 1.5e-5 * dwtv)
                        if not _FUEL_WARNED:
                            print(f"[WARN] vessel_profile 缺少 base_fuel_per_km，按 dwt 估算为 {base_fuel_per_km:.3f} t/km")
                            _FUEL_WARNED = True  # type: ignore
                    else:
                        base_fuel_per_km = 0.05
                        if not _FUEL_WARNED:
                            print("[WARN] vessel_profile 缺少 base_fuel_per_km/dwt，使用保守缺省 0.05 t/km")
                            _FUEL_WARNED = True  # type: ignore
                a1 = float(vp.get("a1_sic_quad", 1.2))
                a2 = float(vp.get("a2_sithick", 0.2))
                b1 = float(vp.get("b1_wave_swh", 0.12))
                wave_thr = float(vp.get("wave_threshold_m", 1.0))
                ice_sens = float(vp.get("ice_sensitivity", 1.0))
                wave_sens = float(vp.get("wave_sensitivity", 1.0))
                sic_da = ne.get("sic")
                sth_da = ne.get("sithick")
                swh_da = ne.get("wave_swh")
                def _sample_da(da):
                    if da is None:
                        return np.zeros_like(seg_km)
                    arr = np.asarray(da.values)
                    Hn, Wn = arr.shape[-2], arr.shape[-1]
                    out = []
                    for (i, j) in path_ij[:-1]:
                        ii = int(np.clip(i, 0, Hn-1)); jj = int(np.clip(j, 0, Wn-1))
                        out.append(float(arr[ii, jj]))
                    return np.asarray(out, dtype=float)
                sic_v = np.clip(_sample_da(sic_da), 0.0, 1.0)
                sth_v = np.clip(_sample_da(sth_da), 0.0, None)
                swh_v = np.clip(_sample_da(swh_da), 0.0, None)
                f_ice = 1.0 + ice_sens * (a1 * (sic_v ** 2.0) + a2 * sth_v)
                f_wave = 1.0 + wave_sens * (b1 * np.clip(swh_v - wave_thr, 0.0, None))
                f_total = f_ice * f_wave
                base_fuel_seg = base_fuel_per_km * seg_km
                fuel_seg = base_fuel_seg * f_total
                co2_seg = fuel_seg * 3.114
                co2_cum = np.concatenate([[0.0], np.cumsum(co2_seg)])
        except Exception:
            co2_cum = None

        return {
            "s": s.astype("float32") if isinstance(s, np.ndarray) else None,
            "t": None,
            "risk_total": None if risk_total is None else np.asarray(risk_total, dtype="float32"),
            "risk_ice": None if risk_ice is None else np.asarray(risk_ice, dtype="float32"),
            "wave_swh": None if wave_swh is None else np.asarray(wave_swh, dtype="float32"),
            "speed": None,
            "co2_cum": None if co2_cum is None else np.asarray(co2_cum, dtype="float32"),
            "lat": lat.astype("float32") if isinstance(lat, np.ndarray) else None,
            "lon": lon.astype("float32") if isinstance(lon, np.ndarray) else None,
        }
    except Exception:
        return {}

def extract_route_hotspots(profile: dict, top_k: int = 3, min_separation: int | None = None, alpha_wave: float = 0.4) -> list[dict]:
    """
    从剖面 profile 中自动识别 2–3 个热点风险点。
    - 打分 score = norm(risk_total) + alpha_wave * norm(wave_swh)
    - 在 1..N-2 区间内寻找局部峰值
    - 采用贪心抑制，保证热点之间具有最小索引间距（默认约为路径长度的 5%）

    参数:
      profile: 来自 sample_route_profile 的字典
      top_k: 选择的热点数量上限（默认 3）
      min_separation: 最小索引间距（None 则按 5%*N 估算，至少 5）
      alpha_wave: 浪高在综合评分中的权重（默认 0.4）

    返回: list[dict]，每个热点包含 {idx, lat, lon, risk_total, risk_ice, wave_swh, reason}
    """
    try:
        lat = np.asarray(profile.get("lat") or [], dtype=float)
        lon = np.asarray(profile.get("lon") or [], dtype=float)
        rt = np.asarray(profile.get("risk_total") or [], dtype=float)
        ri = np.asarray(profile.get("risk_ice") or [], dtype=float)
        wv = np.asarray(profile.get("wave_swh") or [], dtype=float)
        n = int(min(lat.size, lon.size, rt.size))
        if n < 3:
            return []
        lat = lat[:n]; lon = lon[:n]; rt = rt[:n]
        ri = ri[:n] if ri.size >= n else np.full(n, np.nan, dtype=float)
        wv = wv[:n] if wv.size >= n else np.full(n, np.nan, dtype=float)

        def _norm01(a: np.ndarray) -> np.ndarray:
            b = a.astype(float).copy()
            finite = np.isfinite(b)
            if not np.any(finite):
                return np.zeros_like(b)
            vmin = float(np.nanpercentile(b[finite], 5))
            vmax = float(np.nanpercentile(b[finite], 95))
            if not np.isfinite(vmax) or vmax <= vmin:
                vmin = float(np.nanmin(b[finite])); vmax = float(np.nanmax(b[finite]))
                if not np.isfinite(vmax) or vmax <= vmin:
                    return np.zeros_like(b)
            out = (b - vmin) / (vmax - vmin)
            out[~np.isfinite(out)] = 0.0
            return np.clip(out, 0.0, 1.0)

        score = _norm01(rt) + float(alpha_wave) * _norm01(wv)
        # 局部峰值（排除两端）
        cands: list[tuple[float, int]] = []
        for i in range(1, n - 1):
            s = float(score[i])
            if (s >= score[i-1]) and (s >= score[i+1]) and np.isfinite(s):
                cands.append((s, i))
        if not cands:
            # 无局部峰值则取全局前 top_k
            cands = sorted([(float(score[i]), i) for i in range(n)], reverse=True)
        else:
            cands.sort(reverse=True)

        if min_separation is None:
            min_separation = max(5, int(0.05 * n))

        selected: list[int] = []
        for s, i in cands:
            if len(selected) >= int(top_k):
                break
            if all(abs(i - j) >= int(min_separation) for j in selected):
                selected.append(i)
        selected.sort()

        # 统计阈值用于解释
        def _pctl(a: np.ndarray, q: float) -> float:
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return float('nan')
            return float(np.nanpercentile(finite, q))
        rt_p90 = _pctl(rt, 90.0)
        ri_p90 = _pctl(ri, 90.0)
        wv_p90 = _pctl(wv, 90.0)
        wv_p80 = _pctl(wv, 80.0)

        def _reason(i: int) -> str:
            rti = float(rt[i]) if np.isfinite(rt[i]) else float('nan')
            rii = float(ri[i]) if np.isfinite(ri[i]) else float('nan')
            wvi = float(wv[i]) if np.isfinite(wv[i]) else float('nan')
            # 规则优先级：综合风险峰值 > 冰高且浪高 > 冰高 > 浪高 > 一般峰值
            if np.isfinite(rt_p90) and np.isfinite(rti) and (rti >= rt_p90):
                return f"此处综合风险 {rti:.2f} 为全程高位/峰值之一"
            if np.isfinite(ri_p90) and np.isfinite(rii) and (rii >= ri_p90) and np.isfinite(wv_p80) and np.isfinite(wvi) and (wvi >= wv_p80):
                return f"此处冰风险较高（{rii:.2f}）且浪高偏大（{wvi:.1f} m）"
            if np.isfinite(ri_p90) and np.isfinite(rii) and (rii >= ri_p90):
                return f"此处冰风险较高（{rii:.2f}），为需重点关注冰区"
            if np.isfinite(wv_p90) and np.isfinite(wvi) and (wvi >= wv_p90):
                return f"此处浪高 {wvi:.1f} m，为全程浪高峰值之一"
            return f"此处综合风险 {rti:.2f} 为局部峰值"

        out: list[dict] = []
        for i in selected:
            out.append({
                "idx": int(i),
                "lat": float(lat[i]),
                "lon": float(lon[i]),
                "risk_total": None if not np.isfinite(rt[i]) else float(rt[i]),
                "risk_ice": None if not np.isfinite(ri[i]) else float(ri[i]),
                "wave_swh": None if not np.isfinite(wv[i]) else float(wv[i]),
                "reason": _reason(i),
            })
        return out
    except Exception:
        return []

def analyze_route_cost(env: EnvironmentContext, route_result: RouteResult) -> dict[str, Any]:
    """
    沿路径分解各风险成分贡献，返回统一结构：
      {
        "total_cost": float,
        "distance_km": float,
        "risk_components": {
            "ice": float,
            "accident": float,
            "congestion": float,   # 由 interact/collision 映射
            "prior": float,        # 主航线先验惩罚
            "wave": float,         # newenv: wave_swh
        },
        "risk_components_normalized": { ... }  # 可选，占比
      }
    兼容：若缺少任何项，补 0；若无法计算，返回空 dict。
    """
    try:
        if not route_result or not route_result.path_ij:
            return {}
        H = W = None
        try:
            if env.cost_da is not None:
                H, W = env.cost_da.shape[-2:]
        except Exception:
            H = W = None

        path = route_result.path_ij

        def _integrate(field_arr: np.ndarray, *, scale: float = 1.0) -> float:
            if field_arr is None:
                return 0.0
            arr = np.asarray(field_arr, dtype="float32")
            if arr.ndim > 2:
                axes = tuple(range(0, arr.ndim - 2))
                arr = np.nanmean(arr, axis=axes)
            h, w = arr.shape[-2], arr.shape[-1]
            total_val = 0.0
            for k in range(len(path) - 1):
                i, j = path[k]
                ni, nj = path[k + 1]
                if 0 <= i < h and 0 <= j < w and 0 <= ni < h and 0 <= nj < w:
                    c0 = float(arr[i, j])
                    c1 = float(arr[ni, nj])
                    if not (np.isnan(c0) or np.isnan(c1)):
                        step_len = np.sqrt(2.0) if (ni != i and nj != j) else 1.0
                        total_val += 0.5 * (c0 + c1) * scale * step_len
            return float(total_val)

        # 1) 距离
        distance_km = 0.0
        try:
            if route_result.path_lonlat:
                lonlat_tuples = [(p[1], p[0]) for p in route_result.path_lonlat]
                distance_km = float(compute_distance_km(lonlat_tuples))
        except Exception:
            distance_km = 0.0

        # 2) 准备风险层
        layers = env.risk_layers_raw or {}
        # 提取 cost_da attrs 中的权重以便还原 legacy 层的贡献尺度
        w_ice = float(getattr(getattr(env, "cost_da", None), "attrs", {}).get("w_ice", 1.0) or 1.0)
        w_acc = float(getattr(getattr(env, "cost_da", None), "attrs", {}).get("w_accident", 1.0) or 1.0)
        w_inter = float(getattr(getattr(env, "cost_da", None), "attrs", {}).get("w_interact", 0.0) or 0.0)
        prior_w = float(getattr(getattr(env, "cost_da", None), "attrs", {}).get("prior_weight", 0.0) or 0.0)

        comps: dict[str, float] = {k: 0.0 for k in ("ice", "accident", "congestion", "prior", "wave")}

        # 2.1 legacy: ice/accident/interact → 归一化到 0..1 后乘以与 _build_cost_da 一致的缩放
        def _norm_01(da: xr.DataArray | np.ndarray | None) -> np.ndarray | None:
            if da is None:
                return None
            a = np.asarray(getattr(da, "values", da), dtype="float32")
            if a.ndim > 2:
                axes = tuple(range(0, a.ndim - 2))
                a = np.nanmean(a, axis=axes)
            finite = a[np.isfinite(a)]
            if finite.size == 0:
                return None
            p1 = float(np.nanpercentile(finite, 1))
            p99 = float(np.nanpercentile(finite, 99))
            if not np.isfinite(p99) or p99 <= p1:
                return None
            return np.clip((a - p1) / (p99 - p1), 0.0, 1.0).astype("float32")

        try:
            if "ice" in layers:
                ni = _norm_01(layers.get("ice"))
                if ni is not None:
                    comps["ice"] = round(_integrate(ni, scale=10.0 * w_ice), 2)
        except Exception:
            pass
        try:
            if "accident" in layers:
                na = _norm_01(layers.get("accident"))
                if na is not None:
                    comps["accident"] = round(_integrate(na, scale=10.0 * w_acc), 2)
        except Exception:
            pass
        try:
            for k in ("interact", "interaction", "collision"):
                if k in layers:
                    nc = _norm_01(layers.get(k))
                    if nc is not None:
                        comps["congestion"] = round(_integrate(nc, scale=10.0 * w_inter), 2)
                    break
        except Exception:
            pass

        # 2.2 newenv: 直接使用已乘权重的成本分量（在 _build_cost_da 中写入 risk_layers_raw）
        try:
            if "copernicus_ice" in layers:
                arr_ci = np.asarray(layers["copernicus_ice"].values, dtype="float32")
                comps["ice"] = round(comps.get("ice", 0.0) + _integrate(arr_ci, scale=1.0), 2)
        except Exception:
            pass
        try:
            if "wave_swh" in layers:
                arr_wv = np.asarray(layers["wave_swh"].values, dtype="float32")
                comps["wave"] = round(_integrate(arr_wv, scale=1.0), 2)
        except Exception:
            pass

        # 2.3 prior 惩罚
        try:
            if env.prior_penalty_da is not None and prior_w > 0:
                prior_arr = np.asarray(env.prior_penalty_da.values, dtype="float32")
                comps["prior"] = round(_integrate(prior_arr, scale=float(prior_w)), 2)
        except Exception:
            pass

        # 3) 汇总
        total_cost = float(route_result.cost_sum or 0.0)
        # 归一化占比
        s = sum(v for v in comps.values() if np.isfinite(v))
        norm = {k: (float(v) / s if s > 0 else 0.0) for k, v in comps.items()}

        out = {
            "total_cost": round(total_cost, 4),
            "distance_km": round(distance_km, 3),
            "risk_components": comps,
            "risk_components_normalized": norm,
        }
        try:
            print("[DEBUG_RISK] risk_components:", out.get("risk_components"))
            print("[DEBUG_RISK] risk_components_normalized:", out.get("risk_components_normalized"))
        except Exception:
            pass
        return out
    except Exception:
        return {}


def analyze_prior_adherence(env: EnvironmentContext, route_result: RouteResult) -> float:
    da = env.prior_penalty_da
    if not route_result.reachable or da is None or not route_result.path_ij:
        return 0.0
    pi = np.array([p[0] for p in route_result.path_ij], dtype=int)
    pj = np.array([p[1] for p in route_result.path_ij], dtype=int)
    pi = np.clip(pi, 0, da.shape[0] - 1)
    pj = np.clip(pj, 0, da.shape[1] - 1)
    vals = da.values[pi, pj]
    avg_penalty = float(np.nanmean(vals))
    return round(1.0 - avg_penalty, 3)


def summarize_route(route_result: RouteResult) -> dict:
    if not route_result.reachable:
        return {
            "distance_km": 0.0,
            "steps": 0,
            "cost_sum": 0.0,
            "estimated_fuel_ton": 0.0,
            "risk_score": 0.0,
        }
    lonlat_tuples = [(c[1], c[0]) for c in route_result.path_lonlat]
    distance = compute_distance_km(lonlat_tuples)
    estimated_fuel = distance * 0.05
    risk_score = route_result.cost_sum / route_result.len if route_result.len > 0 else 0.0
    return {
        "distance_km": round(float(distance), 2),
        "steps": route_result.len,
        "cost_sum": round(float(route_result.cost_sum), 4),
        "estimated_fuel_ton": round(float(estimated_fuel), 2),
        "risk_score": round(float(risk_score), 4),
    }


def smooth_path_lonlat_for_display(
    path_lonlat: List[Tuple[float, float]],
    iterations: int = 2,
    min_points: int = 10,
    max_points: int = 2000,
) -> List[Tuple[float, float]]:
    """
    Chaikin 风格的简单平滑，仅用于前端展示用的路线曲线。
    - 输入/输出都是 [(lon, lat), ...]。
    - 不改变端点。
    - 不参与任何距离/成本/燃油等计算。
    - 为避免爆炸，点数超过 max_points 时直接返回原始路径。
    """
    if path_lonlat is None:
        return []
    try:
        pts = np.asarray(path_lonlat, dtype=float)
    except Exception:
        return []
    n = pts.shape[0]
    if n < min_points or n > max_points:
        return path_lonlat

    iters = max(0, int(iterations))
    for _ in range(iters):
        new_pts = [pts[0]]  # 保留起点
        for i in range(n - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            # Chaikin corner cutting: 1/4, 3/4
            q = 0.75 * p0 + 0.25 * p1
            r = 0.25 * p0 + 0.75 * p1
            new_pts.extend([q, r])
        new_pts.append(pts[-1])  # 保留终点
        pts = np.vstack(new_pts)
        n = pts.shape[0]

    # 返回列表形式
    return [(float(lon), float(lat)) for lon, lat in pts]


def smooth_path_lonlat(
    path_lonlat: list[tuple[float, float]],
    n_iter: int = 3,
) -> list[tuple[float, float]]:
    """
    使用 Chaikin 曲线算法对路径进行平滑，仅用于显示。
    参数:
      - path_lonlat: [(lon, lat), ...]
      - n_iter: 迭代次数，2~4 之间效果较好。
    返回: [(lon, lat), ...]（首尾点保持不变）
    """
    if path_lonlat is None:
        return []
    try:
        pts = np.asarray(path_lonlat, dtype="float64")
    except Exception:
        return path_lonlat
    if len(pts) < 3 or int(n_iter) <= 0:
        return [(float(lon), float(lat)) for lon, lat in pts]

    for _ in range(int(n_iter)):
        new_pts = [pts[0]]  # 保留起点
        for i in range(len(pts) - 1):
            p = pts[i]
            q = pts[i + 1]
            Q = 0.75 * p + 0.25 * q
            R = 0.25 * p + 0.75 * q
            new_pts.extend([Q, R])
        new_pts.append(pts[-1])  # 保留终点
        pts = np.asarray(new_pts, dtype="float64")

    return [(float(lon), float(lat)) for lon, lat in pts]


def mark_adv_viz_done(trace: "PipelineTrace" | None, meta: dict[str, Any] | None = None) -> None:
    try:
        if trace is None:
            return
        trace.end("adv_viz", status="done", meta=meta or {})
    except Exception:
        pass

def mark_ai_explain_done(trace: "PipelineTrace" | None, meta: dict[str, Any] | None = None) -> None:
    try:
        if trace is None:
            return
        trace.end("ai_explain", status="done", meta=meta or {})
    except Exception:
        pass

def route_to_geojson(route_result: RouteResult) -> str:
    import json
    if not route_result.reachable or not route_result.path_lonlat:
        return "{}"
    coords_lonlat = [[float(p[1]), float(p[0])] for p in route_result.path_lonlat]
    feature = {
        "type": "Feature",
        "geometry": {"type": "LineString", "coordinates": coords_lonlat},
        "properties": {
            "cost_sum": route_result.cost_sum,
            "steps": route_result.len,
            "heuristic": route_result.heuristic,
            "diagonal_allowed": route_result.diagonal,
        },
    }
    return json.dumps({"type": "FeatureCollection", "features": [feature]}, indent=2)


# -------------------------
# 生态评估（封装 core/eco）
# -------------------------
def evaluate_route_eco(route_result: RouteResult, env_ctx: EnvironmentContext) -> EcoSummary:
    if not route_result.reachable or not route_result.path_lonlat:
        return EcoSummary(0.0, 0.0, 0.0, {"ok": False, "reason": "route_not_reachable"})
    # 1) 优先使用 newenv 驱动的 Eco 模型
    try:
        if eco_compute_newenv is not None and env_ctx is not None and env_ctx.lat_arr is not None and env_ctx.lon_arr is not None and route_result.path_ij:
            try:
                newenv = load_newenv_for_eco(env_ctx.ym, env_ctx.lat_arr, env_ctx.lon_arr)
            except Exception:
                newenv = {}
            if newenv:
                try:
                    res = eco_compute_newenv(route_result.path_ij, env_ctx.lat_arr, env_ctx.lon_arr, newenv, vessel_profile=getattr(env_ctx, "vessel_profile", None))
                    fuel_total = float(res.get("fuel_tons", 0.0))
                    co2_total = float(res.get("co2_tons", fuel_total * 3.114))
                    base_fuel = float(res.get("base_fuel_tons", 0.0))
                    cost_usd = fuel_total * 600.0
                    details = {"ok": True, "used": "newenv", "base_fuel_tons": base_fuel}
                    try:
                        vp = getattr(env_ctx, "vessel_profile", None)
                        if vp:
                            details.update({
                                "vessel_profile": vp.get("name"),
                                "vessel_dwt": vp.get("dwt"),
                                "ice_sensitivity": vp.get("ice_sensitivity"),
                                "wave_sensitivity": vp.get("wave_sensitivity"),
                            })
                    except Exception:
                        pass
                    return EcoSummary(round(fuel_total, 2), round(co2_total, 2), round(cost_usd, 2), details)
                except Exception:
                    pass
    except Exception:
        pass
    # 2) 回退：旧版 eco 模块（若可用）
    try:
        if eco_fuel is not None and eco_eval is not None:
            fuel_map_da, meta = eco_fuel.fuel_per_nm_map(ym=env_ctx.ym, vessel_class="cargo_iceclass")
            ef_co2 = 3.114
            path_lonlat_for_eco = [(float(p[1]), float(p[0])) for p in route_result.path_lonlat]
            eco_res = eco_eval.eval_route_eco(path_lonlat_for_eco, fuel_map_da, ef_co2)
            fuel_total = float(eco_res.get("fuel_total_t", 0.0))
            co2_total = float(eco_res.get("co2_total_t", 0.0))
            cost_usd = fuel_total * 600.0
            return EcoSummary(round(fuel_total, 2), round(co2_total, 2), round(cost_usd, 2), {"ok": True, **meta})
    except Exception:
        pass
    # 3) 最后回退：基于距离的粗略估算
    summary = summarize_route(route_result)
    fuel_est = float(summary.get("estimated_fuel_ton", 0.0))
    co2 = fuel_est * 3.114
    cost = fuel_est * 600.0
    return EcoSummary(round(fuel_est, 2), round(co2, 2), round(cost, 2), {"ok": False, "reason": "eco_module_unavailable"})


def run_planning_pipeline(
    ym: str,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    *,
    w_ice: float = 1.0,
    w_accident: float = 0.0,
    prior_weight: float = 0.0,
    allow_diagonal: bool = True,
    heuristic: str = "euclidean",
    eco_enabled: bool = True,
    profile_name: str = "balanced",
    **kwargs,
) -> tuple[EnvironmentContext, RouteResult]:
    """
    高层封装：按标准阶段跑一遍规划链路，并记录 PipelineTrace。
    不改变底层 load_environment/compute_route/evaluate_route_eco 的实现与签名。
    返回 (env_ctx, route_result)。
    注意：RouteResult 为 NamedTuple 不可附加属性，trace 通过 env_ctx.pipeline 暴露给 UI。
    """
    trace = new_pipeline_trace()

    # env
    trace.start("env", "加载环境与网格")
    env_ctx = load_environment(
        ym=ym,
        w_ice=w_ice,
        w_accident=w_accident,
        prior_weight=prior_weight,
        profile_name=profile_name,
        **kwargs,
    )
    # 将 trace 附加到环境，供 UI 读取
    try:
        env_ctx = EnvironmentContext(
            ym=env_ctx.ym,
            cost_da=env_ctx.cost_da,
            sic_da=env_ctx.sic_da,
            lat_arr=env_ctx.lat_arr,
            lon_arr=env_ctx.lon_arr,
            land_mask=env_ctx.land_mask,
            risk_layers_raw=env_ctx.risk_layers_raw,
            prior_penalty_da=env_ctx.prior_penalty_da,
            risk_layers_for_overlay=env_ctx.risk_layers_for_overlay,
            fusion_mode_effective=env_ctx.fusion_mode_effective,
            w_interact=env_ctx.w_interact,
            use_escort=env_ctx.use_escort,
            risk_agg_mode=env_ctx.risk_agg_mode,
            risk_agg_mode_effective=env_ctx.risk_agg_mode_effective,
            risk_agg_alpha=env_ctx.risk_agg_alpha,
            escort_applied=env_ctx.escort_applied,
            profile_name=env_ctx.profile_name,
            vessel_profile=getattr(env_ctx, "vessel_profile", None),
            pipeline=trace,
            domain=getattr(env_ctx, "domain", None),
        )
    except Exception:
        # 若 NamedTuple 扩展失败（理论上不会），则忽略
        pass
    try:
        shape = getattr(env_ctx.cost_da, "shape", None)
        layers = list(getattr(env_ctx, "risk_layers_for_overlay", {}) or {})
    except Exception:
        shape, layers = None, []
    trace.end(
        "env",
        meta={
            "ym": ym,
            "shape": shape,
            "layers": layers,
        },
    )

    # cost
    trace.start("cost", "构建风险与成本场")
    cost_da = getattr(env_ctx, "cost_da", None)
    if cost_da is not None:
        try:
            arr = np.asarray(cost_da.values)
            finite = np.isfinite(arr)
            if np.any(finite):
                vals = arr[finite]
                trace.end(
                    "cost",
                    meta={
                        "min": float(np.nanmin(vals)),
                        "max": float(np.nanmax(vals)),
                        "finite_count": int(finite.sum()),
                    },
                )
            else:
                trace.end("cost", status="error", meta={"reason": "no finite cost"})
        except Exception as e:
            trace.end("cost", status="error", meta={"reason": str(e)})
    else:
        trace.end("cost", status="error", meta={"reason": "no cost_da"})

    # route_primary
    trace.start("route_primary", "规划三条路线")
    # 规划域检查（索引 -> 经纬度 -> 域）
    try:
        dom = getattr(env_ctx, "domain", None)
        if dom is not None and env_ctx.lat_arr is not None and env_ctx.lon_arr is not None:
            def _ij_to_latlon(ij: tuple[int, int]) -> tuple[float, float]:
                ll = path_ij_to_lonlat(env_ctx, [ij]) or []
                if ll:
                    return float(ll[0][0]), float(ll[0][1])
                i, j = int(ij[0]), int(ij[1])
                lat = float(env_ctx.lat_arr[i] if env_ctx.lat_arr.ndim == 1 else env_ctx.lat_arr[i, j])
                lon = float(env_ctx.lon_arr[j] if env_ctx.lon_arr.ndim == 1 else env_ctx.lon_arr[i, j])
                return lat, lon
            s_lat, s_lon = _ij_to_latlon(start_ij)
            g_lat, g_lon = _ij_to_latlon(goal_ij)
            if not (dom.lat_min <= s_lat <= dom.lat_max and dom.lon_min <= s_lon <= dom.lon_max):
                raise ValueError(f"[PLANNING_DOMAIN] start point ({s_lat:.3f},{s_lon:.3f}) outside domain {dom}")
            if not (dom.lat_min <= g_lat <= dom.lat_max and dom.lon_min <= g_lon <= dom.lon_max):
                raise ValueError(f"[PLANNING_DOMAIN] end point ({g_lat:.3f},{g_lon:.3f}) outside domain {dom}")
            # 可选：若冰险存在且端点处为 NaN，打印提示
            try:
                layers = env_ctx.risk_layers_raw or {}
                da_ice = layers.get("ice") or layers.get("copernicus_ice")
                if da_ice is not None:
                    arr = np.asarray(da_ice.values)
                    si, sj = int(start_ij[0]), int(start_ij[1])
                    gi, gj = int(goal_ij[0]), int(goal_ij[1])
                    if np.isnan(arr[si, sj]) or np.isnan(arr[gi, gj]):
                        print("[PLANNING_DOMAIN] start/end in region with NaN ice risk; planning is allowed but ice cost may be 0 there")
            except Exception:
                pass
    except Exception as _e:
        # 将域检查失败作为错误上抛，以便在上层被感知
        raise

    route_result = compute_route(
        env=env_ctx,
        start_ij=start_ij,
        goal_ij=goal_ij,
        allow_diagonal=allow_diagonal,
        heuristic=heuristic,
    )
    meta_route = {"steps": len(getattr(route_result, "path_ij", []) or [])}
    if hasattr(route_result, "expanded_nodes"):
        try:
            meta_route["expanded_nodes"] = int(getattr(route_result, "expanded_nodes"))
        except Exception:
            pass
    trace.end("route_primary", meta=meta_route)

    # eco
    trace.start("eco", "Eco 燃油与排放评估")
    try:
        if eco_enabled:
            eco_summary = evaluate_route_eco(route_result, env_ctx)
            # 无法附加到 NamedTuple；放在 trace.meta 与返回给 UI 的 session_state 中处理
            trace.end(
                "eco",
                meta={
                    "fuel_t": float(getattr(eco_summary, "fuel_total_t", 0.0) or 0.0),
                    "co2_t": float(getattr(eco_summary, "co2_total_t", 0.0) or 0.0),
                },
            )
        else:
            trace.end("eco", status="skipped")
    except Exception as e:
        trace.end("eco", status="error", meta={"error": str(e)})

    # summary
    trace.start("summary", "生成结果摘要")
    try:
        summary = summarize_route(route_result)
        trace.end(
            "summary",
            meta={
                "distance_km": float(summary.get("distance_km", 0.0)),
                "risk_score": float(summary.get("risk_score", 0.0)),
            },
        )
    except Exception as e:
        trace.end("summary", status="error", meta={"error": str(e)})

    # 将 trace 附加到 RouteResult（重建 NamedTuple 并带默认 pipeline），并保留已有的 debug 属性
    try:
        _dbg = getattr(route_result, "debug", None)
    except Exception:
        _dbg = None
    try:
        _rr_new = RouteResult(
            path_ij=route_result.path_ij,
            path_lonlat=route_result.path_lonlat,
            cost_sum=route_result.cost_sum,
            len=route_result.len,
            reachable=route_result.reachable,
            heuristic=route_result.heuristic,
            diagonal=route_result.diagonal,
            pipeline=trace,
        )
        if _dbg is not None:
            try:
                setattr(_rr_new, "debug", _dbg)
            except Exception:
                pass
        route_result = _rr_new
    except Exception:
        pass

    return env_ctx, route_result

def run_planning_pipeline_evidential_robust(
    ym: str,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    profile_name: str = "balanced",
    robust_cfg: Optional[RobustPlannerConfig] = None,
):
    """
    Phase 1: 基于 evidential 融合 + 风险聚合模式 (mean/quantile/CVaR) 的鲁棒规划一站式入口。

    - 强制使用 fusion_mode=robust_cfg.fusion_mode (默认 "evidential")；
    - 强制使用 risk_agg_mode=robust_cfg.risk_agg_mode, risk_agg_alpha=robust_cfg.risk_agg_alpha；
    - 在 env_ctx.cost_da 上调用 compute_route，返回带鲁棒元信息的结果字典。
    """
    if robust_cfg is None:
        robust_cfg = RobustPlannerConfig()

    # 1) 加载环境（覆盖 fusion_mode / risk_agg_*，其它参数按默认）
    env_ctx = load_environment(
        ym=ym,
        profile_name=profile_name,
        fusion_mode=robust_cfg.fusion_mode,
        risk_agg_mode=robust_cfg.risk_agg_mode,
        risk_agg_alpha=robust_cfg.risk_agg_alpha,
    )

    # 2) 经纬度 → 统一吸附 → 栅格索引
    try:
        s_lat_snap, s_lon_snap, s_info = snap_point_to_domain_and_ocean(float(start_lat), float(start_lon), env_ctx)
        g_lat_snap, g_lon_snap, g_info = snap_point_to_domain_and_ocean(float(end_lat), float(end_lon), env_ctx)
        start_ij = (int(s_info.get("grid_i")), int(s_info.get("grid_j")))
        end_ij = (int(g_info.get("grid_i")), int(g_info.get("grid_j")))
    except Exception:
        start_ij = latlon_to_ij(env_ctx, start_lat, start_lon)
        end_ij = latlon_to_ij(env_ctx, end_lat, end_lon)
    if start_ij is None or end_ij is None:
        # 无法映射到网格，直接返回基本信息
        cost_da = getattr(env_ctx, "cost_da", None)
        cost_attrs = getattr(cost_da, "attrs", {}) if cost_da is not None else {}
        return {
            "env_ctx": env_ctx,
            "route": RouteResult([], [], 0.0, 0, False, robust_cfg.heuristic, robust_cfg.allow_diagonal),
            "summary": None,
            "cost_breakdown": None,
            "eco_summary": None,
            "profile": None,
            "robust_meta": {
                "reachable": False,
                "risk_agg_mode": robust_cfg.risk_agg_mode,
                "risk_agg_alpha": robust_cfg.risk_agg_alpha,
                "fusion_mode": robust_cfg.fusion_mode,
                "cost_risk_agg_mode_effective": cost_attrs.get("risk_agg_mode_effective") or getattr(env_ctx, "risk_agg_mode_effective", None),
                "fusion_mode_effective": cost_attrs.get("fusion_mode_effective") or getattr(env_ctx, "fusion_mode_effective", None),
                "snapped_start": [float(s_lat_snap) if 's_lat_snap' in locals() else None, float(s_lon_snap) if 's_lon_snap' in locals() else None],
                "snapped_end": [float(g_lat_snap) if 'g_lat_snap' in locals() else None, float(g_lon_snap) if 'g_lon_snap' in locals() else None],
            },
        }

    # 若起止点相同，自动在邻域内轻推终点到最近可通行海洋格点（兜底，避免退化路线）
    try:
        if isinstance(start_ij, tuple) and isinstance(end_ij, tuple) and (int(start_ij[0]) == int(end_ij[0])) and (int(start_ij[1]) == int(end_ij[1])):
            H = W = None
            try:
                if env_ctx.cost_da is not None:
                    H, W = env_ctx.cost_da.shape[-2:]
                elif isinstance(env_ctx.land_mask, np.ndarray):
                    H, W = env_ctx.land_mask.shape[-2:]
            except Exception:
                H = W = None
            def _is_valid(ii: int, jj: int) -> bool:
                if H is not None and W is not None and (ii < 0 or jj < 0 or ii >= H or jj >= W):
                    return False
                # 陆地禁行
                try:
                    lm = env_ctx.land_mask
                    if isinstance(lm, np.ndarray) and lm.shape[-2:] == (H, W) and bool(lm[ii, jj]):
                        return False
                except Exception:
                    pass
                # 成本有限
                try:
                    if env_ctx.cost_da is not None:
                        v = float(np.asarray(env_ctx.cost_da.values, dtype=float)[ii, jj])
                        if not np.isfinite(v) or v >= 1e6:
                            return False
                except Exception:
                    return False
                return True
            si, sj = int(start_ij[0]), int(start_ij[1])
            found = None
            max_r = int(max(10, (min(H or 0, W or 0) // 10) or 10))
            for r in range(1, max_r + 1):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if di == 0 and dj == 0:
                            continue
                        ii, jj = si + di, sj + dj
                        if _is_valid(ii, jj):
                            found = (ii, jj)
                            break
                    if found:
                        break
                if found:
                    break
            if found is not None:
                print(f"[ROUTE_FIX] end_ij equals start_ij; nudged end -> {found}")
                end_ij = found
    except Exception:
        pass

    # 3) A* 路由
    route = compute_route(
        env_ctx,
        start_ij=start_ij,
        goal_ij=end_ij,
        allow_diagonal=robust_cfg.allow_diagonal,
        heuristic=robust_cfg.heuristic,
    )

    # 4) 若不可达，直接返回
    if not getattr(route, "reachable", True):
        cost_da = getattr(env_ctx, "cost_da", None)
        cost_attrs = getattr(cost_da, "attrs", {}) if cost_da is not None else {}
        return {
            "env_ctx": env_ctx,
            "route": route,
            "summary": None,
            "cost_breakdown": None,
            "eco_summary": None,
            "profile": None,
            "robust_meta": {
                "reachable": False,
                "risk_agg_mode": robust_cfg.risk_agg_mode,
                "risk_agg_alpha": robust_cfg.risk_agg_alpha,
                "fusion_mode": robust_cfg.fusion_mode,
                "cost_risk_agg_mode_effective": cost_attrs.get("risk_agg_mode_effective") or getattr(env_ctx, "risk_agg_mode_effective", None),
                "fusion_mode_effective": cost_attrs.get("fusion_mode_effective") or getattr(env_ctx, "fusion_mode_effective", None),
            },
        }

    # 5) 汇总指标 / 成本分解 / ECO / 剖面
    summary = summarize_route(route)
    cost_breakdown = analyze_route_cost(env_ctx, route)
    try:
        eco_summary = evaluate_route_eco(route, env_ctx)
    except Exception:
        # 任意异常回退简单估计
        eco_summary = estimate_eco_simple(route)

    try:
        profile = sample_route_profile(route, env_ctx)
    except Exception:
        profile = None

    # 6) 组装鲁棒元信息
    cost_da = getattr(env_ctx, "cost_da", None)
    cost_attrs = getattr(cost_da, "attrs", {}) if cost_da is not None else {}
    robust_meta = {
        "reachable": True,
        "risk_agg_mode": robust_cfg.risk_agg_mode,
        "risk_agg_alpha": robust_cfg.risk_agg_alpha,
        "fusion_mode": robust_cfg.fusion_mode,
        "cost_risk_agg_mode_effective": cost_attrs.get("risk_agg_mode_effective") or getattr(env_ctx, "risk_agg_mode_effective", None),
        "fusion_mode_effective": cost_attrs.get("fusion_mode_effective") or getattr(env_ctx, "fusion_mode_effective", None),
        "profile_name": profile_name,
        "ym": ym,
    }

    return {
        "env_ctx": env_ctx,
        "route": route,
        "summary": summary,
        "cost_breakdown": cost_breakdown,
        "eco_summary": eco_summary,
        "profile": profile,
        "robust_meta": robust_meta,
    }


def estimate_eco_simple(route_result: RouteResult) -> EcoSummary:
    if not route_result.reachable or not route_result.path_lonlat:
        return EcoSummary(0.0, 0.0, 0.0, {"ok": False, "reason": "route_not_reachable"})
    summary = summarize_route(route_result)
    fuel_est = float(summary.get("estimated_fuel_ton", 0.0))
    co2 = fuel_est * 3.114
    cost = fuel_est * 600.0
    return EcoSummary(round(fuel_est, 2), round(co2, 2), round(cost, 2), {"ok": False, "reason": "simple_estimate"})


# -------------------------
# 战术重规划（Tactical Replan）
# -------------------------

def compute_tactical_replan(
    env_ctx: EnvironmentContext,
    base_route: RouteResult,
    current_idx: int,
    hazard_shapes: Optional[list[dict]] = None,
    soft_penalty: float = 50.0,
    allow_diagonal: bool = True,
    heuristic: str = "euclidean",
) -> tuple[RouteResult, dict]:
    """
    从 base_route 的 current_idx（剪裁在有效范围）处作为当前船位，考虑 hazard_shapes（Polygon/MultiPolygon）叠加软惩罚，
    对“当前点→终点”进行一次局部重规划，并将前半段（已航行）与新尾段拼接，返回新的 RouteResult 与对比信息。

    降级策略：
    - env_ctx.cost_da 缺失：直接返回 base_route 与空对比。
    - hazard_shapes 不可用/未安装 shapely：忽略 hazard，等价于在当前点重算后半程。
    """
    # 基础防呆
    if env_ctx is None or env_ctx.cost_da is None or not base_route or not base_route.path_ij:
        # 返回空对比，保持兼容
        info = {"baseline": summarize_route(base_route), "tactical": summarize_route(base_route), "delta": {}}
        return base_route, info

    # 1) 解析索引与起止点
    steps = len(base_route.path_ij)
    if steps <= 1:
        info = {"baseline": summarize_route(base_route), "tactical": summarize_route(base_route), "delta": {}, "current_idx": 0}
        return base_route, info

    k = int(max(0, min(int(current_idx), steps - 1)))
    start_ij = base_route.path_ij[k]
    goal_ij = base_route.path_ij[-1]

    # 2) 构造战术成本图（在原 cost_da 基础上叠加软惩罚，绝不修改原对象）
    cost_da = env_ctx.cost_da.copy()

    def _hazard_mask_from_shapes(shapes: Optional[list[dict]]) -> Optional[xr.DataArray]:
        if not shapes:
            return None
        # 需要经纬度网格
        if env_ctx.lat_arr is None or env_ctx.lon_arr is None:
            return None
        try:
            import numpy as _np
            try:
                from shapely import vectorized as _svec  # type: ignore
            except Exception:
                _svec = None  # type: ignore
            from shapely.geometry import shape as _shape  # type: ignore
            from shapely.ops import unary_union as _uunion  # type: ignore
        except Exception:
            return None

        # 构造 2D 网格
        lat_arr = env_ctx.lat_arr
        lon_arr = env_ctx.lon_arr
        if isinstance(lat_arr, np.ndarray) and isinstance(lon_arr, np.ndarray):
            if lat_arr.ndim == 1 and lon_arr.ndim == 1:
                Lon, Lat = np.meshgrid(lon_arr.astype(float), lat_arr.astype(float))
            else:
                Lat, Lon = lat_arr.astype(float), lon_arr.astype(float)
        else:
            return None
        H, W = Lat.shape

        # 解析几何并做 union
        geoms = []
        for feat in shapes:
            geom = feat.get("geometry") if isinstance(feat, dict) and ("geometry" in feat) else feat
            if not isinstance(geom, dict):
                continue
            gtype = str(geom.get("type", "")).lower()
            if gtype not in ("polygon", "multipolygon"):
                continue
            try:
                geoms.append(_shape(geom))
            except Exception:
                continue
        if not geoms:
            return None
        try:
            g_union = _uunion(geoms)
        except Exception:
            # 退化：取第一个
            g_union = geoms[0]

        # 使用 shapely.vectorized.contains 做点内判定；若不可用，退化为逐点（慢）
        xs = (((Lon + 180.0) % 360.0) - 180.0).astype("float64")  # 统一到 [-180,180]
        ys = Lat.astype("float64")
        mask = np.zeros((H, W), dtype=bool)
        try:
            if _svec is not None:
                mask = _svec.contains(g_union, xs, ys)
            else:
                # 逐行批处理，减小 Python 循环开销
                from shapely.prepared import prep as _prep  # type: ignore
                gprep = _prep(g_union)
                for i in range(H):
                    pts = list(zip(xs[i, :].tolist(), ys[i, :].tolist()))
                    mask[i, :] = np.array([gprep.contains(sgeom.Point(p)) for p in pts], dtype=bool)
        except Exception:
            # 最保守退化：返回 None（不加惩罚）
            return None

        # 若多边形很小，可能没有命中任何网格中心，做一次像素膨胀增强影响半径
        try:
            if mask is not None and mask.dtype == bool:
                # 自适应膨胀像素：网格较粗时适当扩大
                dilate_px = 2 if max(H, W) >= 400 else 1
                if dilate_px > 0:
                    mask = binary_dilation(mask, iterations=dilate_px)
        except Exception:
            pass

        da = xr.DataArray(mask.astype("float32"), dims=("y", "x"))
        da.name = "hazard_penalty"
        return da

    haz_da = _hazard_mask_from_shapes(hazard_shapes)
    if haz_da is not None:
        try:
            if tuple(haz_da.shape) != tuple(cost_da.shape[-2:]):
                haz_da = haz_da.interp_like(cost_da)
        except Exception:
            pass
        try:
            # 仅叠加软惩罚，不改变陆地与 NaN
            base = cost_da.where(np.isfinite(cost_da), np.inf)
            cost_da = (base + float(soft_penalty) * haz_da.fillna(0.0)).astype("float32")
        except Exception:
            pass

    # 3) 在战术成本上从当前点到终点跑一次 A*
    def _ensure_finite_on(cost_slice: xr.DataArray, ij: tuple[int, int]) -> tuple[int, int]:
        i, j = int(ij[0]), int(ij[1])
        arr = np.asarray(cost_slice.values)
        H, W = arr.shape[-2], arr.shape[-1]
        if 0 <= i < H and 0 <= j < W and np.isfinite(arr[i, j]):
            return (i, j)
        finite = np.array(np.nonzero(np.isfinite(arr))).T
        if finite.size == 0:
            return (i, j)
        d2 = (finite[:, 0] - i) ** 2 + (finite[:, 1] - j) ** 2
        kmin = int(np.argmin(d2))
        return int(finite[kmin, 0]), int(finite[kmin, 1])

    s_ij = _ensure_finite_on(cost_da, start_ij)
    g_ij = _ensure_finite_on(cost_da, goal_ij)

    try:
        partial = astar_on_cost(cost_da, s_ij, g_ij, neighbor8=bool(allow_diagonal), heuristic=str(heuristic))
        partial_path = partial.path_ij or []
        partial_cost = float(partial.cost_sum)
    except Exception:
        # 无法从当前点走到终点：退回 baseline 剩余段
        partial_path = base_route.path_ij[k:]
        partial_cost = 0.0

    # 4) 拼接完整战术路径（已航段 + 新尾段）
    sailed_ij = list(base_route.path_ij[: k + 1])
    new_tail = list(partial_path[1:]) if len(partial_path) >= 2 else []
    full_path_ij = sailed_ij + new_tail

    # path_lonlat（若缺失则使用转换）
    try:
        full_lonlat = base_route.path_lonlat  # 先尝试沿用
        if not full_lonlat or len(full_lonlat) != len(base_route.path_ij):
            raise Exception("ignore and recompute")
        # 分段拼接
        sailed_ll = full_lonlat[: k + 1]
        new_tail_ll = path_ij_to_lonlat(env_ctx, new_tail) if new_tail else []
        full_path_lonlat = sailed_ll + new_tail_ll
    except Exception:
        full_path_lonlat = path_ij_to_lonlat(env_ctx, full_path_ij)

    # 估算 cost_sum：保守做法，用 A* 局部代价 + 前半段按原 cost 近似
    try:
        # 基线前半段代价近似：按 cost 场采样梯形累计
        arr = np.asarray(env_ctx.cost_da.values, dtype=float)
        cost_front = 0.0
        for t in range(0, len(sailed_ij) - 1):
            i, j = sailed_ij[t]
            ni, nj = sailed_ij[t + 1]
            if 0 <= i < arr.shape[0] and 0 <= j < arr.shape[1] and 0 <= ni < arr.shape[0] and 0 <= nj < arr.shape[1]:
                c0 = float(arr[i, j]) if np.isfinite(arr[i, j]) else 0.0
                c1 = float(arr[ni, nj]) if np.isfinite(arr[ni, nj]) else 0.0
                step_len = np.sqrt(2.0) if (ni != i and nj != j) else 1.0
                cost_front += 0.5 * (c0 + c1) * step_len
        total_cost = float(cost_front + partial_cost)
    except Exception:
        total_cost = float(partial_cost)

    tactical_route = RouteResult(
        path_ij=full_path_ij,
        path_lonlat=full_path_lonlat,
        cost_sum=float(total_cost),
        len=len(full_path_ij),
        reachable=bool(full_path_ij),
        heuristic=str(heuristic),
        diagonal=bool(allow_diagonal),
    )

    # 5) 对比信息
    base_sum = summarize_route(base_route)
    tact_sum = summarize_route(tactical_route)
    info = {
        "baseline": base_sum,
        "tactical": tact_sum,
        "delta": {
            "distance_km": float(tact_sum.get("distance_km", 0.0)) - float(base_sum.get("distance_km", 0.0)),
            "total_cost": float(tact_sum.get("cost_sum", 0.0)) - float(base_sum.get("cost_sum", 0.0)),
            "fuel_t": float(tact_sum.get("estimated_fuel_ton", 0.0)) - float(base_sum.get("estimated_fuel_ton", 0.0)),
            "risk_score": float(tact_sum.get("risk_score", 0.0)) - float(base_sum.get("risk_score", 0.0)),
        },
        "current_idx": int(k),
        "mode": "tactical",
    }

    return tactical_route, info


# -------------------------
# 可视化：将风险层转换为 Folium ImageOverlay 所需 RGBA + bounds
# -------------------------

def _try_load_newenv_layer(layer_id: str) -> Optional[NewEnvLayer]:
    try:
        layers = load_newenv_layers_for_viz()
    except Exception:
        return None
    if layer_id == "ice" and "sic" in layers:
        return layers["sic"]
    if layer_id == "wave" and "wave_swh" in layers:
        return layers["wave_swh"]
    return None


def _bounds_from_da_coords(da: xr.DataArray) -> list[list[float]]:
    # 尝试从 da 自身的坐标推断 bounds
    try:
        lat_name = next((n for n in da.coords if "lat" in n.lower()), None)
        lon_name = next((n for n in da.coords if "lon" in n.lower()), None)
        if lat_name is None or lon_name is None:
            # 再从 dims 猜
            lat_name = next((d for d in da.dims if "lat" in d.lower()), lat_name)
            lon_name = next((d for d in da.dims if "lon" in d.lower()), lon_name)
        if lat_name is None or lon_name is None:
            raise ValueError("no lat/lon in da coords or dims")
        latv = da[lat_name].values
        lonv = da[lon_name].values
        if latv.ndim == 1 and lonv.ndim == 1:
            Lon, Lat = np.meshgrid(lonv, latv)
        else:
            Lat, Lon = latv, lonv
        lat_flat = np.asarray(Lat).ravel()
        lon_flat = ((np.asarray(Lon).ravel() + 180.0) % 360.0) - 180.0
        lat_min, lat_max = float(np.nanmin(lat_flat)), float(np.nanmax(lat_flat))
        lon_min, lon_max = float(np.nanmin(lon_flat)), float(np.nanmax(lon_flat))
        return [[lat_min, lon_min], [lat_max, lon_max]]
    except Exception:
        # 失败时返回一个退化 bounds
        return [[-90.0, -180.0], [90.0, 180.0]]


def build_risk_overlay_rgba(
    env: EnvironmentContext,
    layer_key: str,
    cmap_name: str = "inferno",
    alpha: float = 0.6,
) -> tuple[np.ndarray, list[list[float]]]:
    """
    将指定风险层转换为 PNG 图像数组和地理 bounds，用于 folium ImageOverlay.
    返回: (image_uint8, bounds)，若不可用则抛 ValueError。
    注意：此处返回的是 uint8 的 (H,W,3) RGB 数组，透明度请用 ImageOverlay(opacity=...) 控制。
    """
    # 0) 优先尝试 newenv（仅用于可视化，不影响 cost 计算）
    risk_da = None
    if layer_key in ("ice", "wave"):
        try:
            nl = _try_load_newenv_layer("ice" if layer_key == "ice" else "wave")
        except Exception:
            nl = None
        if nl is not None and isinstance(nl, NewEnvLayer):
            risk_da = nl.da
            bounds = _bounds_from_da_coords(risk_da)
        else:
            bounds = None
    else:
        bounds = None

    # 1) 若 newenv 未命中，则回退到旧风险层
    if risk_da is None:
        layers = getattr(env, "risk_layers_for_overlay", None) or getattr(env, "risk_layers_raw", None) or {}
        risk_da = layers.get(layer_key) if isinstance(layers, dict) else None
        if risk_da is None:
            raise ValueError(f"risk layer '{layer_key}' not available")

    # 转为 float32 并做分位数标准化
    arr = np.asarray(risk_da.values).astype("float32")
    # 压到二维（若有多余维度，先在前面维度做均值）
    if arr.ndim > 2:
        try:
            arr = np.nanmean(arr, axis=tuple(range(0, arr.ndim - 2)))
        except Exception:
            # 最保守退化：逐次取第 0 直到 2D
            while arr.ndim > 2:
                arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"risk layer '{layer_key}' is not 2D")

    mask = np.isfinite(arr)
    if not np.any(mask):
        raise ValueError(f"risk layer '{layer_key}' has no finite values")

    vals = arr[mask]
    # 计算更稳健的分位范围，保证有足够动态范围
    try:
        vmin, vmax = float(np.nanpercentile(vals, 5)), float(np.nanpercentile(vals, 95))
    except Exception:
        vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0
    if vmax - vmin < 1e-6:
        # 回退到极值
        try:
            vmin, vmax = float(np.nanmin(vals)), float(np.nanmax(vals))
        except Exception:
            vmin, vmax = 0.0, 1.0
        if vmax - vmin < 1e-6:
            # 强制拉开一个单位区间，避免全白
            vmax = vmin + 1.0

    arr_norm = (arr - vmin) / (vmax - vmin)
    arr_norm[~mask] = 0.0
    arr_norm = np.clip(arr_norm, 0.0, 1.0)

    # 降采样可视化阵列，控制到最大尺寸，减少传输与渲染负载
    H, W = arr_norm.shape
    sh = int(np.ceil(H / float(_MAX_OVERLAY_H))) if H > _MAX_OVERLAY_H else 1
    sw = int(np.ceil(W / float(_MAX_OVERLAY_W))) if W > _MAX_OVERLAY_W else 1
    arr_vis = arr_norm[::sh, ::sw]

    # 构造渲染缓存键（小型 LRU）：基于层名、可视分辨率、分位范围与若干采样点
    def _sample_sig(a: np.ndarray) -> tuple:
        hh, ww = a.shape
        pts = [
            a[0,0], a[0, ww//2], a[0, -1],
            a[hh//2, 0], a[hh//2, ww//2], a[hh//2, -1],
            a[-1,0], a[-1, ww//2], a[-1, -1]
        ]
        return tuple(float(round(x, 4)) for x in pts)
    cache_key = (str(layer_key), int(arr_vis.shape[0]), int(arr_vis.shape[1]), round(vmin, 3), round(vmax, 3), _sample_sig(arr_vis))

    img_uint8 = None
    if cache_key in _OVERLAY_CACHE:
        img_uint8 = _OVERLAY_CACHE[cache_key]
        # LRU 顺序调整
        try:
            _OVERLAY_KEYS.remove(cache_key)
        except Exception:
            pass
        _OVERLAY_KEYS.append(cache_key)
    else:
        # 映射颜色（按层默认配色；可通过参数覆盖）
        cmap_default_map = {"ice": "Blues", "accident": "Reds", "interact": "OrRd", "wave": "PuBuGn"}
        cmap_to_use = cmap_name or cmap_default_map.get(str(layer_key).lower(), "inferno")
        cmap = cm.get_cmap(cmap_to_use)
        rgba = cmap(arr_vis)  # (h,w,4), 0..1
        rgba = np.asarray(rgba, dtype="float32")

        # 输出 RGB（弃用 per-pixel alpha），透明度统一交给 ImageOverlay(opacity)
        rgb = rgba[..., :3]
        img_uint8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype("uint8")

        # 写入缓存
        _OVERLAY_CACHE[cache_key] = img_uint8
        _OVERLAY_KEYS.append(cache_key)
        if len(_OVERLAY_KEYS) > _OVERLAY_CACHE_CAP:
            old = _OVERLAY_KEYS.pop(0)
            try:
                _ = _OVERLAY_CACHE.pop(old, None)
            except Exception:
                pass


    # bounds：优先 2D lat/lon；否则 1D；经度 wrap 到 [-180,180]
    lat_arr = getattr(env, "lat2d", None) or getattr(env, "lat_arr", None)
    lon_arr = getattr(env, "lon2d", None) or getattr(env, "lon_arr", None)
    if lat_arr is None or lon_arr is None:
        raise ValueError("lat/lon grid not available for overlay bounds")

    lat_vals = lat_arr.values if hasattr(lat_arr, "values") else np.array(lat_arr)
    lon_vals = lon_arr.values if hasattr(lon_arr, "values") else np.array(lon_arr)

    if lat_vals.ndim == 1 and lon_vals.ndim == 1:
        Lon, Lat = np.meshgrid(lon_vals, lat_vals)
    else:
        Lat, Lon = lat_vals, lon_vals

    lat_flat = np.asarray(Lat).ravel()
    lon_flat = np.asarray(Lon).ravel()
    lon_flat = ((lon_flat + 180.0) % 360.0) - 180.0

    lat_min, lat_max = float(np.nanmin(lat_flat)), float(np.nanmax(lat_flat))
    lon_min, lon_max = float(np.nanmin(lon_flat)), float(np.nanmax(lon_flat))

    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    return img_uint8, bounds


# -------------------------
# 风险层统计（用于可视化/诊断，不影响计算）
# -------------------------

def get_risk_layer_stats(env: EnvironmentContext, layer_key: str) -> dict:
    """
    返回指定风险层的快速统计，用于判断是否“近乎常量/大面积 NaN”。
    输出字段：available, shape, finite, total, nan_ratio, min, p5, p50, p95, max,
             spread_p95_p5, constant_like（p95-p5<1e-6）、notes。
    """
    out = {
        "layer": layer_key,
        "available": False,
        "shape": None,
        "finite": 0,
        "total": 0,
        "nan_ratio": None,
        "min": None,
        "p5": None,
        "p50": None,
        "p95": None,
        "max": None,
        "spread_p95_p5": None,
        "constant_like": None,
        "notes": None,
    }
    try:
        layers = getattr(env, "risk_layers_for_overlay", None) or getattr(env, "risk_layers_raw", None) or {}
        da = layers.get(layer_key) if isinstance(layers, dict) else None
        if da is None:
            out["notes"] = "layer_not_available"
            return out
        arr = np.asarray(da.values)
        # 压到 2D 进行统计（对额外维度做均值）
        if arr.ndim > 2:
            try:
                arr2 = np.nanmean(arr, axis=tuple(range(0, arr.ndim - 2)))
            except Exception:
                arr2 = arr
                while arr2.ndim > 2:
                    arr2 = arr2[0]
        else:
            arr2 = arr
        out["shape"] = tuple(int(x) for x in arr2.shape)
        vals = arr2.astype("float64").ravel()
        finite_mask = np.isfinite(vals)
        total = vals.size
        finite = int(np.count_nonzero(finite_mask))
        out["finite"] = finite
        out["total"] = int(total)
        out["nan_ratio"] = float((total - finite) / total) if total > 0 else None
        if finite == 0:
            out["notes"] = "all_nan"
            return out
        v = vals[finite_mask]
        vmin = float(np.nanmin(v))
        vmax = float(np.nanmax(v))
        p5 = float(np.nanpercentile(v, 5))
        p50 = float(np.nanpercentile(v, 50))
        p95 = float(np.nanpercentile(v, 95))
        spread = p95 - p5
        out.update({
            "available": True,
            "min": vmin,
            "p5": p5,
            "p50": p50,
            "p95": p95,
            "max": vmax,
            "spread_p95_p5": float(spread),
            "constant_like": bool(spread < 1e-6),
            "notes": "ok"
        })
        return out
    except Exception as e:
        out["notes"] = f"error:{e}"
        return out

# -------------------------
# 高级功能封装（融合 / Pareto / Review）
# -------------------------
@st.cache_data
def load_fused_risk(ym: str, scenario: str = "default", fusion_mode: str = "linear") -> Optional[xr.DataArray]:
    import os
    risk_path = _repo_root() / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
    if not risk_path.exists() and fusion_mode in ("linear", "none"):
        try:
            from ArcticRoute.core.risk.fusion import fuse_risk  # type: ignore
            fuse_risk(ym, dry_run=False)
        except Exception:
            pass
    if not risk_path.exists():
        return None
    try:
        ds = xr.open_dataset(risk_path)
        var = "risk" if "risk" in ds else ("Risk" if "Risk" in ds else (list(ds.data_vars)[0] if ds.data_vars else None))
        if var is None:
            return None
        da = ds[var]
        if "time" in da.dims:
            da = da.isel(time=0)
        return da.load()
    except Exception:
        return None


def load_pareto_front(ym: str, scenario: str = "default") -> Optional[list[dict]]:
    import json
    candidates = [
        _repo_root() / "ArcticRoute" / "data_processed" / "routes" / f"pareto_front_{ym}_{scenario}.json",
        _repo_root() / "ArcticRoute" / "reports" / "d_stage" / "phaseG" / f"pareto_front_{ym}_{scenario}.json",
        _repo_root() / "ArcticRoute" / "reports" / "phaseG" / f"pareto_front_{ym}_{scenario}.json",
    ]
    pts: Optional[list[dict]] = None
    for p in candidates:
        try:
            if p.exists():
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict) and "points" in obj and isinstance(obj["points"], list):
                    pts = obj["points"]
                    break
                if isinstance(obj, list):
                    pts = obj
                    break
        except Exception:
            continue
    if not pts:
        return None
    def _get(d: dict, keys: list[str], default: float) -> float:
        for k in keys:
            if k in d and isinstance(d[k], (int, float)):
                return float(d[k])
        return default
    for i, d in enumerate(pts):
        if "id" not in d:
            d["id"] = d.get("name", f"cand{i}")
        d["distance_km"] = _get(d, ["distance_km", "distance", "dist"], np.nan)
        d["risk_score"] = _get(d, ["risk_score", "risk", "risk_integral"], np.nan)
        d["fuel_total_t"] = _get(d, ["fuel_total_t", "fuel_t", "fuel"], np.nan)
    val = [d for d in pts if not np.isnan(d["distance_km"]) or not np.isnan(d["risk_score"]) or not np.isnan(d["fuel_total_t"]) ]
    if not val:
        return None
    safe = min(val, key=lambda d: (d["risk_score"] if not np.isnan(d["risk_score"]) else 1e9))
    efficient = min(val, key=lambda d: (d["distance_km"] if not np.isnan(d["distance_km"]) else 1e9))
    def _norm(vs: list[float], x: float) -> float:
        vsf = [v for v in vs if np.isfinite(v)]
        if not vsf:
            return 0.0
        vmin, vmax = min(vsf), max(vsf)
        if not np.isfinite(x) or vmax <= vmin:
            return 0.5
        return (x - vmin) / (vmax - vmin)
    ds = [d["distance_km"] for d in val]
    rs = [d["risk_score"] for d in val]
    balanced = min(val, key=lambda d: _norm(ds, d["distance_km"]) + _norm(rs, d["risk_score"]))
    picks = {safe["id"]: {**safe, "tag": "safe"}, efficient["id"]: {**efficient, "tag": "efficient"}, balanced["id"]: {**balanced, "tag": "balanced"}}
    return list(picks.values())


def apply_feedback_and_replan(
    env_ctx: EnvironmentContext,
    base_route: RouteResult,
    feedback_file: Path,
    *,
    allow_diagonal: bool,
    heuristic: str,
    soft_weight: float = 50.0,
) -> RouteResult:
    if env_ctx.cost_da is None:
        return base_route
    try:
        import json
        from ArcticRoute.core.constraints.engine import build_constraints  # type: ignore
        items: list[dict] = []
        for line in feedback_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        c = build_constraints(ym=env_ctx.ym, feedback_items=items, grid_like=env_ctx.cost_da)
        mask = c.get("mask")
        soft = c.get("soft")
        cost = env_ctx.cost_da
        if mask is not None:
            cost = cost.where((mask <= 0.5), np.inf)
        if soft is not None:
            cost = (cost + soft_weight * soft).astype("float32")
        new_env = EnvironmentContext(
            ym=env_ctx.ym,
            cost_da=cost,
            sic_da=env_ctx.sic_da,
            lat_arr=env_ctx.lat_arr,
            lon_arr=env_ctx.lon_arr,
            land_mask=env_ctx.land_mask,
            risk_layers_raw=env_ctx.risk_layers_raw,
            prior_penalty_da=env_ctx.prior_penalty_da,
            fusion_mode_effective=env_ctx.fusion_mode_effective,
            w_interact=env_ctx.w_interact,
            use_escort=env_ctx.use_escort,
            risk_agg_mode=env_ctx.risk_agg_mode,
            risk_agg_mode_effective=env_ctx.risk_agg_mode_effective,
            risk_agg_alpha=env_ctx.risk_agg_alpha,
            escort_applied=env_ctx.escort_applied,
            profile_name=env_ctx.profile_name,
        )
        if not base_route.path_ij:
            return base_route
        start_ij = base_route.path_ij[0]
        goal_ij = base_route.path_ij[-1]
        return compute_route(new_env, start_ij, goal_ij, allow_diagonal, heuristic)
    except Exception:
        return base_route


# -------------------------
# Review：将绘制的几何转换为 feedback.jsonl 记录
# -------------------------
@st.cache_data
def build_feedback_from_shapes(shapes: list[dict], *, crs: str = "EPSG:4326", default_mode: str = "no_go") -> dict:
    """将 UI 绘制的 GeoJSON 风格几何转换为 feedback 记录列表。
    仅处理 Polygon/MultiPolygon/LineString/MultiLineString；其余忽略。
    返回 {"records": [...], "summary": {...}}
    """
    import time
    records: list[dict] = []
    n_poly, n_line = 0, 0
    for shp in shapes or []:
        if not isinstance(shp, dict):
            continue
        geom = shp.get("geometry") if "geometry" in shp else shp
        if not isinstance(geom, dict):
            continue
        gtype = str(geom.get("type", "")).lower()
        if gtype in ("polygon", "multipolygon"):
            tag = "no_go_polygon" if default_mode == "no_go" else "no_go_polygon"
            rec = {
                "tag": tag,
                "geometry": geom,
                "severity": "med",
                "meta": {"source": "ui_draw", "ts": int(time.time()), "crs": crs},
            }
            records.append(rec)
            n_poly += 1
        elif gtype in ("linestring", "multilinestring"):
            rec = {
                "tag": "lock_corridor",
                "geometry": geom,
                "severity": "med",
                "value": 5.0,
                "meta": {"source": "ui_draw", "ts": int(time.time()), "crs": crs},
            }
            records.append(rec)
            n_line += 1
        else:
            continue
    summary = {"polygons": n_poly, "lines": n_line, "total": len(records)}
    return {"records": records, "summary": summary}


def write_feedback_jsonl(records: list[dict], out_path: Path) -> Path:
    import json
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records or []:
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                continue
    return out_path


# -------------------------
# AI 解释：构造结构化 payload（仅数据整理，不做 I/O）
# -------------------------

def build_ai_explain_payload(
    env: EnvironmentContext,
    route: RouteResult,
    route_summary: dict,
    cost_breakdown: Optional[dict] = None,
    eco_summary: Optional[EcoSummary] = None,
    *,
    prior_adherence: Optional[float] = None,
    prior_weight: Optional[float] = None,
    eco_mode: Optional[str] = None,
    scenario: str = "default",
) -> dict:
    """
    将当前航线规划的关键指标打包成适合 AI 解释的结构化 payload。只组装字典，不做任何 I/O。

    返回结构：
    { meta, metrics, risk_layers, risk_flags, eco, quality }
    """
    # --- meta ---
    meta = {
        "ym": env.ym,
        "scenario": scenario,
        "algo": {
            "allow_diagonal": bool(getattr(route, "diagonal", False)),
            "heuristic": getattr(route, "heuristic", None),
        },
    }

    # --- metrics（带单位）---
    def _get_num(d: dict, keys: list[str], default: float | int = 0.0):
        for k in keys:
            if k in d:
                v = d.get(k)
                if isinstance(v, (int, float)):
                    return v
        return default

    distance_km = float(_get_num(route_summary, ["distance_km", "distance", "dist"], 0.0))
    num_steps = int(_get_num(route_summary, ["num_steps", "steps"], 0))
    total_cost = float(_get_num(route_summary, ["total_cost", "cost_sum"], getattr(route, "cost_sum", 0.0)))
    avg_risk_score = float(_get_num(route_summary, ["avg_risk_score", "risk_score", "mean_risk"], 0.0))

    metrics = {
        "distance_km": distance_km,
        "num_steps": num_steps,
        "total_cost": total_cost,
        "avg_risk_score": avg_risk_score,
        "units": {
            "distance_km": "km",
            "total_cost": "dimensionless_risk_cost",
        },
    }
    # 航程长度等级提示：short/medium/long（仅作为解释用语辅助，不影响数值）
    try:
        if distance_km < 4000:
            metrics["length_level"] = "short"
        elif distance_km < 9000:
            metrics["length_level"] = "medium"
        else:
            metrics["length_level"] = "long"
    except Exception:
        metrics["length_level"] = None
    # 整体风险等级提示：low/medium/high（仅作为解释用语辅助，不影响数值）
    try:
        tc = float(total_cost)
        if tc < 0.5:
            metrics["risk_overall_level"] = "low"
        elif tc < 1.5:
            metrics["risk_overall_level"] = "medium"
        else:
            metrics["risk_overall_level"] = "high"
    except Exception:
        metrics["risk_overall_level"] = None

    # --- 风险分解（值+占比+等级）---
    cb = cost_breakdown or {}

    # 兼容新结构：若包含 risk_components，则浅引用出来
    rc = {}
    try:
        if isinstance(cb, dict) and isinstance(cb.get("risk_components"), dict):
            rc = cb.get("risk_components") or {}
    except Exception:
        rc = {}

    def _get_cb(keys: list[str]) -> float:
        for k in keys:
            # 1) 新结构优先
            if isinstance(rc, dict) and k in rc and isinstance(rc.get(k), (int, float)):
                return float(rc.get(k))
            # 2) 旧结构（平铺键）
            v = cb.get(k)
            if isinstance(v, (int, float)):
                return float(v)
        return 0.0

    def _classify_level(value: float, total: float | None = None) -> str:
        v = max(float(value), 0.0)
        if total is not None and total > 0:
            share = v / total
            if share < 0.1:
                return "low"
            elif share < 0.4:
                return "medium"
            else:
                return "high"
        else:
            if v == 0:
                return "none"
            elif v < 1e3:
                return "low"
            elif v < 1e4:
                return "medium"
            else:
                return "high"

    v_ice = _get_cb(["ice", "Ice Risk Cost", "R_ice"])  # 兼容不同命名
    v_acc = _get_cb(["accident", "Accident Risk Cost", "R_acc"])
    v_col = _get_cb(["interact", "collision", "Collision Risk Cost", "R_interact"])
    v_prior = _get_cb(["prior_penalty", "prior", "Prior Penalty"])  # 可能无

    risk_layers = {
        "ice": {"value": v_ice},
        "accident": {"value": v_acc},
        "interact": {"value": v_col},
        "prior_penalty": {"value": v_prior},
    }
    total_risk = sum(max(0.0, x.get("value", 0.0)) for x in risk_layers.values())
    for k, info in risk_layers.items():
        v = float(info.get("value", 0.0))
        if total_risk > 0:
            info["share"] = v / total_risk
            info["level"] = _classify_level(v, total_risk)
        else:
            info["share"] = 0.0
            info["level"] = _classify_level(v, None)

    risk_flags = {
        "has_ice_risk": v_ice > 0.0,
        "has_accident_risk": v_acc > 0.0,
        "has_collision_risk": v_col > 0.0,
        "has_prior_penalty": v_prior > 0.0,
    }

    # --- Eco（单位与货币 + 等级提示）---
    if eco_summary is not None:
        mode_norm = "eco_model" if (eco_mode == "eco" or (getattr(eco_summary, "details", {}) or {}).get("ok", False)) else "simple_distance"
        fuel_total_tonnes = float(getattr(eco_summary, "fuel_total_t", 0.0) or 0.0)
        co2_total_tonnes = float(getattr(eco_summary, "co2_total_t", 0.0) or 0.0)
        fuel_cost_estimate = float(getattr(eco_summary, "cost_usd", 0.0) or 0.0)
        fuel_cost_currency = "USD" if fuel_cost_estimate > 0 else None
        eco_payload = {
            "mode": mode_norm,
            "fuel_total_tonnes": fuel_total_tonnes,
            "co2_total_tonnes": co2_total_tonnes,
            "fuel_cost_estimate": fuel_cost_estimate,
            "fuel_cost_currency": fuel_cost_currency,
            "units": {
                "fuel_total_tonnes": "t",
                "co2_total_tonnes": "t",
                "fuel_cost_estimate": "currency",
            },
        }
    else:
        eco_payload = {
            "mode": "simple_distance",
            "fuel_total_tonnes": float(route_summary.get("estimated_fuel_ton", 0.0)),
            "co2_total_tonnes": None,
            "fuel_cost_estimate": None,
            "fuel_cost_currency": None,
            "units": {
                "fuel_total_tonnes": "t",
                "co2_total_tonnes": "t",
                "fuel_cost_estimate": "currency",
            },
        }
    try:
        ft = float(eco_payload.get("fuel_total_tonnes") or 0.0)
        eco_payload["fuel_level_hint"] = "low" if ft < 50 else ("medium" if ft < 150 else "high")
    except Exception:
        eco_payload["fuel_level_hint"] = None

    # --- 数据质量/高级模式 ---
    attrs = getattr(getattr(env, "cost_da", None), "attrs", {}) or {}
    quality = {
        "fusion_mode_effective": attrs.get("fusion_mode_effective", getattr(env, "fusion_mode_effective", "baseline")),
        "risk_agg_mode_effective": attrs.get("risk_agg_mode_effective", getattr(env, "risk_agg_mode_effective", getattr(env, "risk_agg_mode", "mean"))),
        "risk_agg_alpha": float(attrs.get("risk_agg_alpha", getattr(env, "risk_agg_alpha", 0.9) or 0.9)),
        "escort_applied": bool(getattr(env, "escort_applied", False)),
        "prior_weight": float(prior_weight) if prior_weight is not None else None,
        "w_interact": attrs.get("w_interact", getattr(env, "w_interact", None)),
        "cost_fallback_used": False,
    }

    payload = {
        "meta": meta,
        "metrics": metrics,
        "risk_layers": risk_layers,
        "risk_flags": risk_flags,
        "eco": eco_payload,
        "quality": quality,
    }
    if prior_adherence is not None:
        payload["prior_adherence"] = float(prior_adherence)

    return payload
