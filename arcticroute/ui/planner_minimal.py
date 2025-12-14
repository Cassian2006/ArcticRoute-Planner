# -*- coding: utf-8 -*-
"""
极简 Streamlit UI + demo A* 集成。

Phase 3：三方案 demo Planner，支持 efficient / edl_safe / edl_robust 三种风险配置。

新增功能（Phase 4）：
- 统一 EDL 模式配置（从 arcticroute.config.edl_modes 导入）
- 场景预设下拉框（从 arcticroute.config.scenarios 导入）
- 一键对比三种模式功能
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime

from arcticroute.core.ais_analysis import evaluate_route_vs_ais_density
from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.core.landmask import (
    load_real_landmask_from_nc,
    evaluate_route_against_landmask,
)
from arcticroute.core.cost import (
    build_demo_cost,
    build_cost_from_real_env,
    list_available_ais_density_files,
    discover_ais_density_candidates,
    compute_grid_signature,
)
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.analysis import compute_route_cost_breakdown, compute_route_profile
from arcticroute.core.eco.vessel_profiles import get_default_profiles, VesselProfile
from arcticroute.core.eco.eco_model import estimate_route_eco

# 导入共享配置
from arcticroute.config import EDL_MODES, list_edl_modes
from arcticroute.core.scenarios import load_all_scenarios
from scripts.export_defense_bundle import build_defense_bundle

# 导入 Pipeline Timeline 组件
from arcticroute.ui.components import (
    Pipeline,
    PipelineStage,
    render_pipeline,
    init_pipeline_in_session,
    get_pipeline,
)

# 导入流动管线 UI 组件
from arcticroute.ui.components.pipeline_flow import (
    PipeNode,
    render_pipeline as render_pipeline_flow,
)

ROUTE_COLORS = {
    "efficient": [56, 189, 248],
    "edl_safe": [251, 146, 60],
    "edl_robust": [248, 113, 113],
}

ROUTE_LABELS_ZH = {
    "efficient": "效率优先",
    "edl_safe": "风险均衡",
    "edl_robust": "稳健安全",
}

# ============================================================================
# 北极固定视角 + 地图控制器配置
# ============================================================================
ARCTIC_VIEW = {
    "latitude": 75.0,
    "longitude": 30.0,
    "zoom": 2.6,
    "min_zoom": 2.2,
    "max_zoom": 6.0,
    "pitch": 0,
}

# 地图控制器配置（禁止拖动，允许滚轮缩放）
MAP_CONTROLLER = {
    "dragPan": False,
    "dragRotate": False,
    "scrollZoom": True,
    "doubleClickZoom": True,
    "touchZoom": True,
    "keyboard": False,
}


# ============================================================================
# 辅助函数：将 EDL_MODES 转换为 ROUTE_PROFILES 格式
# ============================================================================

def build_route_profiles_from_edl_modes() -> list[dict]:
    """
    从共享的 EDL_MODES 配置构建 ROUTE_PROFILES。
    
    Returns:
        ROUTE_PROFILES 列表
    """
    profiles = []
    for mode_key in ["efficient", "edl_safe", "edl_robust"]:
        mode_config = EDL_MODES.get(mode_key)
        if mode_config is None:
            continue
        
        profiles.append({
            "key": mode_key,
            "label": mode_config.get("display_name", mode_config.get("name", mode_key)),
            "ice_penalty_factor": mode_config.get("ice_penalty_factor", 1.0),
            "wave_weight_factor": mode_config.get("wave_weight_factor", 1.0),
            "edl_weight_factor": mode_config.get("edl_weight_factor", 1.0),
            "use_edl_uncertainty": mode_config.get("use_edl_uncertainty", False),
            "edl_uncertainty_weight": mode_config.get("edl_uncertainty_weight", 0.0),
            # 从配置中提取其他参数
            "w_edl": mode_config.get("w_edl", 0.0),
            "use_edl": mode_config.get("use_edl", False),
            "ice_penalty": mode_config.get("ice_penalty", 4.0),
            "wave_penalty": mode_config.get("wave_penalty", 0.0),
        })
    
    return profiles


# 从共享配置构建 ROUTE_PROFILES
ROUTE_PROFILES = build_route_profiles_from_edl_modes()


@dataclass
class RouteInfo:
    """单条路线信息数据类。"""

    mode: str
    label: str
    reachable: bool
    path_lonlat: list[tuple[float, float]]
    distance_km: float | None
    total_cost: float | None
    cost_components: Dict[str, float]
    edl_uncertainty_profile: list[float] | None
    vessel: str | None
    notes: dict[str, Any] = field(default_factory=dict)
    steps: int | None = None
    approx_length_km: float | None = None
    ice_penalty: float = 0.0
    allow_diag: bool = True
    on_land_steps: int = 0
    on_ocean_steps: int = 0
    travel_time_h: float = 0.0
    fuel_total_t: float = 0.0
    co2_total_t: float = 0.0
    coords: list[tuple[float, float]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # 有同步的调用在开发中依赖 route_info.coords，保留同步输出
        self.coords = self.path_lonlat


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间的大圆距离（单位：km）。
    
    Args:
        lat1, lon1: 起点纬度、经度（度）
        lat2, lon2: 终点纬度、经度（度）
    
    Returns:
        距离（km）
    """
    R = 6371.0  # 地球平均半径
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def compute_path_length_km(path: list[tuple[float, float]]) -> float:
    """
    计算路径的总长度（单位：km）。
    
    Args:
        path: [(lat, lon), ...] 路径列表
    
    Returns:
        总长度（km）
    """
    if len(path) < 2:
        return 0.0
    
    total_dist = 0.0
    for (lat1, lon1), (lat2, lon2) in zip(path[:-1], path[1:]):
        total_dist += haversine_km(lat1, lon1, lat2, lon2)
    
    return total_dist


def _wrap_lon(lon: float) -> float:
    """
    将经度归一化到 [-180, 180] 范围内。
    
    Args:
        lon: 原始经度
    
    Returns:
        归一化后的经度
    """
    if lon > 180.0:
        return lon - 360.0
    if lon < -180.0:
        return lon + 360.0
    return lon


def _is_valid_coord(lat: float, lon: float) -> bool:
    """
    检查坐标是否有效（非 NaN、非 inf）。
    
    Args:
        lat: 纬度
        lon: 经度
    
    Returns:
        True 如果坐标有效
    """
    try:
        lat_f = float(lat)
        lon_f = float(lon)
        return np.isfinite(lat_f) and np.isfinite(lon_f)
    except (TypeError, ValueError):
        return False


def _update_pipeline_node(
    idx: int,
    status: str,
    detail: str = "",
    seconds: float | None = None,
) -> None:
    """
    更新流动管线中的节点状态并重新渲染。
    
    Args:
        idx: 节点索引（0-7）
        status: 节点状态 ("pending" | "running" | "done" | "fail")
        detail: 节点详情文本
        seconds: 节点耗时（秒）
    """
    if "pipeline_flow_nodes" not in st.session_state:
        return
    
    nodes = st.session_state.pipeline_flow_nodes
    if idx < 0 or idx >= len(nodes):
        return
    
    # 更新节点
    nodes[idx].status = status
    nodes[idx].detail = detail
    if seconds is not None:
        nodes[idx].seconds = seconds
    
    # 重新渲染管线
    if "pipeline_flow_placeholder" in st.session_state:
        try:
            st.session_state.pipeline_flow_placeholder.empty()
            with st.session_state.pipeline_flow_placeholder.container():
                render_pipeline_flow(
                    nodes,
                    title="🔄 规划流程管线",
                    expanded=st.session_state.get("pipeline_flow_expanded", True),
                )
        except Exception:
            # 如果更新失败，忽略（可能是 placeholder 已被销毁）
            pass


def plan_three_routes(
    grid,
    land_mask,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    allow_diag: bool = True,
    vessel: VesselProfile | None = None,
    cost_mode: str = "demo_icebelt",
    wave_penalty: float = 0.0,
    use_edl: bool = False,
    w_edl: float = 0.0,
    weight_risk: float = 0.33,
    weight_uncertainty: float = 0.33,
    weight_fuel: float = 0.34,
    edl_uncertainty_weight: float = 0.0,
    ais_density: np.ndarray | None = None,
    ais_weight: float = 0.0,
    ais_density_path: Path | None = None,
    ais_density_da=None,
    w_ais_corridor: float = 0.0,
    w_ais_congestion: float = 0.0,
    w_ais: float | None = None,
) -> tuple[dict[str, RouteInfo], dict, dict, dict, str]:
    """
    规划三条路线：efficient / edl_safe / edl_robust（使用 ROUTE_PROFILES 定义的个性化权重）。
    
    Args:
        grid: Grid2D 对象
        land_mask: 陆地掩码数组
        start_lat, start_lon: 起点经纬度
        end_lat, end_lon: 终点经纬度
        allow_diag: 是否允许对角线移动
        vessel: VesselProfile 对象，用于 ECO 估算；若为 None，则不计算 ECO
        cost_mode: 成本模式，"demo_icebelt" 或 "real_sic_if_available"
        wave_penalty: 波浪风险权重（默认 0.0，仅在 real_sic_if_available 模式下有效）
        use_edl: 是否启用 EDL 风险头（默认 False）
        w_edl: EDL 风险权重（默认 0.0，仅在 use_edl=True 且 cost_mode="real_sic_if_available" 时有效）
        weight_risk: EDL 风险权重（用于综合评分）
        weight_uncertainty: EDL 不确定性权重（用于综合评分）
        weight_fuel: 燃油权重（用于综合评分）
    
    Returns:
        (RouteInfo 列表, cost_fields 字典, meta 字典, scores_by_key 字典, recommended_key 字符串)
        其中 cost_fields 为 {profile_key: CostField}
        scores_by_key 为 {profile_key: RouteScore}
        recommended_key 为综合评分最低的路线 key
        meta 包含 {"cost_mode": str, "real_env_available": bool, "fallback_reason": str or None, ...}
    """
    # ========================================================================
    # Step 3: 使用 ROUTE_PROFILES 定义三种个性化方案
    # ========================================================================
    
    routes_info: dict[str, RouteInfo] = {}
    cost_fields = {}
    meta = {
        "cost_mode": cost_mode,
        "real_env_available": False,
        "fallback_reason": None,
        "wave_penalty": wave_penalty,
        "use_edl": bool(use_edl),
        "w_edl": float(w_edl if use_edl else 0.0),
    }
    w_ais_effective = w_ais if w_ais is not None else ais_weight
    
    # 根据 cost_mode 决定是否加载真实环境数据
    real_env = None
    if cost_mode == "real_sic_if_available":
        try:
            real_env = load_real_env_for_grid(grid)
            if real_env is not None and (real_env.sic is not None or real_env.wave_swh is not None):
                meta["real_env_available"] = True
            else:
                meta["fallback_reason"] = "真实环境数据不可用"
        except Exception as e:
            print(f"[WARN] Failed to load real environment data: {e}, falling back to demo cost")
            meta["fallback_reason"] = f"加载真实环境数据失败: {e}"
            real_env = None
    
    # 遍历 ROUTE_PROFILES，为每个方案构建成本场并规划路线
    for profile in ROUTE_PROFILES:
        profile_key = profile["key"]
        profile_label = profile["label"]
        
        # 根据 profile 计算实际的权重参数
        # 基础权重（来自 UI 的全局参数）
        base_ice_penalty = 4.0  # 默认基础冰风险权重
        base_wave_penalty = wave_penalty
        base_w_edl = w_edl if use_edl else 0.0
        
        # 应用 profile 的倍率因子
        actual_ice_penalty = base_ice_penalty * profile["ice_penalty_factor"]
        actual_wave_penalty = base_wave_penalty * profile["wave_weight_factor"]
        actual_w_edl = base_w_edl * profile["edl_weight_factor"]
        use_edl_uncertainty = profile["use_edl_uncertainty"]
        edl_uncertainty_weight = profile["edl_uncertainty_weight"]
        
        # 构建成本场
        try:
            if cost_mode == "real_sic_if_available" and real_env is not None and (real_env.sic is not None or real_env.wave_swh is not None):
                # 使用真实环境成本（包括 sic 和/或 wave 以及可选的 EDL 和 AIS）
                try:
                    cost_field = build_cost_from_real_env(
                        grid, 
                        land_mask, 
                        real_env, 
                        ice_penalty=actual_ice_penalty, 
                        wave_penalty=actual_wave_penalty, 
                        vessel_profile=vessel,
                        use_edl=use_edl, 
                        w_edl=actual_w_edl,
                        use_edl_uncertainty=use_edl_uncertainty,
                        edl_uncertainty_weight=edl_uncertainty_weight,
                        ais_density=ais_density,
                        ais_density_path=ais_density_path,
                        ais_weight=w_ais_effective,
                        ais_density_da=ais_density_da,
                        w_ais_corridor=w_ais_corridor,
                        w_ais_congestion=w_ais_congestion,
                        w_ais=w_ais_effective,
                    )
                except Exception as e:
                    print(f"[WARN] Failed to build cost from real env for {profile_key}: {e}, falling back to demo cost")
                    cost_field = build_demo_cost(
                        grid,
                        land_mask,
                        ice_penalty=actual_ice_penalty,
                        ice_lat_threshold=75.0,
                        w_ais=w_ais_effective,
                        ais_density=ais_density,
                        ais_density_path=ais_density_path,
                        w_ais_corridor=w_ais_corridor,
                        w_ais_congestion=w_ais_congestion,
                    )
            else:
                # 回退到 demo 冰带成本（demo 模式下不启用 EDL）
                cost_field = build_demo_cost(
                    grid,
                    land_mask,
                    ice_penalty=actual_ice_penalty,
                    ice_lat_threshold=75.0,
                    w_ais=w_ais_effective,
                    ais_density=ais_density,
                    ais_density_path=ais_density_path,
                    w_ais_corridor=w_ais_corridor,
                    w_ais_congestion=w_ais_congestion,
                )
            cost_fields[profile_key] = cost_field
        except Exception as e:
            print(f"[ERROR] Unexpected error building cost field for {profile_key}: {e}")
            # 最后的防线：使用 demo 成本
            cost_field = build_demo_cost(
                grid,
                land_mask,
                ice_penalty=actual_ice_penalty,
                ice_lat_threshold=75.0,
                w_ais=w_ais_effective,
                ais_density=ais_density,
                ais_density_path=ais_density_path,
                w_ais_corridor=w_ais_corridor,
                w_ais_congestion=w_ais_congestion,
            )
            cost_fields[profile_key] = cost_field
        
        # 规划路线
        path = plan_route_latlon(
            cost_field,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            neighbor8=allow_diag,
        )
        
        # 构造 RouteInfo
        if path:
            # 计算陆地踩踏统计
            stats = evaluate_route_against_landmask(grid, land_mask, path)
            
            # 计算 ECO 指标
            eco_estimate = None
            if vessel is not None:
                eco_estimate = estimate_route_eco(path, vessel)
            
            route_info = RouteInfo(
                mode=profile_key,
                label=profile_label,
                reachable=True,
                path_lonlat=path,
                distance_km=eco_estimate.distance_km if eco_estimate else compute_path_length_km(path),
                total_cost=None,
                cost_components={},
                edl_uncertainty_profile=None,
                vessel=vessel.name if vessel is not None else None,
                steps=len(path),
                approx_length_km=compute_path_length_km(path),
                ice_penalty=actual_ice_penalty,
                allow_diag=allow_diag,
                on_land_steps=stats.on_land_steps,
                on_ocean_steps=stats.on_ocean_steps,
                travel_time_h=eco_estimate.travel_time_h if eco_estimate else 0.0,
                fuel_total_t=eco_estimate.fuel_total_t if eco_estimate else 0.0,
                co2_total_t=eco_estimate.co2_total_t if eco_estimate else 0.0,
            )
        else:
            route_info = RouteInfo(
                mode=profile_key,
                label=profile_label,
                reachable=False,
                path_lonlat=[],
                distance_km=None,
                total_cost=None,
                cost_components={},
                edl_uncertainty_profile=None,
                vessel=vessel.name if vessel is not None else None,
                steps=None,
                approx_length_km=None,
                ice_penalty=actual_ice_penalty,
                allow_diag=allow_diag,
                on_land_steps=0,
                on_ocean_steps=0,
                travel_time_h=0.0,
                fuel_total_t=0.0,
                co2_total_t=0.0,
            )
        
        routes_info[profile_key] = route_info
    
    # ========================================================================
    # Step 3: 收集打分所需的数据，调用 compute_route_scores
    # ========================================================================
    from arcticroute.core.analysis import compute_route_cost_breakdown, compute_route_scores
    
    # ?? breakdowns_by_key ? eco_by_key
    breakdowns_by_key = {}
    eco_by_key = {}
    
    for profile in ROUTE_PROFILES:
        profile_key = profile['key']
        route_info = routes_info.get(profile_key)
        if route_info is None:
            continue
        
        if route_info.reachable:
            cost_field = cost_fields.get(profile_key)
            if cost_field is not None:
                breakdown = compute_route_cost_breakdown(
                    grid, cost_field, route_info.coords
                )
                breakdowns_by_key[profile_key] = breakdown
                route_info.total_cost = breakdown.total_cost
                route_info.cost_components = breakdown.component_totals or {}
                route_info.notes["breakdown"] = breakdown
                try:
                    profile_result = compute_route_profile(route_info.coords, cost_field)
                    if profile_result.edl_uncertainty is not None:
                        route_info.edl_uncertainty_profile = profile_result.edl_uncertainty.tolist()
                    route_info.notes['profile_result'] = profile_result
                except Exception:
                    route_info.notes['profile_result'] = None
            else:
                from arcticroute.core.analysis import RouteCostBreakdown
                breakdowns_by_key[profile_key] = RouteCostBreakdown(
                    total_cost=0.0,
                    component_totals={},
                    component_fractions={},
                    s_km=[],
                    component_along_path={},
                )
            
            eco_by_key[profile_key] = {
                'fuel_total_t': route_info.fuel_total_t if route_info.fuel_total_t > 0 else None,
                'co2_total_t': route_info.co2_total_t if route_info.co2_total_t > 0 else None,
            }
        else:
            from arcticroute.core.analysis import RouteCostBreakdown
            breakdowns_by_key[profile_key] = RouteCostBreakdown(
                total_cost=0.0,
                component_totals={},
                component_fractions={},
                s_km=[],
                component_along_path={},
            )
            eco_by_key[profile_key] = None

    # 调用 compute_route_scores
    scores_by_key = compute_route_scores(
        breakdowns=breakdowns_by_key,
        eco_by_key=eco_by_key,
        weight_risk=weight_risk,
        weight_uncertainty=weight_uncertainty,
        weight_fuel=weight_fuel,
    )
    
    # 找出综合分数最低的路线（最优）
    recommended_key = min(
        scores_by_key.items(),
        key=lambda kv: kv[1].composite_score,
    )[0]
    
    # ========================================================================
    # DEBUG: 打印路线信息，用于诊断路线不显示问题
    # ========================================================================
    print("\n[DEBUG ROUTES] ===== Route Planning Complete =====")
    for i, (key, r) in enumerate(routes_info.items()):
        try:
            coords = r.coords or []
            print(
                f"[DEBUG ROUTE] #{i} key={key} label={r.label} reachable={r.reachable} "
                f"n_points={len(coords)} "
                f"first={coords[0] if coords else None} "
                f"last={coords[-1] if coords else None}"
            )
        except Exception as e:
            print(f"[DEBUG ROUTE] #{i} error while inspecting route {getattr(r, 'label', '?')}: {e}")
    print("[DEBUG ROUTES] ===== End Route Planning =====\n")
    
    return routes_info, cost_fields, meta, scores_by_key, recommended_key


def render() -> None:
    """
    渲染三方案规划器 UI。
    
    包含：
    - 标题与说明
    - 左侧参数输入（起终点坐标、对角线开关、规划按钮）
    - 主区域：多路径可视化 + 摘要表格
    """
    if not st.session_state.get("_ar_page_config_set"):
        st.set_page_config(
            page_title="ArcticRoute Planner",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_ar_page_config_set"] = True
    
    st.title("ArcticRoute 航线规划驾驶舱")
    st.caption("基于多模态环境场（冰 / 浪 / AIS / 冰级 / EDL）的北极航线智能规划系统")
    
    st.info(
        "当前使用的是 demo 网格和 demo landmask（非真实海陆分布），"
        "在真实底图上看起来可能会压到陆地。"
    )

    ais_density_path: Path | None = None

    # 左侧栏参数输入
    with st.sidebar:
        status_box = st.container()
        st.header("规划参数")
        
        # ====================================================================
        st.subheader("场景与环境")
        scenarios_map = {}
        scenario_load_error = None
        try:
            scenarios_map = load_all_scenarios()
        except Exception as e:
            scenario_load_error = str(e)

        scenario_options = ["manual"] + list(scenarios_map.keys())
        default_scenario_id = st.session_state.get("selected_scenario_id", "manual")
        if default_scenario_id not in scenario_options:
            default_scenario_id = "manual"

        selected_scenario_id = st.selectbox(
            "选择预设场景",
            options=scenario_options,
            index=scenario_options.index(default_scenario_id),
            format_func=lambda sid: "manual（自定义）" if sid == "manual" else f"{sid} - {scenarios_map[sid].title}",
        )
        selected_scenario_name = selected_scenario_id
        st.session_state["selected_scenario_id"] = selected_scenario_id

        if scenario_load_error:
            st.warning(f"预设场景加载失败: {scenario_load_error}")

        if selected_scenario_id != "manual" and selected_scenario_id in scenarios_map:
            scen = scenarios_map[selected_scenario_id]
            st.session_state["start_lat"] = scen.start_lat
            st.session_state["start_lon"] = scen.start_lon
            st.session_state["end_lat"] = scen.end_lat
            st.session_state["end_lon"] = scen.end_lon
            st.session_state["w_ice"] = scen.w_ice
            st.session_state["vessel_profile"] = scen.vessel
            st.session_state["selected_edl_mode"] = scen.base_profile
            st.session_state["wave_penalty"] = scen.w_wave
            st.session_state["w_ais"] = scen.w_ais
            st.session_state["w_ais_corridor"] = getattr(scen, "w_ais_corridor", scen.w_ais)
            st.session_state["w_ais_congestion"] = getattr(scen, "w_ais_congestion", 0.0)
            st.session_state["use_edl_pref"] = scen.use_edl
            st.session_state["use_edl_uncertainty_pref"] = scen.use_edl_uncertainty
            st.session_state["ym"] = scen.ym
            st.session_state["grid_mode_pref"] = scen.grid_mode
            st.caption(f"{scen.description} | ym={scen.ym}")
        else:
            st.caption("手动输入起终点和权重参数")

        # ??????????
        start_lat_default = st.session_state.get("start_lat", 66.0)
        start_lon_default = st.session_state.get("start_lon", 5.0)
        end_lat_default = st.session_state.get("end_lat", 78.0)
        end_lon_default = st.session_state.get("end_lon", 150.0)
        grid_mode_pref = st.session_state.get("grid_mode_pref", "demo")

        st.subheader("网格模式")
        grid_mode_options = ["demo", "real"]
        grid_mode_default = grid_mode_pref if grid_mode_pref in grid_mode_options else "demo"
        grid_mode = st.radio(
            "栅格模式",
            options=grid_mode_options,
            index=grid_mode_options.index(grid_mode_default),
            format_func=lambda s: "演示 (demo)" if s == "demo" else "真实数据",
            horizontal=True,
        )
        st.session_state["grid_mode_pref"] = grid_mode

        # ====================================================================
        # 任务 C：Grid Signature 计算与 Session State 隔离
        # ====================================================================
        # 计算当前网格的签名
        try:
            if grid_mode == "demo":
                current_grid, _ = make_demo_grid()
            else:
                # 对于真实网格，需要先加载（这里假设已有 ym 参数）
                ym = st.session_state.get("ym", "202401")
                current_grid = load_real_grid_from_nc(ym=ym)
            
            current_grid_signature = compute_grid_signature(current_grid)
            
            # 检查 grid_signature 是否发生变化
            prev_grid_signature = st.session_state.get("grid_signature", None)
            if prev_grid_signature != current_grid_signature:
                # Grid 发生变化，清空 AIS 相关的 session_state
                st.session_state["grid_signature"] = current_grid_signature
                st.session_state["ais_density_path_selected"] = None
                st.session_state["ais_density_cache_key"] = None
                print(f"[UI] Grid signature changed: {prev_grid_signature} -> {current_grid_signature}")
            else:
                st.session_state["grid_signature"] = current_grid_signature
        except Exception as e:
            print(f"[UI] warning: failed to compute grid signature: {e}")
            st.session_state["grid_signature"] = None

        cost_mode_options = ["demo_icebelt", "real_sic_if_available"]
        cost_mode_default = "real_sic_if_available" if grid_mode == "real" else "demo_icebelt"
        cost_mode = st.selectbox(
            "成本模式",
            options=cost_mode_options,
            index=cost_mode_options.index(cost_mode_default),
            format_func=lambda s: "演示成本 (demo_icebelt)" if s == "demo_icebelt" else "真实 SIC / 波浪（可用则启用）",
        )
        st.subheader("起点")
        start_lat = st.number_input(
            "起点纬度",
            min_value=60.0,
            max_value=85.0,
            value=start_lat_default,
            step=0.1,
        )
        start_lon = st.number_input(
            "起点经度",
            min_value=-180.0,
            max_value=180.0,
            value=start_lon_default,
            step=0.1,
        )
        
        st.subheader("终点")
        end_lat = st.number_input(
            "终点纬度",
            min_value=60.0,
            max_value=85.0,
            value=end_lat_default,
            step=0.1,
        )
        end_lon = st.number_input(
            "终点经度",
            min_value=-180.0,
            max_value=180.0,
            value=end_lon_default,
            step=0.1,
        )
        
        st.subheader("寻路配置")
        allow_diag = st.checkbox("允许对角线移动 (8 邻接)", value=True)
        
        st.subheader("风险权重")
        wave_penalty = st.slider(
            "波浪权重 (wave_penalty)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("wave_penalty", 2.0)),
            step=0.5,
            help="仅在成本模式为真实环境数据时有影响；若缺少 wave 数据则自动退回为 0。",
        )
        st.session_state["wave_penalty"] = wave_penalty

        st.subheader("AIS 成本权重")
        default_corridor = float(st.session_state.get("w_ais_corridor", 2.0))
        default_congestion = float(st.session_state.get("w_ais_congestion", 1.0))
        col_corr, col_cong = st.columns(2)
        with col_corr:
            w_ais_corridor = st.slider(
                "AIS 主航线偏好 (w_corridor)",
                min_value=0.0,
                max_value=10.0,
                value=default_corridor,
                step=0.5,
                help="Corridor：越接近高密度主航线，成本越低。",
            )
        with col_cong:
            w_ais_congestion = st.slider(
                "AIS 拥挤惩罚 (w_congestion)",
                min_value=0.0,
                max_value=10.0,
                value=default_congestion,
                step=0.5,
                help="Congestion：仅对密度高分位区域（如 P90+）施加惩罚。",
            )
        st.session_state["w_ais_corridor"] = w_ais_corridor
        st.session_state["w_ais_congestion"] = w_ais_congestion
        st.caption("Corridor：偏好成熟航道 | Congestion：避开极端拥挤")

        with st.expander("旧版 AIS 权重 (w_ais, deprecated)", expanded=False):
            w_ais = st.slider(
                "AIS 旧版权重 w_ais",
                min_value=0.0,
                max_value=10.0,
                value=float(st.session_state.get("w_ais", 0.0)),
                step=0.1,
                help="向后兼容参数，若新权重均为 0，会自动映射为 corridor。",
            )
            st.caption("建议使用上方 corridor/congestion 权重，新项目不再直接使用 w_ais。")
        st.session_state["w_ais"] = w_ais
        ais_weights_enabled = any(weight > 0 for weight in [w_ais, w_ais_corridor, w_ais_congestion])

        # ====================================================================
        # 任务 C1：网格变化检测 - 自动清空旧 AIS 选择
        # ====================================================================
        # 检查网格是否发生变化，若变化则清空 AIS 密度选择以避免维度错配
        previous_grid_signature = st.session_state.get("previous_grid_signature", None)
        current_grid_signature = st.session_state.get("grid_signature", None)
        
        if (previous_grid_signature is not None and 
            current_grid_signature is not None and 
            previous_grid_signature != current_grid_signature):
            # 网格已切换，清空 AIS 密度选择
            st.session_state["ais_density_path"] = None
            st.session_state["ais_density_path_selected"] = None
            st.session_state["ais_density_cache_key"] = None
            st.info(f"🔄 网格已切换（{previous_grid_signature[:25]}... → {current_grid_signature[:25]}...），已清空 AIS 密度选择以避免维度错配")
            print(f"[UI] Grid changed, cleared AIS selection: {previous_grid_signature[:30]}... -> {current_grid_signature[:30]}...")
        
        # 更新当前网格 signature
        if current_grid_signature is not None:
            st.session_state["previous_grid_signature"] = current_grid_signature

        # ====================================================================
        # 任务 C1：按 grid_signature 优先选择 AIS 密度文件
        # ====================================================================
        # 自动发现 AIS 密度候选文件（按 grid_signature 优先级排序）
        grid_sig = st.session_state.get("grid_signature", None)
        ais_candidates = discover_ais_density_candidates(grid_signature=grid_sig)
        
        ais_options = ["自动选择 (推荐)"]
        ais_path_map = {"自动选择 (推荐)": None}
        ais_match_type_map = {"自动选择 (推荐)": "auto"}
        
        for cand in ais_candidates:
            label = cand["label"]
            match_type = cand.get("match_type", "generic")
            
            # 在标签中显示匹配类型
            if match_type == "exact":
                label_with_type = f"{label} ✓ (精确匹配)"
            elif match_type == "demo":
                label_with_type = f"{label} (演示)"
            else:
                label_with_type = f"{label}"
            
            ais_options.append(label_with_type)
            ais_path_map[label_with_type] = cand["path"]
            ais_match_type_map[label_with_type] = match_type

        ais_choice = st.selectbox(
            "AIS 密度数据源 (.nc)",
            options=ais_options,
            help="自动：在 data_real/ais/density 与 data_real/ais/derived 中自动选取；也可手动固定某个 .nc 文件。",
        )

        ais_density_path = ais_path_map.get(ais_choice)  # 可能是 None
        ais_match_type = ais_match_type_map.get(ais_choice, "unknown")

        


        # ====================================================================
        # ====================================================================  
        # 任务 C：AIS 密度数据检查提示 + 重新扫描按钮
        # ====================================================================  
        ais_data_available = False
        if not ais_weights_enabled:
            st.info("AIS 未启用（corridor/congestion/legacy 权重均为 0）。")
        else:
            # 检查 AIS 密度数据是否可用（根据 NC 文件）
            ais_data_available = False
            ais_status_text = ""
            ais_status_color = "gray"
            ais_file_info = ""

            try:
                from arcticroute.core import cost as cost_core
                from pathlib import Path

                # 根据网格模式选择优先级
                prefer_real = (grid_mode == "real")
                detected_ais_density = cost_core.has_ais_density_data(grid=None, prefer_real=prefer_real)

                if ais_density_path is not None:
                    ais_data_available = True
                    # 确保 ais_density_path 是 Path 对象
                    ais_path_obj = Path(ais_density_path) if isinstance(ais_density_path, str) else ais_density_path
                    
                    # 构建详细的状态信息
                    match_label = ""
                    if ais_match_type == "exact":
                        match_label = "[精确匹配]"
                    elif ais_match_type == "demo":
                        match_label = "[演示文件]"
                    else:
                        match_label = "[通用]"
                    
                    ais_file_info = f"{ais_path_obj.name} {match_label}"
                    ais_status_text = f"✅ AIS density: {ais_file_info}"
                    ais_status_color = "green"
                elif detected_ais_density:
                    ais_data_available = True
                    ais_status_text = "✅ 已检测到 AIS 拥挤度密度数据（自动选择模式）"
                    ais_status_color = "green"
                else:
                    ais_status_text = "⚠ 未找到匹配当前网格的 AIS density，需启用权重时再检查或运行生成脚本"
                    ais_status_color = "orange"
            except Exception as e:
                ais_status_text = f"⚠ AIS 数据检查失败: {str(e)[:60]}"
                ais_status_color = "orange"

            # 显示 AIS 状态（左侧栏）
            st.markdown("**AIS 密度状态**")
            if ais_status_color == "green":
                st.success(ais_status_text)
            elif ais_status_color == "orange":
                st.warning(ais_status_text)
            else:
                st.info(ais_status_text)
            
            # 添加重新扫描按钮
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 重新扫描 AIS", key="rescan_ais_btn"):
                    # 清空 AIS 相关的缓存
                    st.session_state["ais_density_path_selected"] = None
                    st.session_state["ais_density_cache_key"] = None
                    st.rerun()
            
            with col2:
                if st.button("ℹ️ 网格信息", key="grid_info_btn"):
                    st.info(f"当前网格签名: {st.session_state.get('grid_signature', 'N/A')}")

        
        # ====================================================================
        # Phase 4: 规划风格下拉框（统一 EDL 模式）
        # ====================================================================
        st.subheader("规划风格")
        edl_modes = list_edl_modes()
        selected_edl_mode_default = st.session_state.get("selected_edl_mode", edl_modes[0] if edl_modes else None)
        if selected_edl_mode_default not in edl_modes:
            selected_edl_mode_default = edl_modes[0]
        selected_edl_mode = st.selectbox(
            "选择规划风格",
            options=edl_modes,
            index=edl_modes.index(selected_edl_mode_default),
            format_func=lambda m: EDL_MODES[m].get("display_name", m),
            help="选择不同的规划风格会自动调整 EDL 权重、不确定性权重等参数。",
        )
        st.session_state["selected_edl_mode"] = selected_edl_mode
        
        # 从选定的 EDL 模式获取参数
        edl_mode_config = EDL_MODES.get(selected_edl_mode, {})
        use_edl = edl_mode_config.get("use_edl", False)
        w_edl = edl_mode_config.get("w_edl", 0.0)
        use_edl_uncertainty = edl_mode_config.get("use_edl_uncertainty", False)
        edl_uncertainty_weight = edl_mode_config.get("edl_uncertainty_weight", 0.0)
        
        # 显示当前模式的参数信息
        st.caption(
            f"当前模式参数：w_edl={w_edl:.1f}, "
            f"use_uncertainty={use_edl_uncertainty}, "
            f"unc_weight={edl_uncertainty_weight:.1f}"
        )
        
        # ====================================================================
        # Step 2: 路线偏好滑条（多目标权重）
        # ====================================================================
        st.subheader("路线偏好（多目标权重）")
        
        # 安全性 vs 燃油权衡
        risk_vs_fuel = st.slider(
            "安全性 vs 燃油（0=更省油，1=更安全）",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="向左倾向选择燃油消耗少的路线；向右倾向选择风险低的路线。",
        )
        
        # 不确定性重要性
        uncertainty_importance = st.slider(
            "不确定性重要性",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="越大越倾向避开 EDL 不确定性高的区域。",
        )
        
        # 根据滑条值计算三个权重
        # 设计思路：
        # - weight_fuel = 1.0 - risk_vs_fuel（燃油权重与安全性反向）
        # - weight_risk = risk_vs_fuel * (1.0 - 0.3 * uncertainty_importance)
        # - weight_uncertainty = risk_vs_fuel * (0.3 * uncertainty_importance)
        # - 最后归一化使总和为 1
        weight_fuel = 1.0 - risk_vs_fuel
        weight_risk = risk_vs_fuel * (1.0 - 0.3 * uncertainty_importance)
        weight_uncertainty = risk_vs_fuel * (0.3 * uncertainty_importance)
        
        # 归一化权重，使总和为 1
        weight_sum = weight_fuel + weight_risk + weight_uncertainty
        if weight_sum > 0:
            weight_fuel /= weight_sum
            weight_risk /= weight_sum
            weight_uncertainty /= weight_sum
        
        # 显示权重分配
        st.caption(
            f"权重分配：燃油 {weight_fuel:.1%} | 风险 {weight_risk:.1%} | 不确定性 {weight_uncertainty:.1%}"
        )
        
        st.subheader("船舶配置")
        vessel_profiles = get_default_profiles()
        vessel_keys = list(vessel_profiles.keys())
        vessel_default = st.session_state.get("vessel_profile", vessel_keys[0] if vessel_keys else None)
        if vessel_default not in vessel_keys:
            vessel_default = vessel_keys[0]
        selected_vessel_key = st.selectbox(
            "选择船型",
            options=vessel_keys,
            index=vessel_keys.index(vessel_default),
            format_func=lambda k: f"{vessel_profiles[k].name} ({k})",
        )
        selected_vessel = vessel_profiles[selected_vessel_key]
        
        # ====================================================================
        # 任务 C：Health Check - 添加 AIS density grid_signature 验证
        # ====================================================================
        # 构建状态信息，包括 grid_signature 验证
        # 任务 C1：检查网格是否变化，若变化则清空 AIS 选择
        try:
            current_grid_sig = compute_grid_signature(grid)
        except Exception as e:
            print(f"[UI] Warning: failed to compute grid signature: {e}")
            current_grid_sig = None
        
        previous_grid_sig = st.session_state.get("previous_grid_signature", None)
        
        if (previous_grid_sig is not None and 
            current_grid_sig is not None and 
            previous_grid_sig != current_grid_sig):
            # 网格已切换，清空 AIS 密度选择
            st.session_state["ais_density_path"] = None
            st.session_state["ais_density_path_selected"] = None
            st.session_state["ais_density_cache_key"] = None
            st.info(f"🔄 网格已切换（{previous_grid_sig[:20]}... → {current_grid_sig[:20]}...），已清空 AIS 密度选择以避免维度错配")
        
        if current_grid_sig is not None:
            st.session_state["previous_grid_signature"] = current_grid_sig
        grid_sig = current_grid_sig
        ais_status_check = "✓" if ais_data_available else "✗"
        
        # 处理 grid_sig 可能为 None 的情况
        if grid_sig is None:
            grid_sig = "N/A"
        grid_sig_display = f"{grid_sig[:20]}..." if grid_sig != "N/A" and len(grid_sig) > 20 else grid_sig
        
        status_box.markdown(
            f"**当前网格**：{'真实' if grid_mode == 'real' else '演示'} (签名: {grid_sig_display})  \n"
            f"**船型**：{selected_vessel.name}  \n"
            f"**EDL**：{'开启' if use_edl else '关闭'}  \n"
            f"**AIS**：{'开启' if ais_weights_enabled else '关闭'} {ais_status_check} "
            f"(corridor={w_ais_corridor:.1f}, congestion={w_ais_congestion:.1f}, legacy={w_ais:.1f})"
        )
        
        st.caption("当前仅支持 demo 风险：高纬冰带成本；真实多模态风险后续接入。")
        
        do_plan = st.button("规划三条方案", type="primary")

    # 初始化流动管线相关的 session state
    if "pipeline_flow_nodes" not in st.session_state:
        st.session_state.pipeline_flow_nodes = []
    if "pipeline_flow_expanded" not in st.session_state:
        st.session_state.pipeline_flow_expanded = True
    if "pipeline_flow_start_time" not in st.session_state:
        st.session_state.pipeline_flow_start_time = None
    
    # 规划按钮被点击时，初始化流动管线
    if do_plan:
        st.session_state.pipeline_flow_expanded = True
        st.session_state.pipeline_flow_start_time = datetime.now()
        # 初始化 8 个节点
        st.session_state.pipeline_flow_nodes = [
            PipeNode(key="parse", label="① 解析场景/参数", status="pending"),
            PipeNode(key="grid_landmask", label="② 加载网格与 landmask", status="pending"),
            PipeNode(key="env_layers", label="③ 加载环境层", status="pending"),
            PipeNode(key="ais_density", label="④ 加载 AIS 密度", status="pending"),
            PipeNode(key="cost_field", label="⑤ 构建成本场", status="pending"),
            PipeNode(key="astar", label="⑥ A* 规划", status="pending"),
            PipeNode(key="analysis", label="⑦ 分析与诊断", status="pending"),
            PipeNode(key="render", label="⑧ 渲染与导出", status="pending"),
        ]
    
    # 初始化旧 Pipeline（保留向后兼容）
    pipeline = init_pipeline_in_session()
    
    # 定义 Pipeline stages
    pipeline_stages = [
        ("grid_env", "加载网格"),
        ("ais", "加载 AIS"),
        ("cost_build", "构建成本场"),
        ("snap", "起止点吸附"),
        ("astar", "A* 路由"),
        ("analysis", "成本分析"),
        ("render", "数据准备"),
    ]
    
    # 添加所有 stages 到 pipeline
    for stage_key, stage_label in pipeline_stages:
        pipeline.add_stage(stage_key, stage_label)
    
    # 初始化 session state 中的 pipeline 控制变量
    if "pipeline_expanded" not in st.session_state:
        st.session_state.pipeline_expanded = True
    
    # 规划按钮被点击时，强制展开 pipeline
    if do_plan:
        st.session_state.pipeline_expanded = True
    
    if ais_density_path is None and ais_weights_enabled:
        st.warning("当前未选择 AIS density 文件，AIS corridor/congestion 成本可能不会生效（将尝试自动匹配）。")
    
    # 主区域逻辑
    # 创建流动管线展示容器
    pipeline_flow_placeholder = st.empty()
    st.session_state.pipeline_flow_placeholder = pipeline_flow_placeholder
    
    # 初始渲染流动管线
    if do_plan and st.session_state.pipeline_flow_nodes:
        with pipeline_flow_placeholder.container():
            render_pipeline_flow(
                st.session_state.pipeline_flow_nodes,
                title="🔄 规划流程管线",
                expanded=st.session_state.get("pipeline_flow_expanded", True),
            )
    
    # [removed] 简化版 Pipeline（Timeline）渲染已删除，避免与上方卡片管道重复
    if not do_plan:
        st.info("在左侧设置起止点并点击『规划三条方案』。")
        return
    
    # 根据 grid_mode 加载网格
    grid_source_label = "demo"
    ais_density = None
    ais_density_da = None
    with st.spinner("加载网格与规划路线..."):
        # 更新第 1 个节点：解析场景/参数
        _update_pipeline_node(0, "running", "正在解析...")
        
        # 更新第 2 个节点：加载网格与 landmask
        _update_pipeline_node(1, "running", "正在加载...")
        
        if grid_mode == "real":
            # 尝试加载真实网格
            real_grid = load_real_grid_from_nc()
            if real_grid is not None:
                grid = real_grid
                # 尝试加载真实 landmask
                land_mask = load_real_landmask_from_nc(grid)
                if land_mask is not None:
                    grid_source_label = "real"
                else:
                    # 使用 demo landmask
                    st.warning("真实 landmask 不可用，使用演示 landmask。")
                    _, land_mask = make_demo_grid(ny=grid.shape()[0], nx=grid.shape()[1])
                    grid_source_label = "real_grid_demo_landmask"
            else:
                # 回退到 demo
                st.warning("真实网格不可用，使用演示网格。")
                grid, land_mask = make_demo_grid()
                grid_source_label = "demo"
        else:
            # 使用 demo 网格
            grid, land_mask = make_demo_grid()
            grid_source_label = "demo"
        
        # 完成第 1、2 个节点
        grid_shape = grid.shape() if hasattr(grid, 'shape') else (0, 0)
        _update_pipeline_node(0, "done", f"grid={grid_shape[0]}×{grid_shape[1]}", seconds=0.5)
        _update_pipeline_node(1, "done", f"landmask={grid_source_label}", seconds=0.3)
        
        # 尝试加载 AIS 密度（从 NC 文件）
        # 完成 grid_env stage
        pipeline.done('grid_env', extra_info=f'grid={grid_shape[0]}×{grid_shape[1]}')
        # [removed] render_pipeline timeline (simplified) disabled to avoid duplicate UI
        
        # ====================================================================
        # 任务 A：AIS 密度加载与状态管理
        # 确保 AIS 步骤完成时不停留在 pending（成功加载或跳过都标记为 done）
        # ====================================================================
        ais_info = {"loaded": False, "error": None, "shape": None, "num_points": 0, "num_binned": 0}
        ais_da_loaded = None
        
        if not ais_weights_enabled:
            # 权重为 0，直接标记 AIS 为 done（skip）
            _update_pipeline_node(3, "done", "跳过：权重为 0", seconds=0.1)
        else:
            # 权重 > 0，尝试加载 AIS 密度
            _update_pipeline_node(3, "running", "正在加载 AIS 密度...")
            
            try:
                from arcticroute.core import cost as cost_core
                import xarray as xr
                from pathlib import Path

                prefer_real = (grid_mode == "real")
                ais_density_path_obj = Path(ais_density_path) if ais_density_path is not None else None
                
                # 情况 1：用户未选择 AIS 文件（自动模式，交由成本构建阶段匹配/重采样）
                if ais_density_path_obj is None:
                    _update_pipeline_node(3, "done", "自动选择：运行时加载", seconds=0.1)
                    st.info("ℹ️ AIS 采用自动选择/重采样，将在成本阶段按网格自动匹配。")
                
                # 情况 2：文件存在，尝试加载
                elif ais_density_path_obj.exists():
                    try:
                        with xr.open_dataset(ais_density_path_obj) as ds:
                            if "ais_density" in ds:
                                ais_da_loaded = ds["ais_density"].load()
                            elif ds.data_vars:
                                ais_da_loaded = list(ds.data_vars.values())[0].load()
                            else:
                                ais_da_loaded = None
                        
                        if ais_da_loaded is not None:
                            ais_density = ais_da_loaded.values if hasattr(ais_da_loaded, "values") else np.asarray(ais_da_loaded)
                            ais_info.update({
                                "loaded": True,
                                "shape": ais_density.shape,
                            })
                            # 成功加载，标记为 done
                            _update_pipeline_node(3, "done", f"AIS={ais_density.shape[0]}×{ais_density.shape[1]} source={ais_density_path_obj.name}", seconds=0.3)
                            st.success(f"✅ 已加载 AIS 拥挤度密度数据，栅格={ais_info['shape']}")
                        else:
                            # 文件无效
                            _update_pipeline_node(3, "done", "跳过：文件格式无效", seconds=0.1)
                            st.warning("⚠️ AIS 密度文件格式无效，已跳过")
                            w_ais = 0.0
                            w_ais_corridor = 0.0
                            w_ais_congestion = 0.0
                            ais_weights_enabled = False
                    
                    except Exception as e:
                        # 加载失败
                        _update_pipeline_node(3, "fail", f"加载失败：{str(e)[:50]}", seconds=0.2)
                        st.error(f"❌ 加载 AIS 密度失败：{e}")
                        w_ais = 0.0
                        w_ais_corridor = 0.0
                        w_ais_congestion = 0.0
                        ais_weights_enabled = False
                
                # 情况 3：文件不存在
                else:
                    _update_pipeline_node(3, "done", f"跳过：文件不存在", seconds=0.1)
                    st.warning(f"⚠️ AIS 密度文件不存在：{ais_density_path_obj}")
                    w_ais = 0.0
                    w_ais_corridor = 0.0
                    w_ais_congestion = 0.0
                    ais_weights_enabled = False
            
            except Exception as e:
                # 意外错误
                _update_pipeline_node(3, "fail", f"异常：{str(e)[:50]}", seconds=0.2)
                st.error(f"❌ AIS 加载异常：{e}")
                w_ais = 0.0
                w_ais_corridor = 0.0
                w_ais_congestion = 0.0
                ais_weights_enabled = False

        # 重新计算权重启用状态（可能在上方被置零）
        ais_weights_enabled = any(weight > 0 for weight in [w_ais, w_ais_corridor, w_ais_congestion])
        
        # 更新流动管线显示
        if "pipeline_flow_placeholder" in st.session_state:
            try:
                st.session_state.pipeline_flow_placeholder.empty()
                with st.session_state.pipeline_flow_placeholder.container():
                    render_pipeline_flow(
                        st.session_state.pipeline_flow_nodes,
                        title="🔧 规划流程管线",
                        expanded=st.session_state.get("pipeline_flow_expanded", True),
                    )
            except Exception:
                pass

        # 若 AIS 已加载但与当前网格尺寸不一致，明确提示原因（将被成本构建跳过）
        if ais_weights_enabled and ais_density is not None:
            if hasattr(grid, 'lat2d') and ais_density.shape != grid.lat2d.shape:
                st.warning(
                    f"AIS 拥挤度已加载，但栅格维度不匹配，将在成本计算中被跳过："
                    f"AIS={ais_density.shape} vs GRID={grid.lat2d.shape}。\n"
                    f"可能原因：先用 demo 网格生成了 AIS，再切换到真实网格。建议刷新页面或重启 UI 后重试。"
                )
        
        # 尝试加载预处理的 AIS 密度栅格（用于主航道成本）
        try:
            from arcticroute.core import cost as cost_core
            prefer_real = (grid_mode == "real")
            ais_da_for_corridor = ais_da_loaded if ais_da_loaded is not None else cost_core.load_ais_density_for_grid(grid, prefer_real=prefer_real)
            if ais_da_for_corridor is not None:
                if hasattr(grid, "lat2d") and ais_da_for_corridor.shape == grid.lat2d.shape:
                    ais_density_da = ais_da_for_corridor
                    # 若前面未设置 ais_density，则复用
                    if ais_density is None:
                        ais_density = ais_da_for_corridor.values if hasattr(ais_da_for_corridor, "values") else np.asarray(ais_da_for_corridor)
                else:
                    st.warning(
                        f"AIS 密度数据已加载但维度不匹配，将跳过 AIS 主航道成本："
                        f"AIS={ais_da_for_corridor.shape} vs GRID={grid.lat2d.shape}"
                    )
            else:
                if ais_weights_enabled and w_ais_corridor > 0:
                    st.warning("未找到 AIS 密度数据，将不使用 AIS 主航道成本。")
        except Exception as e:
            st.warning(f"加载 AIS 密度数据失败：{e}")

        # 依赖项提示（便于定位渲染/重采样问题）
        with st.expander("诊断与依赖状态 (可展开)"):
            # pydeck
            try:
                import pydeck  # type: ignore
                st.caption("可视化: pydeck 可用 ✅")
            except Exception:
                st.warning("可视化: pydeck 未安装，将无法在地图上绘制路径。请运行 `pip install pydeck`。")
            
            # scipy（用于更高质量的 landmask 重采样）
            try:
                import scipy  # type: ignore
                st.caption("重采样: SciPy 可用 ✅（landmask 将使用 KDTree 最近邻，质量更好）")
            except Exception:
                st.info("重采样: SciPy 未安装，将使用简易最近邻重采样（已自动降级）。建议 `pip install scipy` 提升质量与速度。")
            
            # torch（用于 EDL 模型）
            try:
                import torch  # type: ignore
                st.caption("EDL: PyTorch 可用 ✅")
            except Exception:
                st.info("EDL: PyTorch 未安装，EDL 风险将使用占位/常数风险（日志中含有 EDL fallback 提示）。")
            
            # 显示当前栅格与 AIS 尺寸
            st.caption(
                f"当前 GRID 维度: {grid.shape() if grid is not None else 'N/A'} | "
                f"AIS 维度: {ais_density.shape if ais_density is not None else '未加载'}"
            )
        
        # 完成第 3 个节点：加载环境层
        _update_pipeline_node(2, "done", "SIC/Wave 已加载", seconds=0.2)
        
        # 规划路线（使用从 EDL 模式获取的参数）
        # 启动后续 stages
        
        # 更新第 5 个节点：构建成本场
        _update_pipeline_node(4, "running", "构建成本场...")
        
        routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(
            grid, land_mask, start_lat, start_lon, end_lat, end_lon, allow_diag, selected_vessel, cost_mode, wave_penalty, use_edl, w_edl,
            weight_risk=weight_risk, weight_uncertainty=weight_uncertainty, weight_fuel=weight_fuel,
            edl_uncertainty_weight=edl_uncertainty_weight,
            ais_density=ais_density,
            ais_density_path=ais_density_path,
            ais_density_da=ais_density_da,
            w_ais_corridor=w_ais_corridor,
            w_ais_congestion=w_ais_congestion,
            w_ais=w_ais,
        )
        
        # 完成第 5、6 个节点
        _update_pipeline_node(4, "done", "3 种成本场", seconds=0.6)
        
        # 更新第 6 个节点：A* 规划
        _update_pipeline_node(5, "running", "规划路线...")
        
        # 完成 cost_build/snap/astar stages
        pipeline.done('cost_build')
        pipeline.done('snap')
        num_reachable = sum(1 for r in routes_info.values() if r.reachable)
        pipeline.done('astar', extra_info=f'routes reachable={num_reachable}/3')
        
        # 完成第 6 个节点
        _update_pipeline_node(5, "done", f"可达={num_reachable}/3", seconds=0.8)
        # [removed] render_pipeline timeline (simplified) disabled to avoid duplicate UI
        
        # 如果真实环境数据不可用，显示警告并给出可能原因
        if cost_mode == "real_sic_if_available" and not cost_meta["real_env_available"]:
            st.warning(
                "真实环境数据不可用，已自动回退为演示冰带成本。\n"
                "可能原因：\n"
                "- 未设置或设置了错误的 ARCTICROUTE_DATA_ROOT\n"
                "- env_clean.nc / ice_copernicus_sic.nc / wave_swh.nc 文件不存在\n"
                "- NetCDF 结构与预期不一致（坐标变量命名/维度）\n"
                "- 文件权限或路径不可读"
            )

    # 网格加载状态提示
    if grid_source_label == "real":
        ny, nx = grid.shape()
        st.success(f"✅ 使用真实环境网格（{ny}×{nx}）")
    elif grid_mode == "real":
        st.warning("⚠️ 真实环境不可用，已回退到 demo 网格")
    else:
        st.info("当前使用演示网格")


    # 检查是否有可达的路线
    reachable_routes = {k: v for k, v in routes_info.items() if v.reachable}
    
    if not reachable_routes:
        st.error("三条方案均不可达，请调整起止点后重试。")
        return

    modes = ["efficient", "edl_safe", "edl_robust"]
    default_mode = st.session_state.get("selected_mode", "edl_safe")
    if default_mode not in modes:
        default_mode = "edl_safe"
    selected_mode = st.radio(
        "当前对比方案",
        options=modes,
        index=modes.index(default_mode),
        format_func=lambda m: ROUTE_LABELS_ZH.get(m, m),
        horizontal=True,
        key="selected_mode",
        help="用于 KPI 卡片与地图高亮",
    )

    ROUTE_DESC = {
        "efficient": "偏距离 / 燃油",
        "edl_safe": "偏安全 / 平衡",
        "edl_robust": "EDL + 不确定性",
    }
    ROUTE_LABELS = ROUTE_LABELS_ZH

    def _get_breakdown_for_route(key: str, route: RouteInfo):
        breakdown = route.notes.get("breakdown")
        if breakdown is None:
            cost_field = cost_fields.get(key)
            if cost_field is not None and route.reachable:
                breakdown = compute_route_cost_breakdown(grid, cost_field, route.coords)
                route.notes["breakdown"] = breakdown
        return breakdown

    def _get_profile_for_route(key: str, route: RouteInfo):
        profile_result = route.notes.get("profile_result")
        if profile_result is None:
            cost_field = cost_fields.get(key)
            if cost_field is not None and route.reachable:
                try:
                    profile_result = compute_route_profile(route.coords, cost_field)
                    route.notes["profile_result"] = profile_result
                except Exception:
                    profile_result = None
        return profile_result

    def _get_total_cost_value(key: str, route: RouteInfo | None) -> float | None:
        if route is None or not route.reachable:
            return None
        if route.total_cost is not None:
            return route.total_cost
        breakdown = _get_breakdown_for_route(key, route)
        return breakdown.total_cost if breakdown else None

    def _get_risk_cost(key: str, route: RouteInfo | None) -> float | None:
        if route is None or not route.reachable:
            return None
        breakdown = _get_breakdown_for_route(key, route)
        components = {}
        if breakdown and breakdown.component_totals:
            components = breakdown.component_totals
        elif route.cost_components:
            components = route.cost_components
        if not components:
            return None
        risk_keys = [
            "ice_risk",
            "wave_risk",
            "ais_density",
            "edl_risk",
            "edl_uncertainty_penalty",
            "ice_class_soft",
            "ice_class_hard",
        ]
        has_value = False
        total_val = 0.0
        for rk in risk_keys:
            v = components.get(rk)
            if v is None:
                continue
            try:
                if np.isfinite(float(v)):
                    total_val += float(v)
                    has_value = True
            except Exception:
                continue
        return total_val if has_value else None

    def _get_distance_value(route: RouteInfo | None) -> float | None:
        if route is None or not route.reachable:
            return None
        if route.distance_km is not None:
            return route.distance_km
        return route.approx_length_km

    baseline_route = routes_info.get("efficient")
    current_route = routes_info.get(selected_mode) if selected_mode in routes_info else None
    if current_route is None or not current_route.reachable:
        current_route = baseline_route

    baseline_distance = _get_distance_value(baseline_route)
    current_distance = _get_distance_value(current_route)
    baseline_cost = _get_total_cost_value("efficient", baseline_route)
    current_cost = _get_total_cost_value(current_route.mode if current_route else "efficient", current_route)
    risk_baseline = _get_risk_cost("efficient", baseline_route)
    risk_current = _get_risk_cost(current_route.mode if current_route else "efficient", current_route)

    profile_available = False
    mean_uncertainty = None
    high_uncertainty_frac = None
    if current_route is not None and current_route.reachable:
        profile_result = _get_profile_for_route(current_route.mode, current_route)
        if profile_result is not None and getattr(profile_result, "edl_uncertainty", None) is not None:
            arr = np.asarray(profile_result.edl_uncertainty)
            valid = np.isfinite(arr)
            if np.any(valid):
                profile_available = True
                vals = arr[valid]
                mean_uncertainty = float(np.mean(vals))
                high_uncertainty_frac = float(np.sum(vals > 0.5)) / float(len(vals))

    # 启动 analysis stage
    pipeline.start('analysis')
    
    st.subheader("KPI 总览")
    c1, c2, c3 = st.columns(3)

    with c1:
        if baseline_route is None or not baseline_route.reachable:
            st.metric("基准方案（效率优先）", "—", help="基线不可用")
        else:
            st.metric(
                "基准方案（效率优先）",
                f"{baseline_distance:.0f} km / 成本 {baseline_cost:.2f}" if baseline_distance and baseline_cost is not None else "—",
                help="不考虑 EDL 额外惩罚的基线方案",
            )

    with c2:
        if current_route is None or not current_route.reachable or baseline_route is None or not baseline_route.reachable:
            st.metric("当前方案", "—", help="缺少可比方案")
        else:
            delta_dist_pct = None
            delta_cost_pct = None
            if baseline_distance and baseline_distance > 0 and current_distance is not None:
                delta_dist_pct = (current_distance - baseline_distance) / baseline_distance * 100
            if baseline_cost and baseline_cost > 0 and current_cost is not None:
                delta_cost_pct = (current_cost - baseline_cost) / baseline_cost * 100
            st.metric(
                f"当前方案：{ROUTE_LABELS_ZH.get(selected_mode, selected_mode)}",
                f"{current_distance:.0f} km / 成本 {current_cost:.2f}" if current_distance and current_cost is not None else "—",
                help=(
                    f"相对基线：距离 {delta_dist_pct:+.2f}%，成本 {delta_cost_pct:+.2f}%"  # type: ignore[operator]
                    if delta_dist_pct is not None and delta_cost_pct is not None
                    else "相对基线暂无对比"
                ),
            )

    with c3:
        if risk_baseline is None or risk_baseline <= 0 or risk_current is None:
            st.metric("风险与不确定性", "—", help="基线风险为 0，无法计算相对降低")
        else:
            risk_red_pct = (risk_baseline - risk_current) / risk_baseline * 100
            if profile_available and mean_uncertainty is not None and high_uncertainty_frac is not None:
                unc_text = f"均值不确定性 {mean_uncertainty:.2f}, 高不确定性路段 {high_uncertainty_frac*100:.1f}%"
            else:
                unc_text = "EDL 不确定性数据不足"
            st.metric(
                "风险与不确定性",
                f"风险降低 {risk_red_pct:.1f}%",
                help=unc_text,
            )

    # 顶部地图
    # 完成 analysis 并启动 render
    # 更新第 7 个节点：分析与诊断
    _update_pipeline_node(6, "running", "分析成本...")
    pipeline.done('analysis')
    # [removed] render_pipeline timeline (simplified) disabled to avoid duplicate UI
    _update_pipeline_node(6, "done", "分析完成", seconds=0.3)
    
    # 更新第 8 个节点：渲染与导出
    _update_pipeline_node(7, "running", "渲染地图...")
    pipeline.start('render')
    
    st.subheader("路线对比地图")
    path_data = []
    for mode in ["efficient", "edl_safe", "edl_robust"]:
        route = routes_info.get(mode)
        if route is None or not route.reachable:
            continue
        coords = route.coords or []
        valid_coords = [(lat, lon) for lat, lon in coords if _is_valid_coord(lat, lon)]
        if not valid_coords:
            continue
        path_list = [[_wrap_lon(float(lon)), float(lat)] for lat, lon in valid_coords]
        distance_val = route.distance_km or route.approx_length_km or 0.0
        total_cost_val = route.total_cost if route.total_cost is not None else None
        color_val = ROUTE_COLORS.get(mode, [200, 200, 200])
        width_val = 6 if mode == selected_mode else 2
        path_data.append({
            "name": ROUTE_LABELS_ZH.get(mode, mode),
            "label": ROUTE_LABELS_ZH.get(mode, mode),
            "mode": mode,
            "path": path_list,
            "color": color_val,
            "width": width_val,
            "distance_km": distance_val,
            "total_cost": total_cost_val if total_cost_val is not None else "-",
        })

    if path_data:
        try:
            import pydeck as pdk

            all_points = [pt for item in path_data for pt in item["path"]]
            avg_lon = float(np.mean([p[0] for p in all_points]))
            avg_lat = float(np.mean([p[1] for p in all_points]))

            layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_width="width",
                get_color="color",
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=ARCTIC_VIEW["latitude"],
                longitude=ARCTIC_VIEW["longitude"],
                zoom=ARCTIC_VIEW["zoom"],
                pitch=ARCTIC_VIEW["pitch"],
                min_zoom=ARCTIC_VIEW["min_zoom"],
                max_zoom=ARCTIC_VIEW["max_zoom"],
            )
            # 兼容不同版本的 pydeck：部分版本不支持 controller 参数
            try:
                deck_obj = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={
                        "text": "{label}\n距离: {distance_km} km\n总成本: {total_cost}"
                    },
                    controller=MAP_CONTROLLER,
                )
            except TypeError:
                deck_obj = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={
                        "text": "{label}\n距离: {distance_km} km\n总成本: {total_cost}"
                    },
                )
            st.pydeck_chart(
                deck_obj,
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"pydeck 渲染失败：{e}")
    else:
        st.info("当前没有可视化的路径数据。")

    st.markdown("#### 路线图例")
    col1, col2, col3 = st.columns(3)
    for col, mode in zip((col1, col2, col3), ["efficient", "edl_safe", "edl_robust"]):
        color = ROUTE_COLORS[mode]
        label = ROUTE_LABELS_ZH[mode]
        col.markdown(
            f"""
            <div style="display:flex;align-items:center;gap:0.5rem;">
              <div style="width:18px;height:4px;background:rgb({color[0]},{color[1]},{color[2]});border-radius:999px;"></div>
              <span>{label}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 方案摘要卡片
    st.subheader("方案摘要卡片")
    card_cols = st.columns(3)

    def _format_main_risk(route: RouteInfo) -> str:
        components = route.cost_components or {}
        if not components:
            breakdown = _get_breakdown_for_route(route.mode, route)
            if breakdown:
                components = breakdown.component_totals or {}
        total = route.total_cost if route.total_cost is not None else sum(components.values()) if components else 0.0
        if not components or total <= 0:
            return "主风险：—"
        filtered = {k: v for k, v in components.items() if v is not None and k != "base_distance"}
        if not filtered:
            return "主风险：—"
        main_key, main_val = max(filtered.items(), key=lambda kv: kv[1])
        frac = main_val / total if total else 0.0
        emoji_map = {
            "ice_risk": "🧊",
            "wave_risk": "🌊",
            "ais_density": "🚢",
            "edl_risk": "🧠",
            "edl_uncertainty_penalty": "❓",
        }
        return f"主风险：{emoji_map.get(main_key, '')} {main_key} {frac:.0%}"

    for idx, mode in enumerate(["efficient", "edl_safe", "edl_robust"]):
        route = routes_info.get(mode)
        label = ROUTE_LABELS.get(mode, mode)
        with card_cols[idx]:
            if route is None or not route.reachable:
                st.error(f"{label}：不可达")
                st.caption("距离 / 成本：-")
                continue
            tag_bits = [
                "real" if grid_source_label == "real" else "demo",
                selected_vessel.name,
                f"grid={grid_source_label}",
            ]
            st.markdown(f"**{label}**")
            st.caption(" / ".join(tag_bits))
            distance_val = route.distance_km or route.approx_length_km
            total_cost_val = route.total_cost
            fuel = route.fuel_total_t if route.fuel_total_t > 0 else None
            co2 = route.co2_total_t if route.co2_total_t > 0 else None
            cols = st.columns(2)
            cols[0].metric("距离 (km)", f"{distance_val:.1f}" if distance_val else "-")
            cols[1].metric("总成本", f"{total_cost_val:.2f}" if total_cost_val else "-")
            cols2 = st.columns(2)
            cols2[0].metric("燃油 (t)", f"{fuel:.2f}" if fuel else "-")
            cols2[1].metric("CO₂ (t)", f"{co2:.2f}" if co2 else "-")
            st.caption(_format_main_risk(route))

    # === UX-1：三方案“小仪表盘 + 雷达图” ===
    st.markdown("---")
    st.subheader("EDL 三模式对比：小仪表盘 + 雷达图")

    def _risk_total_for_route(key: str, route: RouteInfo) -> float | None:
        """综合风险（EDL 风险 + 不确定性 + 冰 + 浪）。若均缺失则返回 None。"""
        breakdown = _get_breakdown_for_route(key, route)
        if breakdown is None or not breakdown.component_totals:
            return None
        comp = breakdown.component_totals
        vals = []
        for k in ["edl_risk", "edl_uncertainty_penalty", "ice_risk", "wave_risk"]:
            v = comp.get(k, None)
            if v is not None:
                vals.append(float(v))
        return (sum(vals) if vals else None)

    # 统计三条方案的核心指标
    metrics_rows = []
    for m in ["efficient", "edl_safe", "edl_robust"]:
        r = routes_info.get(m)
        if r is None or not r.reachable:
            continue
        risk_total = _risk_total_for_route(m, r)
        total_cost = r.total_cost
        if total_cost is None:
            b = _get_breakdown_for_route(m, r)
            total_cost = b.total_cost if b else None
        metrics_rows.append({
            "mode": m,
            "distance": (r.distance_km or r.approx_length_km or 0.0),
            "risk_total": risk_total,
            "total_cost": total_cost,
        })

    if metrics_rows:
        # 三张 metric 卡片
        cols = st.columns(3)
        # 距离最短
        shortest = min(metrics_rows, key=lambda d: d["distance"])
        eff_row = next((d for d in metrics_rows if d["mode"] == "efficient"), None)
        delta_km = None
        if eff_row is not None:
            delta_km = shortest["distance"] - eff_row["distance"]
        cols[0].metric(
            "距离最短方案",
            f"{shortest['distance']:.1f} km" if shortest["distance"] else "-",
            (f"{delta_km:+.1f} km 相对 Efficient" if delta_km is not None else None),
            help=f"{ROUTE_LABELS.get(shortest['mode'], shortest['mode'])}",
        )

        # 风险最低（基于综合风险）
        risk_rows = [d for d in metrics_rows if d["risk_total"] is not None]
        if risk_rows:
            safest = min(risk_rows, key=lambda d: d["risk_total"])
            delta_risk = None
            if eff_row is not None and eff_row.get("risk_total") is not None and eff_row["risk_total"] > 0:
                delta_risk = (safest["risk_total"] - eff_row["risk_total"]) / eff_row["risk_total"] * 100.0
            cols[1].metric(
                "风险最低方案",
                f"{safest['risk_total']:.2f}" if safest["risk_total"] is not None else "-",
                (f"{delta_risk:+.1f}% 相对 Efficient" if delta_risk is not None else None),
                help=f"{ROUTE_LABELS.get(safest['mode'], safest['mode'])}（EDL 风险+不确定性+冰+浪）",
            )
        else:
            cols[1].metric("风险最低方案", "未启用/数据不可用")

        # 折中方案（edl_safe）
        safe_row = next((d for d in metrics_rows if d["mode"] == "edl_safe"), None)
        if safe_row is not None:
            delta_km_safe = None
            if eff_row is not None:
                delta_km_safe = safe_row["distance"] - eff_row["distance"]
            cols[2].metric(
                "折中方案（edl_safe）",
                f"{safe_row['distance']:.1f} km",
                (f"{delta_km_safe:+.1f} km 相对 Efficient" if delta_km_safe is not None else None),
            )
        else:
            cols[2].metric("折中方案（edl_safe）", "不可达")

        # 解释文本（edl_safe 相对 efficient）
        if eff_row is not None and safe_row is not None and eff_row["distance"] > 0:
            dist_inc_pct = (safe_row["distance"] - eff_row["distance"]) / eff_row["distance"] * 100.0
            risk_red_pct = None
            if eff_row.get("risk_total") and safe_row.get("risk_total") and eff_row["risk_total"] > 0:
                risk_red_pct = (eff_row["risk_total"] - safe_row["risk_total"]) / eff_row["risk_total"] * 100.0
            txt = f"EDL-Safe 相比 Efficient，多走约 {dist_inc_pct:.1f}% 路程。"
            if risk_red_pct is not None:
                txt += f" 同时降低约 {risk_red_pct:.1f}% 综合风险。"
            st.caption(txt)

        # 雷达图/蜘蛛图（Altair）
        try:
            import altair as alt
            dims = []
            # 归一化准备
            distances = [r["distance"] for r in metrics_rows]
            max_dist = max(distances) if distances else 0.0
            costs = [r["total_cost"] for r in metrics_rows if r["total_cost"] is not None]
            max_cost = max(costs) if costs else None
            risks = [r["risk_total"] for r in metrics_rows if r["risk_total"] is not None]
            max_risk = max(risks) if risks else None

            radar_rows = []
            if max_dist and max_dist > 0:
                dims.append("distance_norm")
            if max_cost is not None and max_cost > 0:
                dims.append("total_cost_norm")
            if max_risk is not None and max_risk > 0:
                dims.append("risk_score_norm")

            # 如果某些维度不存在则自动隐藏
            for row in metrics_rows:
                for dim in dims:
                    if dim == "distance_norm":
                        val = row["distance"] / max_dist if max_dist else 0.0
                    elif dim == "total_cost_norm":
                        v = row["total_cost"] if row["total_cost"] is not None else 0.0
                        val = v / max_cost if (max_cost and max_cost > 0) else 0.0
                    elif dim == "risk_score_norm":
                        v = row["risk_total"] if row["risk_total"] is not None else 0.0
                        val = v / max_risk if (max_risk and max_risk > 0) else 0.0
                    else:
                        val = 0.0
                    radar_rows.append({
                        "mode": ROUTE_LABELS.get(row["mode"], row["mode"]),
                        "metric": dim,
                        "value": float(val),
                    })

            if dims and radar_rows:
                # 分配角度
                angle_map = {m: i for i, m in enumerate(dims)}
                for r in radar_rows:
                    r["angle"] = float(angle_map[r["metric"]]) / max(1, len(dims)) * 2 * math.pi

                # 为闭合多边形重复首点
                closed_rows = []
                for mode_label in {r["mode"] for r in radar_rows}:
                    rows = [r for r in radar_rows if r["mode"] == mode_label]
                    rows_sorted = sorted(rows, key=lambda x: x["angle"])  # 以角度排序
                    if rows_sorted:
                        rows_sorted.append({**rows_sorted[0]})
                    closed_rows.extend(rows_sorted)

                df_radar = pd.DataFrame(closed_rows)
                chart = (
                    alt.Chart(df_radar)
                    .mark_line(point=True)
                    .encode(
                        theta=alt.Theta("angle:Q", stack=None),
                        radius=alt.Radius("value:Q", scale=alt.Scale(domain=[0, 1]), title="归一化值"),
                        color=alt.Color("mode:N", title="方案"),
                        tooltip=["mode:N", "metric:N", alt.Tooltip("value:Q", format=".2f")],
                    )
                )
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info("雷达图维度不足（例如 EDL 未启用或总成本缺失），已隐藏。")
        except Exception as e:
            st.info(f"雷达图绘制失败：{e}")

    tab_cost, tab_profile, tab_edl, tab_ais = st.tabs(
        ["📊 成本分解（balanced/edl_safe）", "📈 沿程剖面", "🧠 EDL 不确定性", "🚢 AIS 拥挤度 & 拥堵"]
    )

    with tab_cost:
        base_key = "edl_safe" if routes_info.get("edl_safe") else "efficient"
        base_route = routes_info.get(base_key)
        if base_route is None or not base_route.reachable:
            st.info("当前没有可用的 edl_safe/balanced 路线。")
        else:
            breakdown = _get_breakdown_for_route(base_key, base_route)
            if breakdown and breakdown.component_totals:
                df_cost = pd.DataFrame({
                    "component": list(breakdown.component_totals.keys()),
                    "cost": list(breakdown.component_totals.values()),
                    "fraction": [breakdown.component_fractions.get(k, 0.0) for k in breakdown.component_totals.keys()],
                })
                st.dataframe(df_cost, use_container_width=True)
                st.bar_chart(df_cost.set_index("component")["cost"])
            else:
                st.info("成本分解不可用。")

    with tab_profile:
        for mode in ["efficient", "edl_safe", "edl_robust"]:
            route = routes_info.get(mode)
            if route is None or not route.reachable:
                st.caption(f"{ROUTE_LABELS.get(mode, mode)}：不可达，跳过剖面。")
                continue
            profile_result = _get_profile_for_route(mode, route)
            if profile_result is None or len(profile_result.distance_km) == 0:
                st.info(f"{ROUTE_LABELS.get(mode, mode)}：暂无剖面数据。")
                continue
            st.markdown(f"**{ROUTE_LABELS.get(mode, mode)} 剖面**")
            distance = profile_result.distance_km
            for comp_key, title in [("ice_risk", "冰风险"), ("wave_risk", "波浪风险")]:
                comp_arr = profile_result.components.get(comp_key)
                if comp_arr is not None and np.any(np.isfinite(comp_arr)) and float(np.nanmax(np.abs(comp_arr))) > 0:
                    df = pd.DataFrame({"distance_km": distance, title: comp_arr})
                    st.line_chart(df.set_index("distance_km"))
                else:
                    st.caption(f"{title}：该方案中此风险维度无显著贡献。")
            if profile_result.ais_density is not None and np.any(np.isfinite(profile_result.ais_density)):
                df = pd.DataFrame({"distance_km": distance, "AIS 拥挤度": profile_result.ais_density})
                st.line_chart(df.set_index("distance_km"))
            else:
                st.caption("AIS 拥挤度：该方案中此风险维度无显著贡献。")

    with tab_edl:
        route = routes_info.get("edl_robust")
        if route is None or not route.reachable:
            st.info("edl_robust 路线不可达或缺失。")
        else:
            profile_result = _get_profile_for_route("edl_robust", route)
            if profile_result is None or profile_result.edl_uncertainty is None or len(profile_result.edl_uncertainty) == 0:
                st.info("当前无 EDL 不确定性剖面。")
            else:
                df = pd.DataFrame({
                    "distance_km": profile_result.distance_km,
                    "uncertainty": profile_result.edl_uncertainty,
                })
                st.line_chart(df.set_index("distance_km"))
                valid = np.isfinite(profile_result.edl_uncertainty)
                if np.any(valid):
                    vals = profile_result.edl_uncertainty[valid]
                    avg_unc = float(np.mean(vals))
                    frac_high = float(np.sum(vals > 0.5)) / float(len(vals))
                    st.caption(f"平均不确定性：{avg_unc:.2f}，超过 0.5 的比例约 {frac_high*100:.1f}%")
                    if frac_high > 0.3:
                        st.warning(f"该路线约 {frac_high*100:.0f}% 的里程存在较高模型不确定性，建议结合传统路径规划结果审慎评估。")

            # === UX-2：沿程多维剖面（SIC / 波高 / EDL） ===
            st.markdown("---")
            st.markdown("**沿程多维风险剖面（SIC / 波高 / EDL）**")
            if not cost_meta.get("real_env_available", False):
                st.info("真实环境数据不可用（当前为 demo 成本），隐藏多维剖面。")
            else:
                try:
                    # 选择一条路线（优先 edl_robust）
                    base_key = "edl_robust" if routes_info.get("edl_robust") and routes_info["edl_robust"].reachable else "edl_safe"
                    base_route2 = routes_info.get(base_key)
                    if base_route2 is None or not base_route2.reachable:
                        st.info("无可用于剖面的路线。")
                    else:
                        prof = _get_profile_for_route(base_key, base_route2)
                        if prof is None or len(getattr(prof, "distance_km", [])) == 0:
                            st.info("当前无沿程剖面数据。")
                        else:
                            # 取组件
                            dist = np.asarray(prof.distance_km)
                            comp = getattr(prof, "components", {}) or {}
                            sic = comp.get("sic")
                            wave = comp.get("wave_swh")
                            edl_risk = comp.get("edl_risk")
                            edl_unc = getattr(prof, "edl_uncertainty", None)

                            # 如缺 EDL risk，尝试从 breakdown 沿程
                            if edl_risk is None:
                                b2 = _get_breakdown_for_route(base_key, base_route2)
                                if b2 is not None and b2.component_along_path and b2.s_km:
                                    # 需要对齐长度；粗略插值到 prof.distance_km
                                    from numpy import interp
                                    try:
                                        edl_risk_raw = np.asarray(b2.component_along_path.get("edl_risk")) if b2.component_along_path.get("edl_risk") is not None else None
                                        if edl_risk_raw is not None and len(edl_risk_raw) == len(b2.s_km):
                                            edl_risk = interp(dist, np.asarray(b2.s_km), edl_risk_raw)
                                    except Exception:
                                        edl_risk = None

                            # 校验至少有 sic/wave 与一条 EDL 曲线
                            avail_series = {
                                "海冰浓度 (SIC)": sic,
                                "波高 (SWH)": wave,
                                "EDL 风险": edl_risk,
                                "EDL 不确定性": edl_unc,
                            }
                            # 过滤可用且有有效数值
                            series = {k: np.asarray(v) for k, v in avail_series.items() if v is not None and np.any(np.isfinite(v))}

                            if len(series) < 2:
                                st.info("可用剖面曲线不足（至少需要两条），已隐藏。")
                            else:
                                # 统一截断为同一长度（取最短）
                                min_len = min(len(dist), *[len(v) for v in series.values()])
                                dist2 = dist[:min_len]
                                series2 = {k: v[:min_len] for k, v in series.items()}

                                # 将点数压缩至不超过 100（均匀采样）
                                max_n = 100
                                if len(dist2) > max_n:
                                    idx = np.linspace(0, len(dist2) - 1, num=max_n).astype(int)
                                    dist2 = dist2[idx]
                                    series2 = {k: v[idx] for k, v in series2.items()}

                                # 归一化到 0-1
                                def _norm(a: np.ndarray) -> np.ndarray:
                                    a = a.astype(float)
                                    mask = np.isfinite(a)
                                    if not np.any(mask):
                                        return np.zeros_like(a)
                                    amin = float(np.nanmin(a[mask]))
                                    amax = float(np.nanmax(a[mask]))
                                    if amax > amin:
                                        out = (a - amin) / (amax - amin)
                                    else:
                                        out = np.zeros_like(a)
                                    out[~mask] = np.nan
                                    return out

                                long_rows = []
                                for name, arr in series2.items():
                                    vals = _norm(np.asarray(arr))
                                    for s, v in zip(dist2, vals):
                                        long_rows.append({"distance_km": float(s), "变量": name, "值": float(v) if np.isfinite(v) else np.nan})

                                df_long = pd.DataFrame(long_rows)
                                try:
                                    import altair as alt
                                    chart = (
                                        alt.Chart(df_long.dropna())
                                        .mark_line()
                                        .encode(
                                            x=alt.X("distance_km:Q", title="路径累计距离 (km)"),
                                            y=alt.Y("值:Q", scale=alt.Scale(domain=[0, 1]), title="标准化值 (0-1)"),
                                            color=alt.Color("变量:N", title="维度"),
                                            tooltip=[alt.Tooltip("distance_km:Q", format=".0f"), "变量:N", alt.Tooltip("值:Q", format=".2f")],
                                        )
                                    )
                                    st.altair_chart(chart, use_container_width=True)
                                except Exception as e:
                                    st.info(f"Altair 绘制失败：{e}")

                                # 自动解释文本：寻找同时升高的区间
                                try:
                                    # 简单滑窗平均 + 阈值
                                    def smooth(x, k=5):
                                        x = np.asarray(x, dtype=float)
                                        if len(x) < k:
                                            return x
                                        w = np.ones(k) / k
                                        return np.convolve(x, w, mode="same")

                                    sic_n = series2.get("海冰浓度 (SIC)")
                                    wave_n = series2.get("波高 (SWH)")
                                    edl_risk_n = series2.get("EDL 风险")
                                    edl_unc_n = series2.get("EDL 不确定性")

                                    candidates = []
                                    if sic_n is not None and wave_n is not None:
                                        s_s = smooth(_norm(sic_n))
                                        w_s = smooth(_norm(wave_n))
                                        e_s = smooth(_norm(edl_risk_n)) if edl_risk_n is not None else None
                                        u_s = smooth(_norm(edl_unc_n)) if edl_unc_n is not None else None
                                        high = (s_s > 0.6) & (w_s > 0.6) & ((e_s > 0.6) if e_s is not None else (u_s > 0.6 if u_s is not None else False))
                                        if np.any(high):
                                            idx = np.where(high)[0]
                                            seg_lo = int(idx[0])
                                            seg_hi = int(idx[-1])
                                            d0, d1 = dist2[seg_lo], dist2[seg_hi]
                                            candidates.append((float(d0), float(d1)))
                                    if candidates:
                                        d0, d1 = candidates[0]
                                        st.caption(f"在 {d0:.0f}–{d1:.0f} km 区间，海冰和波高同时升高，EDL 风险/不确定性也显著增加，因此 {ROUTE_LABELS.get(base_key, base_key)} 方案在此段选择更保守的绕行策略。")
                                    else:
                                        st.caption(f"沿程上未检测到显著的危险叠加区间；{ROUTE_LABELS.get(base_key, base_key)} 方案总体保持对高风险区域的规避。")
                                except Exception:
                                    pass
                except Exception as e:
                    st.info(f"多维剖面生成失败：{e}")

    with tab_ais:
        ais_rows = []
        for mode in ["efficient", "edl_safe", "edl_robust"]:
            route = routes_info.get(mode)
            if route is None or not route.reachable:
                continue
            components = route.cost_components or {}
            ais_cost = sum(v for k, v in components.items() if k.startswith("ais")) if components else 0.0
            if ais_cost > 0:
                ais_rows.append({"mode": ROUTE_LABELS.get(mode, mode), "ais_cost": ais_cost, "total": route.total_cost or 0.0})
        if ais_rows:
            df_ais = pd.DataFrame(ais_rows)
            st.dataframe(df_ais, use_container_width=True)
            st.bar_chart(df_ais.set_index("mode")["ais_cost"])
        else:
            st.info("当前成本构建未启用 AIS 成本（权重为 0 或缺少 AIS 数据）。")

    st.subheader("📥 导出当前规划结果")
    export_data = []
    for mode, route in routes_info.items():
        if route.reachable:
            breakdown = _get_breakdown_for_route(mode, route)
            export_data.append({
                "scenario": selected_scenario_name,
                "mode": mode,
                "reachable": True,
                "distance_km": route.approx_length_km,
                "total_cost": breakdown.total_cost if breakdown else route.total_cost,
                "edl_risk_cost": breakdown.component_totals.get("edl_risk", None) if breakdown else None,
                "edl_unc_cost": breakdown.component_totals.get("edl_uncertainty_penalty", None) if breakdown else None,
                "ice_cost": breakdown.component_totals.get("ice_risk", None) if breakdown else None,
                "wave_cost": breakdown.component_totals.get("wave_risk", None) if breakdown else None,
                "ice_class_soft_cost": breakdown.component_totals.get("ice_class_soft", None) if breakdown else None,
                "ice_class_hard_cost": breakdown.component_totals.get("ice_class_hard", None) if breakdown else None,
                "vessel_profile": selected_vessel_key,
                "use_real_data": cost_mode == "real_sic_if_available",
                "cost_mode": cost_mode,
                "grid_source": grid_source_label,
            })
    
    if export_data:
        df_export = pd.DataFrame(export_data)
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 下载当前规划结果 (CSV)",
            data=csv_bytes,
            file_name=f"{selected_scenario_name}_{selected_edl_mode}_results.csv",
            mime="text/csv",
            key="download_csv_new",
        )

        # === UX-3：一键导出当前规划报告（Markdown） ===
        st.subheader("🧾 导出本次规划报告 (Markdown)")

        def _get_costs_for_row(mode_key: str, route_obj: RouteInfo):
            b = _get_breakdown_for_route(mode_key, route_obj)
            total = b.total_cost if b else route_obj.total_cost
            edl_r = b.component_totals.get("edl_risk", None) if b else None
            edl_u = b.component_totals.get("edl_uncertainty_penalty", None) if b else None
            return total, edl_r, edl_u

        def _fmt(v, fmt=".2f"):
            try:
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    return "-"
                return f"{float(v):{fmt}}"
            except Exception:
                return str(v)

        # 标题与概览
        now_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        title = f"# ArcticRoute 规划报告 | 场景：{selected_scenario_name} | {now_str}\n\n"
        subtitle = (
            f"- Grid 模式：{'real' if grid_mode == 'real' else 'demo'}\n"
            f"- 成本模式：{cost_mode}\n"
            f"- 船型：{selected_vessel.name} ({selected_vessel_key})\n"
            f"- EDL 模式：{EDL_MODES.get(selected_edl_mode, {}).get('display_name', selected_edl_mode)}"
        )

        # 参数区
        params_md = (
            "\n## 参数配置\n\n"
            f"- 起点：({start_lat:.2f}, {start_lon:.2f})\n"
            f"- 终点：({end_lat:.2f}, {end_lon:.2f})\n"
            f"- 冰带权重（基础）：4.0\n"
            f"- 波浪权重 wave_penalty：{wave_penalty:.2f}\n"
            f"- AIS 主航线偏好 w_corridor：{w_ais_corridor:.2f}\n"
            f"- AIS 拥挤惩罚 w_congestion：{w_ais_congestion:.2f}\n"
            f"- AIS 旧版权重 w_ais（兼容）：{w_ais:.2f}\n"
            f"- EDL 启用：{'是' if use_edl else '否'}\n"
            f"- EDL 权重 w_edl：{(w_edl if use_edl else 0.0):.2f}\n"
            f"- 不确定性权重：{edl_uncertainty_weight:.2f}\n"
        )

        # 三方案表格
        rows_md = ["\n## 三条路线摘要\n\n", "| 方案 | 距离 (km) | 总成本 | EDL 风险 | EDL 不确定性 | 踩陆 |", "|---|---:|---:|---:|---:|:--:|"]
        for mk in ["efficient", "edl_safe", "edl_robust"]:
            r = routes_info.get(mk)
            if r is None:
                continue
            total, edl_r, edl_u = _get_costs_for_row(mk, r) if r.reachable else (None, None, None)
            rows_md.append(
                f"| {ROUTE_LABELS.get(mk, mk)} | "
                f"{_fmt(r.distance_km or r.approx_length_km, '.1f')} | "
                f"{_fmt(total)} | {_fmt(edl_r)} | {_fmt(edl_u)} | "
                f"{'是' if (r.on_land_steps or 0) > 0 else '否'} |"
            )
        routes_md = "\n".join(rows_md) + "\n\n"

        # 评估结果（如果存在）
        eval_md = ""
        try:
            eval_path = Path(__file__).resolve().parents[2] / "reports" / "eval_mode_comparison.csv"
            if eval_path.exists():
                df_eval = pd.read_csv(eval_path)
                # 尝试匹配场景列
                scen_col = None
                for c in ["scenario", "scenario_id", "scenario_name"]:
                    if c in df_eval.columns:
                        scen_col = c
                        break
                if scen_col is not None:
                    sub = df_eval[df_eval[scen_col] == selected_scenario_name]
                    if not sub.empty:
                        # 尝试解析 edl_safe / edl_robust 指标
                        def pick_row(mode_name):
                            mode_col = "mode" if "mode" in sub.columns else None
                            return sub[sub[mode_col] == mode_name] if mode_col else pd.DataFrame()

                        def fmt_eval_row(r: pd.Series) -> str:
                            keys = {
                                "Δdist(km)": ["delta_distance_km", "delta_dist_km", "d_dist_km", "Δdist(km)"],
                                "Δdist(%)": ["delta_distance_pct", "Δdist(%)"],
                                "Δcost": ["delta_cost", "Δcost"],
                                "Δcost(%)": ["delta_cost_pct", "Δcost(%)"],
                                "risk_red(%)": ["risk_reduction_pct", "risk_red(%)"],
                            }
                            parts = []
                            for label, cands in keys.items():
                                val = None
                                for cc in cands:
                                    if cc in r:
                                        val = r[cc]
                                        break
                                if val is not None and pd.notna(val):
                                    parts.append(f"{label}: {val}")
                            return " | ".join(parts) if parts else "指标缺失"

                        row_safe = pick_row("edl_safe")
                        row_rob = pick_row("edl_robust")
                        eval_md = "## 场景评估（相对 Efficient）\n\n"
                        if not row_safe.empty:
                            eval_md += f"- EDL-Safe：{fmt_eval_row(row_safe.iloc[0])}\n"
                        else:
                            eval_md += "- EDL-Safe：未找到评估行\n"
                        if not row_rob.empty:
                            eval_md += f"- EDL-Robust：{fmt_eval_row(row_rob.iloc[0])}\n"
                        else:
                            eval_md += "- EDL-Robust：未找到评估行\n"
                        eval_md += "\n"
        except Exception as e:
            eval_md = f"_评估结果读取失败：{e}_\n\n"

        # 自动结论：基于本次规划结果
        concl_md = "## 结论摘要\n\n"
        eff = routes_info.get("efficient")
        esafe = routes_info.get("edl_safe")
        erob = routes_info.get("edl_robust")
        def _risk_total_of(rk, r):
            b = _get_breakdown_for_route(rk, r)
            if not b:
                return None
            comp = b.component_totals or {}
            vals = [comp.get(k, 0.0) for k in ["edl_risk", "edl_uncertainty_penalty", "ice_risk", "wave_risk"]]
            vals = [float(v) for v in vals if v is not None]
            return sum(vals) if vals else None
        lines = []
        if eff and esafe and eff.reachable and esafe.reachable and (eff.distance_km or eff.approx_length_km):
            d0 = eff.distance_km or eff.approx_length_km
            d1 = esafe.distance_km or esafe.approx_length_km
            if d0 and d0 > 0 and d1 is not None:
                dist_pct = (d1 - d0) / d0 * 100.0
                r0 = _risk_total_of("efficient", eff)
                r1 = _risk_total_of("edl_safe", esafe)
                risk_pct = None
                if r0 and r0 > 0 and r1 is not None:
                    risk_pct = (r0 - r1) / r0 * 100.0
                s = f"EDL-Safe 相比 Efficient，距离变化约 {dist_pct:.1f}%。"
                if risk_pct is not None:
                    s += f" 风险下降约 {risk_pct:.1f}%。"
                lines.append(s)
        if eff and erob and eff.reachable and erob.reachable and (eff.distance_km or eff.approx_length_km):
            d0 = eff.distance_km or eff.approx_length_km
            d1 = erob.distance_km or erob.approx_length_km
            if d0 and d0 > 0 and d1 is not None:
                dist_pct = (d1 - d0) / d0 * 100.0
                r0 = _risk_total_of("efficient", eff)
                r1 = _risk_total_of("edl_robust", erob)
                risk_pct = None
                if r0 and r0 > 0 and r1 is not None:
                    risk_pct = (r0 - r1) / r0 * 100.0
                s = f"EDL-Robust 相比 Efficient，距离变化约 {dist_pct:.1f}%。"
                if risk_pct is not None:
                    s += f" 风险下降约 {risk_pct:.1f}%。"
                lines.append(s)
        if not lines:
            lines.append("本次规划未能生成可比较的两条路线，或关键指标缺失。")
        concl_md += "\n\n".join(lines) + "\n\n"

        full_md = title + subtitle + "\n\n" + params_md + routes_md + eval_md + concl_md
        st.download_button(
            label="导出本次规划报告 (Markdown)",
            data=full_md.encode("utf-8"),
            file_name=f"{selected_scenario_name}_{selected_edl_mode}_report.md",
            mime="text/markdown",
            key="download_md_report",
        )
    else:
        st.warning("当前无可导出的结果。")

    eval_tab, = st.tabs(["📑 EDL 评估结果"]) 
    with eval_tab:
        try:
            from arcticroute.ui import eval_results as eval_ui_results
            df_eval = eval_ui_results.load_eval_results()
            if df_eval is None:
                st.info("reports/eval_mode_comparison.csv 不存在，先运行 `python -m scripts.eval_scenario_results` 生成。")
            else:
                st.markdown("**评估结果（相对 efficient 基线）**")
                eval_ui_results.render_scenario_table(df_eval)
                col_l, col_r = st.columns([2, 1])
                with col_l:
                    eval_ui_results.render_scatter_plot(df_eval)
                with col_r:
                    summary = eval_ui_results.generate_global_summary(df_eval)
                    eval_ui_results.render_summary_stats(summary)
                st.markdown(eval_ui_results.generate_conclusion_text(summary))
        except Exception as e:
            st.warning(f"评估结果展示失败：{e}")

    results_tab, = st.tabs(["📊 方案对比"])
    with results_tab:
        st.caption("展示当前场景三条方案的距离 / 成本 / 风险对比，地图与 KPI 卡片位于上方，可使用上方单选转换高亮方案。")

        def _risk_total_for_summary(route_key: str, route: RouteInfo) -> float | None:
            breakdown = _get_breakdown_for_route(route_key, route)
            if not breakdown:
                return None
            comp = breakdown.component_totals or {}
            vals = [comp.get(k, 0.0) for k in ["edl_risk", "edl_uncertainty_penalty", "ice_risk", "wave_risk"]]
            vals = [float(v) for v in vals if v is not None]
            return sum(vals) if vals else None

        eff_route = routes_info.get("efficient")
        base_risk = _risk_total_for_summary("efficient", eff_route) if eff_route and eff_route.reachable else None

        summary_rows = []
        for mode_key in ["efficient", "edl_safe", "edl_robust"]:
            route = routes_info.get(mode_key)
            if route is None or not route.reachable:
                continue
            risk_total = _risk_total_for_summary(mode_key, route)
            risk_reduction = None
            if mode_key != "efficient" and base_risk is not None and base_risk > 0 and risk_total is not None:
                risk_reduction = (base_risk - risk_total) / base_risk * 100.0
            summary_rows.append({
                "mode": mode_key,
                "distance": route.distance_km or route.approx_length_km,
                "total_cost": route.total_cost,
                "risk_reduction": risk_reduction,
            })

        if summary_rows:
            df_small = pd.DataFrame([
                {
                    "方案": ROUTE_LABELS_ZH.get(item["mode"], item["mode"]),
                    "距离 (km)": item["distance"],
                    "总成本": item["total_cost"],
                    "风险降低 (%)": item["risk_reduction"],
                }
                for item in summary_rows
            ])
            st.dataframe(df_small, use_container_width=True, hide_index=True)
        else:
            st.info("暂无可比较的方案，请先完成规划。")

        if st.button("🗂 导出当前场景结果包（ZIP）"):
            with st.spinner("正在生成结果包..."):
                env_meta = {
                    "ym": st.session_state.get("ym"),
                    "grid_mode": grid_mode,
                    "cost_mode": cost_mode,
                    "wave_penalty": wave_penalty,
                    "w_ais": w_ais,
                    "w_ais_corridor": w_ais_corridor,
                    "w_ais_congestion": w_ais_congestion,
                    "edl_mode": selected_edl_mode,
                    "use_edl": use_edl,
                    "edl_uncertainty_weight": edl_uncertainty_weight,
                    "vessel": selected_vessel.name if selected_vessel else None,
                }
                zip_path = build_defense_bundle(
                    scenario_id=selected_scenario_id,
                    routes_info=list(routes_info.values()),
                    env_meta=env_meta,
                    eval_summary=None,
                )
                data = zip_path.read_bytes()
            st.download_button(
                "下载结果包",
                data=data,
                file_name=zip_path.name,
                mime="application/zip",
            )

        with st.expander("查看全部批量评估结果（高级）", expanded=False):
            st.caption("该表来自脚本 run_scenario_suite 的批量离线评估，可用于论文和答辩数据支撑。")
            results_path = Path(__file__).resolve().parents[2] / "reports" / "scenario_suite_results.csv"
            if results_path.exists():
                df_results = pd.read_csv(results_path)
                scenario_ids = ["全部"] + sorted(df_results["scenario_id"].unique())
                grid_modes = ["全部"] + sorted(df_results["grid_mode"].unique())
                modes = ["全部"] + sorted(df_results["mode"].unique())

                scen_choice = st.selectbox("场景 ID", options=scenario_ids, index=0)
                grid_choice = st.selectbox("选择 grid_mode", options=grid_modes, index=0)
                mode_choice = st.selectbox("选择模式", options=modes, index=0)

                filtered = df_results.copy()
                if scen_choice != "全部":
                    filtered = filtered[filtered["scenario_id"] == scen_choice]
                if grid_choice != "全部":
                    filtered = filtered[filtered["grid_mode"] == grid_choice]
                if mode_choice != "全部":
                    filtered = filtered[filtered["mode"] == mode_choice]

                st.dataframe(filtered, use_container_width=True)

                if not filtered.empty:
                    try:
                        import altair as alt

                        chart = (
                            alt.Chart(filtered)
                            .mark_circle(size=70, opacity=0.8)
                            .encode(
                                x=alt.X("distance_km:Q", title="Distance (km)"),
                                y=alt.Y("total_cost:Q", title="Total cost"),
                                color=alt.Color("mode:N", title="Mode"),
                                tooltip=["scenario_id", "mode", "grid_mode", "distance_km", "total_cost"],
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)
                    except Exception as e:
                        st.info(f"可视化加载失败：{e}")
            else:
                st.info("reports/scenario_suite_results.csv 不存在，暂无结果浏览。")

    return
    # 构造地图数据
    path_data = []
    color_map = {
        "efficient": [0, 128, 255],      # 蓝色
        "balanced": [255, 140, 0],       # 橙色
        "safe": [255, 0, 80],            # 红色
    }
    
    print("[DEBUG RENDER] ===== Constructing path_data for map =====")
    for route_info in reachable_routes:
        coords = route_info.coords or []
        
        # 防守性检查：过滤掉无效的坐标
        valid_coords = []
        for lat, lon in coords:
            if _is_valid_coord(lat, lon):
                valid_coords.append((lat, lon))
            else:
                print(f"[DEBUG RENDER] Skipping invalid coord: lat={lat}, lon={lon}")
        
        if not valid_coords:
            print(f"[DEBUG RENDER] Route '{route_info.label}' has no valid coordinates, skipping")
            continue
        
        # 构造 pydeck 路径数据（经度在前，纬度在后）
        path_list = []
        for lat, lon in valid_coords:
            wrapped_lon = _wrap_lon(float(lon))
            path_list.append([wrapped_lon, float(lat)])
        
        if not path_list:
            print(f"[DEBUG RENDER] Route '{route_info.label}' has no valid path points after wrapping, skipping")
            continue
        
        path_data.append({
            "name": route_info.label,
            "color": color_map.get(route_info.label, [128, 128, 128]),
            "path": path_list,
        })
        
        print(f"[DEBUG RENDER] Route '{route_info.label}': {len(path_list)} valid points, "
              f"first=[{path_list[0][1]}, {path_list[0][0]}], "
              f"last=[{path_list[-1][1]}, {path_list[-1][0]}]")
    
    print(f"[DEBUG RENDER] Total routes in path_data: {len(path_data)}")
    print("[DEBUG RENDER] ===== End Constructing path_data =====\n")
    
    # 绘制地图（支持 Plotly 备用渲染）
    st.subheader("规划路线（三方案对比）")
    use_plotly_fallback = st.checkbox(
        "地图显示异常？使用 Plotly 备用渲染",
        value=False,
        help="若 pydeck 在本机浏览器/WebGL 环境下无法显示，可勾选此项。",
    )

    def _render_with_plotly(data_items):
        try:
            import plotly.graph_objects as go
            fig = go.Figure()
            for item in data_items:
                lons = [pt[0] for pt in item["path"]]
                lats = [pt[1] for pt in item["path"]]
                color = item.get("color", [128, 128, 128])
                width = item.get("width", 2)
                color_str = f"rgb({color[0]},{color[1]},{color[2]})"
                fig.add_trace(
                    go.Scattergeo(
                        lon=lons,
                        lat=lats,
                        mode="lines",
                        line=dict(width=width, color=color_str),
                        name=item.get("name", "route"),
                    )
                )
            fig.update_layout(
                geo=dict(showland=True, showcountries=True, projection_type="natural earth"),
                margin=dict(l=0, r=0, t=0, b=0),
                legend=dict(orientation="h"),
            )
            st.plotly_chart(fig, use_container_width=True)
            return True
        except Exception as e:
            st.error(f"Plotly 备用渲染失败：{e}")
            return False

    rendered = False
    if not use_plotly_fallback:
        try:
            import pydeck as pdk
            # 计算地图中心
            all_points = [pt for item in path_data for pt in item["path"]]
            avg_lon = np.mean([p[0] for p in all_points])
            avg_lat = np.mean([p[1] for p in all_points])
            # 创建路径层
            layer = pdk.Layer(
                "PathLayer",
                data=path_data,
                get_path="path",
                get_width="width",
                get_color="color",
                pickable=True,
            )
            view_state = pdk.ViewState(
                latitude=ARCTIC_VIEW["latitude"],
                longitude=ARCTIC_VIEW["longitude"],
                zoom=ARCTIC_VIEW["zoom"],
                pitch=ARCTIC_VIEW["pitch"],
                min_zoom=ARCTIC_VIEW["min_zoom"],
                max_zoom=ARCTIC_VIEW["max_zoom"],
            )
            # 兼容不同版本的 pydeck：部分版本不支持 controller 参数
            try:
                deck_obj_2 = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={"text": "{name}"},
                    controller=MAP_CONTROLLER,
                )
            except TypeError:
                deck_obj_2 = pdk.Deck(
                    layers=[layer],
                    initial_view_state=view_state,
                    map_style="mapbox://styles/mapbox/dark-v11",
                    tooltip={"text": "{name}"},
                )
            st.pydeck_chart(
                deck_obj_2,
                use_container_width=True
            )
            rendered = True
        except ImportError:
            st.warning("pydeck 未安装，将尝试使用 Plotly 备用渲染。请运行 `pip install pydeck` 以启用原生地图渲染。")
        except Exception as e:
            st.warning(f"pydeck 渲染失败：{e}，自动切换到 Plotly 备用渲染。")

    if not rendered:
        _render_with_plotly(path_data)
    
    # 显示船舶冰级信息
    st.subheader("当前船舶配置")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("船型", selected_vessel.name)
    with col2:
        st.metric("最大冰厚 (m)", f"{selected_vessel.max_ice_thickness_m:.2f}")
    with col3:
        effective_thickness = selected_vessel.get_effective_max_ice_thickness()
        st.metric("有效最大冰厚 (m)", f"{effective_thickness:.2f}")
    
    st.caption(f"安全裕度系数: {selected_vessel.ice_margin_factor:.0%}")
    
    # ========================================================================
    # Step 4: 显示摘要表格（包含 EDL 成本列）
    # ========================================================================
    st.subheader("方案摘要")
    st.caption(f"Grid source: {grid_source_label}, Cost mode: {cost_mode}, wave_penalty={wave_penalty}")
    
    summary_data = []
    for i, route_info in enumerate(routes_info):
        # 计算该路线的成本分解（用于获取 EDL 成本）
        edl_risk_cost = 0.0
        edl_uncertainty_cost = 0.0
        
        if route_info.reachable:
            # 从 cost_fields 中获取对应的成本场
            # cost_fields 的 key 是 profile_key（efficient, edl_safe, edl_robust）
            profile_key = ROUTE_PROFILES[i]["key"]
            cost_field = cost_fields.get(profile_key)
            
            if cost_field is not None and cost_field.components:
                breakdown = compute_route_cost_breakdown(
                    grid, cost_field, route_info.coords
                )
                edl_risk_cost = breakdown.component_totals.get("edl_risk", 0.0)
                edl_uncertainty_cost = breakdown.component_totals.get("edl_uncertainty_penalty", 0.0)
        
        summary_data.append({
            "方案": route_info.label,
            "可达": "✓" if route_info.reachable else "✗",
            "路径点数": route_info.steps if route_info.steps is not None else "-",
            "粗略距离_km": (
                f"{route_info.approx_length_km:.1f}"
                if route_info.approx_length_km is not None
                else "-"
            ),
            "distance_km": f"{route_info.distance_km:.1f}" if route_info.distance_km > 0 else "-",
            "travel_time_h": f"{route_info.travel_time_h:.1f}" if route_info.travel_time_h > 0 else "-",
            "fuel_total_t": f"{route_info.fuel_total_t:.2f}" if route_info.fuel_total_t > 0 else "-",
            "co2_total_t": f"{route_info.co2_total_t:.2f}" if route_info.co2_total_t > 0 else "-",
            "EDL风险成本": f"{edl_risk_cost:.2f}" if edl_risk_cost > 0 else "-",
            "EDL不确定性成本": f"{edl_uncertainty_cost:.2f}" if edl_uncertainty_cost > 0 else "-",
            "冰带权重": route_info.ice_penalty,
            "允许对角线": "是" if route_info.allow_diag else "否",
            "on_land_steps": route_info.on_land_steps,
            "on_ocean_steps": route_info.on_ocean_steps,
        })
    
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, width='stretch')
    
    st.caption("ECO 模块为简化版估算，仅用于 demo，对绝对数值不要过度解读。")
    
    # ========================================================================
    # Step 4: 显示推荐路线和评分表
    # ========================================================================
    st.subheader("🎯 路线推荐与评分")
    
    # 显示推荐路线
    recommended_label = None
    for i, route_info in enumerate(routes_info):
        profile_key = ROUTE_PROFILES[i]["key"]
        if profile_key == recommended_key:
            recommended_label = route_info.label
            break
    
    if recommended_label:
        st.success(f"✅ 当前偏好下推荐路线：**{recommended_label}**（综合评分最低）")
        
        # 根据推荐路线给出提示
        if recommended_key == "edl_robust":
            st.info("💡 EDL-Robust 方案更保守，更规避高不确定性区域，适合风险厌恶型用户。")
        elif recommended_key == "edl_safe":
            st.info("💡 EDL-Safe 方案平衡风险和燃油，适合综合考虑的用户。")
        elif recommended_key == "efficient":
            st.info("💡 Efficient 方案偏向燃油经济性，适合成本敏感型用户。")
    
    # 构造评分表
    st.write("**各方案综合评分对比**")
    
    score_rows = []
    for i, route_info in enumerate(routes_info):
        profile_key = ROUTE_PROFILES[i]["key"]
        score = scores_by_key.get(profile_key)
        
        if score is None:
            continue
        
        # 标记推荐方案
        is_recommended = "⭐" if profile_key == recommended_key else ""
        
        score_rows.append({
            "方案": f"{is_recommended} {route_info.label}".strip(),
            "距离_km": f"{score.distance_km:.1f}",
            "燃油_t": f"{score.fuel_t:.2f}" if score.fuel_t is not None else "-",
            "EDL风险成本": f"{score.edl_risk_cost:.2f}",
            "EDL不确定性成本": f"{score.edl_uncertainty_cost:.2f}",
            "综合评分": f"{score.composite_score:.3f}",
        })
    
    df_scores = pd.DataFrame(score_rows)
    st.dataframe(df_scores, width='stretch')
    
    # 显示归一化指标（用于理解评分）
    st.write("**归一化指标（0=最优，1=最差）**")
    
    norm_rows = []
    for i, route_info in enumerate(routes_info):
        profile_key = ROUTE_PROFILES[i]["key"]
        score = scores_by_key.get(profile_key)
        
        if score is None:
            continue
        
        norm_rows.append({
            "方案": route_info.label,
            "距离": f"{score.norm_distance:.2f}",
            "燃油": f"{score.norm_fuel:.2f}",
            "EDL风险": f"{score.norm_edl_risk:.2f}",
            "EDL不确定性": f"{score.norm_edl_uncertainty:.2f}",
        })
    
    df_norm = pd.DataFrame(norm_rows)
    st.dataframe(df_norm, width='stretch')
    
    # 绘制对比图表
    st.write("**综合评分条形图**")
    
    chart_data = []
    for i, route_info in enumerate(routes_info):
        profile_key = ROUTE_PROFILES[i]["key"]
        score = scores_by_key.get(profile_key)
        
        if score is not None:
            chart_data.append({
                "方案": route_info.label,
                "综合评分": score.composite_score,
            })
    
    if chart_data:
        df_chart = pd.DataFrame(chart_data)
        st.bar_chart(df_chart.set_index("方案"))
    
    # EDL 成本对比
    st.write("**EDL 成本对比（风险 vs 不确定性）**")
    
    edl_chart_data = []
    for i, route_info in enumerate(routes_info):
        profile_key = ROUTE_PROFILES[i]["key"]
        score = scores_by_key.get(profile_key)
        
        if score is not None:
            edl_chart_data.append({
                "方案": route_info.label,
                "EDL风险": score.edl_risk_cost,
                "EDL不确定性": score.edl_uncertainty_cost,
            })
    
    if edl_chart_data:
        df_edl = pd.DataFrame(edl_chart_data)
        st.bar_chart(df_edl.set_index("方案"))
    
    # 检查是否有路线踩陆
    if any((info.get("on_land_steps", 0) or 0) > 0 for info in summary_data):
        st.error("警告：根据当前 landmask，有路线踩到了陆地，请检查成本场或掩码数据。")
    else:
        st.success("根据当前 landmask，三条路线均未踩陆（demo 世界下行为正常）。")
    
    # ========================================================================
    # Step 4: 三路线成本对比图表
    # ========================================================================
    st.subheader("三方案成本对比")
    
    # 提取可达路线的成本数据
    comparison_data = []
    for i, route_info in enumerate(routes_info):
        if route_info.reachable:
            profile_key = ROUTE_PROFILES[i]["key"]
            cost_field = cost_fields.get(profile_key)
            
            if cost_field is not None and cost_field.components:
                breakdown = compute_route_cost_breakdown(
                    grid, cost_field, route_info.coords
                )
                
                comparison_data.append({
                    "方案": route_info.label,
                    "总成本": breakdown.total_cost,
                    "EDL风险": breakdown.component_totals.get("edl_risk", 0.0),
                    "EDL不确定性": breakdown.component_totals.get("edl_uncertainty_penalty", 0.0),
                })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        
        # 绘制对比柱状图
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**总成本对比**")
            chart_data = df_comparison[["方案", "总成本"]].set_index("方案")
            st.bar_chart(chart_data)
        
        with col2:
            st.write("**EDL 成本对比**")
            edl_data = df_comparison[["方案", "EDL风险", "EDL不确定性"]].set_index("方案")
            st.bar_chart(edl_data)
        
        # 检查是否有高不确定性路线
        for _, row in df_comparison.iterrows():
            if row["EDL不确定性"] > 0.5:
                st.warning(
                    f"⚠️ {row['方案']} 在 EDL 不确定性成本上较高（{row['EDL不确定性']:.2f}），"
                    f"建议与其它方案对比权衡。"
                )
    
    # 显示详细信息
    st.subheader("详细信息")
    
    for route_info in reachable_routes:
        with st.expander(f"方案：{route_info.label}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**冰带权重**: {route_info.ice_penalty}")
                st.write(f"**路径点数**: {route_info.steps}")
                st.write(f"**粗略距离**: {route_info.approx_length_km:.1f} km")
            
            with col2:
                st.write(f"**起点**: {route_info.coords[0]}")
                st.write(f"**终点**: {route_info.coords[-1]}")
                st.write(f"**允许对角线**: {'是' if route_info.allow_diag else '否'}")
            
            # 显示部分路径点
            st.write("**部分路径点（前 5 / 后 5）**")
            head = route_info.coords[:5]
            tail = route_info.coords[-5:] if len(route_info.coords) > 5 else []
            st.write({
                "head (前 5)": head,
                "tail (后 5)": tail,
            })
    
    # ========================================================================
    # Phase EDL-CORE Step 4: UI 端的来源感知展示优化
    # ========================================================================
    # 成本分解标签映射
    COMPONENT_LABELS = {
        "base_distance": "基础距离成本",
        "ice_risk": "冰风险",
        "wave_risk": "波浪风险",
        "ice_class_soft": "⚠️ 冰级软约束",
        "ice_class_hard": "🚫 冰级硬限制",
        "edl_risk": "🧠 EDL 风险",
        "edl_uncertainty_penalty": "❓ EDL 不确定性",
        "ais_density": "🚢 AIS 拥挤度 (deprecated)",
        "ais_corridor": "🧭 AIS 主航线偏好（corridor）",
        "ais_congestion": "🚦 AIS 拥挤惩罚（congestion）",
    }
    
    # 成本分解展示（重点看 edl_safe 方案）
    st.subheader("成本分解（edl_safe 方案）")
    
    # 查找 edl_safe 方案
    edl_safe_route = None
    for idx, route_info in enumerate(routes_info):
        if ROUTE_PROFILES[idx]["key"] == "edl_safe" and route_info.reachable:
            edl_safe_route = route_info
            break
    
    if edl_safe_route is None:
        st.info("edl_safe 方案不可达或未规划，无法显示成本分解。")
    else:
        # 获取 edl_safe 方案的成本场
        cost_field = cost_fields.get("edl_safe")
        if cost_field is None or not cost_field.components:
            st.warning("成本场未包含组件分解信息，无法显示成本分解。")
        else:
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(
                grid, cost_field, edl_safe_route.coords
            )
            
            # 显示成本分解表格
            if breakdown.component_totals:
                # 若开启 EDL 但未产生 edl_risk 分量，给出提示
                if use_edl and "edl_risk" not in breakdown.component_totals:
                    st.info("EDL 已开启，但当前环境下未产生有效的 EDL 风险分量（可能是缺少模型或真实环境数据）。")
                
                # 若开启 AIS 但未产生 corridor/congestion 分量，给出提示
                if ais_weights_enabled and not any(
                    key in breakdown.component_totals for key in ["ais_corridor", "ais_congestion", "ais_density"]
                ):
                    st.info("AIS 权重已开启，但当前环境下未产生有效的 AIS 分量（可能是缺少 AIS 数据或维度不匹配）。")

                breakdown_data = []
                for comp_name in sorted(breakdown.component_totals.keys()):
                    comp_value = breakdown.component_totals[comp_name]
                    comp_frac = breakdown.component_fractions.get(comp_name, 0.0)
                    
                    # 使用友好的标签映射
                    comp_label = COMPONENT_LABELS.get(comp_name, comp_name)
                    
                    # 为冰级组件添加特殊标记
                    if comp_name == "ice_class_hard":
                        comp_label = f"🚫 {comp_label}"
                    elif comp_name == "ice_class_soft":
                        comp_label = f"⚠️ {comp_label}"
                    elif comp_name == "edl_risk":
                        # 添加 EDL 来源标记
                        edl_source = cost_field.components.get("edl_risk_source", "unknown")
                        if hasattr(cost_field, 'meta') and cost_field.meta:
                            edl_source = cost_field.meta.get("edl_source", "unknown")
                        
                        # 根据来源添加标签
                        if edl_source == "miles-guess":
                            comp_label = f"🧠 {comp_label} [miles-guess]"
                        elif edl_source == "pytorch":
                            comp_label = f"🧠 {comp_label} [PyTorch]"
                        else:
                            comp_label = f"🧠 {comp_label}"
                    elif comp_name == "ais_density":
                        # AIS 标签已经包含 🚢 emoji，这里保持原样
                        pass
                    
                    breakdown_data.append({
                        "component": comp_label,
                        "total_contribution": f"{comp_value:.2f}",
                        "fraction": f"{comp_frac:.2%}",
                    })
                
                df_breakdown = pd.DataFrame(breakdown_data)
                st.dataframe(df_breakdown, width='stretch')
                
                # AIS 走廊贴合度
                st.markdown("### AIS 走廊贴合度")
                ais_grid = None
                if "ais_density" in locals():
                    ais_grid = ais_density
                if ais_grid is None and "ais_density" in cost_field.components:
                    ais_grid = cost_field.components["ais_density"]
                
                if ais_grid is None:
                    st.caption("当前未加载 AIS 拥挤度数据，无法评估走廊贴合度。")
                else:
                    try:
                        stats = evaluate_route_vs_ais_density(
                            route_latlon=edl_safe_route.coords,
                            grid_lats=grid.lat2d,
                            grid_lons=grid.lon2d,
                            ais_density=ais_grid,
                        )
                        st.write(
                            f"- 平均 AIS 密度：{stats.mean_density:.4f}\n"
                            f"- 高密度走廊占比：{stats.frac_high_corridor*100:.0f}%\n"
                            f"- 低使用水域占比：{stats.frac_low_usage*100:.0f}%"
                        )
                        st.caption(
                            "这条路线有约 "
                            f"{stats.frac_high_corridor*100:.0f}% 的路段位于历史 AIS 高密度走廊，"
                            "越高说明与真实航道越贴合。"
                        )
                        if stats.notes:
                            st.caption("备注：" + "；".join(stats.notes))
                    except Exception as e:
                        st.caption(f"AIS 贴合度计算失败：{e}")
                
                # ================================================================
                # Phase 3 Step 4: EDL 风险贡献度检查与提示
                # ================================================================
                # 检查 EDL 风险贡献是否过小
                if use_edl and "edl_risk" in breakdown.component_totals:
                    edl_risk_cost = breakdown.component_totals["edl_risk"]
                    total_cost = breakdown.total_cost
                    
                    if total_cost > 0:
                        edl_risk_fraction = edl_risk_cost / total_cost
                        
                        # 如果 EDL 风险占比 < 5%，显示提示
                        if edl_risk_fraction < 0.05:
                            st.info(
                                f"💡 **EDL 风险贡献很小**（占比 {edl_risk_fraction*100:.1f}%）。"
                                f"这可能表示：\n"
                                f"1. 当前区域本身环境风险不高（海冰、波浪等较少）\n"
                                f"2. EDL 模型在该区域的预测不敏感\n"
                                f"3. 建议检查 w_edl 权重是否设置过低"
                            )
                
                # 如果有冰级硬约束被触发，显示警告
                if "ice_class_hard" in breakdown.component_totals and breakdown.component_totals["ice_class_hard"] > 0:
                    st.warning(
                        f"⚠️ 警告：该路线经过了冰厚超过船舶能力的区域（硬禁区）。"
                        f"当前船舶最大安全冰厚约 {selected_vessel.get_effective_max_ice_thickness():.2f}m。"
                    )
                
                # 显示柱状图
                st.write("**成本组件贡献（柱状图）**")
                chart_data = pd.DataFrame({
                    "component": list(breakdown.component_totals.keys()),
                    "contribution": list(breakdown.component_totals.values()),
                })
                st.bar_chart(chart_data.set_index("component"))
                
                # 显示剖面图（沿程冰带成本）
                if "ice_risk" in breakdown.component_along_path and breakdown.s_km:
                    st.write("**沿程冰带成本剖面**")
                    profile_data = pd.DataFrame({
                        "distance_km": breakdown.s_km,
                        "ice_risk": breakdown.component_along_path["ice_risk"],
                    })
                    st.line_chart(profile_data.set_index("distance_km"))
            else:
                st.info("该方案的成本分解为空。")
    
    # EDL 不确定性剖面展示
    if use_edl:
        # 查找 edl_robust 方案
        edl_robust_route = None
        for idx, route_info in enumerate(routes_info):
            if ROUTE_PROFILES[idx]["key"] == "edl_robust" and route_info.reachable:
                edl_robust_route = route_info
                break
        
        if edl_robust_route is not None:
            st.subheader("EDL 不确定性沿程剖面（edl_robust）")
            
            cost_field = cost_fields.get("edl_robust")
            if cost_field is not None:
                # 计算路线剖面
                profile = compute_route_profile(edl_robust_route.coords, cost_field)
                
                if profile.edl_uncertainty is not None and np.any(np.isfinite(profile.edl_uncertainty)):
                    # 构造数据框
                    df_unc = pd.DataFrame({
                        "距离_km": profile.distance_km,
                        "EDL_不确定性": profile.edl_uncertainty,
                    })
                    
                    # 显示折线图
                    st.line_chart(df_unc.set_index("距离_km"))
                    
                    # 计算高不确定性占比
                    valid = np.isfinite(profile.edl_uncertainty)
                    if np.any(valid):
                        vals = profile.edl_uncertainty[valid]
                        high_mask = vals > 0.7
                        frac_high = float(np.sum(high_mask)) / float(len(vals))
                        
                        st.caption(f"路线中不确定性 > 0.7 的路段比例约为 {frac_high*100:.1f}%")
                        
                        if frac_high > 0.3:
                            st.warning("⚠️ EDL 不确定性较高，建议结合物理风险和人工判断谨慎使用。")
                else:
                    st.info("已启用 EDL，但当前未能获得有效的不确定性剖面（可能是模型或数据未提供 uncertainty）。")
    
    # AIS 成本沿程剖面（仅当有 AIS 分量且启用 AIS 时显示）
    balanced_route = None
    for idx, route_info in enumerate(routes_info):
        if ROUTE_PROFILES[idx]["key"] == "edl_safe" and route_info.reachable:
            balanced_route = route_info
            break

    if balanced_route is not None:
        cost_field = cost_fields.get("edl_safe")
        if cost_field is not None and ais_weights_enabled:
            profile = compute_route_profile(balanced_route.coords, cost_field)
            ais_series = {}
            if profile.components:
                if "ais_corridor" in profile.components:
                    ais_series["AIS_corridor_cost"] = profile.components["ais_corridor"]
                if "ais_congestion" in profile.components:
                    ais_series["AIS_congestion_cost"] = profile.components["ais_congestion"]
                if not ais_series and "ais_density" in profile.components:
                    ais_series["AIS_legacy_cost"] = profile.components["ais_density"]
            if ais_series:
                df_ais = pd.DataFrame({"距离_km": profile.distance_km, **ais_series})
                st.subheader("AIS 成本沿程剖面（edl_safe）")
                st.line_chart(df_ais.set_index("距离_km"))
                st.caption("Corridor：偏好航道；Congestion：惩罚极端拥挤区。")
            else:
                st.info("当前未启用 AIS 成本或未找到 AIS 栅格，暂无 AIS 剖面可展示。")
    
    # ========================================================================
    # Phase 5: 导出当前规划结果
    # ========================================================================
    
    # 完成 render stage 并保存结果到 session_state
    # [removed] render_pipeline timeline (simplified) disabled to avoid duplicate UI
    pipeline.done('render')
    
    # 完成第 8 个节点：渲染与导出
    _update_pipeline_node(7, "done", "渲染完成", seconds=0.5)
    
    # 计算总耗时
    if st.session_state.get("pipeline_flow_start_time") is not None:
        total_time = (datetime.now() - st.session_state.pipeline_flow_start_time).total_seconds()
        # 更新所有节点的总耗时显示（通过重新渲染）
        with st.session_state.pipeline_flow_placeholder.container():
            render_pipeline_flow(
                st.session_state.pipeline_flow_nodes,
                title="🔄 规划流程管线 ✅ 完成",
                expanded=False,  # 完成后自动折叠
            )
    
    # 将规划结果保存到 session_state，以便在 rerun 后仍可用
    st.session_state['last_plan_result'] = {
        'routes_info': routes_info,
        'cost_fields': cost_fields,
        'cost_meta': cost_meta,
        'scores_by_key': scores_by_key,
        'recommended_key': recommended_key,
    }
    
    # 规划完成后自动折叠 pipeline
    st.session_state['pipeline_expanded'] = False
    st.rerun()
    st.subheader("📥 导出当前规划结果")
    
    # 为每个可达的路线生成导出数据
    export_data = []
    for i, route_info in enumerate(routes_info):
        if route_info.reachable:
            profile_key = ROUTE_PROFILES[i]["key"]
            cost_field = cost_fields.get(profile_key)
            
            # 计算成本分解
            breakdown = compute_route_cost_breakdown(
                grid, cost_field, route_info.coords
            ) if cost_field else None
            
            # 构建导出记录
            export_record = {
                "scenario": selected_scenario_name,
                "mode": profile_key,
                "reachable": True,
                "distance_km": route_info.approx_length_km,
                "total_cost": breakdown.total_cost if breakdown else 0.0,
                "edl_risk_cost": breakdown.component_totals.get("edl_risk", None) if breakdown else None,
                "edl_unc_cost": breakdown.component_totals.get("edl_uncertainty_penalty", None) if breakdown else None,
                "ice_cost": breakdown.component_totals.get("ice_risk", None) if breakdown else None,
                "wave_cost": breakdown.component_totals.get("wave_risk", None) if breakdown else None,
                "ice_class_soft_cost": breakdown.component_totals.get("ice_class_soft", None) if breakdown else None,
                "ice_class_hard_cost": breakdown.component_totals.get("ice_class_hard", None) if breakdown else None,
                "vessel_profile": selected_vessel_key,
                "use_real_data": cost_mode == "real_sic_if_available",
                "cost_mode": cost_mode,
                "grid_source": grid_source_label,
            }
            export_data.append(export_record)
    
    if export_data:
        # 创建 DataFrame
        df_export = pd.DataFrame(export_data)
        
        # CSV 导出
        csv_bytes = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 下载当前规划结果 (CSV)",
            data=csv_bytes,
            file_name=f"{selected_scenario_name}_{selected_edl_mode}_results.csv",
            mime="text/csv",
            key="download_csv",
        )
        
        # JSON 导出
        import json
        
        def convert_to_serializable(obj):
            """将 numpy 类型转换为可序列化的 Python 类型。"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        json_data = json.dumps(
            [convert_to_serializable(record) for record in export_data],
            indent=2,
            ensure_ascii=False
        ).encode("utf-8")
        
        st.download_button(
            label="📥 下载当前规划结果 (JSON)",
            data=json_data,
            file_name=f"{selected_scenario_name}_{selected_edl_mode}_results.json",
            mime="application/json",
            key="download_json",
        )
        
        st.caption("✓ 导出数据包含所有可达方案的规划结果，包括距离、成本分量等详细信息。")
    else:
        st.warning("⚠️ 当前无可达方案，无法导出结果。")

    # 批量评测结果
    results_tab, = st.tabs(["批量测试结果"])
    with results_tab:
        results_path = Path(__file__).resolve().parents[2] / "reports" / "scenario_suite_results.csv"
        if results_path.exists():
            df_results = pd.read_csv(results_path)
            scenario_ids = ["全部"] + sorted(df_results["scenario_id"].unique())
            grid_modes = ["全部"] + sorted(df_results["grid_mode"].unique())
            modes = ["全部"] + sorted(df_results["mode"].unique())

            scen_choice = st.selectbox("筛选场景 ID", options=scenario_ids, index=0)
            grid_choice = st.selectbox("筛选 grid_mode", options=grid_modes, index=0)
            mode_choice = st.selectbox("筛选模式", options=modes, index=0)

            filtered = df_results.copy()
            if scen_choice != "全部":
                filtered = filtered[filtered["scenario_id"] == scen_choice]
            if grid_choice != "全部":
                filtered = filtered[filtered["grid_mode"] == grid_choice]
            if mode_choice != "全部":
                filtered = filtered[filtered["mode"] == mode_choice]

            st.dataframe(filtered, use_container_width=True)

            if not filtered.empty:
                try:
                    import altair as alt

                    chart = (
                        alt.Chart(filtered)
                        .mark_circle(size=70, opacity=0.8)
                        .encode(
                            x=alt.X("distance_km:Q", title="Distance (km)"),
                            y=alt.Y("total_cost:Q", title="Total cost"),
                            color=alt.Color("mode:N", title="Mode"),
                            tooltip=["scenario_id", "mode", "grid_mode", "distance_km", "total_cost"],
                        )
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception as e:
                    st.info(f"可视化失败: {e}")
        else:
            st.info("reports/scenario_suite_results.csv 暂未生成，无法展示批量结果。")
