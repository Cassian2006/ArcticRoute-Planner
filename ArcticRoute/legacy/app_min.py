from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as _plt
import numpy as np
import streamlit as st

from ArcticRoute.core import planner_service

# Optional imports
_HAS_PRIOR_UI = False # Default to false
try:
    # Check for module existence without an unused import
    from ArcticRoute.apps import layers_prior
    if layers_prior:
        _HAS_PRIOR_UI = True
except ImportError:
    pass # Module doesn't exist, _HAS_PRIOR_UI remains False
try:
    from streamlit_folium import st_folium
    import folium
    _HAS_FOLIUM = True
except Exception:
    _HAS_FOLIUM = False

st.set_page_config(page_title="ArcticRoute Demo", layout="wide")
st.title("ArcticRoute Minimal Demo")

# --- Constants and Helpers ---
_DEF_CMAPS = ["viridis", "plasma", "magma", "cividis", "Blues", "Reds"]
_ERROR_HINTS = {
    "file_missing:sic": "未找到 SIC 文件。请更换月份或检查 merged 目录。",
    "file_missing:ice": "未找到 ICE COST 文件。仅显示可用图层，路由可能不可用。",
    "no_latlon": "数据不含经纬度坐标，隐藏经纬输入。请使用 (i,j) 索引。",
    "coord_mismatch": "坐标维度不一致或缺失，已回退到简单显示。",
    "route_fail": "规划失败：请尝试调整起止点、时间或关闭斜向移动。",
}

def _hint(kind: str, detail: str | None = None, level: str = "warn") -> None:
    msg = _ERROR_HINTS.get(kind, kind)
    if detail:
        msg = f"{msg} · {detail}"
    if level == "info":
        st.info(msg)
    else:
        st.warning(msg)

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _merged_dir() -> Path:
    return _repo_root() / "ArcticRoute" / "data_processed" / "ice_forecast" / "merged"

def _list_available_months() -> list[str]:
    d = _merged_dir()
    if not d.exists():
        return []
    months: set[str] = set()
    pat = re.compile(r"(sic_fcst|ice_cost)_(\d{6})\.nc$")
    for p in d.glob("*.nc"):
        m = pat.search(p.name)
        if m:
            months.add(m.group(2))
    return sorted(months)

def _pick_latest_month(months: list[str]) -> str | None:
    return months[-1] if months else None

# --- Main App Logic ---

with st.sidebar:
    st.markdown("### 外观与主题")
    _dark_mode = st.checkbox("深色模式", value=False)

# ... (Theme and CSS logic can be kept as is)

with st.sidebar:
    st.subheader("F1-06 · 月份/图层设置")
    months = _list_available_months()
    latest = _pick_latest_month(months)
    use_latest = st.checkbox("使用最新快照", value=True)
    sel_ym = st.selectbox("可用月份 (扫描 merged/)", options=months or ["无可用"], index=(len(months) - 1 if months else 0), disabled=use_latest or not months)
    ym_input = st.text_input("手动输入 YYYYMM", value=(latest or "202412"))
    ym = (latest if use_latest else (sel_ym if months else ym_input)) or ym_input
    alpha = st.slider("α_ice", 0.0, 2.0, 0.6, 0.05)

@st.cache_data
def get_env_ctx(year_month, alpha_val):
    return planner_service.load_environment(year_month, alpha_val)

env_ctx = get_env_ctx(ym, alpha)
st.session_state['env_ctx'] = env_ctx
sic_da = env_ctx.sic_da
ice_da = env_ctx.cost_da

with st.sidebar:
    if st.button("刷新数据", width='stretch'):
        get_env_ctx.clear()
        st.rerun()

    st.markdown("---")
    st.subheader("图层面板")
    # ... (Layer panel UI)

    st.markdown("---")
    st.subheader("Pipeline 运行 / 路由设置")
    time_index = st.number_input("time_index", value=0, step=1)
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("start_i (y)", value=200, step=1, key="inp_si")
        st.number_input("start_j (x)", value=150, step=1, key="inp_sj")
    with c2:
        st.number_input("goal_i (y)", value=320, step=1, key="inp_gi")
        st.number_input("goal_j (x)", value=480, step=1, key="inp_gj")

    grid_shape = None
    if env_ctx.cost_da is not None:
        grid_shape = env_ctx.cost_da.shape[-2:]
    elif env_ctx.sic_da is not None:
        grid_shape = env_ctx.sic_da.shape[-2:]
    
    if grid_shape:
        st.caption(f"有效索引范围: i(y)∈[0, {grid_shape[0]-1}], j(x)∈[0, {grid_shape[1]-1}]")
    else:
        st.caption("无法获取栅格尺寸，请确保数据文件存在。")

    allow_diag = st.checkbox("允许斜向移动", value=True, key="inp_diag")
    heuristic = st.selectbox("启发函数", options=["manhattan", "euclidean", "octile"], index=0, key="inp_heuristic")

    if env_ctx.has_latlon:
        st.caption("已检测到经纬度坐标，可输入经纬度→转换为(i,j)")
        c3, c4 = st.columns(2)
        with c3:
            start_lat = st.number_input("start_lat", value=70.0, step=0.1, key="inp_start_lat")
            start_lon = st.number_input("start_lon", value=-150.0, step=0.1, key="inp_start_lon")
        with c4:
            goal_lat = st.number_input("goal_lat", value=72.0, step=0.1, key="inp_goal_lat")
            goal_lon = st.number_input("goal_lon", value=-160.0, step=0.1, key="inp_goal_lon")
        if st.button("经纬转索引并填入", width='stretch'):
            start_ij = planner_service.latlon_to_ij(env_ctx, start_lat, start_lon)
            goal_ij = planner_service.latlon_to_ij(env_ctx, goal_lat, goal_lon)
            if start_ij and goal_ij:
                st.session_state["inp_si"], st.session_state["inp_sj"] = start_ij[0], start_ij[1]
                st.session_state["inp_gi"], st.session_state["inp_gj"] = goal_ij[0], goal_ij[1]
                st.success(f"映射成功：start=({start_ij[0]},{start_ij[1]}) goal=({goal_ij[0]},{goal_ij[1]})")
                st.rerun()
            else:
                _hint("coord_mismatch", "无法将经纬度映射到索引。")
    else:
        _hint("no_latlon", level="info")

    st.markdown("---")
    if st.button("规划", width='stretch'):
        st.session_state["_do_plan_now"] = True

# --- Tabs ---
live_tab, pipeline_tab, layers_tab, route_tab, summ_tab, report_tab, export_tab, review_tab, health_tab, cache_tab = st.tabs(["Live", "Pipeline", "Layers", "Route", "Summary", "Report", "Export", "Review", "Health", "Cache & Clean"])

def _compute_route_and_store():
    if env_ctx.cost_da is None:
        _hint("file_missing:ice", "缺少 ice_cost 网格，无法规划。")
        return
    try:
        si = int(st.session_state.get("inp_si", 0))
        sj = int(st.session_state.get("inp_sj", 0))
        gi = int(st.session_state.get("inp_gi", 0))
        gj = int(st.session_state.get("inp_gj", 0))
        neighbor8 = bool(st.session_state.get("inp_diag", True))
        heuristic = str(st.session_state.get("inp_heuristic", "manhattan"))
        route_result = planner_service.compute_route(env=env_ctx, start_ij=(si, sj), goal_ij=(gi, gj), allow_diagonal=neighbor8, heuristic=heuristic)
        metrics = planner_service.summarize_route(route_result)
        st.session_state["_last_route_path"] = route_result.path_ij
        st.session_state["_last_route_metrics"] = metrics
    except Exception as _e:
        st.session_state["_last_route_path"] = []
        st.session_state["_last_route_metrics"] = {}
        _hint("route_fail", str(_e))
    finally:
        st.session_state.pop("_do_plan_now", None)
        st.rerun()

if st.session_state.get("_do_plan_now"):
    _compute_route_and_store()

with layers_tab:
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("SIC 预测")
        if sic_da is not None:
            fig, ax = _plt.subplots()
            sic_da.plot.imshow(ax=ax)
            st.pyplot(fig, clear_figure=True)
    with col_r:
        st.subheader("ICE COST 显示")
        if ice_da is not None:
            fig, ax = _plt.subplots()
            ice_da.plot.imshow(ax=ax)
            if st.session_state.get("_last_route_path"):
                rr = st.session_state["_last_route_path"]
                xs = [j for (_i, j) in rr]
                ys = [i for (i, _j) in rr]
                ax.plot(xs, ys, color="#ff006e", lw=2.0, alpha=0.9)
            st.pyplot(fig, clear_figure=True)

with route_tab:
    st.subheader("交互地图")
    if not _HAS_FOLIUM or not env_ctx.has_latlon:
        st.info("Folium 未安装或数据缺少经纬度坐标。")
    else:
        center_lat, center_lon = (np.nanmean(env_ctx.lat_arr), np.nanmean(env_ctx.lon_arr)) if env_ctx.has_latlon else (75, 0)
        fmap = folium.Map(location=[center_lat, center_lon], zoom_start=3)
        if st.session_state.get("_last_route_path"):
            coords = planner_service.path_ij_to_lonlat(env_ctx, st.session_state["_last_route_path"])
            if coords:
                folium.PolyLine(locations=[(c[1], c[0]) for c in coords], color="#ff006e").add_to(fmap)
        st_folium(fmap, height=520)





