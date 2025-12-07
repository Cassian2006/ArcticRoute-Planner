from __future__ import annotations

from typing import Dict, Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from ArcticRoute.apps.registry import UIRegistry

# 可选：plotly 事件联动（若不可用则降级为无联动）
try:  # type: ignore
    from streamlit_plotly_events import plotly_events  # type: ignore
    _HAS_PLOTLY_EVENTS = True
except Exception:  # pragma: no cover
    plotly_events = None  # type: ignore
    _HAS_PLOTLY_EVENTS = False


def _route_key_to_label(key: str) -> str:
    mapping = {
        "balanced": "均衡",
        "safe": "安全优先",
        "efficient": "效率优先",
    }
    return mapping.get(str(key), str(key))


def render_profile_view(route_variants: Dict[str, Any], env_ctx: Any, *, mode: str = "distance") -> None:
    """
    在地图下方渲染“剖面视图”卡片（距离/时间 vs 风险/浪高/CO₂）。

    参数：
    - route_variants: {'balanced'|'safe'|'efficient': RouteResult}
    - env_ctx: EnvironmentContext（用于采样剖面）
    - mode: x 轴模式，"distance" 或 "time"（当前版本仅 distance 有效）

    交互：
    - 鼠标 hover 时，将 {route_key, idx} 写入 st.session_state['profile_hover']，用于地图联动高亮
    """
    if not route_variants or env_ctx is None:
        return

    y_field_map = {
        "综合风险": "risk_total",
        "冰风险": "risk_ice",
        "浪高": "wave_swh",
        "速度": "speed",
        "CO₂ 累积": "co2_cum",
    }

    # UI 卡片外观（玻璃拟态）
    with st.container():
        st.markdown(
            """
            <div style="background: rgba(12,22,42,0.75); border:1px solid rgba(255,255,255,0.08); border-radius:18px; padding:14px 16px; box-shadow: 0 18px 45px rgba(0,0,0,0.65);">
              <div style="display:flex;align-items:center;justify-content:space-between;">
                <div style="font-weight:600;color:#E6EDF7;">剖面视图（多方案沿程对比）</div>
                <div style="font-size:12px;color:#9FB2C8;">Hover 可在地图中联动高亮</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 选择纵轴变量
    try:
        y_name = st.segmented_control("纵轴变量", list(y_field_map.keys()), index=0, key="profile_y_axis_choice")  # type: ignore[attr-defined]
    except Exception:
        y_name = st.radio("纵轴变量", list(y_field_map.keys()), index=0, horizontal=True, key="profile_y_axis_choice")

    # 采样/缓存：尽量复用 st.session_state['route_profiles']，键为 route_key
    prof_cache: Dict[str, Dict[str, Any]] = st.session_state.get("route_profiles", {}) or {}

    # 后端采样函数（通过 planner_service 暴露，避免在组件内重算复杂逻辑）
    try:
        from ArcticRoute.core import planner_service as _ps
    except Exception:
        _ps = None  # type: ignore

    # 收集 DataFrame 行
    rows: List[Dict[str, Any]] = []
    hover_customdata: List[List[Any]] = []  # 与行索引对应

    for rkey, rr in route_variants.items():
        if rr is None or not getattr(rr, "reachable", False):
            continue
        prof = prof_cache.get(rkey)
        if prof is None and _ps is not None:
            try:
                prof = _ps.sample_route_profile(rr, env_ctx)
            except Exception:
                prof = None
            if prof:
                prof_cache[rkey] = prof
        if not prof:
            continue
        # x 轴：距离（km）或时间（h）
        x = np.asarray(prof.get("s"))
        if not isinstance(x, np.ndarray) or x.size == 0:
            continue
        if str(mode) == "time" and isinstance(prof.get("t"), (list, np.ndarray)):
            x = np.asarray(prof.get("t"))
        y_field = y_field_map.get(y_name, "risk_total")
        y = prof.get(y_field)
        if y is None:
            # 无此变量则跳过该路线
            continue
        y_arr = np.asarray(y)
        n = min(len(x), len(y_arr))
        x = x[:n]
        y_arr = y_arr[:n]
        lat_arr = np.asarray(prof.get("lat"))[:n] if prof.get("lat") is not None else np.full(n, np.nan)
        lon_arr = np.asarray(prof.get("lon"))[:n] if prof.get("lon") is not None else np.full(n, np.nan)
        label = _route_key_to_label(rkey)
        for i in range(n):
            rows.append({
                "x": float(x[i]),
                "y": float(y_arr[i]) if np.isfinite(y_arr[i]) else None,
                "route": label,
                "route_key": rkey,
                "idx": i,
                "lat": float(lat_arr[i]) if np.isfinite(lat_arr[i]) else None,
                "lon": float(lon_arr[i]) if np.isfinite(lon_arr[i]) else None,
            })
            hover_customdata.append([rkey, i, float(lat_arr[i]) if np.isfinite(lat_arr[i]) else None, float(lon_arr[i]) if np.isfinite(lon_arr[i]) else None])

    if not rows:
        st.info("暂无可用的剖面数据。")
        return

    df = pd.DataFrame(rows)
    fig = px.line(
        df,
        x="x",
        y="y",
        color="route",
        labels={"x": "距离 (km)" if mode != "time" else "时间 (h)", "y": y_name},
        line_group="route",
        hover_data={"route": True, "idx": True, "lat": True, "lon": True},
    )
    # 提供自定义数据以便事件插件读取 route_key + idx
    try:
        fig.update_traces(customdata=hover_customdata, selector=dict(mode="lines"))
    except Exception:
        pass
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=300, legend_title_text="路线")

    # 热点标记覆盖（可选）
    try:
        ureg = UIRegistry()
        if bool(ureg.is_advanced_enabled("enable_hotspot_markers", False)):
            hotspots = st.session_state.get("route_hotspots_main") or []
            if hotspots:
                # 仅以主线（balanced）为参照叠加
                prof_bal = prof_cache.get("balanced")
                if prof_bal is None and _ps is not None and (route_variants.get("balanced") is not None):
                    try:
                        prof_bal = _ps.sample_route_profile(route_variants.get("balanced"), env_ctx)
                        if prof_bal:
                            prof_cache["balanced"] = prof_bal
                    except Exception:
                        prof_bal = None
                if prof_bal:
                    s_arr = np.asarray(prof_bal.get("s") or [])
                    y_field = y_field_map.get(y_name, "risk_total")
                    y_arr_bal = np.asarray(prof_bal.get(y_field) or [])
                    xs, ys, texts = [], [], []
                    for h in hotspots:
                        try:
                            i = int(h.get("idx", -1))
                            if 0 <= i < len(s_arr):
                                xs.append(float(s_arr[i]))
                                yv = float(y_arr_bal[i]) if i < len(y_arr_bal) and np.isfinite(y_arr_bal[i]) else None
                                ys.append(yv)
                                texts.append(str(h.get("reason", "热点")))
                        except Exception:
                            continue
                    if xs and ys:
                        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="热点", marker=dict(color="rgba(255,100,0,0.95)", size=10, symbol="diamond"), text=texts, hovertemplate="热点: %{text}<br>x=%{x:.1f}<br>y=%{y}", showlegend=True))
                        # 竖线更淡一些，避免干扰
                        for xv in xs:
                            try:
                                fig.add_vline(x=xv, line_width=1, line_dash="dot", line_color="rgba(255,100,0,0.35)")
                            except Exception:
                                pass
    except Exception:
        pass

    # 事件/联动
    if _HAS_PLOTLY_EVENTS and plotly_events is not None:
        pts = plotly_events(fig, click_event=False, hover_event=True, select_event=False, override_height=320, override_width="100%")  # type: ignore
        if pts:
            pt = pts[0]
            cd = pt.get("customdata") or []
            if isinstance(cd, list) and len(cd) >= 2:
                st.session_state["profile_hover"] = {"route_key": cd[0], "idx": int(cd[1])}
        else:
            # 鼠标移出图后清空（避免高亮卡住）
            st.session_state.pop("profile_hover", None)
    else:
        st.plotly_chart(fig, use_container_width=True)
        # 降级：无 hover 事件
        st.caption("提示：未安装 streamlit-plotly-events，联动高亮不可用。可在 requirements 中加入 streamlit-plotly-events 以启用。")

    # 回写缓存
    st.session_state["route_profiles"] = prof_cache

