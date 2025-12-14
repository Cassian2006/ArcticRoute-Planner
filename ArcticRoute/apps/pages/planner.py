from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import copy
from typing import Any

# Correctly import the service layer
from ArcticRoute.core import planner_service
from ArcticRoute.core.planner_service import RobustPlannerConfig, run_planning_pipeline_evidential_robust
from ArcticRoute.apps.theme import inject_theme, read_theme_flag
from ArcticRoute.apps.registry import UIRegistry
from ArcticRoute.apps.services import ai_explainer
from ArcticRoute.apps.components import advanced_viz
from ArcticRoute.apps.components.profile_view import render_profile_view
from ArcticRoute.apps.services import scenarios as scenarios_service
from ArcticRoute.core.eco import vessel_profiles as eco_vessel_profiles

# --- Planner 页面通用样式（主容器与玻璃卡片） ---
st.markdown(
    """
    <style>
    /* Planner 主容器调宽，留出中间空间（更自适应） */
    .main .block-container {
        max-width: min(96vw, 1680px);
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 1.0rem;
        padding-right: 1.0rem;
        margin-left: auto;
        margin-right: auto;
    }

    .planner-main-container {
        min-height: calc(100vh - 7rem);
    }

    /* 通用玻璃卡片 */
    .glass-panel {
        background: radial-gradient(circle at top left,
                    rgba(0, 255, 209, 0.16),
                    rgba(6, 15, 32, 0.96));
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        padding: 1.4rem 1.6rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.65);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        width: 100%;
        box-sizing: border-box;
    }

    /* 左右窄一点的卡片 */
    .glass-panel--tight { padding: 1.1rem 1.3rem; }

    .glass-panel h3, .glass-panel h4 {
        margin-top: 0; margin-bottom: 0.6rem;
    }

    /* 隐藏旧版中心 arc 卡片，避免重复显示 */
    .arc-card-center { display: none !important; }

    /* 响应式：窄屏时三列改为纵向堆叠 */
    @media (max-width: 1280px) {
        .main .block-container { max-width: 98vw; padding-left: 0.8rem; padding-right: 0.8rem; }
        div[data-testid="stHorizontalBlock"] { display: block !important; }
        div[data-testid="column"] { width: 100% !important; display: block !important; }
        .art-main-block, .glass-panel, .art-side-panel { margin-bottom: 1rem; }
    }

    /* 超宽屏时适度放宽容器上限 */
    @media (min-width: 1800px) {
        .main .block-container { max-width: 1800px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 侧栏 About（独立运行时也显示）
with st.sidebar:
    with st.expander("关于 ArcticRoute Planner", expanded=False):
        st.markdown(
            """
**Author**：蔡元祺  
**项目名称**：ArcticRoute Planner — 北极多模态智能航线规划平台  
**联系方式**：  
- Email：`caiyuanqi2006@outlook.com`  
- GitHub：`https://github.com/Cassian2006`  

**开发者声明**：  
- 本项目由作者独立完成，用于课程 / 科研 / 比赛演示。  
- 禁止抄袭、未授权转载或用于商业用途。  

**数据与模型来源**：  
- Copernicus Marine Service（海冰、海浪等再分析/预报产品）  
- NSIDC Sea Ice Index / CDR（海冰浓度）  
- GEBCO 2025（海底地形与海陆线）  

**使用提醒**：  
- 本系统仅用于技术演示与决策辅助，不构成实际航行导航建议。  
- 航运决策需结合更高精度资料与专业人员判断。
            """
        )

# 追加：左右侧玻璃面板与主区间距
st.markdown(
    """
    <style>
    /* 左右侧玻璃面板（比标题略深一些） */
    .art-side-panel {
        background: rgba(5, 19, 40, 0.88);
        border-radius: 24px;
        padding: 18px 20px;
        border: 1px solid rgba(0, 255, 255, 0.10);
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(18px);
    }
    /* 让 map 区域和侧边卡片之间有一点间距 */
    .art-main-block { margin-top: 0.75rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 船型选择（修复：不再仅显示两个巴拿马型；展示所有 ui_visible 的 profile）
try:
    _all_vp = eco_vessel_profiles.load_all_profiles() or {}
except Exception:
    _all_vp = {}

if isinstance(_all_vp, dict) and _all_vp:
    try:
        _vessel_keys = [k for k, v in _all_vp.items() if isinstance(v, dict) and (v.get("ui_visible", True))]
        _vessel_keys.sort()
        def _vessel_label(k: str) -> str:
            prof = _all_vp.get(k, {})
            name = prof.get("display_name") or prof.get("name") or k
            dwt = prof.get("dwt") or prof.get("deadweight_ton")
            try:
                return f"{name} ({float(dwt)/1000:.0f}k DWT)" if dwt is not None else name
            except Exception:
                return str(name)
        with st.sidebar:
            st.caption("选择用于 ECO 评估与风险缩放的船型（若留空将回退默认 panamax）")
            _default_key = st.session_state.get("planner_vessel_key")
            _index = _vessel_keys.index(_default_key) if (_default_key in _vessel_keys) else 0
            _sel_key = st.selectbox(
                "船型",
                options=_vessel_keys,
                index=int(_index) if _vessel_keys else 0,
                format_func=_vessel_label,
                key="planner_vessel_key",
                help="展示所有配置中 ui_visible=true 的船型",
            )
        # 便于后续流程读取
        st.session_state["planner_vessel_profile"] = _all_vp.get(_sel_key, {})
        st.session_state["vessel_profile_name"] = _sel_key
    except Exception:
        pass


# 策略预设（显式区分 Balanced / Safe / Efficient），用于一次性规划三条“性格不同”的路线
PROFILE_PRESETS = {
    "balanced": {
        "label": "均衡",
        "profile_name": "balanced",
        "w_ice": 1.0,
        "w_accident": 0.5,
        "prior_weight": 0.3,
        "color": "#ff006e",  # 洋红
    },
    "safe": {
        "label": "安全优先",
        "profile_name": "safe",
        "w_ice": 2.0,
        "w_accident": 1.5,
        "prior_weight": 0.5,
        "color": "#00b894",  # 绿色
    },
    "efficient": {
        "label": "效率优先",
        "profile_name": "efficient",
        "w_ice": 0.2,
        "w_accident": 0.1,
        "prior_weight": 0.0,
        "color": "#ffa502",  # 橙色
    },
}
# 显示顺序
PROFILE_ORDER = ["balanced", "safe", "efficient"]


# 计算流程的步骤定义（按实际 pipeline 顺序，供回退使用）
COMPUTE_PIPELINE_STEPS = [
    {"key": "input", "label": "输入与场景"},
    {"key": "env",   "label": "加载环境与网格"},
    {"key": "cost",  "label": "构建风险与成本场"},
    {"key": "route_primary", "label": "规划三条路线"},
    {"key": "eco",   "label": "Eco 评估"},
    {"key": "summary",   "label": "汇总与导出"},
    {"key": "adv_viz",   "label": "高级可视化"},
    {"key": "ai_explain",   "label": "AI 航线解读"},
]

PIPELINE_ORDER = [
    ("input", "输入与场景"),
    ("env", "加载环境与网格"),
    ("cost", "构建风险与成本场"),
    ("route_primary", "规划三条路线"),
    ("eco", "Eco 评估"),
    ("summary", "汇总与导出"),
    ("adv_viz", "高级可视化"),
    ("ai_explain", "AI 航线解读"),
]

def _render_compute_pipeline(current_stage: str | None = None, finished: bool = False) -> None:
    # 若有服务层的 PipelineTrace，则以其为准；否则退回旧逻辑。
    trace = None
    try:
        route = st.session_state.get("route_result")
        env_ctx = st.session_state.get("env_ctx")
        trace = getattr(route, "pipeline", None) or getattr(env_ctx, "pipeline", None)
    except Exception:
        trace = None
    """
    在当前 Streamlit 容器中渲染“计算流程管线”。

    参数:
    - current_stage: 当前所在阶段的 key（见 COMPUTE_PIPELINE_STEPS），为空时视为尚未开始。
    - finished: 若 True，则所有节点视为已完成。
    """
    # 为简化逻辑：如果 finished=True，则强制当前阶段为最后一步
    if finished and COMPUTE_PIPELINE_STEPS:
        current_stage = COMPUTE_PIPELINE_STEPS[-1]["key"]

    # 计算每个节点的状态: "done" / "active" / "todo"
    statuses: list[tuple[str, str, str]] = []  # (key, label, status)
    passed_current = False
    for step in COMPUTE_PIPELINE_STEPS:
        key = step["key"]
        label = step["label"]
        if current_stage is None:
            status = "todo"
        elif key == current_stage:
            status = "active"
            passed_current = True
        elif not passed_current:
            status = "done"
        else:
            status = "todo"
        statuses.append((key, label, status))

    css = """
    <style>
    .ar-pipeline-wrapper { display: flex; align-items: flex-start; justify-content: space-between; margin: 0.75rem 0 1.0rem 0; position: relative; }
    .ar-pipeline-step { flex: 1; text-align: center; position: relative; min-width: 0; }
    .ar-pipeline-step:not(:last-child)::after { content: ""; position: absolute; top: 11px; left: 50%; right: -50%; height: 2px; background: linear-gradient(90deg, rgba(148,163,184,0.9), rgba(148,163,184,0.3)); background-size: 40px 2px; animation: flowX 3s linear infinite; z-index: 0; }
    .ar-pipeline-circle { width: 18px; height: 18px; border-radius: 999px; border: 2px solid #64748b; background: #020617; margin: 0 auto; position: relative; z-index: 1; box-sizing: border-box; }
    .ar-pipeline-circle.done { background: #22c55e; border-color: #22c55e; }
    .ar-pipeline-circle.active { background: #3b82f6; border-color: #3b82f6; animation: pulse 1.6s ease-in-out infinite; }
    .ar-pipeline-circle.todo { background: #020617; border-color: #475569; opacity: 0.6; }
    .ar-pipeline-circle.error { background: #ef4444; border-color: #ef4444; }
    .ar-pipeline-circle.skipped { background: #6b7280; border-color: #6b7280; opacity: 0.6; }
    .ar-pipeline-label { margin-top: 0.35rem; font-size: 0.72rem; line-height: 1.1rem; color: #cbd5f5; white-space: normal; }
    .ar-pipeline-sub { font-size: 0.62rem; color: #94a3b8; margin-top: 0.15rem; }
    @media (max-width: 900px) { .ar-pipeline-label { font-size: 0.68rem; } }
    @keyframes flowX { 0% { background-position: 0 0; } 100% { background-position: 40px 0; } }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.35);} 70% { box-shadow: 0 0 0 8px rgba(59,130,246,0);} 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0);} }
    </style>
    """

    step_html_parts: list[str] = []

    if trace is not None:
        for stage_id, stage_label in PIPELINE_ORDER:
            stg = trace.stages.get(stage_id)
            status = stg.status if stg else "pending"
            css_status = "active" if status == "running" else status
            circle_cls = f"ar-pipeline-circle {css_status}"
            safe_label = stage_label.replace("<", "&lt;").replace(">", "&gt;")
            # 简要 meta 展示
            sub = ""
            if stg and stg.duration_ms is not None:
                sub_pieces = [f"{stg.duration_ms:.0f} ms"]
                m = stg.meta or {}
                try:
                    if stage_id == "env":
                        if m.get("shape"):
                            sub_pieces.append(f"网格 {m.get('shape')}")
                        if m.get("layers"):
                            sub_pieces.append("层:" + ",".join(list(m.get("layers"))[:3]))
                    if stage_id == "cost":
                        if "min" in m and "max" in m:
                            sub_pieces.append(f"{float(m['min']):.2f}~{float(m['max']):.2f}")
                    if stage_id == "route_primary":
                        if "steps" in m:
                            sub_pieces.append(f"步数 {int(m['steps'])}")
                    if stage_id == "eco":
                        if "fuel_t" in m:
                            sub_pieces.append(f"燃油 {float(m['fuel_t']):.1f}t")
                except Exception:
                    pass
                if sub_pieces:
                    sub = f"<div class='ar-pipeline-sub'>{' · '.join(sub_pieces)}</div>"
            step_html_parts.append(
                f"""
                <div class=\"ar-pipeline-step\">
                    <div class=\"{circle_cls}\"></div>
                    <div class=\"ar-pipeline-label\">{safe_label}{sub}</div>
                </div>
                """
            )
    else:
        for key, label, status in statuses:
            circle_cls = f"ar-pipeline-circle {status}"
            safe_label = label.replace("<", "&lt;").replace(">", "&gt;")
            step_html_parts.append(
                f"""
                <div class="ar-pipeline-step">
                    <div class="{circle_cls}"></div>
                    <div class="ar-pipeline-label">{safe_label}</div>
                </div>
                """
            )

    html = css + f"""
    <div style="margin-top: 0.5rem;">
      <div style="font-size:0.78rem;color:#9ca3af;margin-bottom:0.25rem;">计算流程管线</div>
      <div class="ar-pipeline-wrapper">{''.join(step_html_parts)}</div>
    </div>
    """
    try:
        import streamlit.components.v1 as components  # type: ignore
        # Use components.html to ensure HTML/CSS render reliably (avoids Markdown escaping on some setups)
        components.html(html, height=120, scrolling=False)
    except Exception:
        st.markdown(html, unsafe_allow_html=True)


def _render_compute_pipeline_vertical(current_stage: str | None = None, finished: bool = False, height: int | None = None) -> None:
    """
    竖向版本的流程管线（侧边栏用）。从上到下排列，避免文字挤在一起。
    若检测到服务层的 PipelineTrace，则使用 trace 的状态；否则回退到旧的阶段推断。
    """
    # 优先读取 trace
    trace = None
    try:
        route = st.session_state.get("route_result")
        env_ctx = st.session_state.get("env_ctx")
        trace = getattr(route, "pipeline", None) or getattr(env_ctx, "pipeline", None)
    except Exception:
        trace = None

    if finished and COMPUTE_PIPELINE_STEPS:
        current_stage = COMPUTE_PIPELINE_STEPS[-1]["key"]

    statuses: list[tuple[str, str, str]] = []  # (key, label, status)
    if trace is not None:
        for sid, slabel in PIPELINE_ORDER:
            stg = trace.stages.get(sid)
            status = stg.status if stg else "pending"
            css_status = "active" if status == "running" else status
            statuses.append((sid, slabel, css_status))
    else:
        passed_current = False
        for step in COMPUTE_PIPELINE_STEPS:
            key = step["key"]
            label = step["label"]
            if current_stage is None:
                status = "todo"
            elif key == current_stage:
                status = "active"
                passed_current = True
            elif not passed_current:
                status = "done"
            else:
                status = "todo"
            statuses.append((key, label, status))

    css = """
    <style>
    .ar-pipelineV-wrapper { position: relative; margin: 0.5rem 0 0.5rem 0; }
    .ar-pipelineV-step { position: relative; display: flex; align-items: flex-start; gap: 8px; padding: 6px 0 10px 0; }
    .ar-pipelineV-circle { width: 16px; height: 16px; border-radius: 999px; border: 2px solid #64748b; background: #020617; flex: 0 0 auto; z-index: 1; }
    .ar-pipelineV-circle.done { background: #22c55e; border-color: #22c55e; }
    .ar-pipelineV-circle.active { background: #3b82f6; border-color: #3b82f6; animation: pulse 1.6s ease-in-out infinite; }
    .ar-pipelineV-circle.todo { background: #020617; border-color: #475569; opacity: 0.6; }
    .ar-pipelineV-circle.error { background: #ef4444; border-color: #ef4444; }
    .ar-pipelineV-circle.skipped { background: #6b7280; border-color: #6b7280; opacity: 0.6; }
    .ar-pipelineV-label { color: #cbd5f5; font-size: 0.78rem; line-height: 1.2rem; }
    .ar-pipelineV-step:not(:last-child)::after { content: ""; position: absolute; left: 7px; top: 20px; bottom: -2px; width: 2px; background: linear-gradient(180deg, rgba(148,163,184,0.9), rgba(148,163,184,0.3)); background-size: 2px 40px; animation: flowY 3s linear infinite; z-index: 0; }
    @keyframes flowY { 0% { background-position: 0 0; } 100% { background-position: 0 40px; } }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(59,130,246,0.35);} 70% { box-shadow: 0 0 0 8px rgba(59,130,246,0);} 100% { box-shadow: 0 0 0 0 rgba(59,130,246,0);} }
    </style>
    """

    items = []
    for key, label, status in statuses:
        safe_label = label.replace("<", "&lt;").replace(">", "&gt;")
        items.append(
            f"""
            <div class=\"ar-pipelineV-step\">
                <div class=\"ar-pipelineV-circle {status}\"></div>
                <div class=\"ar-pipelineV-label\">{safe_label}</div>
            </div>
            """
        )

    html = css + f"""
    <div>
      <div style=\"font-size:0.78rem;color:#9ca3af;margin-bottom:0.25rem;\">计算流程（侧边栏）</div>
      <div class=\"ar-pipelineV-wrapper\">{''.join(items)}</div>
    </div>
    """

    try:
        import streamlit.components.v1 as components  # type: ignore
        components.html(html, height=height or (62 * len(COMPUTE_PIPELINE_STEPS)), scrolling=False)
    except Exception:
        st.markdown(html, unsafe_allow_html=True)

# Optional imports for map display
try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

@st.cache_resource
def get_base_map(center_lat, center_lon, zoom_start):
    """Creates and caches the base Folium map object.
    兼容 _HAS_FOLIUM=False 的情况；尽量最小化实现，避免语法错误。
    """
    if not _HAS_FOLIUM:
        return None
    try:
        m = folium.Map(
            location=[float(center_lat), float(center_lon)],
            zoom_start=int(zoom_start),
            tiles="CartoDB Positron",
        )
        return m
    except Exception:
        # 兜底返回 None，调用方需判空
        return None

# 最小占位：若后续代码未完整加载，确保模块提供 render()，避免 AttributeError
from typing import Optional

def render(ctx: Optional[dict] = None) -> None:
    st.title("ArcticRoute Planner")
    st.caption("Planner 页面模块已加载（最小占位版）。如需完整功能，请确保文件未被截断并刷新页面。")

