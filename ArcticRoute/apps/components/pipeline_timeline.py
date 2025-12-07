# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import List, Dict, Optional
import streamlit as st

_STATUS_STYLE = {
    "pending": {"bg": "#6b7280", "glow": "rgba(107,114,128,0.45)", "icon": "⏳"},  # gray
    "running": {"bg": "#3b82f6", "glow": "rgba(59,130,246,0.55)", "icon": "⏱️"},  # blue
    "done": {"bg": "#10b981", "glow": "rgba(16,185,129,0.55)", "icon": "✔"},   # teal/green
    "warning": {"bg": "#f59e0b", "glow": "rgba(245,158,11,0.55)", "icon": "⚠"},  # amber
    "error": {"bg": "#ef4444", "glow": "rgba(239,68,68,0.55)", "icon": "✖"},    # red
}

_DEFAULT_STEPS = [
    {"key": "input", "label": "输入与场景", "desc": "读取参数与场景配置", "status": "done"},
    {"key": "env", "label": "加载环境与网格", "desc": "海冰/浪场/地形裁剪与插值", "status": "done"},
    {"key": "cost", "label": "构建风险与成本场", "desc": "SIC+事故+浪+主航线", "status": "done"},
    {"key": "route_primary", "label": "规划多条路线", "desc": "均衡/安全/效率", "status": "done"},
    {"key": "eco", "label": "Eco 评估", "desc": "燃油与 CO₂ 估算", "status": "done"},
    {"key": "summary", "label": "结果摘要", "desc": "距离/风险/燃油", "status": "done"},
]


def _build_css() -> str:
    return """
    <style>
    .timeline-wrapper { width: 100%; margin: 10px 0 6px 0; }
    .timeline { display: flex; align-items: center; justify-content: space-between; position: relative; }
    .timeline::before { content: ""; position: absolute; top: 20px; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, #1f2937, #374151); filter: blur(0.5px); }
    .step { position: relative; text-align: center; flex: 1 1 0; }
    .dot { width: 18px; height: 18px; border-radius: 50%; margin: 0 auto; position: relative; box-shadow: 0 0 10px rgba(0,0,0,0.3); }
    .dot::after { content: attr(data-icon); position: absolute; top: 20px; left: 50%; transform: translateX(-50%); font-size: 12px; color: #d1d5db; }
    .label { margin-top: 28px; font-size: 12px; color: #e5e7eb; font-weight: 600; }
    .desc { margin-top: 2px; font-size: 11px; color: #9ca3af; }
    </style>
    """


def render_pipeline_timeline(steps: Optional[List[Dict]] = None, *, title: Optional[str] = "计算流程管线") -> None:
    """在页面绘制水平时间线。
    - steps: 列表 [{key,label,desc,status}]；缺省时使用 _DEFAULT_STEPS
    - status in {pending|running|done|warning|error}
    """
    st.markdown(_build_css(), unsafe_allow_html=True)
    if title:
        st.markdown(f"#### {title}")
    steps = steps or _DEFAULT_STEPS

    # 渲染
    st.markdown('<div class="timeline-wrapper">', unsafe_allow_html=True)
    cols = st.columns(len(steps), gap="small")
    for i, step in enumerate(steps):
        with cols[i]:
            s = (step.get("status") or "pending").lower()
            style = _STATUS_STYLE.get(s, _STATUS_STYLE["pending"])
            bg = style["bg"]
            glow = style["glow"]
            icon = style["icon"]
            dot_html = f'<div class="dot" style="background:{bg}; box-shadow: 0 0 10px {glow}, 0 0 18px {glow};" data-icon="{icon}"></div>'
            st.markdown(dot_html, unsafe_allow_html=True)
            st.markdown(f'<div class="label">{step.get("label","-")}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="desc">{step.get("desc","")}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


def steps_from_trace(trace: Optional[object]) -> List[Dict]:
    """将 planner_service.PipelineTrace 转换为 steps 列表。"""
    if trace is None:
        return _DEFAULT_STEPS
    try:
        stages = getattr(trace, "stages", {}) or {}
        # 映射顺序与展示
        order = [
            ("input", "输入与场景", "读取参数与场景配置"),
            ("env", "加载环境与网格", "海冰/浪场/地形裁剪与插值"),
            ("cost", "构建风险与成本场", "SIC+事故+浪+主航线"),
            ("route_primary", "规划多条路线", "均衡/安全/效率"),
            ("eco", "Eco 评估", "燃油与 CO₂ 估算"),
            ("summary", "结果摘要", "距离/风险/燃油"),
            ("adv_viz", "可视化", "地图与叠加图"),
            ("ai_explain", "AI 解读", "说明与建议"),
        ]
        out: List[Dict] = []
        for k, label, desc in order:
            stg = stages.get(k)
            status = getattr(stg, "status", None) or "done"
            out.append({"key": k, "label": label, "desc": desc, "status": str(status)})
        return out
    except Exception:
        return _DEFAULT_STEPS




















