from __future__ import annotations

import streamlit as st
import html as html_module


def render() -> None:
    """Render the overview landing page used for presentation/demo."""
    # 生成无 JS 的打字机逐字显现 HTML（逐字符 span + 延迟）
    typing_text = "基于 EDL 的智能北极航线规划系统"
    _tw_len = len(typing_text)
    _tw_speed = 0.08  # 每字秒数
    _tw_hold = 1.2    # 打完/清空停留
    _tw_duration = max(3.0, _tw_len * _tw_speed * 2 + _tw_hold)  # 一轮：打字->停留->回退
    typed_html = f'<span class="typewriter" style="--typing-chars:{_tw_len}; --tw-duration:{_tw_duration}s">{html_module.escape(typing_text)}</span>'

    st.markdown(
        f"""
        <style>
        [data-testid="stAppViewContainer"] {{
            background: #020617;
        }}
        /* Hero + typewriter */
        .ar-hero {{
            padding: 2rem 1.5rem 1rem 1.5rem;
            background: linear-gradient(135deg, #0f172a, #020617);
            border-radius: 18px;
            border: 1px solid rgba(255,255,255,0.08);
            position: relative;
            overflow: hidden;
            color: #f9fafb;
        }}
        .ar-hero .hero-title {{
            font-size: 2.2rem;
            margin: 0;
            color: #f9fafb;
            font-weight: 700;
        }}
        .ar-hero .typewriter {{
            font-family: "Source Code Pro", Consolas, monospace;
            font-size: 1rem;
            display: inline-block;
            white-space: nowrap;
            overflow: hidden;
            border-right: 2px solid #38bdf8; /* 光标随文本末尾移动 */
            width: 0ch;
            animation: ar-typing var(--tw-duration) steps(var(--typing-chars), end) infinite, ar-blink 1s step-end infinite;
        }}
        @keyframes ar-typing {{
            0%   {{ width: 0ch; }}
            40%  {{ width: var(--typing-chars)ch; }}
            60%  {{ width: var(--typing-chars)ch; }}
            100% {{ width: 0ch; }}
        }}
        @keyframes ar-blink {{ 50% {{ border-color: transparent; }} }}
        .ar-hero .subtitle-list {{
            margin-top: 0.75rem;
            color: #e5e7eb;
            opacity: 0.95;
            font-size: 1rem;
        }}
        .ar-hero .subtitle-line {{
            display: block;
            margin-bottom: 0.25rem;
            opacity: 0;
            transform: translateY(6px);
            animation: fadeIn 0.8s ease forwards;
        }}
        .ar-hero .subtitle-line:nth-child(1) {{ animation-delay: 0.8s; }}
        .ar-hero .subtitle-line:nth-child(2) {{ animation-delay: 1.4s; }}
        .ar-hero .subtitle-line:nth-child(3) {{ animation-delay: 2.0s; }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        .ar-card {{
            background: #0b1120;
            border-radius: 14px;
            padding: 1rem;
            box-shadow: 0 18px 40px rgba(15, 23, 42, 0.16);
            border: 1px solid rgba(255, 255, 255, 0.06);
            height: 100%;
        }}
        .ar-section-title {{
            color: #f9fafb;
            font-size: 1.25rem;
            font-weight: 600;
            margin: 1.5rem 0 0.75rem;
        }}
        .ar-card .feature-card-title {{
            margin-top: 0;
            margin-bottom: 0.35rem;
            color: #f9fafb;
            font-weight: 600;
        }}
        .ar-card .feature-card-subtitle {{
            margin: 0;
            color: #e5e7eb;
            font-size: 0.95rem;
            line-height: 1.4;
        }}
        /* 光标 */
        .ar-hero .typewriter .caret{{ display:inline-block; width:2px; height:1em; background:#38bdf8; vertical-align:-0.2em; animation: blink 1s step-end infinite; }}
        @keyframes blink {{ 50% {{ opacity: 0; }} }}
        </style>
        <div class="ar-hero">
            <h1 class="hero-title">ArcticRoute: 智能北极航线规划系统</h1>
            <div class="subtitle-list">
                {typed_html}
                <span class="subtitle-line">多模态环境场：海冰 / 海浪 / AIS / 冰级</span>
                <span class="subtitle-line">基于 EDL 的不确定性感知风险规避</span>
                <span class="subtitle-line">真实地理网格 + AIS 拥挤度 + Copernicus SIC数据</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 移除 iframe 内的 JS 打字机脚本，避免不同 DOM 上下文导致无效

    st.markdown('<div class="ar-section-title">项目亮点</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
            <div class="ar-card">
                <h3 class="feature-card-title"> 多模态成本</h3>
                <p class="feature-card-subtitle">海冰 + 海浪 + AIS 拥挤度 + 冰级约束</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="ar-card">
                <h3 class="feature-card-title"> EDL 风险与不确定性</h3>
                <p class="feature-card-subtitle">miles-guess / PyTorch · 风险 + 不确定性评估</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
            <div class="ar-card">
                <h3 class="feature-card-title"> 智能航线规划</h3>
                <p class="feature-card-subtitle">三种策略：效率优先 / 风险均衡 / 稳健安全</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="ar-section-title">从这里开始体验</div>', unsafe_allow_html=True)
    cta_container = st.container()
    with cta_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(" 进入航线规划驾驶舱", use_container_width=True):
                st.session_state["active_page"] = "planner"
