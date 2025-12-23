from __future__ import annotations
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

ASSET = Path(__file__).parent / "assets" / "arctic_ui_cover.html"


def _goto(page: str) -> None:
    st.session_state["nav_page"] = page
    try:
        st.query_params["page"] = page
    except Exception:
        pass


def render_cover() -> None:
    # 注意：page_config 由 run_ui.py 统一管理，此处不再设置
    html = None
    error_msg = None
    try:
        if ASSET.exists():
            html = ASSET.read_text(encoding="utf-8", errors="ignore")
            if not html or len(html.strip()) == 0:
                error_msg = "封面文件为空"
                html = None
        else:
            error_msg = f"封面文件不存在: {ASSET}"
    except Exception as e:
        error_msg = f"读取封面文件失败: {e}"
        html = None

    # 约定：封面里按钮 id = btnStart / btnFakeShot
    # 用 postMessage -> Streamlit 侧监听很麻烦，这里走"最稳"的按钮兜底：
    # HTML 只做视觉，跳转用 Streamlit 按钮（始终可用）
    if html:
        components.html(html, height=900, scrolling=False)
    else:
        # Fallback: 显示一个简单的封面
        st.markdown(
            """
            <style>
            .cover-fallback {
                text-align: center;
                padding: 4rem 2rem;
                background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
                border-radius: 12px;
                color: #f9fafb;
            }
            .cover-fallback h1 {
                font-size: 3.5rem;
                font-weight: 700;
                background: linear-gradient(135deg, #38bdf8 0%, #0ea5e9 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                background-clip: text;
                margin-bottom: 1rem;
            }
            .cover-fallback p {
                font-size: 1.2rem;
                color: #e5e7eb;
                margin-bottom: 2rem;
            }
            </style>
            <div class="cover-fallback">
                <h1>ArcticRoute</h1>
                <p>基于 EDL 的智能北极航线规划系统</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if error_msg:
            st.warning(f"⚠️ {error_msg}")

    # 始终显示进入按钮（即使 HTML 加载成功，作为备用）
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        if st.button("🚀 进入航线规划驾驶舱", use_container_width=True, type="primary"):
            _goto("planner")
        if html:
            st.caption("💡 提示：若上方封面按钮无响应，请使用此按钮进入。")

