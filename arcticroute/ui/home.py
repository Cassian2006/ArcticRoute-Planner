from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


def _assets_dir() -> Path:
    return Path(__file__).resolve().parent / "assets"


def render() -> None:
    """
    封面页：唯一来源为 assets/arctic_ui_cover.html。

    不再依赖 HTML 内按钮与 Streamlit 交互，只提供一个可靠的 Streamlit 按钮：
      - 点击后写入 query params: ?page=planner
      - 同时设置 st.session_state["page"] = "planner"
      - 然后 st.rerun()
    """
    cover_path = _assets_dir() / "arctic_ui_cover.html"

    if cover_path.exists():
        html = cover_path.read_text(encoding="utf-8", errors="ignore")
        components.html(html, height=900, scrolling=True)
    else:
        st.warning(f"封面 HTML 未找到：{cover_path}")

    st.markdown("### 从这里进入系统")
    if st.button("进入系统 / Start", use_container_width=True, type="primary"):
        try:
            qp = st.query_params
            qp["page"] = "planner"
        except Exception:
            # 即使 query params 不可用，也至少保证 session_state 生效
            pass
        st.session_state["page"] = "planner"
        st.rerun()
