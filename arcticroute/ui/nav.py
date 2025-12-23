from __future__ import annotations
import streamlit as st

PAGES = {
    "cover": "封面",
    "planner": "规划",
    "data": "数据",
    "diagnostics": "规划诊断",
    "about": "关于",
}

def get_page(default: str = "cover") -> str:
    # query param 优先，其次 session_state
    try:
        qp = st.query_params
        if "page" in qp and qp["page"]:
            return str(qp["page"])
    except Exception:
        pass

    if "page" in st.session_state and st.session_state["page"]:
        return str(st.session_state["page"])

    return default

def set_page(page: str) -> None:
    if page not in PAGES:
        page = "cover"
    st.session_state["page"] = page
    # 同步到 URL
    try:
        st.query_params["page"] = page
    except Exception:
        try:
            st.experimental_set_query_params(page=page)
        except Exception:
            pass
    st.rerun()

def render_sidebar_nav(current_page: str) -> None:
    # 单一导航：只允许这一套导航存在
    labels = [PAGES[k] for k in PAGES.keys()]
    keys = list(PAGES.keys())
    try:
        idx = keys.index(current_page)
    except Exception:
        idx = 0

    choice = st.sidebar.radio("页面导航", labels, index=idx, key="__nav_radio__")
    chosen_page = keys[labels.index(choice)]
    if chosen_page != current_page:
        set_page(chosen_page)

