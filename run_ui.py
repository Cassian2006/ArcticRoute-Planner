"""
Streamlit entrypoint for the ArcticRoute UI shell.

重构版本 - 使用统一路由系统，避免双导航问题
"""

from __future__ import annotations

import streamlit as st

from arcticroute.ui.nav import get_page, render_sidebar_nav
from arcticroute.ui.pages.cover_page import render_cover_page
from arcticroute.ui.pages.planner_page import render_planner_page
from arcticroute.ui.pages.data_page import render_data_page
from arcticroute.ui.pages.diagnostics_page import render_diagnostics_page
from arcticroute.ui.pages.about_page import render_about_page


def main() -> None:
    """应用主入口"""
    
    # 设置页面配置（必须在最开始）
    st.set_page_config(
        page_title="ArcticRoute",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 获取当前页面
    current = get_page(default="cover")
    
    # 渲染侧边栏导航（唯一的导航入口）
    render_sidebar_nav(current)
    
    # 路由分发
    if current == "cover":
        render_cover_page()
    elif current == "planner":
        render_planner_page()
    elif current == "data":
        render_data_page()
    elif current == "diagnostics":
        render_diagnostics_page()
    elif current == "about":
        render_about_page()
    else:
        # 默认显示封面页
        render_cover_page()


if __name__ == "__main__":
    main()
