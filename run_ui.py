"""
Streamlit entrypoint for the ArcticRoute UI shell.

重构版本 - 使用统一路由系统，避免双导航问题
"""

from __future__ import annotations

import streamlit as st

from arcticroute.ui.ui_style import inject_global_style
from arcticroute.ui.error_boundary import safe_render
from arcticroute.ui.shell_skin import inject_all_styles
from arcticroute.ui.app_router import (
    get_router,
    PAGE_COVER,
    PAGE_PLANNER,
    PAGE_DATA,
    PAGE_RULES,
    PAGE_DOCTOR,
    PAGE_ABOUT,
)
from arcticroute.ui.pages_cover import render_cover
from arcticroute.ui.pages_data import render_data
from arcticroute.ui.pages_rules import render_rules
from arcticroute.ui.pages_about import render_about
from arcticroute.ui.pages.doctor_page import render_doctor_page
from arcticroute.ui import planner_minimal


def main() -> None:
    """应用主入口"""
    
    # 设置页面配置（必须在最开始）
    st.set_page_config(
        page_title="ArcticRoute",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # 全局字体栈
    inject_global_style()

    # 注入 UI 样式
    inject_all_styles()
    
    # 获取路由器并注册页面
    router = get_router()
    router.register(PAGE_COVER, lambda: safe_render("cover", render_cover))
    router.register(PAGE_PLANNER, lambda: safe_render("planner", planner_minimal.render))
    router.register(PAGE_DATA, lambda: safe_render("data", render_data))
    router.register(PAGE_RULES, lambda: safe_render("rules", render_rules))
    router.register(PAGE_DOCTOR, lambda: safe_render("doctor", render_doctor_page))
    router.register(PAGE_ABOUT, lambda: safe_render("about", render_about))
    
    # 运行路由器（渲染导航和当前页面）
    router.run()


if __name__ == "__main__":
    main()
