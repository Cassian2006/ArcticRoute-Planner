"""
应用路由器 - 统一管理页面导航
确保只有一个导航入口，避免双导航问题
"""

from __future__ import annotations

from typing import Callable

import streamlit as st


# 页面名称常量
PAGE_COVER = "封面"
PAGE_PLANNER = "规划"
PAGE_DATA = "数据"
PAGE_RULES = "规则诊断"
PAGE_ABOUT = "关于"

# 所有页面列表（顺序很重要）
ALL_PAGES = [PAGE_COVER, PAGE_PLANNER, PAGE_DATA, PAGE_RULES, PAGE_ABOUT]


def get_current_page() -> str:
    """
    获取当前页面
    
    Returns:
        当前页面名称
    """
    # 首先尝试从 query params 获取
    query_params = st.query_params
    if "page" in query_params:
        page_param = query_params["page"]
        # 映射 URL 参数到页面名称
        page_mapping = {
            "cover": PAGE_COVER,
            "planner": PAGE_PLANNER,
            "data": PAGE_DATA,
            "rules": PAGE_RULES,
            "about": PAGE_ABOUT,
        }
        if page_param in page_mapping:
            return page_mapping[page_param]
    
    # 然后从 session_state 获取
    if "current_page" in st.session_state:
        page = st.session_state["current_page"]
        if page in ALL_PAGES:
            return page
    
    # 默认返回封面页
    return PAGE_COVER


def set_current_page(page: str) -> None:
    """
    设置当前页面
    
    Args:
        page: 页面名称
    """
    if page not in ALL_PAGES:
        st.error(f"无效的页面: {page}")
        return
    
    # 保存到 session_state
    st.session_state["current_page"] = page
    
    # 同步到 query params（可选，用于分享链接）
    page_mapping = {
        PAGE_COVER: "cover",
        PAGE_PLANNER: "planner",
        PAGE_DATA: "data",
        PAGE_RULES: "rules",
        PAGE_ABOUT: "about",
    }
    if page in page_mapping:
        st.query_params["page"] = page_mapping[page]


def navigate_to(page: str) -> None:
    """
    导航到指定页面（会触发 rerun）
    
    Args:
        page: 页面名称
    """
    set_current_page(page)
    st.rerun()


def render_navigation() -> str:
    """
    渲染侧边栏导航（唯一的导航入口）
    
    Returns:
        当前选中的页面
    """
    current_page = get_current_page()
    
    # 确保当前页面在列表中
    if current_page not in ALL_PAGES:
        current_page = PAGE_COVER
    
    # 渲染单选框
    selected_page = st.sidebar.radio(
        "页面导航",
        options=ALL_PAGES,
        index=ALL_PAGES.index(current_page),
        key="page_navigation",
    )
    
    # 如果选择改变，更新状态
    if selected_page != current_page:
        set_current_page(selected_page)
    
    return selected_page


def create_page_button(label: str, target_page: str, icon: str = "", use_container_width: bool = True) -> bool:
    """
    创建页面跳转按钮
    
    Args:
        label: 按钮文字
        target_page: 目标页面
        icon: 图标（可选）
        use_container_width: 是否占满容器宽度
    
    Returns:
        是否点击了按钮
    """
    button_label = f"{icon} {label}" if icon else label
    
    if st.button(button_label, use_container_width=use_container_width):
        navigate_to(target_page)
        return True
    
    return False


# 页面渲染函数类型
PageRenderer = Callable[[], None]


class Router:
    """
    路由器类 - 管理页面注册和渲染
    """
    
    def __init__(self):
        self._pages: dict[str, PageRenderer] = {}
    
    def register(self, page_name: str, renderer: PageRenderer) -> None:
        """
        注册页面渲染函数
        
        Args:
            page_name: 页面名称
            renderer: 渲染函数
        """
        if page_name not in ALL_PAGES:
            raise ValueError(f"无效的页面名称: {page_name}")
        
        self._pages[page_name] = renderer
    
    def render(self, page_name: str) -> None:
        """
        渲染指定页面
        
        Args:
            page_name: 页面名称
        """
        if page_name not in self._pages:
            st.error(f"页面未注册: {page_name}")
            st.info(f"已注册的页面: {list(self._pages.keys())}")
            return
        
        renderer = self._pages[page_name]
        renderer()
    
    def run(self) -> None:
        """
        运行路由器（渲染导航和当前页面）
        """
        # 渲染导航
        current_page = render_navigation()
        
        # 渲染当前页面
        self.render(current_page)


# 全局路由器实例
_router = Router()


def get_router() -> Router:
    """
    获取全局路由器实例
    
    Returns:
        路由器实例
    """
    return _router
