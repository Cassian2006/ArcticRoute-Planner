"""
UI Router Smoke Tests - 确保路由器和样式注入不报错
"""

from __future__ import annotations

import pytest


def test_import_app_router():
    """测试导入 app_router 模块"""
    try:
        import arcticroute.ui.app_router
        assert arcticroute.ui.app_router is not None
    except Exception as e:
        pytest.fail(f"Failed to import app_router: {e}")


def test_import_shell_skin():
    """测试导入 shell_skin 模块"""
    try:
        import arcticroute.ui.shell_skin
        assert arcticroute.ui.shell_skin is not None
    except Exception as e:
        pytest.fail(f"Failed to import shell_skin: {e}")


def test_router_constants():
    """测试路由器常量定义"""
    from arcticroute.ui.app_router import (
        PAGE_COVER,
        PAGE_PLANNER,
        PAGE_DATA,
        PAGE_RULES,
        PAGE_ABOUT,
        ALL_PAGES,
    )
    
    # 确保所有页面常量都定义了
    assert PAGE_COVER is not None
    assert PAGE_PLANNER is not None
    assert PAGE_DATA is not None
    assert PAGE_RULES is not None
    assert PAGE_ABOUT is not None
    
    # 确保 ALL_PAGES 包含所有页面
    assert len(ALL_PAGES) == 5
    assert PAGE_COVER in ALL_PAGES
    assert PAGE_PLANNER in ALL_PAGES
    assert PAGE_DATA in ALL_PAGES
    assert PAGE_RULES in ALL_PAGES
    assert PAGE_ABOUT in ALL_PAGES


def test_get_router():
    """测试获取路由器实例"""
    from arcticroute.ui.app_router import get_router
    
    router = get_router()
    assert router is not None
    assert hasattr(router, "register")
    assert hasattr(router, "render")
    assert hasattr(router, "run")


def test_inject_shell_css_callable():
    """测试 inject_shell_css 可调用（不实际渲染）"""
    from arcticroute.ui.shell_skin import inject_shell_css
    
    # 确保函数可调用
    assert callable(inject_shell_css)
    
    # 注意：不实际调用，因为需要 Streamlit 上下文


def test_inject_all_styles_callable():
    """测试 inject_all_styles 可调用"""
    from arcticroute.ui.shell_skin import inject_all_styles
    
    # 确保函数可调用
    assert callable(inject_all_styles)


def test_page_modules_importable():
    """测试所有页面模块可导入"""
    try:
        from arcticroute.ui import pages_cover
        from arcticroute.ui import pages_data
        from arcticroute.ui import pages_rules
        from arcticroute.ui import pages_about
        
        assert pages_cover is not None
        assert pages_data is not None
        assert pages_rules is not None
        assert pages_about is not None
    except Exception as e:
        pytest.fail(f"Failed to import page modules: {e}")


def test_page_render_functions_exist():
    """测试所有页面渲染函数存在"""
    from arcticroute.ui.pages_cover import render_cover
    from arcticroute.ui.pages_data import render_data
    from arcticroute.ui.pages_rules import render_rules
    from arcticroute.ui.pages_about import render_about
    
    assert callable(render_cover)
    assert callable(render_data)
    assert callable(render_rules)
    assert callable(render_about)


def test_router_registration():
    """测试路由器注册功能"""
    from arcticroute.ui.app_router import Router, PAGE_COVER
    
    router = Router()
    
    # 定义一个简单的渲染函数
    def dummy_render():
        pass
    
    # 测试注册
    router.register(PAGE_COVER, dummy_render)
    
    # 确保注册成功
    assert PAGE_COVER in router._pages
    assert router._pages[PAGE_COVER] == dummy_render


def test_router_invalid_page_registration():
    """测试注册无效页面会抛出异常"""
    from arcticroute.ui.app_router import Router
    
    router = Router()
    
    def dummy_render():
        pass
    
    # 尝试注册无效页面
    with pytest.raises(ValueError):
        router.register("invalid_page", dummy_render)


def test_create_page_button_callable():
    """测试 create_page_button 函数可调用"""
    from arcticroute.ui.app_router import create_page_button
    
    assert callable(create_page_button)


def test_navigate_to_callable():
    """测试 navigate_to 函数可调用"""
    from arcticroute.ui.app_router import navigate_to
    
    assert callable(navigate_to)

