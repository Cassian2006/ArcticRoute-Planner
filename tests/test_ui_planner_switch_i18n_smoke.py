"""Smoke tests for planner selector and i18n functionality."""

from arcticroute.ui.i18n import t
from arcticroute.core.planners.selector import select_backend


def test_i18n_basic():
    # 不依赖 streamlit runtime，t(key) 直接返回字符串（默认 zh）
    # 注意：在没有 streamlit session 的情况下，t() 会使用默认语言
    result = t("planner_engine")
    assert isinstance(result, str)
    assert len(result) > 0


def test_selector_auto_fallback():
    backend, sel = select_backend(
        mode="auto",
        pipeline_dir=None,
        external_vessel_mesh=None,
        external_route_config=None
    )
    assert sel.planner_used in ("astar", "polarroute")
    assert sel.requested_mode == "auto"
    assert sel.planner_mode in ("astar", "polarroute_pipeline", "polarroute_external")


def test_selector_astar_force():
    backend, sel = select_backend(mode="astar")
    assert sel.planner_used == "astar"
    assert sel.planner_mode == "astar"
    assert sel.fallback_reason is None


def test_selector_polarroute_pipeline_missing():
    backend, sel = select_backend(
        mode="polarroute_pipeline",
        pipeline_dir="/nonexistent/path"
    )
    assert sel.planner_used == "astar"
    assert sel.fallback_reason is not None
    assert "pipeline" in sel.fallback_reason.lower() or "unavailable" in sel.fallback_reason.lower()

