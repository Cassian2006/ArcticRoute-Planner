"""
烟雾测试：验证包结构与基本导入。
"""


def test_can_import_arcticroute():
    """验证能够导入 arcticroute 包。"""
    import arcticroute
    assert arcticroute is not None


def test_can_import_core_modules():
    """验证能够导入 core 子模块。"""
    from arcticroute import core
    assert core is not None


def test_can_import_ui_modules():
    """验证能够导入 ui 子模块。"""
    from arcticroute import ui
    assert ui is not None


def test_planner_minimal_has_render():
    """验证 planner_minimal 模块有 render 函数。"""
    from arcticroute.ui import planner_minimal
    assert hasattr(planner_minimal, "render")
    assert callable(planner_minimal.render)


def test_core_submodules_exist():
    """验证 core 的各个子模块存在。"""
    from arcticroute.core import grid, landmask, cost, astar
    assert grid is not None
    assert landmask is not None
    assert cost is not None
    assert astar is not None


def test_eco_submodule_exists():
    """验证 eco 子模块存在。"""
    from arcticroute.core import eco
    assert eco is not None













