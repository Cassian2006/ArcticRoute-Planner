"""代价模型子模块（package）。

注意：本仓库同时存在 arcticroute/core/cost.py（模块）与 arcticroute/core/cost/（包）。
为兼容历史代码，这里桥接导出 cost.py 中的核心 API。
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

# 解析 cost.py 的真实路径（包的上一级目录下）
_pkg_dir = Path(__file__).resolve().parent
_core_dir = _pkg_dir.parent
_cost_py = _core_dir / "cost.py"

__all__ = []

if _cost_py.exists():
    try:
        # 使用一个不与本包冲突的模块名加载 cost.py
        mod_name = "arcticroute.core._cost_file"
        if mod_name in sys.modules:
            _mod = sys.modules[mod_name]
        else:
            spec = importlib.util.spec_from_file_location(mod_name, str(_cost_py))
            assert spec and spec.loader
            _mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = _mod
            spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

        # 选择性导出常用符号
        exports = [
            "CostField",
            "build_demo_cost",
            "build_cost_from_real_env",
            "build_cost_from_sic",
            "_add_ais_cost_component",
            "_normalize_ais_density_array",
            "_regrid_ais_density_to_grid",
            "load_ais_density_for_grid",
            "AIS_DENSITY_PATH_DEMO",
            "AIS_DENSITY_PATH_REAL",
            "AIS_DENSITY_PATH",
            "list_available_ais_density_files",
            "discover_ais_density_candidates",
            "compute_grid_signature",
        ]
        g = globals()
        for name in exports:
            if hasattr(_mod, name):
                g[name] = getattr(_mod, name)
                __all__.append(name)
    except Exception as e:
        # 加载失败时不抛出，以避免影响其它子模块导入
        import warnings
        warnings.warn(f"cost package bridge failed: {e}")
else:
    import warnings
    warnings.warn(f"cost.py not found at {_cost_py}")
