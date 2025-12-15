"""代价模型子模块（package）。

注意：本仓库同时存在 arcticroute/core/cost.py（模块）与 arcticroute/core/cost/（包）。
为兼容历史代码，这里桥接导出 cost.py 中的核心 API。
"""
from __future__ import annotations

from pathlib import Path
import sys
import importlib
import importlib.util
import types

# 解析 cost.py 的真实路径（包的上一级目录下）
_pkg_dir = Path(__file__).resolve().parent
_core_dir = _pkg_dir.parent
_cost_py = _core_dir / "cost.py"

__all__: list[str] = []

if _cost_py.exists():
    try:
        # 确保父包已在 sys.modules 中
        # arcticroute 与 arcticroute.core 需要可用，供相对导入（from .grid 等）使用
        try:
            import arcticroute as _arcticroute
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"failed to import arcticroute: {e}")
        # arcticroute.core 作为一个命名空间包（已存在）
        import arcticroute.core as _arctic_core  # noqa: F401

        # 兼容旧模块名：将 "ArcticRoute" 映射到当前包对象
        sys.modules.setdefault("ArcticRoute", _arcticroute)

        # 使用 importlib 正规方式加载文件模块
        mod_name = "arcticroute.core._cost_file"
        spec = importlib.util.spec_from_file_location(mod_name, str(_cost_py))
        if spec is None or spec.loader is None:  # pragma: no cover
            raise RuntimeError("spec_from_file_location failed for cost.py")

        module = importlib.util.module_from_spec(spec)
        # 关键：设置 __package__ 以支持相对导入（from .grid 等）
        module.__package__ = "arcticroute.core"
        module.__file__ = str(_cost_py)
        # 注册到 sys.modules 以便子模块互相引用
        sys.modules[mod_name] = module

        # 执行模块
        spec.loader.exec_module(module)  # type: ignore[arg-type]

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
            if hasattr(module, name):
                g[name] = getattr(module, name)
                __all__.append(name)
    except Exception as e:  # pragma: no cover
        # 加载失败时不抛出，以避免影响其它子模块导入
        import warnings
        warnings.warn(f"cost package bridge failed: {e}")
else:  # pragma: no cover
    import warnings
    warnings.warn(f"cost.py not found at {_cost_py}")
