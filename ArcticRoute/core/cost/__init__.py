"""代价模型子模块。

此模块重新导出上级目录 cost.py 中的所有公共函数和类，
以解决包/模块冲突问题（同时存在 cost.py 和 cost/ 目录）。
"""

import sys
from pathlib import Path
import importlib.util

# 直接导入上级目录的 cost 模块
_core_dir = Path(__file__).parent.parent
_parent_module_name = __name__.rsplit('.', 1)[0]  # 'arcticroute.core'

try:
    # 加载 cost.py 作为独立模块
    _cost_path = _core_dir / "cost.py"
    _spec = importlib.util.spec_from_file_location("_cost_core", _cost_path)
    _cost_core = importlib.util.module_from_spec(_spec)
    
    # 设置 __package__ 以支持相对导入
    _cost_core.__package__ = _parent_module_name
    
    # 注册到 sys.modules 以便 dataclasses 等机制可见
    sys.modules[_spec.name] = _cost_core
    _spec.loader.exec_module(_cost_core)
    
    # 重新导出所有公共符号
    from _cost_core import (
        CostField,
        AIS_DENSITY_PATH_DEMO,
        AIS_DENSITY_PATH_REAL,
        AIS_DENSITY_PATH,
        AIS_DENSITY_SEARCH_DIRS,
        AIS_DENSITY_PATTERNS,
        load_fitted_exponents,
        get_default_exponents,
        compute_grid_signature,
        discover_ais_density_candidates,
        list_available_ais_density_files,
        load_ais_density_for_demo_grid,
        load_ais_density_for_grid,
        has_ais_density_data,
        build_demo_cost,
        build_cost_from_real_env,
        build_cost_from_sic,
        _add_ais_cost_component,
        _normalize_ais_density_array,
        _load_normalized_ais_density,
        _regrid_ais_density_to_grid,
        _validate_ais_density_for_grid,
        _nearest_neighbor_resample_no_scipy,
        _resolve_ais_weights,
        _resolve_data_root,
        _save_resampled_ais_density,
        _warn_ais_once,
    )
    
    __all__ = [
        "CostField",
        "AIS_DENSITY_PATH_DEMO",
        "AIS_DENSITY_PATH_REAL",
        "AIS_DENSITY_PATH",
        "AIS_DENSITY_SEARCH_DIRS",
        "AIS_DENSITY_PATTERNS",
        "load_fitted_exponents",
        "get_default_exponents",
        "compute_grid_signature",
        "discover_ais_density_candidates",
        "list_available_ais_density_files",
        "load_ais_density_for_demo_grid",
        "load_ais_density_for_grid",
        "has_ais_density_data",
        "build_demo_cost",
        "build_cost_from_real_env",
        "build_cost_from_sic",
        "_add_ais_cost_component",
        "_normalize_ais_density_array",
        "_load_normalized_ais_density",
        "_regrid_ais_density_to_grid",
        "_validate_ais_density_for_grid",
        "_nearest_neighbor_resample_no_scipy",
        "_resolve_ais_weights",
        "_resolve_data_root",
        "_save_resampled_ais_density",
        "_warn_ais_once",
    ]
    
except Exception as e:
    print(f"[ERROR] Failed to load cost module: {e}")
    raise
