"""
实验与导出模块。

提供统一的"运行一次规划并返回 DataFrame/字典"的封装。
"""

from .runner import SingleRunResult, run_single_case, run_case_grid

__all__ = [
    "SingleRunResult",
    "run_single_case",
    "run_case_grid",
]







