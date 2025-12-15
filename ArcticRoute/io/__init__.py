"""
I/O 模块：数据加载和导出

包含从各种格式（NetCDF、JSON 等）加载数据的工具。
"""

from .cmems_loader import (
    find_latest_nc,
    load_sic_from_nc,
    load_swh_from_nc,
)

__all__ = [
    "find_latest_nc",
    "load_sic_from_nc",
    "load_swh_from_nc",
]

