"""
POLARIS 诊断信息展示模块。

沿程解释表格：展示路由采样点的 RIO / level / speed_limit
统计信息：special/elevated 命中比例
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd


def extract_route_diagnostics(
    route_points: List[Tuple[float, float]],
    polaris_meta: Dict[str, Any],
    grid: Optional[Any] = None,
) -> pd.DataFrame:
    """
    提取路由沿程的 POLARIS 诊断信息。

    Args:
        route_points: 路由采样点列表 [(lat, lon), ...]
        polaris_meta: 来自 apply_hard_constraints 的 polaris_meta
        grid: 可选的网格对象（用于坐标转换）

    Returns:
        DataFrame，包含每个采样点的 RIO / level / speed_limit
    """
    rio_field = polaris_meta.get("rio_field")
    level_field = polaris_meta.get("level_field")
    speed_field = polaris_meta.get("speed_field")

    if rio_field is None or level_field is None:
        return pd.DataFrame()

    rows = []
    for idx, (lat, lon) in enumerate(route_points):
        # 这里假设 route_points 已经是网格索引 (i, j)
        # 如果是地理坐标，需要通过 grid 转换
        if isinstance(lat, (int, np.integer)) and isinstance(lon, (int, np.integer)):
            i, j = int(lat), int(lon)
        else:
            # 假设 grid 有 latlon_to_ij 方法
            if grid is not None and hasattr(grid, "latlon_to_ij"):
                i, j = grid.latlon_to_ij(lat, lon)
            else:
                continue

        # 检查边界
        if i < 0 or i >= rio_field.shape[0] or j < 0 or j >= rio_field.shape[1]:
            continue

        rio = float(rio_field[i, j]) if not np.isnan(rio_field[i, j]) else None
        level = str(level_field[i, j]) if level_field[i, j] else None
        speed = float(speed_field[i, j]) if not np.isnan(speed_field[i, j]) else None

        rows.append({
            "采样点": idx,
            "纬度": lat,
            "经度": lon,
            "RIO": rio,
            "操作等级": level,
            "速度限制(节)": speed,
        })

    return pd.DataFrame(rows)


def compute_route_statistics(
    polaris_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    计算路由的 POLARIS 统计信息。

    Args:
        polaris_meta: 来自 apply_hard_constraints 的 polaris_meta

    Returns:
        Dict 包含统计信息
    """
    stats = {
        "rio_min": polaris_meta.get("rio_min"),
        "rio_mean": polaris_meta.get("rio_mean"),
        "special_fraction": polaris_meta.get("special_fraction", 0.0),
        "elevated_fraction": polaris_meta.get("elevated_fraction", 0.0),
        "special_count": polaris_meta.get("special_count", 0),
        "elevated_count": polaris_meta.get("elevated_count", 0),
        "total_valid_cells": polaris_meta.get("total_valid_cells", 0),
        "riv_table_used": polaris_meta.get("riv_table_used", "table_1_3"),
    }
    return stats


def format_diagnostics_summary(
    polaris_meta: Dict[str, Any],
) -> str:
    """
    格式化诊断信息摘要（用于 UI 展示）。

    Args:
        polaris_meta: 来自 apply_hard_constraints 的 polaris_meta

    Returns:
        格式化的文本摘要
    """
    stats = compute_route_statistics(polaris_meta)

    summary = f"""
**POLARIS 诊断摘要**

- **RIO 范围**: {stats['rio_min']:.1f} ~ {stats['rio_mean']:.1f}
- **特殊等级比例**: {stats['special_fraction']:.1%} ({stats['special_count']} 个格点)
- **提升等级比例**: {stats['elevated_fraction']:.1%} ({stats['elevated_count']} 个格点)
- **有效格点数**: {stats['total_valid_cells']}
- **使用表格**: {stats['riv_table_used']}
"""
    return summary.strip()


def aggregate_route_by_segment(
    route_diagnostics: pd.DataFrame,
    segment_size: int = 10,
) -> pd.DataFrame:
    """
    按区段聚合路由诊断信息。

    Args:
        route_diagnostics: 来自 extract_route_diagnostics 的 DataFrame
        segment_size: 每个区段的采样点数

    Returns:
        聚合后的 DataFrame
    """
    if route_diagnostics.empty:
        return pd.DataFrame()

    route_diagnostics["区段"] = route_diagnostics["采样点"] // segment_size

    agg_dict = {
        "RIO": ["min", "mean", "max"],
        "速度限制(节)": ["min", "mean"],
        "操作等级": lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
    }

    aggregated = route_diagnostics.groupby("区段").agg(agg_dict)
    aggregated.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in aggregated.columns]

    return aggregated


def render_polaris_diagnostics_panel(
    polaris_meta: Optional[Dict[str, Any]],
    route_points: Optional[List[Tuple[float, float]]] = None,
    grid: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    渲染 POLARIS 诊断面板（返回结构化数据，供 UI 使用）。

    Args:
        polaris_meta: 来自 apply_hard_constraints 的 polaris_meta
        route_points: 可选的路由采样点
        grid: 可选的网格对象

    Returns:
        Dict 包含诊断信息、表格、统计等
    """
    if polaris_meta is None:
        return {"error": "No POLARIS metadata available"}

    result = {
        "summary": format_diagnostics_summary(polaris_meta),
        "statistics": compute_route_statistics(polaris_meta),
    }

    if route_points is not None:
        route_diag = extract_route_diagnostics(route_points, polaris_meta, grid)
        if not route_diag.empty:
            result["route_table"] = route_diag.to_dict("records")
            result["route_aggregated"] = aggregate_route_by_segment(route_diag).to_dict("records")

    return result


__all__ = [
    "extract_route_diagnostics",
    "compute_route_statistics",
    "format_diagnostics_summary",
    "aggregate_route_by_segment",
    "render_polaris_diagnostics_panel",
]

