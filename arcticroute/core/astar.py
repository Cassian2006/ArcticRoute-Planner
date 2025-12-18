"""
A* 路由算法模块。

提供 A* 寻路算法的实现。
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .cost import CostField
from .grid import Grid2D


@dataclass
class AStarResult:
    path_ij: list[tuple[int, int]]
    reachable: bool
    reason: Optional[str]
    expanded: int


def grid_astar(
    cost_field: CostField,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    neighbor8: bool = True,
) -> list[tuple[int, int]]:
    """
    兼容旧接口：仅返回路径索引列表；若不可达则返回空列表。
    """
    res = grid_astar_with_info(cost_field, start_ij, goal_ij, neighbor8=neighbor8)
    return res.path_ij


def grid_astar_with_info(
    cost_field: CostField,
    start_ij: tuple[int, int],
    goal_ij: tuple[int, int],
    neighbor8: bool = True,
    max_expansions: int | None = None,
) -> AStarResult:
    """
    在 cost_field.cost 上做 A* 网格搜索，返回带可达性与失败原因的信息。

    失败原因：
      - start_blocked / goal_blocked
      - max_expansions_reached
      - no_path（open set 耗尽）
    """
    cost = cost_field.cost
    land_mask = cost_field.land_mask
    ny, nx = cost.shape

    # 检查起点和终点是否有效
    si, sj = start_ij
    gi, gj = goal_ij

    if not (0 <= si < ny and 0 <= sj < nx):
        return AStarResult([], False, "start_blocked", 0)
    if not (0 <= gi < ny and 0 <= gj < nx):
        return AStarResult([], False, "goal_blocked", 0)

    if land_mask[si, sj] or np.isinf(cost[si, sj]):
        return AStarResult([], False, "start_blocked", 0)
    if land_mask[gi, gj] or np.isinf(cost[gi, gj]):
        return AStarResult([], False, "goal_blocked", 0)

    # 定义邻接关系
    if neighbor8:
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
        step_costs = [
            np.sqrt(2),
            1.0,
            np.sqrt(2),
            1.0,
            1.0,
            np.sqrt(2),
            1.0,
            np.sqrt(2),
        ]
    else:
        directions = [(-1, 0), (0, -1), (0, 1), (1, 0)]
        step_costs = [1.0, 1.0, 1.0, 1.0]

    # A*
    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start_ij))

    g_score = {start_ij: 0.0}
    f_score = {start_ij: _heuristic(start_ij, goal_ij)}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    closed_set: set[tuple[int, int]] = set()

    expanded = 0

    while open_set:
        _, current = heapq.heappop(open_set)

        if current in closed_set:
            continue

        if max_expansions is not None and expanded >= max_expansions:
            return AStarResult([], False, "max_expansions_reached", expanded)

        if current == goal_ij:
            # 重建路径
            path: list[tuple[int, int]] = []
            node = goal_ij
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start_ij)
            path.reverse()
            return AStarResult(path, True, None, expanded)

        closed_set.add(current)
        ci, cj = current
        expanded += 1

        # 探索邻接格点
        for (di, dj), step_cost in zip(directions, step_costs):
            ni, nj = ci + di, cj + dj

            # 边界检查
            if not (0 <= ni < ny and 0 <= nj < nx):
                continue

            # 不能走陆地或无穷成本
            if land_mask[ni, nj] or np.isinf(cost[ni, nj]):
                continue

            if (ni, nj) in closed_set:
                continue

            # 计算新的 g_score
            tentative_g = g_score[current] + step_cost * cost[ni, nj]

            if (ni, nj) not in g_score or tentative_g < g_score[(ni, nj)]:
                came_from[(ni, nj)] = current
                g_score[(ni, nj)] = tentative_g
                f = tentative_g + _heuristic((ni, nj), goal_ij)
                f_score[(ni, nj)] = f
                heapq.heappush(open_set, (f, (ni, nj)))

    # 不可达
    return AStarResult([], False, "no_path", expanded)


def _heuristic(current: tuple[int, int], goal: tuple[int, int]) -> float:
    """欧氏距离启发函数。"""
    ci, cj = current
    gi, gj = goal
    di = gi - ci
    dj = gj - cj
    return np.hypot(di, dj)


def _nearest_ocean_cell(
    grid: Grid2D,
    land_mask: np.ndarray,
    lat: float,
    lon: float,
    max_radius: int = 10,
) -> tuple[int, int] | None:
    """
    在 (i0,j0) 附近一个小方框内（半径 max_radius）找最近的海洋格点。

    找不到则返回 None。

    Args:
        grid: Grid2D 对象
        land_mask: bool 数组，True = 陆地
        lat: 目标纬度
        lon: 目标经度
        max_radius: 搜索半径（格点数）

    Returns:
        (i, j) 或 None
    """
    # 先找最近的格点
    lat2d = grid.lat2d
    lon2d = grid.lon2d
    ny, nx = grid.shape()

    # 计算距离
    dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
    i0, j0 = np.unravel_index(np.argmin(dist), dist.shape)

    # 在半径内搜索海洋格点
    best_cell = None
    best_dist = float("inf")

    for di in range(-max_radius, max_radius + 1):
        for dj in range(-max_radius, max_radius + 1):
            ni, nj = i0 + di, j0 + dj

            if not (0 <= ni < ny and 0 <= nj < nx):
                continue

            if land_mask[ni, nj]:
                continue

            # 这是一个海洋格点，计算距离
            cell_dist = np.sqrt((lat2d[ni, nj] - lat) ** 2 + (lon2d[ni, nj] - lon) ** 2)
            if cell_dist < best_dist:
                best_dist = cell_dist
                best_cell = (ni, nj)

    return best_cell


@dataclass
class PlanRouteResult:
    path_latlon: list[tuple[float, float]]
    reachable: bool
    reason: Optional[str]
    expanded: int
    start_ij: Optional[tuple[int, int]]
    goal_ij: Optional[tuple[int, int]]
    snapped_start: Optional[tuple[float, float]]
    snapped_goal: Optional[tuple[float, float]]


def plan_route_latlon(
    cost_field: CostField,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    neighbor8: bool = True,
    return_info: bool = False,
    max_expansions: int | None = None,
) -> list[tuple[float, float]] | PlanRouteResult:
    """
    兼容旧接口：默认返回路径列表；若 return_info=True 则返回 PlanRouteResult。
    """
    res = plan_route_latlon_with_info(
        cost_field,
        start_lat,
        start_lon,
        end_lat,
        end_lon,
        neighbor8=neighbor8,
        max_expansions=max_expansions,
    )
    return res if return_info else res.path_latlon


def plan_route_latlon_with_info(
    cost_field: CostField,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    neighbor8: bool = True,
    max_expansions: int | None = None,
) -> PlanRouteResult:
    """
    在给定 CostField 上，以经纬度为输入规划一条路径，并返回诊断信息。
    """
    grid = cost_field.grid
    land_mask = cost_field.land_mask

    # 1) 映射起点到最近的海洋格点
    start_ij = _nearest_ocean_cell(grid, land_mask, start_lat, start_lon)
    if start_ij is None:
        return PlanRouteResult([], False, "no_ocean_start", 0, None, None, None, None)

    # 2) 映射终点到最近的海洋格点
    goal_ij = _nearest_ocean_cell(grid, land_mask, end_lat, end_lon)
    if goal_ij is None:
        return PlanRouteResult([], False, "no_ocean_goal", 0, start_ij, None, None, None)

    # 3) 调用 A* 算法
    res = grid_astar_with_info(cost_field, start_ij, goal_ij, neighbor8=neighbor8, max_expansions=max_expansions)
    if not res.reachable or not res.path_ij:
        # 生成吸附后的经纬度信息（起终点）
        s_latlon = (grid.lat2d[start_ij], grid.lon2d[start_ij]) if start_ij is not None else None
        g_latlon = (grid.lat2d[goal_ij], grid.lon2d[goal_ij]) if goal_ij is not None else None
        return PlanRouteResult([], False, res.reason, res.expanded, start_ij, goal_ij, s_latlon, g_latlon)

    # 4) 将路径转换为 lat/lon
    lat2d = grid.lat2d
    lon2d = grid.lon2d
    path_latlon = [(lat2d[i, j], lon2d[i, j]) for i, j in res.path_ij]

    s_latlon = (grid.lat2d[start_ij], grid.lon2d[start_ij])
    g_latlon = (grid.lat2d[goal_ij], grid.lon2d[goal_ij])

    return PlanRouteResult(path_latlon, True, None, res.expanded, start_ij, goal_ij, s_latlon, g_latlon)





