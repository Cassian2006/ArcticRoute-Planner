from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush
from math import sqrt
from typing import List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr


@dataclass
class RouteSummary:
    path_ij: List[Tuple[int, int]]
    cost_sum: float


def _neighbors8() -> Sequence[Tuple[int, int, float]]:
    # di, dj, step_len
    return [
        (-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
        (-1, -1, sqrt(2.0)), (-1, 1, sqrt(2.0)), (1, -1, sqrt(2.0)), (1, 1, sqrt(2.0)),
    ]


def _neighbors4() -> Sequence[Tuple[int, int, float]]:
    return [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]


def _heuristic(i: int, j: int, gi: int, gj: int, kind: str = "manhattan") -> float:
    dx = abs(i - gi)
    dy = abs(j - gj)
    if kind == "euclidean":
        return float((dx * dx + dy * dy) ** 0.5)
    if kind == "octile":
        # 常用于8邻接网格：max + (sqrt(2)-1)*min
        m, n = (dx, dy) if dx >= dy else (dy, dx)
        return float(m + (sqrt(2.0) - 1.0) * n)
    # 默认 manhattan
    return float(dx + dy)


def _safe_cost(arr: np.ndarray, i: int, j: int) -> Optional[float]:
    v = float(arr[i, j])
    if np.isnan(v) or v >= 1e9:  # 大值或 NaN 视为不可通行
        return None
    return v


def astar_on_cost(
    cost_slice: xr.DataArray,
    start_ij: Tuple[int, int],
    goal_ij: Tuple[int, int],
    *,
    neighbor8: bool = True,
    heuristic: str = "manhattan",
) -> RouteSummary:
    if not {"y", "x"}.issubset(cost_slice.dims):
        raise ValueError("cost_slice 需要包含 y,x 维度")
    Ny = int(cost_slice.sizes["y"])  # i 对应 y 轴
    Nx = int(cost_slice.sizes["x"])  # j 对应 x 轴

    arr = cost_slice.values.astype("float32", copy=False)

    si, sj = int(start_ij[0]), int(start_ij[1])
    gi, gj = int(goal_ij[0]), int(goal_ij[1])
    if not (0 <= si < Ny and 0 <= sj < Nx and 0 <= gi < Ny and 0 <= gj < Nx):
        raise ValueError("起止点索引越界")

    if _safe_cost(arr, si, sj) is None or _safe_cost(arr, gi, gj) is None:
        raise ValueError("起点或终点不可达（NaN/掩膜/超大代价）")

    moves = _neighbors8() if neighbor8 else _neighbors4()

    start = (si, sj)
    goal = (gi, gj)

    g_cost = {start: 0.0}
    parent = {}
    pq: List[Tuple[float, float, int, int]] = []
    heappush(pq, (_heuristic(si, sj, gi, gj, heuristic), 0.0, si, sj))

    while pq:
        f, g, i, j = heappop(pq)
        if g > g_cost.get((i, j), float("inf")):
            continue
        if (i, j) == goal:
            break
        c0 = _safe_cost(arr, i, j)
        if c0 is None:
            continue
        for di, dj, step_len in moves:
            ni, nj = i + di, j + dj
            if not (0 <= ni < Ny and 0 <= nj < Nx):
                continue
            c1 = _safe_cost(arr, ni, nj)
            if c1 is None:
                continue
            # The risk cost (c0, c1) is too small compared to the heuristic.
            # We add a base cost of 1 for distance and scale the risk cost to make it significant.
            step_cost = (1 + 0.5 * (c0 + c1) * 1000) * step_len
            new_g = g + step_cost
            if new_g < g_cost.get((ni, nj), float("inf")):
                g_cost[(ni, nj)] = new_g
                parent[(ni, nj)] = (i, j)
                h = _heuristic(ni, nj, gi, gj, heuristic)
                heappush(pq, (new_g + h, new_g, ni, nj))

    if (gi, gj) not in parent and (si, sj) != (gi, gj):
        # 不可达
        raise RuntimeError("未找到可行路径")

    # 回溯
    path: List[Tuple[int, int]] = []
    node = (gi, gj)
    path.append(node)
    while node != (si, sj):
        node = parent[node]
        path.append(node)
    path.reverse()

    # 汇总代价
    cost_sum = 0.0
    for k in range(len(path) - 1):
        i, j = path[k]
        ni, nj = path[k + 1]
        c0 = _safe_cost(arr, i, j) or 1.0
        c1 = _safe_cost(arr, ni, nj) or 1.0
        step_len = sqrt(2.0) if (ni != i and nj != j) else 1.0
        cost_sum += 0.5 * (c0 + c1) * step_len

    return RouteSummary(path_ij=path, cost_sum=float(cost_sum))

