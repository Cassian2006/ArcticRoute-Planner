from __future__ import annotations

import math
from heapq import heappop, heappush
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ..interfaces import CostProvider, PredictorOutput, RoutePlanner, RouteResult


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


class LayerCache:
    def __init__(self, data_array, base_time_index: int):
        if "time" not in data_array.dims:
            raise ValueError("数据缺少时间维度")
        self.data_array = data_array
        self.base_idx = int(np.clip(base_time_index, 0, data_array.sizes["time"] - 1))
        self.max_idx = data_array.sizes["time"] - 1
        self.max_offset = self.max_idx - self.base_idx
        self._cache: Dict[int, np.ndarray] = {}

    def _clamp_offset(self, offset: int) -> int:
        if offset <= 0:
            return 0
        return min(offset, self.max_offset)

    def offset_for_step(self, step: int, time_step_nodes: int) -> int:
        if time_step_nodes <= 0 or self.max_offset == 0:
            return 0
        offset = step // time_step_nodes
        return self._clamp_offset(offset)

    def index_from_offset(self, offset: int) -> int:
        return self.base_idx + self._clamp_offset(offset)

    def slice_from_offset(self, offset: int) -> np.ndarray:
        idx = self.index_from_offset(offset)
        if idx not in self._cache:
            arr = self.data_array.isel(time=idx).values
            arr = np.asarray(arr, dtype="float32")
            # Ensure 2D [lat/y, lon/x]
            if arr.ndim > 2:
                arr2 = np.squeeze(arr)
                if arr2.ndim > 2:
                    # 对除最后两维外的维度取均值，保留 (y,x)
                    axes = tuple(range(0, arr2.ndim - 2))
                    arr2 = arr2.mean(axis=axes)
                arr = arr2
            self._cache[idx] = arr
        return self._cache[idx]

    def value(self, offset: int, i: int, j: int) -> float:
        arr = self.slice_from_offset(offset)
        # 边界保护：若外部索引越界，则夹取到最近的有效像元
        hi = max(0, arr.shape[-2] - 1)
        hj = max(0, arr.shape[-1] - 1)
        ii = 0 if hi == 0 else int(np.clip(i, 0, hi))
        jj = 0 if hj == 0 else int(np.clip(j, 0, hj))
        return float(arr[ii, jj])


class AStarGridTimePlanner(RoutePlanner):
    """支持时间层推进的 A* 网格规划。

    扩展（Phase O）：可从 cost_provider 上读取可选属性 _forbid_mask 与 _soft_penalty（均为 LayerCache 或 xr.DataArray），
    以实现禁行像元与软惩罚叠加；默认缺失时不影响既有行为（# REUSE）。
    """

    def _neighbors(self, neighbor8: bool) -> Sequence[Tuple[int, int]]:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if neighbor8:
            moves += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        return moves

    def plan(
        self,
        predictor_output: PredictorOutput,
        cost_provider: CostProvider,
        start_latlon: Tuple[float, float],
        goal_latlon: Tuple[float, float],
        time_step_nodes: int = 0,
        neighbor8: bool = True,
    ) -> RouteResult:
        lat = predictor_output.lat
        lon = predictor_output.lon
        risk_cache = LayerCache(predictor_output.risk, predictor_output.base_time_index)
        corridor_cache = (
            LayerCache(predictor_output.corridor, predictor_output.base_time_index)
            if predictor_output.corridor is not None
            else None
        )
        accident_cache = (
            LayerCache(predictor_output.accident, predictor_output.base_time_index)
            if predictor_output.accident is not None
            else None
        )
        # Phase O: 可选 forbid_mask 与 soft_penalty（若 cost_provider 暂存了对应 DataArray，则构建 LayerCache）
        forbid_cache = None
        softp_cache = None
        try:
            fm = getattr(cost_provider, "_forbid_mask", None)
            if fm is not None:
                forbid_cache = LayerCache(fm, predictor_output.base_time_index)
        except Exception:
            forbid_cache = None
        try:
            sp = getattr(cost_provider, "_soft_penalty", None)
            if sp is not None:
                softp_cache = LayerCache(sp, predictor_output.base_time_index)
        except Exception:
            softp_cache = None
        # Phase M: ECO normalized cost cache (optional)
        eco_cache = (
            LayerCache(predictor_output.eco_norm, predictor_output.base_time_index)
            if getattr(predictor_output, "eco_norm", None) is not None
            else None
        )

        si = int(np.abs(lat - start_latlon[0]).argmin())
        sj = int(np.abs(lon - start_latlon[1]).argmin())
        gi = int(np.abs(lat - goal_latlon[0]).argmin())
        gj = int(np.abs(lon - goal_latlon[1]).argmin())

        neighbors = self._neighbors(neighbor8)

        start_state = (si, sj, 0)
        goal_state = None
        g_cost: Dict[Tuple[int, int, int], float] = {start_state: 0.0}
        parent: Dict[Tuple[int, int, int], Tuple[int, int, int]] = {}
        queue: List[Tuple[float, float, int, int, int]] = []

        initial_h = haversine_m(lat[si], lon[sj], lat[gi], lon[gj])
        heappush(queue, (initial_h, 0.0, si, sj, 0))

        while queue:
            f_val, cost_so_far, i, j, step = heappop(queue)
            state = (i, j, step)
            if cost_so_far > g_cost.get(state, float("inf")):
                continue
            if (i, j) == (gi, gj):
                goal_state = state
                break

            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if not (0 <= ni < len(lat) and 0 <= nj < len(lon)):
                    continue
                next_step = step + 1
                curr_offset = risk_cache.offset_for_step(step, time_step_nodes)
                next_offset = risk_cache.offset_for_step(next_step, time_step_nodes)

                # Phase O: 禁行区检查（若提供 forbid_mask）
                if forbid_cache is not None:
                    try:
                        if float(forbid_cache.value(next_offset, ni, nj)) > 0.5:
                            continue
                    except Exception:
                        pass

                risk_curr = risk_cache.value(curr_offset, i, j)
                risk_next = risk_cache.value(next_offset, ni, nj)
                corridor_curr = corridor_cache.value(curr_offset, i, j) if corridor_cache is not None else None
                corridor_next = corridor_cache.value(next_offset, ni, nj) if corridor_cache is not None else None

                accident_curr = accident_cache.value(curr_offset, i, j) if accident_cache is not None else None
                accident_next = accident_cache.value(next_offset, ni, nj) if accident_cache is not None else None
                coef_curr = cost_provider.compute(risk_curr, corridor_curr, accident_curr)
                coef_next = cost_provider.compute(risk_next, corridor_next, accident_next)
                distance = haversine_m(lat[i], lon[j], lat[ni], lon[nj])
                base = distance * 0.5 * (coef_curr + coef_next)
                # Phase G: optional distance weight from cost provider (非破坏)
                dw = getattr(cost_provider, "distance_weight", 1.0)
                new_cost = cost_so_far + float(dw) * base
                # Phase M: 叠加 ECO 归一化代价（与距离成正比）
                try:
                    we = float(getattr(cost_provider, "eco_weight", 0.0))
                except Exception:
                    we = 0.0
                if eco_cache is not None and we > 0.0:
                    eco_curr = eco_cache.value(curr_offset, i, j)
                    eco_next = eco_cache.value(next_offset, ni, nj)
                    eco_term = max(0.0, 0.5 * (float(eco_curr) + float(eco_next))) * float(we) * distance
                    new_cost += eco_term

                # Phase O: 软惩罚叠加
                if softp_cache is not None:
                    try:
                        soft_p_val = softp_cache.value(next_offset, ni, nj)
                        # 假设软惩罚权重由 cost_provider 提供（或默认值）
                        soft_p_weight = getattr(cost_provider, "soft_penalty_weight", 1.0)
                        # 惩罚项与距离和代价系数成正比
                        penalty = soft_p_val * soft_p_weight * base
                        new_cost += penalty
                    except Exception:
                        pass

                next_state = (ni, nj, next_step)
                if new_cost < g_cost.get(next_state, float("inf")):
                    g_cost[next_state] = new_cost
                    parent[next_state] = state
                    h = haversine_m(lat[ni], lon[nj], lat[gi], lon[gj])
                    heappush(queue, (new_cost + h, new_cost, ni, nj, next_step))

        if goal_state is None:
            raise RuntimeError("未找到可行路径")

        path_states: List[Tuple[int, int, int]] = []
        state = goal_state
        while True:
            path_states.append(state)
            if state == start_state:
                break
            state = parent[state]
        path_states.reverse()

        lat_path = np.array([lat[s[0]] for s in path_states], dtype="float32")
        lon_path = np.array([lon[s[1]] for s in path_states], dtype="float32")
        path_idx = np.array([[s[0], s[1]] for s in path_states], dtype=int)

        risk_samples = []
        accident_samples = [] if accident_cache is not None else None
        corridor_samples = [] if corridor_cache is not None else None
        time_indices: List[int] = []
        time_change_events: List[Tuple[int, int]] = []
        time_switch_nodes: List[Dict[str, float]] = []
        prev_idx = None
        for idx, (_, _, step) in enumerate(path_states):
            offset = risk_cache.offset_for_step(step, time_step_nodes)
            time_idx = risk_cache.index_from_offset(offset)
            time_indices.append(time_idx)
            if prev_idx is not None and time_idx != prev_idx:
                time_change_events.append((idx, time_idx))
                i_node, j_node, _ = path_states[idx]
                time_switch_nodes.append(
                    {
                        "node_index": idx,
                        "time_idx": time_idx,
                        "lat": float(lat[i_node]),
                        "lon": float(lon[j_node]),
                    }
                )
            prev_idx = time_idx

            i, j, _ = path_states[idx]
            risk_samples.append(risk_cache.value(offset, i, j))
            if corridor_cache is not None:
                corridor_samples.append(corridor_cache.value(offset, i, j))
            if accident_cache is not None:
                accident_samples.append(accident_cache.value(offset, i, j))

        risk_samples_arr = np.array(risk_samples, dtype="float32")
        corridor_samples_arr = np.array(corridor_samples, dtype="float32") if corridor_samples is not None else None
        accident_samples_arr = np.array(accident_samples, dtype="float32") if accident_samples is not None else None

        distance_m = 0.0
        for idx in range(len(lat_path) - 1):
            distance_m += haversine_m(lat_path[idx], lon_path[idx], lat_path[idx + 1], lon_path[idx + 1])

        return RouteResult(
            lat_path=lat_path,
            lon_path=lon_path,
            total_cost=g_cost[goal_state],
            path_idx=path_idx,
            risk_samples=risk_samples_arr,
            corridor_samples=corridor_samples_arr,
            accident_samples=accident_samples_arr,
            time_indices=time_indices,
            time_change_events=time_change_events,
            time_switch_nodes=time_switch_nodes,
            node_count=len(path_states),
            distance_m=distance_m,
        )
