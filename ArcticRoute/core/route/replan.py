from __future__ import annotations

"""
Phase N · 实时重规划引擎

提供：
- should_replan(state, deltas, cfg) -> (bool, reason)
- stitch_and_plan(current_pos, route_old, risk, prior_penalty, params) -> route_new(list[lon,lat])

约定：
state: {
  "last_replan_ts": float | None,
  "last_reason": str | None,
  "last_route": list[[lon,lat]] | None,
}

deltas: {
  "periodic": bool,
  "risk_jump": float | None,
  "risk_mean_next": float | None,
  "interact_delta": float | None,
  "eco_delta_pct": float | None,
  "has_new_surface": bool | None,
  "now_ts": float,
}

cfg: 见 docs Phase N 与 configs/replan.yaml
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math
import time

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.route.metrics import haversine_m  # REUSE
from ArcticRoute.core.planners.astar_grid_time import AStarGridTimePlanner  # REUSE
from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider  # REUSE


def nm_to_m(x: float) -> float:
    return float(x) * 1852.0


def _polyline_length_m(coords: Sequence[Sequence[float]]) -> float:
    if not coords or len(coords) < 2:
        return 0.0
    d = 0.0
    for i in range(len(coords) - 1):
        lon1, lat1 = coords[i]
        lon2, lat2 = coords[i + 1]
        d += haversine_m(lat1, lon1, lat2, lon2)
    return d


def _project_on_segment(p: Tuple[float, float], a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[Tuple[float, float], float, float]:
    """将点p投影到线段ab（经纬度近似为平面），返回 (proj_lonlat, t in [0,1], dist_m)。"""
    lon, lat = p; x1, y1 = a; x2, y2 = b
    # 简易等矩形近似（小范围足够）：按纬度缩放经度
    sx = math.cos(math.radians((y1 + y2) * 0.5))
    ax, ay = x1 * sx, y1
    bx, by = x2 * sx, y2
    px, py = lon * sx, lat
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    vv = vx * vx + vy * vy
    t = 0.0 if vv <= 1e-12 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
    qx, qy = ax + t * vx, ay + t * vy
    # 还原经纬度
    qlon = qx / sx if abs(sx) > 1e-8 else lon
    qlat = qy
    d = haversine_m(lat, lon, qlat, qlon)
    return (qlon, qlat), float(t), float(d)


def _progress_projection(current_pos: Tuple[float, float], route_old: Sequence[Sequence[float]]) -> Tuple[int, float, Tuple[float, float]]:
    """返回 (seg_idx, t, proj_point) 其中 seg_idx 表示 [i,i+1]，t 为段内比例。若路线不足两点，退化为(0,0,p0)。"""
    if not route_old:
        return 0, 0.0, current_pos
    if len(route_old) == 1:
        return 0, 0.0, route_old[0]
    best = (0, 0.0, route_old[0])
    best_d = float("inf")
    for i in range(len(route_old) - 1):
        a = route_old[i]; b = route_old[i + 1]
        q, t, d = _project_on_segment(current_pos, a, b)
        if d < best_d:
            best_d = d; best = (i, t, q)
    return best


def should_replan(state: Dict[str, Any], deltas: Dict[str, Any], cfg: Dict[str, Any]) -> Tuple[bool, str]:
    now_ts = float(deltas.get("now_ts", time.time()))
    last_ts = float(state.get("last_replan_ts", 0.0) or 0.0)
    cool = float((cfg.get("replan") or {}).get("cool_down_sec", cfg.get("cool_down_sec", 900)))
    period = float((cfg.get("replan") or {}).get("period_sec", cfg.get("period_sec", 1800)))
    risk_thr = float((cfg.get("replan") or {}).get("risk_threshold", cfg.get("risk_threshold", 0.55)))
    risk_delta_thr = float((cfg.get("replan") or {}).get("risk_delta", cfg.get("risk_delta", 0.15)))
    inter_thr = float((cfg.get("replan") or {}).get("interact_delta", cfg.get("interact_delta", 0.10)))
    eco_thr = float((cfg.get("replan") or {}).get("eco_delta_pct", cfg.get("eco_delta_pct", 5.0)))

    # 冷却期
    if last_ts > 0 and (now_ts - last_ts) < cool:
        return False, f"cooldown({int(cool)}s)"

    if bool(deltas.get("periodic", False)):
        # 周期触发：距上次超过 period
        if last_ts <= 0 or (now_ts - last_ts) >= period:
            return True, "periodic"

    # 风险窗口
    mean_next = deltas.get("risk_mean_next", None)
    if isinstance(mean_next, (int, float)) and mean_next >= risk_thr:
        return True, f"risk_mean>{risk_thr}"
    dj = deltas.get("risk_jump", None)
    if isinstance(dj, (int, float)) and dj >= risk_delta_thr:
        return True, f"risk_jump>{risk_delta_thr}"

    # 交互
    inter_d = deltas.get("interact_delta", None)
    if isinstance(inter_d, (int, float)) and inter_d >= inter_thr:
        return True, "interact_surge"

    # ECO（可选）
    eco_pct = deltas.get("eco_delta_pct", None)
    if isinstance(eco_pct, (int, float)) and eco_pct >= eco_thr:
        return True, "eco_change"

    # 新的 live surface 可作为弱触发（若 period==0）
    if bool(deltas.get("has_new_surface", False)) and period <= 0:
        return True, "new_surface"

    return False, "stable"


def _predictor_from_layers(risk: "xr.DataArray", prior: Optional["xr.DataArray"], interact: Optional["xr.DataArray"]) -> Any:
    # REUSE: 与 scan._predictor_from_layers 相同
    if "time" not in risk.dims:
        risk = risk.expand_dims({"time": [0]})
    latn = "lat" if "lat" in risk.coords else ("latitude" if "latitude" in risk.coords else None)
    lonn = "lon" if "lon" in risk.coords else ("longitude" if "longitude" in risk.coords else None)
    if not (latn and lonn):
        raise RuntimeError("risk layer missing lat/lon coordinates")
    lat = np.asarray(risk.coords[latn].values, dtype="float32")
    lon = np.asarray(risk.coords[lonn].values, dtype="float32")
    if prior is not None and "time" not in prior.dims and "time" in risk.dims:
        prior = prior.expand_dims({"time": risk.coords["time"]})
    if interact is not None and "time" not in interact.dims and "time" in risk.dims:
        interact = interact.expand_dims({"time": risk.coords["time"]})
    class PO:
        def __init__(self, risk, corridor, lat, lon):
            self.risk = risk; self.corridor = corridor; self.lat = lat; self.lon = lon; self.base_time_index = 0; self.accident = interact
    return PO(risk, prior, lat, lon)


def _handover_point(route_old: Sequence[Sequence[float]], idx: int, handover_nm: float) -> Tuple[int, Sequence[float]]:
    if not route_old:
        return 0, []
    target = nm_to_m(max(0.0, float(handover_nm)))
    if target <= 0:
        return idx, route_old[idx]
    cum = 0.0
    j = idx
    while j < len(route_old) - 1 and cum < target:
        lon1, lat1 = route_old[j]
        lon2, lat2 = route_old[j + 1]
        cum += haversine_m(lat1, lon1, lat2, lon2)
        j += 1
    return j, route_old[j]


def stitch_and_plan(
    current_pos: Tuple[float, float],
    route_old: Sequence[Sequence[float]],
    risk: "xr.DataArray",
    prior_penalty: Optional["xr.DataArray"],
    params: Dict[str, Any],
) -> List[Sequence[float]]:
    """从当前进度向前 handover 冻结，再规划至终点，并缝合。
    params: {
      "handover_nm": float,
      "weights": {"w_r":float, "w_c":float, "w_p":float},
      "neighbor8": bool,
      "heuristic": str,
    }
    """
    if xr is None:
        raise RuntimeError("xarray required")
    if not route_old or len(route_old) < 2:
        # 退化为全程规划：current_pos -> last point
        start = current_pos
        goal = current_pos if not route_old else route_old[-1]
        route_prefix: List[Sequence[float]] = [start]
    else:
        seg_idx, t, proj = _progress_projection(current_pos, route_old)
        # 从投影点向前 handover_nm，可能跨多段
        hand_nm = float(params.get("handover_nm", 8.0))
        remain_m = nm_to_m(max(0.0, hand_nm))
        cur_point = proj
        i = seg_idx
        # 先扣除本段余量
        if remain_m > 0:
            # 段内从 t 到 1 的距离
            a = route_old[i]; b = route_old[i+1]
            _, t2, _ = _project_on_segment(cur_point, a, b)  # 归一到同一段参数
            # 迭代向后推进
            while remain_m > 0 and i < len(route_old) - 1:
                a = cur_point; b = route_old[i+1]
                d_step = haversine_m(a[1], a[0], b[1], b[0])
                if d_step >= remain_m:
                    # 在段内插入插值点
                    ratio = 0.0 if d_step <= 1e-6 else remain_m / d_step
                    lon = a[0] + (b[0] - a[0]) * ratio
                    lat = a[1] + (b[1] - a[1]) * ratio
                    cur_point = (lon, lat)
                    remain_m = 0.0
                    break
                else:
                    cur_point = b
                    remain_m -= d_step
                    i += 1
                    if i >= len(route_old) - 1:
                        break
        H = cur_point
        # 构建冻结段：0..seg_idx 的原点 + 投影点（若不等于原节点）+ 中间必要点直到 H 所在位置
        prefix: List[Sequence[float]] = []
        if seg_idx > 0:
            prefix.extend(route_old[0:seg_idx])
        # 加入段起点
        prefix.append(route_old[seg_idx])
        # 加入投影点（若不同于段起点）
        if proj != route_old[seg_idx]:
            prefix.append(proj)
        # 若 H 跨越到后续节点，填入经过的整点
        # 找 H 在何处：如果 H 等于某个节点则止步，否则保持到 H
        # 从 seg_idx+1 依次加入，直到超过 H 所在段末端
        j = seg_idx + 1
        while j < len(route_old) and route_old[j] != H:
            # 若 H 在 route_old[j-1] 与 route_old[j] 之间且已到 H，则停止
            # 这里用距离判断是否到达 H 所在段：当 H 等于 route_old[j] 或超过最后一段时退出
            # 简化：直到 H 等于 route_old[j] 才停止，否则继续追加整点
            if haversine_m(route_old[j-1][1], route_old[j-1][0], H[1], H[0]) + haversine_m(H[1], H[0], route_old[j][1], route_old[j][0]) - haversine_m(route_old[j-1][1], route_old[j-1][0], route_old[j][1], route_old[j][0]) <= 1e-3:
                break
            prefix.append(route_old[j])
            j += 1
        # 最后加入 H
        if not prefix or prefix[-1] != H:
            prefix.append(H)
        route_prefix = prefix
        start = H
        goal = route_old[-1]

    # 规划 H->goal
    po = _predictor_from_layers(risk, prior_penalty, None)
    planner = AStarGridTimePlanner()
    w = (params.get("weights") or {})
    beta = float(w.get("w_r", 1.0))
    w_c = float(w.get("w_c", 0.0))
    w_p = float(w.get("w_p", 0.0))
    cost = EnvRiskCostProvider(beta=beta, p_exp=1.0, gamma=0.0, interact_weight=w_c, prior_penalty_weight=w_p)
    try:
        cost.distance_weight = float(w.get("w_d", 1.0))  # REUSE
    except Exception:
        pass
    res = planner.plan(predictor_output=po, cost_provider=cost, start_latlon=(float(start[1]), float(start[0])), goal_latlon=(float(goal[1]), float(goal[0])))
    seg = [(float(lon), float(lat)) for lon, lat in zip(res.lon_path.tolist(), res.lat_path.tolist())]

    # 缝合：避免回退重复点
    if route_prefix:
        if seg and seg[0] == route_prefix[-1]:
            seg = seg[1:]
        route_new = list(route_prefix) + seg
    else:
        route_new = seg

    # 防止向后折返：简单检测单调前进（允许曲折，仅禁止明显倒退回 prefix 中更早的点）
    # 此处保守不做强约束，仅返回缝合结果供上层做 Δlen / Δrisk 检查
    return route_new


__all__ = ["should_replan", "stitch_and_plan", "nm_to_m"]

