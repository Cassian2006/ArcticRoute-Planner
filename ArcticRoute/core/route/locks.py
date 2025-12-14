from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from ArcticRoute.core.interfaces import PredictorOutput
from ArcticRoute.core.planners.astar_grid_time import AStarGridTimePlanner
from ArcticRoute.core.cost.env_risk_cost import EnvRiskCostProvider


def plan_with_locks(
    *,
    predictor: PredictorOutput,
    planner: AStarGridTimePlanner,
    cost: EnvRiskCostProvider,
    start_latlon: Tuple[float, float],
    goal_latlon: Tuple[float, float],
    lock_points: Optional[Sequence[Tuple[float, float]]],
) -> List[Tuple[float, float]]:
    """
    最小锁点分段规划：
    - 若 lock_points 为空：直接规划 start→goal
    - 若存在：按 start→lock1→lock2→...→goal 逐段规划，拼接路径（去重相邻节点）
    - 不实现回退禁用/自交检查（后续 O-03 完整化），当前仅满足 Phase O 最小闭环。
    """
    ordered: List[Tuple[float, float]] = [start_latlon]
    if lock_points:
        ordered.extend([(float(lat), float(lon)) if isinstance(lat, float) and isinstance(lon, float) else (float(p[0]), float(p[1])) for lat, lon in lock_points])  # type: ignore[misc]
    ordered.append(goal_latlon)

    full: List[Tuple[float, float]] = []
    for i in range(len(ordered) - 1):
        a = ordered[i]
        b = ordered[i + 1]
        res = planner.plan(predictor_output=predictor, cost_provider=cost, start_latlon=a, goal_latlon=b)
        seg = [(float(lon), float(lat)) for lon, lat in zip(res.lon_path.tolist(), res.lat_path.tolist())]
        if i == 0:
            full.extend(seg)
        else:
            # 去掉与上一段重叠的首点
            full.extend(seg[1:])
    return full


__all__ = ["plan_with_locks"]
