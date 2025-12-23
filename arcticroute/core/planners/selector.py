from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Tuple

from ..astar import plan_route_latlon
from ..cost import CostField
from ..grid import Grid2D


class RoutePlannerBackend(Protocol):
    name: str

    def plan(
        self,
        grid: Grid2D,
        land_mask,
        cost_field: CostField,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        allow_diag: bool = True,
    ):
        ...


@dataclass
class AStarBackend:
    name: str = "astar"

    def plan(
        self,
        grid: Grid2D,
        land_mask,
        cost_field: CostField,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        allow_diag: bool = True,
    ):
        return plan_route_latlon(
            cost_field,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            neighbor8=allow_diag,
        )


def _polarroute_available() -> bool:
    """轻量检查 polarroute 是否可用（不强求安装）。"""
    return importlib.util.find_spec("polarroute") is not None


def select_planner_backend(
    mode: str,
    pipeline_dir: str | None = None,
    external_vessel_mesh: str | None = None,
    external_route_config: str | None = None,
) -> Tuple[RoutePlannerBackend, dict[str, Any]]:
    """
    选择规划后端，带自动回退逻辑。

    Returns:
        (backend, meta)
        meta:
          planner_used, planner_mode, fallback_reason,
          pipeline_dir, external_vessel_mesh, external_route_config
    """
    meta: dict[str, Any] = {
        "planner_mode": mode,
        "planner_used": None,
        "fallback_reason": None,
        "pipeline_dir": pipeline_dir,
        "external_vessel_mesh": external_vessel_mesh,
        "external_route_config": external_route_config,
    }

    def _fallback(reason: str):
        meta["planner_used"] = "astar"
        meta["fallback_reason"] = reason
        return AStarBackend(), meta

    # 强制 A*
    if mode == "astar":
        meta["planner_used"] = "astar"
        return AStarBackend(), meta

    # PolarRoute（pipeline）
    if mode == "polarroute_pipeline":
        if not pipeline_dir:
            return _fallback("pipeline_dir 未提供，回退 A*")
        if not Path(pipeline_dir).exists():
            return _fallback("pipeline_dir 不存在，回退 A*")
        if not _polarroute_available():
            return _fallback("PolarRoute 未安装，回退 A*")
        # 当前仓库未集成 PolarRoute 运行器，仍回退 A*
        return _fallback("PolarRoute 后端未集成，回退 A*")

    # PolarRoute（external 文件）
    if mode == "polarroute_external":
        if not external_vessel_mesh or not external_route_config:
            return _fallback("外部 mesh/route_config 缺失，回退 A*")
        if not Path(external_vessel_mesh).exists() or not Path(external_route_config).exists():
            return _fallback("外部文件不存在，回退 A*")
        if not _polarroute_available():
            return _fallback("PolarRoute 未安装，回退 A*")
        return _fallback("PolarRoute 外部模式未集成，回退 A*")

    # Auto：按优先级尝试，全部不可用时回退 A*
    if mode == "auto":
        if pipeline_dir and Path(pipeline_dir).exists() and _polarroute_available():
            return _fallback("PolarRoute 后端未集成（auto pipeline），回退 A*")
        if (
            external_vessel_mesh
            and external_route_config
            and Path(external_vessel_mesh).exists()
            and Path(external_route_config).exists()
            and _polarroute_available()
        ):
            return _fallback("PolarRoute 后端未集成（auto external），回退 A*")
        return _fallback("PolarRoute 不可用，自动回退 A*")

    # 未知模式
    return _fallback(f"未知规划模式 {mode}，回退 A*")

