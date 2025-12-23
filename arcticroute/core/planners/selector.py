from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Any

from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
from arcticroute.core.planners.astar_backend import AStarBackend
from arcticroute.core.planners.base import PlannerBackendError

Mode = Literal["auto", "astar", "polarroute_pipeline", "polarroute_external"]

@dataclass
class PlannerSelection:
    requested_mode: Mode
    planner_used: str           # "astar" / "polarroute"
    planner_mode: str           # resolved mode
    fallback_reason: str | None
    pipeline_dir: str | None
    external_vessel_mesh: str | None
    external_route_config: str | None

def _exists(p: Optional[str]) -> bool:
    return bool(p) and Path(p).exists()

def select_backend(
    mode: Mode = "auto",
    pipeline_dir: str | None = None,
    external_vessel_mesh: str | None = None,
    external_route_config: str | None = None,
) -> tuple[Any, PlannerSelection]:
    requested = mode

    def use_astar(reason: str | None) -> tuple[Any, PlannerSelection]:
        return AStarBackend(), PlannerSelection(
            requested_mode=requested,
            planner_used="astar",
            planner_mode="astar",
            fallback_reason=reason,
            pipeline_dir=pipeline_dir,
            external_vessel_mesh=external_vessel_mesh,
            external_route_config=external_route_config,
        )

    # force astar
    if mode == "astar":
        return use_astar(None)

    # pipeline mode
    if mode == "polarroute_pipeline":
        if not pipeline_dir or not Path(pipeline_dir).exists():
            return use_astar("pipeline_unavailable: pipeline_dir_not_set_or_missing")
        try:
            backend = PolarRouteBackend(pipeline_dir=pipeline_dir)
            return backend, PlannerSelection(requested, "polarroute", "polarroute_pipeline", None, pipeline_dir, external_vessel_mesh, external_route_config)
        except Exception as e:
            return use_astar(f"pipeline_unavailable: {type(e).__name__}")

    # external mode
    if mode == "polarroute_external":
        if not (_exists(external_vessel_mesh) and _exists(external_route_config)):
            return use_astar("external_unavailable: missing vessel_mesh or route_config")
        try:
            backend = PolarRouteBackend(
                vessel_mesh_path=external_vessel_mesh,
                route_config_path=external_route_config,
            )
            return backend, PlannerSelection(requested, "polarroute", "polarroute_external", None, pipeline_dir, external_vessel_mesh, external_route_config)
        except Exception as e:
            return use_astar(f"external_unavailable: {type(e).__name__}")

    # auto mode: pipeline -> external -> astar
    if pipeline_dir and Path(pipeline_dir).exists():
        try:
            backend = PolarRouteBackend(pipeline_dir=pipeline_dir)
            return backend, PlannerSelection(requested, "polarroute", "polarroute_pipeline", None, pipeline_dir, external_vessel_mesh, external_route_config)
        except Exception as e:
            # fallthrough
            pass
    if _exists(external_vessel_mesh) and _exists(external_route_config):
        try:
            backend = PolarRouteBackend(vessel_mesh_path=external_vessel_mesh, route_config_path=external_route_config)
            return backend, PlannerSelection(requested, "polarroute", "polarroute_external", None, pipeline_dir, external_vessel_mesh, external_route_config)
        except Exception:
            pass

    return use_astar("auto_fallback: polarroute_unavailable")

