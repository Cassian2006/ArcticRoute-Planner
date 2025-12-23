"""A* planner backend implementation."""

from __future__ import annotations
from typing import Any

from arcticroute.core.astar import plan_route_latlon
from arcticroute.core.cost import CostField
from .base import PlannerBackend


class AStarBackend(PlannerBackend):
    """A* planner backend - always available fallback."""
    
    def plan(
        self,
        cost_field: CostField,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        **kwargs: Any,
    ) -> list[tuple[float, float]]:
        """Plan route using A* algorithm.
        
        Args:
            cost_field: Cost field
            start_lat, start_lon: Start coordinates
            end_lat, end_lon: End coordinates
            **kwargs: Additional parameters (neighbor8, etc.)
            
        Returns:
            List of (lat, lon) tuples representing the route
        """
        neighbor8 = kwargs.get("neighbor8", True)
        return plan_route_latlon(
            cost_field,
            start_lat,
            start_lon,
            end_lat,
            end_lon,
            neighbor8=neighbor8,
        )

