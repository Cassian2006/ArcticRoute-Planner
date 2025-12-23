"""Base classes for planner backends."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class PlannerBackendError(Exception):
    """Base exception for planner backend errors."""
    pass


class PlannerBackend(ABC):
    """Abstract base class for planner backends."""
    
    @abstractmethod
    def plan(
        self,
        cost_field: Any,
        start_lat: float,
        start_lon: float,
        end_lat: float,
        end_lon: float,
        **kwargs: Any,
    ) -> Any:
        """Plan a route from start to end coordinates.
        
        Args:
            cost_field: Cost field object
            start_lat, start_lon: Start coordinates
            end_lat, end_lon: End coordinates
            **kwargs: Additional planning parameters
            
        Returns:
            Route planning result (format depends on backend)
        """
        pass

