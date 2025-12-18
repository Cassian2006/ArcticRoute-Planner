"""Route evaluation functions for economic cost assessment.

This module provides functions to evaluate economic costs along routes
in the Arctic routing system.
"""

from __future__ import annotations

import numpy as np
from typing import Optional


def eval_route_eco(
    route_data: Optional[np.ndarray] = None,
    land_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Evaluate economic cost for a route or grid.
    
    This function computes economic cost values, potentially based on route data
    and land mask constraints. If no route data is provided, returns a default
    zero-cost field.
    
    Args:
        route_data: Optional route data array (not used in current implementation)
        land_mask: Optional land mask array indicating land cells (True = land)
    
    Returns:
        Economic cost field as a 2D numpy array. Returns zeros for all cells,
        or matches the shape of land_mask if provided.
    """
    if land_mask is not None:
        # Return zero cost field with same shape as land mask
        return np.zeros_like(land_mask, dtype=float)
    else:
        # Return a default zero cost field (demo grid size)
        # Using a small default grid size for fallback
        return np.zeros((10, 10), dtype=float)



