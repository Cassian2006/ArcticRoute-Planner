"""Fuel consumption and economic cost calculations for Arctic routing.

This module provides functions to estimate fuel consumption per nautical mile
based on vessel speed and ice conditions, and to normalize economic cost fields.
"""

from __future__ import annotations

import numpy as np
from typing import Union


def fuel_per_nm_map(
    speed_knots: Union[float, np.ndarray],
    ice_penalty: Union[float, np.ndarray] = 0.0,
) -> Union[float, np.ndarray]:
    """
    Estimate fuel consumption per nautical mile as a function of speed and ice penalty.
    
    Fuel consumption increases with speed (typically cubic relationship) and with
    ice conditions (ice_penalty).
    
    Args:
        speed_knots: Vessel speed in knots (scalar or array)
        ice_penalty: Ice condition penalty factor, typically in [0, 1] range
                    (scalar or array, same shape as speed_knots)
    
    Returns:
        Fuel consumption per nautical mile (same shape as inputs)
    """
    # Base fuel consumption model: fuel ~ speed^2 (simplified cubic model)
    # Typical fuel consumption increases significantly with speed
    base_fuel = 0.1 + 0.02 * np.asarray(speed_knots) ** 2
    
    # Ice penalty multiplier: increases fuel consumption in ice conditions
    # ice_penalty in [0, 1] maps to multiplier in [1.0, 1.5]
    ice_multiplier = 1.0 + 0.5 * np.asarray(ice_penalty)
    
    return base_fuel * ice_multiplier


def eco_cost_norm(cost_field: np.ndarray) -> np.ndarray:
    """
    Normalize an economic cost field to [0, 1] range using min-max normalization.
    
    Handles NaN values gracefully by ignoring them in min/max calculation.
    If all values are constant (or NaN), returns zeros.
    
    Args:
        cost_field: 2D or higher dimensional cost array (may contain NaN)
    
    Returns:
        Normalized cost field in [0, 1] range (NaN values preserved)
    """
    cost_array = np.asarray(cost_field, dtype=float)
    
    # Calculate min and max ignoring NaN
    with np.errstate(invalid='ignore'):
        cost_min = np.nanmin(cost_array)
        cost_max = np.nanmax(cost_array)
    
    # Handle edge cases
    if not np.isfinite(cost_min) or not np.isfinite(cost_max):
        # All NaN or empty
        return np.full_like(cost_array, np.nan, dtype=float)
    
    if np.isclose(cost_min, cost_max):
        # Constant values -> return zeros (no variation)
        return np.zeros_like(cost_array, dtype=float)
    
    # Min-max normalization
    normalized = (cost_array - cost_min) / (cost_max - cost_min)
    
    return normalized



