"""
AIS route evaluation helpers.
Provides utilities to compare planned routes against historical AIS density.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence, Tuple

import numpy as np


def _ensure_grid_2d(
    grid_lats: np.ndarray, grid_lons: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return 2D lat/lon grids (matching broadcasting rules)."""
    if grid_lats.ndim == 1 and grid_lons.ndim == 1:
        lon2d, lat2d = np.meshgrid(grid_lons, grid_lats)
        return lat2d, lon2d

    if grid_lats.ndim == 2 and grid_lons.ndim == 2:
        lat2d, lon2d = np.broadcast_arrays(grid_lats, grid_lons)
        return lat2d, lon2d

    # Fallback: broadcast whatever shapes we have to the same 2D shape
    lat2d, lon2d = np.broadcast_arrays(grid_lats, grid_lons)
    return lat2d, lon2d


def _map_route_to_indices(
    route_latlon: Iterable[Tuple[float, float]],
    lat2d: np.ndarray,
    lon2d: np.ndarray,
) -> list[tuple[int, int]]:
    """
    Map (lat, lon) points to nearest grid indices using the same nearest-neighbor
    logic as existing cost/analysis utilities.
    """
    ny, nx = lat2d.shape
    ij_path: list[tuple[int, int]] = []

    for lat, lon in route_latlon:
        dist = np.sqrt((lat2d - lat) ** 2 + (lon2d - lon) ** 2)
        i, j = np.unravel_index(np.argmin(dist), dist.shape)
        i = int(np.clip(i, 0, ny - 1))
        j = int(np.clip(j, 0, nx - 1))
        ij_path.append((i, j))

    return ij_path


@dataclass
class AISRouteStats:
    total_steps: int
    mean_density: float
    max_density: float
    p80_threshold: float
    frac_high_corridor: float
    frac_low_usage: float
    num_nan: int
    notes: List[str] = field(default_factory=list)


def evaluate_route_vs_ais_density(
    route_latlon: Sequence[Tuple[float, float]],
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
    ais_density: np.ndarray,
) -> AISRouteStats:
    """
    Sample AIS density along a route and compute corridor adherence metrics.

    Behavior:
      1. Map each (lat, lon) to nearest grid index (consistent with existing
         cost/profile sampling logic).
      2. Collect density values, tracking NaNs separately.
      3. Compute mean/max density, p80 threshold (from full map), fractions of
         points above p80 and below p20, and NaN counts.
      4. Percentiles are computed from the full ais_density, ignoring NaNs.
      5. If all along-route samples are NaN, return zeros and add a note.
    """
    lat2d, lon2d = _ensure_grid_2d(grid_lats, grid_lons)

    # Basic validations/shape alignment
    if ais_density.shape != lat2d.shape:
        try:
            ais_density = np.broadcast_to(ais_density, lat2d.shape)
        except ValueError:
            raise ValueError(
                f"ais_density shape {ais_density.shape} is not compatible with grid {lat2d.shape}"
            )

    total_steps = len(route_latlon)
    ij_path = _map_route_to_indices(route_latlon, lat2d, lon2d)

    # Global percentiles (ignore NaNs)
    global_valid = ais_density[np.isfinite(ais_density)]
    notes: List[str] = []
    if global_valid.size == 0:
        p20 = 0.0
        p80 = 0.0
        notes.append("ais_density has no finite values; percentiles set to 0")
    else:
        p20 = float(np.nanpercentile(global_valid, 20))
        p80 = float(np.nanpercentile(global_valid, 80))

    sampled: list[float] = []
    num_nan = 0
    for i, j in ij_path:
        val = ais_density[i, j]
        if np.isnan(val):
            num_nan += 1
        else:
            sampled.append(float(val))

    if total_steps == 0:
        return AISRouteStats(
            total_steps=0,
            mean_density=0.0,
            max_density=0.0,
            p80_threshold=p80,
            frac_high_corridor=0.0,
            frac_low_usage=0.0,
            num_nan=0,
            notes=["empty route"],
        )

    if not sampled:
        notes.append("all sampled AIS densities are NaN")
        return AISRouteStats(
            total_steps=total_steps,
            mean_density=0.0,
            max_density=0.0,
            p80_threshold=p80,
            frac_high_corridor=0.0,
            frac_low_usage=0.0,
            num_nan=num_nan,
            notes=notes,
        )

    sampled_arr = np.asarray(sampled, dtype=float)
    mean_density = float(np.mean(sampled_arr))
    max_density = float(np.max(sampled_arr))

    frac_high_corridor = float(np.sum(sampled_arr > p80) / len(sampled_arr))
    frac_low_usage = float(np.sum(sampled_arr < p20) / len(sampled_arr))

    return AISRouteStats(
        total_steps=total_steps,
        mean_density=mean_density,
        max_density=max_density,
        p80_threshold=p80,
        frac_high_corridor=frac_high_corridor,
        frac_low_usage=frac_low_usage,
        num_nan=num_nan,
        notes=notes,
    )
