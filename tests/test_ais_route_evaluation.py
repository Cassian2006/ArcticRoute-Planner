import numpy as np
import pytest

from arcticroute.core.ais_analysis import evaluate_route_vs_ais_density


def _make_test_grid(n: int = 10):
    lats = np.linspace(0, n - 1, n)
    lons = np.linspace(0, n - 1, n)
    lon2d, lat2d = np.meshgrid(lons, lats)
    # Row-wise gradient from 0 -> 1
    row_vals = np.linspace(0.0, 1.0, n).reshape(n, 1)
    ais_density = np.tile(row_vals, (1, n))
    return lat2d, lon2d, ais_density


def test_route_in_high_corridor():
    lat2d, lon2d, ais_density = _make_test_grid()
    # Route along the last row (highest densities)
    route = [(lat2d[-1, j], lon2d[-1, j]) for j in range(lat2d.shape[1])]

    stats = evaluate_route_vs_ais_density(route, lat2d, lon2d, ais_density)

    assert stats.total_steps == len(route)
    assert stats.frac_high_corridor > 0.9
    assert stats.frac_low_usage < 0.05


def test_route_in_low_usage():
    lat2d, lon2d, ais_density = _make_test_grid()
    # Route along the first row (lowest densities)
    route = [(lat2d[0, j], lon2d[0, j]) for j in range(lat2d.shape[1])]

    stats = evaluate_route_vs_ais_density(route, lat2d, lon2d, ais_density)

    assert stats.frac_low_usage > 0.9
    assert stats.frac_high_corridor < 0.05


def test_route_with_nans():
    lat2d, lon2d, ais_density = _make_test_grid()
    ais_density[:, 5] = np.nan  # insert a NaN stripe
    row_idx = 5
    route = [(lat2d[row_idx, j], lon2d[row_idx, j]) for j in range(lat2d.shape[1])]

    stats = evaluate_route_vs_ais_density(route, lat2d, lon2d, ais_density)

    assert stats.num_nan > 0
    assert stats.mean_density == pytest.approx(ais_density[row_idx, 0], rel=1e-6)
    assert stats.total_steps == len(route)


def test_empty_route():
    lat2d, lon2d, ais_density = _make_test_grid()

    stats = evaluate_route_vs_ais_density([], lat2d, lon2d, ais_density)

    assert stats.total_steps == 0
    assert stats.mean_density == 0.0
    assert stats.frac_high_corridor == 0.0
    assert stats.frac_low_usage == 0.0
