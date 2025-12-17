import numpy as np
from arcticroute.core.eco.fuel import fuel_per_nm_map, eco_cost_norm

def test_fuel_per_nm_map_monotonic_speed():
    a = fuel_per_nm_map(5.0, 0.0)
    b = fuel_per_nm_map(10.0, 0.0)
    assert b > a

def test_fuel_per_nm_map_increases_with_ice_penalty():
    a = fuel_per_nm_map(10.0, 0.0)
    b = fuel_per_nm_map(10.0, 1.0)
    assert b > a

def test_eco_cost_norm_range():
    x = np.array([[0.0, 1.0], [2.0, 3.0]])
    y = eco_cost_norm(x)
    assert np.nanmin(y) >= 0.0
    assert np.nanmax(y) <= 1.0

def test_eco_cost_norm_constant_returns_zero():
    x = np.ones((3, 3))
    y = eco_cost_norm(x)
    assert np.allclose(y, 0.0)

