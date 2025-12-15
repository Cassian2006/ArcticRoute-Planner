import numpy as np
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_cost_from_real_env, _load_normalized_ais_density
from arcticroute.core.env_real import RealEnvLayers

grid, land_mask = make_demo_grid(ny=20, nx=20)
ny, nx = grid.shape()
ais_density = np.zeros((ny, nx), dtype=float)

ocean_idx = None
for i in range(ny):
    for j in range(nx):
        if not land_mask[i, j]:
            ocean_idx = (i, j)
            break
    if ocean_idx:
        break

print(f"Ocean index: {ocean_idx}")
ais_density[ocean_idx] = 1.0

# Test normalization
normalized = _load_normalized_ais_density(
    grid=grid,
    density_source=ais_density,
    ais_density_path=None,
    prefer_real=True,
    warn_if_missing=False,
    cache_resampled=False,
)

print(f"Normalized AIS density shape: {normalized.shape if normalized is not None else None}")
if normalized is not None:
    print(f"Normalized value at ocean point: {normalized[ocean_idx]}")
    print(f"Max normalized value: {normalized.max()}")
    print(f"Min normalized value: {normalized.min()}")

env = RealEnvLayers(
    grid=grid,
    sic=np.ones((ny, nx), dtype=float) * 0.5,
    wave_swh=None,
    ice_thickness_m=None,
    land_mask=land_mask,
)

cost_0 = build_cost_from_real_env(
    grid, land_mask, env, ais_density=ais_density, w_ais_corridor=0.0
)

cost_1 = build_cost_from_real_env(
    grid, land_mask, env, ais_density=ais_density, w_ais_corridor=1.0
)

i, j = ocean_idx
print(f"\nCost at ocean point (w=0): {cost_0.cost[i, j]}")
print(f"Cost at ocean point (w=1): {cost_1.cost[i, j]}")
print(f"Difference: {cost_1.cost[i, j] - cost_0.cost[i, j]}")
print(f"Test passes: {cost_1.cost[i, j] < cost_0.cost[i, j]}")

if "ais_corridor" in cost_1.components:
    print(f"AIS corridor component at ocean point: {cost_1.components['ais_corridor'][i, j]}")


