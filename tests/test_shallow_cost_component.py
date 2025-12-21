from __future__ import annotations

from pathlib import Path

import numpy as np

from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import make_demo_grid
import arcticroute.core.cost as cost_module


def test_shallow_penalty_component(monkeypatch, tmp_path):
    grid, land_mask = make_demo_grid(ny=2, nx=2)
    env = RealEnvLayers(grid=grid, sic=np.zeros((2, 2)), land_mask=land_mask)

    depth_grid = np.array([[-5.0, -20.0], [-1.0, -30.0]], dtype=float)

    def fake_loader(path: str, grid_obj):
        return depth_grid, {"source_path": str(path)}

    monkeypatch.setattr(cost_module, "load_depth_to_grid", fake_loader)
    dummy_path = tmp_path / "dummy.nc"
    dummy_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cost_module, "get_static_asset_path", lambda asset_id: dummy_path)

    cost_field = build_cost_from_real_env(
        grid,
        land_mask,
        env,
        min_depth_m=10,
        w_shallow=1.0,
    )

    shallow = cost_field.components.get("shallow_penalty")
    assert shallow is not None
    assert set(np.unique(shallow)).issubset({0.0, 1.0})
    assert shallow[0, 0] == 1.0
    assert shallow[0, 1] == 0.0


def test_shallow_penalty_disabled(monkeypatch, tmp_path):
    grid, land_mask = make_demo_grid(ny=2, nx=2)
    env = RealEnvLayers(grid=grid, sic=np.zeros((2, 2)), land_mask=land_mask)

    monkeypatch.setattr(cost_module, "load_depth_to_grid", lambda path, grid_obj: (np.zeros((2, 2)), {}))
    dummy_path = tmp_path / "dummy.nc"
    dummy_path.write_text("x", encoding="utf-8")
    monkeypatch.setattr(cost_module, "get_static_asset_path", lambda asset_id: dummy_path)

    cost_field = build_cost_from_real_env(
        grid,
        land_mask,
        env,
        min_depth_m=None,
        w_shallow=0.0,
    )

    assert "shallow_penalty" not in cost_field.components
