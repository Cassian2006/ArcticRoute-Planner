from __future__ import annotations

import time
import numpy as np
import xarray as xr

from ArcticRoute.core.route.replan import should_replan, stitch_and_plan  # REUSE


def test_should_replan_hysteresis_and_periodic():
    now = time.time()
    cfg = {"replan": {"cool_down_sec": 900, "period_sec": 600, "risk_threshold": 0.55, "risk_delta": 0.15, "interact_delta": 0.10, "eco_delta_pct": 5}}
    # 冷却期内：不触发
    state = {"last_replan_ts": now - 100}
    ok, reason = should_replan(state, {"now_ts": now, "periodic": True}, cfg)
    assert ok is False
    assert "cooldown" in reason
    # 超过周期：触发
    state2 = {"last_replan_ts": now - 3600}
    ok2, reason2 = should_replan(state2, {"now_ts": now, "periodic": True}, cfg)
    assert ok2 is True
    assert reason2 in ("periodic",)


def _mk_grid(n=16):
    # 构造一个带 lat/lon 的 2D 风险网格 (time 维自动广播)
    lat = np.linspace(0.0, float(n-1), n, dtype=float)
    lon = np.linspace(0.0, float(n-1), n, dtype=float)
    y = np.arange(n); x = np.arange(n)
    arr = np.zeros((n, n), dtype=float)
    da = xr.DataArray(arr, dims=("y","x"), coords={"y": y, "x": x, "lat": ("y", lat), "lon": ("x", lon)}, name="risk")
    return da


def test_stitch_and_plan_no_duplicate_and_forward_progress():
    risk = _mk_grid(20)
    # 旧路线（lon,lat）对角线
    route_old = [(1.0, 1.0), (5.0, 5.0), (10.0, 10.0), (15.0, 15.0)]
    current = route_old[0]
    params = {"handover_nm": 1.0, "weights": {"w_r": 1.0, "w_d": 1.0}}
    new_coords = stitch_and_plan(current, route_old, risk, None, params)
    assert len(new_coords) >= len(route_old) - 1  # 不比冻结前更短
    # 接缝处不重复
    for i in range(len(new_coords) - 1):
        assert new_coords[i] != new_coords[i+1]

