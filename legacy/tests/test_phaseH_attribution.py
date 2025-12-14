from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ArcticRoute.core.route.attribution import explain_route

REPO_ROOT = Path(__file__).resolve().parents[1]


def _weighted_objective_from_geojson(route_path: Path) -> float:
    data = json.loads(route_path.read_text(encoding="utf-8"))
    props = data["features"][0].get("properties", {})
    w_r = float(props.get("w_r", props.get("beta", 1.0)))
    w_p = float(props.get("w_p", props.get("prior_penalty_weight", 0.0)))
    w_c = float(props.get("w_c", props.get("interact_weight", 0.0)))
    w_d = float(props.get("w_d", props.get("distance_weight", 1.0)))
    risk = float(props.get("risk_integral", 0.0))
    prior = float(props.get("prior_integral", 0.0))
    cong = float(props.get("congest_integral", 0.0))
    dist_m = float(props.get("distance_km", 0.0)) * 1000.0
    return w_r * risk + w_p * prior + w_c * cong + w_d * dist_m


def test_contribution_sum_matches_objective_smoke():
    # 使用仓库中已存在的一条路线（Phase G 产物）
    route = REPO_ROOT / "ArcticRoute" / "data_processed" / "routes" / "route_202412_nsr_wbound_smoke_balanced.geojson"
    if not route.exists():
        # 若样本缺失则跳过
        import pytest
        pytest.skip("sample route missing")
    ym = "202412"
    out_dir = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"
    payload = explain_route(route, ym, out_dir=out_dir)
    js = json.loads(Path(payload["json"]).read_text(encoding="utf-8"))
    totals = js.get("totals", {})
    # attribution 总和（包含距离项贡献）
    total_attr = float(totals.get("c_risk", 0.0)) + float(totals.get("c_prior", 0.0)) + float(totals.get("c_interact", 0.0)) + float(totals.get("c_distance", 0.0))
    # 路径属性中的规划目标（按权重合成）
    obj = _weighted_objective_from_geojson(route)
    # 容差 ±2%
    denom = max(1.0, abs(obj))
    rel_err = abs(total_attr - obj) / denom
    assert rel_err <= 0.02, f"attribution sum {total_attr} vs objective {obj} (rel_err={rel_err:.4f})"

