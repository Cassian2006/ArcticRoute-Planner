from __future__ import annotations

from pathlib import Path
import json

from ArcticRoute.core.feedback.schema import TagConfig, validate_feedback_lines, digest_items
from ArcticRoute.core.constraints.checker import check_route
from ArcticRoute.core.constraints.engine import Constraints


def test_feedback_parse_and_digest(tmp_path: Path):
    # write review config
    (tmp_path/"configs").mkdir(exist_ok=True)
    cfg = {"tags": {"no_go_polygon": {}, "min_clearance_km": {"min":1, "max":20, "default":3}}}
    tag_cfg = TagConfig(cfg)
    # feedback jsonl
    data = [
        {"route_id": "r1", "tag": "no_go_polygon", "severity":"med", "geometry": {"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1],[0,0]]]}},
        {"route_id": "r1", "tag": "min_clearance_km", "note": "5"},
    ]
    lines = [json.dumps(x, ensure_ascii=False) for x in data]
    items = validate_feedback_lines(lines, tag_cfg)
    assert len(items) == 2
    dg = digest_items(items)
    assert isinstance(dg, str) and len(dg) == 32


def test_constraints_checker_no_go(tmp_path: Path):
    # 简单正方形 no-go 与一条穿过的路线
    geo_no_go = {"type":"Polygon","coordinates":[[[0,0],[2,0],[2,2],[0,2],[0,0]]]} 
    route = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[-1,1],[3,1]]}, "properties": {}}
        ]
    }
    constraints = Constraints(mask_meta={}, soft_cost_meta={}, no_go_polygons=[geo_no_go], locks=[])
    res = check_route(route, constraints)
    assert isinstance(res, dict)
    assert res.get("stats", {}).get("no_go_polygons") == 1
    assert len(res.get("violations", [])) >= 1

