from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.eco.fuel import fuel_per_nm_map, eco_cost_norm  # REUSE
from ArcticRoute.core.eco.route_eval import eval_route_eco  # REUSE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "reports", "d_stage", "phaseM")
ROUTES_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "data_processed", "routes")
CFG_ECO = os.path.join(REPO_ROOT, "ArcticRoute", "config", "eco.yaml")

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore


def _load_cfg() -> Dict[str, Any]:
    if yaml is None or not os.path.exists(CFG_ECO):
        return {}
    try:
        with open(CFG_ECO, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_route_geojson(path: str) -> List[Tuple[float, float]]:
    data = json.loads(open(path, "r", encoding="utf-8").read())
    feat = (data.get("features") or [{}])[0]
    coords = (feat.get("geometry") or {}).get("coordinates") or []
    return [(float(x), float(y)) for x, y in coords]


def build_eco_summary(ym: str, scenario: str, vessel_class: str = "cargo_iceclass") -> Dict[str, Any]:
    os.makedirs(REPORT_DIR, exist_ok=True)
    cfg = _load_cfg()
    ef = float(((cfg.get("eco") or {}).get("ef_co2_t_per_t_fuel", 3.114)))
    # 查找一条代表路线（优先 balanced）
    cand = [
        os.path.join(ROUTES_DIR, f"route_{ym}_{scenario}_balanced.geojson"),
        os.path.join(ROUTES_DIR, f"route_{ym}_{scenario}_efficient.geojson"),
        os.path.join(ROUTES_DIR, f"route_{ym}_{scenario}_safe.geojson"),
    ]
    route_path = next((p for p in cand if os.path.exists(p)), None)
    if route_path is None:
        # 尝试默认输出
        route_path = os.path.join(REPO_ROOT, "outputs", "route.geojson")
    if not os.path.exists(route_path):
        raise FileNotFoundError("代表路线缺失，先运行 route.scan 或提供 route.geojson")
    way = _load_route_geojson(route_path)
    eco_da, meta = fuel_per_nm_map(ym, vessel_class=vessel_class, alpha_ice=None, alpha_wave=None)
    totals = eval_route_eco(way, eco_da, ef)

    # JSON
    out_json = os.path.join(REPORT_DIR, f"eco_summary_{ym}_{scenario}.json")
    payload = {"ym": ym, "scenario": scenario, "vessel_class": vessel_class, **meta, **totals}
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(out_json + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(out_json)}, f, ensure_ascii=False, indent=2)

    # HTML（最小实现：表格 + 前10段条形）
    out_html = os.path.join(REPORT_DIR, f"eco_summary_{ym}_{scenario}.html")
    rows = payload["per_segment"][: min(10, len(payload["per_segment"]))]
    bars = "\n".join(
        f"<div style='display:flex;gap:6px'><span>#{int(r['seg'])} ({r['d_nm']:.2f}nm)</span>"
        f"<div style='background:#2b8; height:12px; width:{max(1,int(r['fuel_t']*100))}px'></div>"
        f"<div style='background:#8b2; height:12px; width:{max(1,int(r['co2_t']*40))}px'></div></div>" for r in rows
    )
    html = [
        "<html><head><meta charset='utf-8'><title>Eco Summary</title></head><body>",
        f"<h1>Eco Summary {ym} / {scenario}</h1>",
        f"<p>Vessel class: {vessel_class}; Fuel total: {payload['fuel_total_t']:.3f} t; CO₂ total: {payload['co2_total_t']:.3f} t</p>",
        "<h2>Top segments preview</h2>",
        bars,
        "</body></html>",
    ]
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    with open(out_html + ".meta.json", "w", encoding="utf-8") as f:
        json.dump({"logical_id": os.path.basename(out_html)}, f, ensure_ascii=False, indent=2)

    return {"json": out_json, "html": out_html, **payload}


__all__ = ["build_eco_summary"]

