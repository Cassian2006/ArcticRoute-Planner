from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

from ArcticRoute.core.route.metrics import summarize_route  # REUSE

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
REPORT_DIR = os.path.join(REPO_ROOT, "ArcticRoute", "reports", "d_stage", "phaseN")


def _load_geojson(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _coords_from_geojson(gj: dict) -> List[Tuple[float, float]]:
    try:
        feat = (gj.get("features") or [])[0]
        coords = feat.get("geometry", {}).get("coordinates") or []
        return [(float(lon), float(lat)) for lon, lat in coords]
    except Exception:
        return []


def build_replan_summary(
    *,
    scenario: str,
    ym: str,
    risk_da: Optional["xr.DataArray"],
    route_old_path: Optional[str],
    route_new_path: str,
    trigger: str,
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if xr is None:
        raise RuntimeError("xarray required")
    out_dir2 = out_dir or REPORT_DIR
    os.makedirs(out_dir2, exist_ok=True)
    # 读取路线坐标以汇总指标
    old_coords: List[Tuple[float, float]] = []
    if route_old_path and os.path.exists(route_old_path):
        gj = _load_geojson(route_old_path)
        if gj:
            old_coords = _coords_from_geojson(gj)
    gj_new = _load_geojson(route_new_path)
    new_coords = _coords_from_geojson(gj_new or {})

    prior = None
    interact = None
    metrics_new = summarize_route(new_coords, risk=risk_da, prior_penalty=prior, interact=interact)
    metrics_old = summarize_route(old_coords, risk=risk_da, prior_penalty=prior, interact=interact) if old_coords else None

    # 写 JSON
    run_id = os.path.splitext(os.path.basename(route_new_path))[0]
    json_path = os.path.join(out_dir2, f"replan_summary_{scenario}_{ym}.json")
    payload: Dict[str, Any] = {
        "scenario": scenario,
        "ym": ym,
        "trigger": trigger,
        "route_new": os.path.abspath(route_new_path),
        "route_old": os.path.abspath(route_old_path) if route_old_path else None,
        "metrics_new": metrics_new,
        "metrics_old": metrics_old,
        "delta": None,
    }
    if metrics_old:
        payload["delta"] = {k: (float(metrics_new.get(k, 0.0)) - float(metrics_old.get(k, 0.0))) for k in metrics_new.keys() if k in metrics_old}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 简单 HTML
    html_path = os.path.join(out_dir2, f"replan_summary_{scenario}_{ym}.html")
    try:
        lines = [
            "<html><head><meta charset='utf-8'><title>Replan Summary</title></head><body>",
            f"<h1>Replan Summary · {scenario} · {ym}</h1>",
            f"<p>Trigger: <b>{trigger}</b></p>",
            f"<p>Route new: {os.path.basename(route_new_path)}<br/>Route old: {os.path.basename(route_old_path or '-')}</p>",
            "<h2>Metrics</h2>",
            f"<pre>{json.dumps({'new': metrics_new, 'old': metrics_old, 'delta': payload.get('delta')}, ensure_ascii=False, indent=2)}</pre>",
            "</body></html>",
        ]
        with open(html_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    except Exception:
        html_path = ""

    return {"json": json_path, "html": html_path}


__all__ = ["build_replan_summary"]

