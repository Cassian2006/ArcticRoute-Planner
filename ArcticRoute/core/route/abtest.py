from __future__ import annotations

# A/B 对比（最小实现） # REUSE
from typing import Any, Dict, List, Sequence, Tuple
import json
from pathlib import Path

from ArcticRoute.core.route.metrics import compute_distance_km  # REUSE


def _extract_path_lonlat(route_geojson: Dict[str, Any]) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    feats = route_geojson.get("features") or []
    for ft in feats:
        geom = (ft or {}).get("geometry") or {}
        if geom.get("type") == "LineString":
            for p in (geom.get("coordinates") or []):
                if isinstance(p, (list, tuple)) and len(p) >= 2:
                    pts.append((float(p[0]), float(p[1])))
    return pts


def compare_routes(baseline_path: Path, new_path: Path) -> Dict[str, Any]:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    p0 = _extract_path_lonlat(base)
    p1 = _extract_path_lonlat(new)
    dist0 = compute_distance_km([(lon, lat) for lon, lat in p0]) if p0 else 0.0
    dist1 = compute_distance_km([(lon, lat) for lon, lat in p1]) if p1 else 0.0
    return {
        "baseline_points": len(p0),
        "new_points": len(p1),
        "distance_km_before": dist0,
        "distance_km_after": dist1,
        "delta_km": float(dist1 - dist0),
    }


__all__ = ["compare_routes"]

