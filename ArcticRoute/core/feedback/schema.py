from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

ALLOWED_TAGS = {
    "avoid_ice_edge",
    "avoid_shallow",
    "avoid_traffic_hotspot",
    "avoid_mpa",
    "lock_corridor",
    "no_go_polygon",
    "min_clearance_km",
    "prefer_prior",
    "prefer_smoother",
    "comment",
}

ALLOWED_SEVERITY = {"low", "med", "high"}


@dataclass
class FeedbackItem:
    route_id: str
    tag: str
    severity: Optional[str] = None
    note: Optional[str] = None
    segment_idx: Optional[int] = None
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON Geometry
    value: Optional[float] = None  # for min_clearance_km


def _is_geojson_geometry(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    t = obj.get("type")
    if t in {"Point", "LineString", "Polygon", "MultiPolygon", "MultiLineString"}:
            return True
    return False


def validate_item(obj: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    if not isinstance(obj, dict):
        return False, "not a dict"
    if not isinstance(obj.get("route_id"), str):
        return False, "route_id missing or not str"
    tag = obj.get("tag")
    if tag not in ALLOWED_TAGS:
        return False, f"tag {tag!r} not allowed"
    sev = obj.get("severity")
    if sev is not None and sev not in ALLOWED_SEVERITY:
        return False, f"severity {sev!r} invalid"
    if obj.get("geometry") is not None and not _is_geojson_geometry(obj.get("geometry")):
        return False, "geometry must be GeoJSON geometry"
    if tag == "min_clearance_km":
        try:
            v = float(obj.get("value"))
            if v <= 0:
                return False, "min_clearance_km must be >0"
        except Exception:
            return False, "min_clearance_km requires numeric value"
    if obj.get("segment_idx") is not None:
        try:
            int(obj.get("segment_idx"))
        except Exception:
            return False, "segment_idx must be int"
    return True, None


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            ok, err = validate_item(obj)
            if not ok:
                raise ValueError(f"invalid feedback line: {err}: {obj}")
            out.append(obj)
    return out


def dedup(items: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    outs: List[Dict[str, Any]] = []
    for it in items:
        key = (
            it.get("route_id"),
            it.get("tag"),
            it.get("severity"),
            int(it.get("segment_idx")) if it.get("segment_idx") is not None else None,
            json.dumps(it.get("geometry"), sort_keys=True, ensure_ascii=False) if it.get("geometry") else None,
            it.get("value"),
        )
        if key in seen:
            continue
        seen.add(key)
        outs.append(it)
    return outs


def build_digest(items: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    tags: Dict[str, int] = {}
    for it in items:
        t = str(it.get("tag"))
        tags[t] = tags.get(t, 0) + 1
    return {"count": len(list(items)), "by_tag": tags}


__all__ = ["FeedbackItem", "validate_item", "load_jsonl", "dedup", "build_digest"]
