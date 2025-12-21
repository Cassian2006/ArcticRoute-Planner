from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_geojson(path: str | Path) -> dict[str, Any]:
    path_obj = Path(path)
    return json.loads(path_obj.read_text(encoding="utf-8"))


def _iter_features(obj: dict[str, Any]) -> list[dict[str, Any]]:
    if obj.get("type") == "FeatureCollection":
        return [f for f in obj.get("features", []) if isinstance(f, dict)]
    if obj.get("type") == "Feature":
        return [obj]
    return []


def read_geojson_points(path: str | Path) -> list[dict[str, Any]]:
    obj = load_geojson(path)
    points: list[dict[str, Any]] = []
    for feature in _iter_features(obj):
        geom = feature.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if len(coords) < 2:
            continue
        props = feature.get("properties") or {}
        points.append(
            {
                "name": props.get("name") or props.get("port_name") or "",
                "lon": float(coords[0]),
                "lat": float(coords[1]),
            }
        )
    return points


def read_geojson_lines(path: str | Path) -> list[list[tuple[float, float]]]:
    obj = load_geojson(path)
    lines: list[list[tuple[float, float]]] = []
    for feature in _iter_features(obj):
        geom = feature.get("geometry") or {}
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "LineString":
            line = []
            for coord in coords:
                if len(coord) < 2:
                    continue
                line.append((float(coord[0]), float(coord[1])))
            if line:
                lines.append(line)
        elif gtype == "MultiLineString":
            for part in coords:
                line = []
                for coord in part:
                    if len(coord) < 2:
                        continue
                    line.append((float(coord[0]), float(coord[1])))
                if line:
                    lines.append(line)
    return lines
