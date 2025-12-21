from __future__ import annotations

import json

from arcticroute.io.geojson_light import read_geojson_lines, read_geojson_points


def test_read_geojson_points(tmp_path):
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"name": "Port A"},
                "geometry": {"type": "Point", "coordinates": [10.0, 20.0]},
            },
            {
                "type": "Feature",
                "properties": {"name": "Port B"},
                "geometry": {"type": "Point", "coordinates": [30.0, 40.0]},
            },
        ],
    }
    path = tmp_path / "ports.geojson"
    path.write_text(json.dumps(data), encoding="utf-8")

    points = read_geojson_points(path)
    assert len(points) == 2
    assert points[0]["name"] == "Port A"
    assert points[0]["lon"] == 10.0
    assert points[0]["lat"] == 20.0


def test_read_geojson_lines(tmp_path):
    data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
                },
            },
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": [
                        [[10.0, 10.0], [11.0, 11.0]],
                        [[20.0, 20.0], [21.0, 21.0]],
                    ],
                },
            },
        ],
    }
    path = tmp_path / "corridors.geojson"
    path.write_text(json.dumps(data), encoding="utf-8")

    lines = read_geojson_lines(path)
    assert len(lines) == 3
    assert len(lines[0]) == 3
    assert lines[0][0] == (0.0, 0.0)
