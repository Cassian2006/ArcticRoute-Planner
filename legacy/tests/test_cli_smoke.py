from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.e2e_stub
def test_cli_smoke_fallback(tmp_path: Path, sample_paths: dict[str, Path]) -> None:
    for key in ("sat_demo", "ais_demo", "coastline_stub"):
        assert sample_paths[key].exists()

    project_root = sample_paths["sat_demo"].resolve().parents[2]
    output_dir = tmp_path / "cli"
    output_dir.mkdir(parents=True, exist_ok=True)

    tag = "smoke"
    cmd = [
        sys.executable,
        "-m",
        "api.cli",
        "plan",
        "--cfg",
        "ArcticRoute/config/runtime.yaml",
        "--tag",
        tag,
        "--start",
        "72,-160",
        "--goal",
        "74,-150",
        "--output-dir",
        str(output_dir),
    ]

    result = subprocess.run(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"CLI fallback failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    geojson_path = output_dir / f"route_{tag}.geojson"
    assert geojson_path.exists()

    payload = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert payload["features"], "GeoJSON missing features"
    feature = payload["features"][0]
    props = feature["properties"]

    assert props.get("fallback") is True
    waypoints = props.get("waypoints")
    assert isinstance(waypoints, list) and len(waypoints) >= 2
    assert all(isinstance(pt, list) and len(pt) == 2 for pt in waypoints)

    eta_hours = props.get("eta_hours")
    assert isinstance(eta_hours, (int, float))
    assert eta_hours >= 0

    cost = props.get("cost")
    assert isinstance(cost, (int, float))
    assert cost >= 0

    # Smoke-check run report alignment
    report_path = output_dir / f"run_report_{tag}.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("fallback") is True
    assert report.get("waypoints") == waypoints
