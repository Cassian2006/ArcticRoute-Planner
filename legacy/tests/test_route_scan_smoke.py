from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.e2e_stub
def test_route_scan_and_report_build(tmp_path: Path):
    # Use provided minimal scenario in configs/scenarios.yaml
    cmd_scan = [
        sys.executable,
        "-m",
        "api.cli",
        "route.scan",
        "--scenario",
        "nsr_wbound_smoke",
        "--ym",
        "202412",
        "--risk-source",
        "fused",
        "--grid",
        "configs/scenarios.yaml",
        "--export",
        "3",
        "--out",
        "ArcticRoute/reports/d_stage/phaseG/",
    ]
    result = subprocess.run(cmd_scan, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert result.returncode == 0, f"route.scan failed: {result.stderr}"
    payload = json.loads(result.stdout)
    front = Path(payload["front"]) if isinstance(payload, dict) and payload.get("front") else None
    assert front and front.exists(), "pareto_front json not found"

    cmd_report = [
        sys.executable,
        "-m",
        "api.cli",
        "report.build",
        "--ym",
        "202412",
        "--include",
        "pareto",
        "--scenario",
        "nsr_wbound_smoke",
    ]
    result2 = subprocess.run(cmd_report, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert result2.returncode == 0, f"report.build failed: {result2.stderr}"
    payload2 = json.loads(result2.stdout)
    html = Path(payload2.get("html"))
    assert html.exists(), "pareto html not found"









