from __future__ import annotations

from pathlib import Path
import os

from ArcticRoute.core.reporting.animate import animate_layers

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_animation_creates_non_empty_file():
    ym = "202412"
    nc = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
    if not nc.exists():
        import pytest
        pytest.skip("risk fused file missing")
    out_dir = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"anim_{ym}_risk.gif"
    res = animate_layers([nc], out, fps=2)
    assert Path(res).exists(), "animation output missing"
    assert os.path.getsize(res) > 0, "animation file is empty"



