from __future__ import annotations

from pathlib import Path
import json
import numpy as np

from ArcticRoute.core.reporting.calibration import build_month, reliability_curve

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_reliability_bins_monotonic_in_expectation():
    ym = "202412"
    payload = build_month(ym)
    js_path = Path(payload["json"])
    obj = json.loads(js_path.read_text(encoding="utf-8"))
    rel = obj["reliability"]
    rates = np.array([r if (r is not None) else np.nan for r in rel.get("pos_rate", [])], dtype=float)
    # 仅保留非 NaN
    v = rates[np.isfinite(rates)]
    if v.size <= 1:
        import pytest
        pytest.skip("insufficient bins for monotonic check")
    # 允许少量随机波动：检查是否近似单调（允许 3% 下降总幅度）
    diffs = np.diff(v)
    total_drop = float(np.abs(diffs[diffs < 0]).sum())
    assert total_drop <= 0.03 * max(1.0, float(v.max()) - float(v.min())), f"reliability curve not monotonic enough: total_drop={total_drop:.4f}"



