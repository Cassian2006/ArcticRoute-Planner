from __future__ import annotations
import pandas as pd
from ArcticRoute.core.domain.bucketer import Bucketer


def _cfg():
    return {
        "regions": [
            {"name": "NSR", "bbox": [30.0, 66.0, 180.0, 85.0]},
            {"name": "NWP", "bbox": [-170.0, 65.0, -40.0, 85.0]},
        ],
        "season_rules": {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]},
        "vessel_map": {"cargo": "cargo", "tanker": "tanker", "fishing": "fishing"},
        "default_bucket": "global"
    }


def test_bucketer_basic_region_season_vessel():
    b = Bucketer(_cfg())
    ts = pd.Timestamp("2024-12-15")
    bucket = b.infer_bucket(69.0, 60.0, ts, "cargo")
    assert bucket.startswith("NSR_DJF"), bucket
    assert bucket.endswith("cargo"), bucket


def test_bucketer_fallback_unknown_vessel():
    b = Bucketer(_cfg())
    ts = pd.Timestamp("2024-12-15")
    bucket = b.infer_bucket(69.0, 60.0, ts, "unknown_type")
    assert bucket == "NSR_DJF", bucket


def test_bucketer_fallback_global_when_outside():
    b = Bucketer(_cfg())
    ts = pd.Timestamp("2024-07-01")
    bucket = b.infer_bucket(10.0, 10.0, ts, "cargo")
    assert bucket == "global", bucket

