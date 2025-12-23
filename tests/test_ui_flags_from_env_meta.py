from __future__ import annotations

from arcticroute.ui import data_discovery
from arcticroute.ui import planner_minimal


def test_ui_flags_from_env_meta():
    summary = {
        "sic": {"found": True},
        "swh": {"found": True},
        "sit": {"found": False, "reason": "missing"},
        "drift": {"found": True},
        "bathymetry": {"found": False},
        "ais": {"found": False},
    }

    flags = data_discovery.availability_flags(summary)
    assert flags["sic"] is True
    assert flags["drift"] is True
    assert flags["sit"] is False
    assert flags["bathymetry"] is False

    # summarize_data_sources 应能接收摘要并回填 note
    meta = planner_minimal.summarize_data_sources(
        real_env=None,
        resolved_files=None,
        ais_density=None,
        ais_density_da=None,
        ais_density_path=None,
        discovery_summary=summary,
    )
    assert meta["sit"]["note"] == "missing"
    assert meta["drift"]["found"] is True

