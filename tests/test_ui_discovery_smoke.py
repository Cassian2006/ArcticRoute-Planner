from __future__ import annotations

from arcticroute.ui import data_discovery


def test_scan_all_structure():
    snapshot = data_discovery.scan_all()
    assert "roots_used" in snapshot
    assert "env" in snapshot
    assert "items" in snapshot

    summary = data_discovery.summarize_discovery(snapshot)
    for key in ["sic", "swh", "sit", "drift", "bathymetry", "ais"]:
        assert key in summary
        info = summary[key]
        assert "found_paths" in snapshot["items"].get(f"cmems_nc_{key}", {}) or key in ["bathymetry", "ais"]
        assert "found" in info
        assert "searched_paths" in info



