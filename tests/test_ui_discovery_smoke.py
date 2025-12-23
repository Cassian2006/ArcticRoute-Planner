from __future__ import annotations

from arcticroute.ui import data_discovery


def test_scan_all_structure():
    snapshot = data_discovery.scan_all()
    assert "roots_used" in snapshot
    assert "env" in snapshot
    assert "hits" in snapshot

    hits = snapshot["hits"]
    for key in [
        "ais_density_nc",
        "cmems_nc_sic",
        "cmems_nc_swh",
        "cmems_nc_sit",
        "cmems_nc_drift",
        "static_assets",
    ]:
        assert key in hits
        info = hits[key]
        assert "count" in info
        assert "examples" in info
        assert "roots_used" in info


