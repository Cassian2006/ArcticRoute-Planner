from pathlib import Path

from arcticroute.io.data_discovery import discover_cmems_layers


def test_cmems_status_paths_with_newenv_and_cache(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    newenv_dir = tmp_path / "data_processed" / "newenv"
    cache_dir = tmp_path / "data" / "cmems_cache"
    newenv_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)

    (newenv_dir / "ice_copernicus_sic.nc").write_text("sic")
    (newenv_dir / "wave_swh.nc").write_text("swh")
    (newenv_dir / "ice_thickness.nc").write_text("sit")
    (newenv_dir / "ice_drift.nc").write_text("drift")

    layers = discover_cmems_layers(
        newenv_dirs=[str(newenv_dir)],
        cache_dirs=[str(cache_dir)],
    )

    assert layers["sic"].found
    assert layers["sic"].source == "newenv"
    assert layers["sic"].path is not None
    assert layers["sic"].size_mb is not None

    assert layers["swh"].found
    assert layers["swh"].source == "newenv"

    assert layers["sit"].found
    assert layers["sit"].source == "newenv"

    assert layers["drift"].found
    assert layers["drift"].source == "newenv"
