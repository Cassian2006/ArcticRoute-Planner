import numpy as np
from pathlib import Path

from arcticroute.core.ais_ingest import build_ais_density_da_for_demo_grid


def test_build_ais_density_da_for_demo_grid(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    sample = Path(__file__).parent / "data" / "ais_sample.csv"
    (raw_dir / "a.csv").write_text(sample.read_text(), encoding="utf-8")
    (raw_dir / "b.csv").write_text(sample.read_text(), encoding="utf-8")

    grid_lat, grid_lon = np.meshgrid(np.linspace(74.5, 76.5, 4), np.linspace(20.0, 22.0, 4))
    da = build_ais_density_da_for_demo_grid(raw_dir, grid_lat, grid_lon)

    assert da.dims == ("y", "x")
    assert da.shape == grid_lat.shape
    assert np.count_nonzero(da.values > 0) > 0
    assert da.attrs.get("source") == "real_ais"
