from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest


@pytest.fixture(scope="session")
def sample_paths() -> Dict[str, Path]:
    """Provide stable handles to bundled synthetic sample data."""
    project_root = Path(__file__).resolve().parents[1]
    samples_dir = project_root / "data" / "samples"
    mapping: Dict[str, Path] = {
        "sat_demo": samples_dir / "sat_demo.tif",
        "ais_demo": samples_dir / "ais_demo.geojson",
        "coastline_stub": samples_dir / "coastline_stub.geojson",
    }
    missing = [name for name, candidate in mapping.items() if not candidate.exists()]
    if missing:
        formatted = ", ".join(missing)
        raise FileNotFoundError(
            f"Sample data missing for keys: {formatted}. "
            "Run `python scripts/gen_minidata.py` to create the minimal dataset bundle."
        )
    return mapping
