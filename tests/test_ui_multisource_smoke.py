from __future__ import annotations

from pathlib import Path

from arcticroute.core.eco.vessel_profiles import get_profile_catalog
from arcticroute.ui import planner_minimal
from arcticroute.ui.data_discovery import (
    build_search_dirs,
    discover_ais_density,
    discover_newenv_cmems,
    discover_static_assets,
)


def test_import_planner_minimal() -> None:
    assert planner_minimal is not None


def test_build_search_dirs_returns_list() -> None:
    dirs = build_search_dirs()
    assert isinstance(dirs, list)


def test_profile_catalog_expanded() -> None:
    catalog = get_profile_catalog()
    assert isinstance(catalog, dict)
    assert len(catalog) > 3
    for key in ["handy", "panamax", "ice_class"]:
        assert key in catalog


def test_discovery_empty_dirs(tmp_path: Path) -> None:
    df, meta = discover_ais_density([tmp_path], grid=None)
    assert hasattr(df, "empty")
    assert df.empty
    assert isinstance(meta, dict)

    cmems_meta = discover_newenv_cmems(tmp_path)
    assert isinstance(cmems_meta, dict)

    static_meta = discover_static_assets(tmp_path / "manifest.json")
    assert isinstance(static_meta, dict)
