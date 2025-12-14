from pathlib import Path

import pytest

from arcticroute.core.scenarios import load_all_scenarios


def test_load_default_scenarios_parses_file():
    scenarios = load_all_scenarios()
    assert scenarios, "scenarios.yaml should yield at least one scenario"
    assert "barents_to_chukchi_edl" in scenarios
    assert set(s.grid_mode for s in scenarios.values()).issubset({"demo", "real"})


def test_barents_fields_are_loaded_correctly():
    scenarios = load_all_scenarios()
    scen = scenarios["barents_to_chukchi_edl"]
    assert pytest.approx(69.0) == scen.start_lat
    assert scen.base_profile == "edl_safe"
    assert scen.use_edl is True
    assert scen.grid_mode == "real"
    assert scen.w_ais == pytest.approx(4.0)


def test_grid_modes_are_limited_to_allowed_values(tmp_path: Path):
    bad_yaml = """scenarios:
  bad_case:
    title: oops
    description: invalid grid mode
    start_lat: 0
    start_lon: 0
    end_lat: 1
    end_lon: 1
    ym: "202412"
    grid_mode: "invalid"
    base_profile: "efficient"
    vessel: "handy"
    w_ice: 1.0
    w_wave: 1.0
    w_ais: 1.0
    use_edl: false
    use_edl_uncertainty: false
"""
    yaml_path = tmp_path / "scenarios.yaml"
    yaml_path.write_text(bad_yaml, encoding="utf-8")

    with pytest.raises(ValueError):
        load_all_scenarios(yaml_path)
