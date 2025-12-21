import yaml
from pathlib import Path

def _cfg():
    p = Path("arcticroute/config/polar_rules.yaml")
    return yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore")) or {}

def test_phase14_polaris_thresholds_pinned():
    cfg = _cfg()
    pol = cfg.get("polaris") or {}
    thr = pol.get("rio_thresholds") or {}
    assert thr.get("normal_min") == 0
    assert thr.get("elevated_min") == -10
    assert thr.get("special_lt") == -10

def test_phase14_polaris_speed_limits_pinned():
    cfg = _cfg()
    pol = cfg.get("polaris") or {}
    sp = pol.get("elevated_speed_limits_knots") or {}
    assert sp.get("PC1") == 11
    assert sp.get("PC2") == 8
    assert sp.get("PC3_PC5") == 5
    assert sp.get("BELOW_PC5") == 3

def test_phase14_polar_code_defs_present():
    cfg = _cfg()
    pc = cfg.get("polar_code_definitions") or {}
    assert pc.get("open_water_max_concentration_fraction") == 0.1
    assert pc.get("thin_first_year_ice_thickness_m") == [0.3, 0.7]
    assert pc.get("medium_first_year_ice_thickness_m") == [0.7, 1.2]
    assert pc.get("first_year_ice_thickness_m") == [0.3, 2.0]
