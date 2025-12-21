from __future__ import annotations

from pathlib import Path
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "arcticroute" / "config" / "polar_rules.yaml"
DOC = ROOT / "docs" / "ICE_RULES_SOURCES.md"
TEST = ROOT / "tests" / "test_phase14_authoritative_rules.py"
DUMP = ROOT / "scripts" / "phase14_dump_authoritative_rules.py"

def safe_read_yaml(p: Path) -> dict:
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8", errors="ignore")) or {}

def safe_write_yaml(p: Path, data: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True) + "\n", encoding="utf-8")

def upsert_authoritative_rules(cfg: dict) -> dict:
    # --- POLARIS (MSC.1/Circ.1519) ---
    polaris = cfg.get("polaris") or {}
    polaris.setdefault("enabled", True)
    polaris.setdefault("use_decayed_table", False)
    polaris.setdefault("hard_block_level", "special")

    # 权威：RIO 分档（Table 1.1）
    thr = polaris.get("rio_thresholds") or {}
    thr.setdefault("normal_min", 0)     # normal: RIO >= 0
    thr.setdefault("elevated_min", -10) # elevated: -10 <= RIO < 0
    thr.setdefault("special_lt", -10)   # special: RIO < -10
    polaris["rio_thresholds"] = thr

    # 权威语义：below PC7 / no class 在 elevated 风险下按 special consideration 对待
    policy = polaris.get("policy_for_below_pc7_or_no_class") or {}
    policy.setdefault("treat_elevated_as_special", True)
    polaris["policy_for_below_pc7_or_no_class"] = policy

    # 权威：Elevated 风险下建议限速（Table 1.2）
    sp = polaris.get("elevated_speed_limits_knots") or {}
    sp.setdefault("PC1", 11)
    sp.setdefault("PC2", 8)
    sp.setdefault("PC3_PC5", 5)
    sp.setdefault("BELOW_PC5", 3)
    polaris["elevated_speed_limits_knots"] = sp

    # 保留你们现有的 elevated_penalty/expose_speed_limit 等
    polaris.setdefault("elevated_penalty", {"enabled": True, "scale": 1.0})
    polaris.setdefault("expose_speed_limit", True)

    cfg["polaris"] = polaris

    # --- Polar Code definitions (MSC.385(94)) ---
    # 这些主要用于"语义与解释"，不强行作为通用 hard-block
    pc = cfg.get("polar_code_definitions") or {}
    pc.setdefault("open_water_max_concentration_fraction", 0.1)  # < 1/10

    # FYI thickness ranges（用于解释/文档，不替代船舶 PWOM）
    pc.setdefault("thin_first_year_ice_thickness_m", [0.3, 0.7])
    pc.setdefault("medium_first_year_ice_thickness_m", [0.7, 1.2])
    pc.setdefault("first_year_ice_thickness_m", [0.3, 2.0])

    cfg["polar_code_definitions"] = pc
    return cfg

def upsert_sources_doc() -> None:
    DOC.parent.mkdir(parents=True, exist_ok=True)
    cur = DOC.read_text(encoding="utf-8", errors="ignore") if DOC.exists() else ""

    block = """
## Phase 14 – Authoritative thresholds pinned (POLARIS + Polar Code)

### POLARIS (IMO MSC.1/Circ.1519)
- RIO → Operation level thresholds (Table 1.1)
  - normal: RIO >= 0
  - elevated: -10 <= RIO < 0
  - special consideration: RIO < -10
  - below PC7 / no ice class: elevated risk is treated as special consideration
- Speed limitations in elevated risk (Table 1.2)
  - PC1: 11 kn, PC2: 8 kn, PC3–PC5: 5 kn, below PC5: 3 kn
- Decayed ice (Table 1.4) usage
  - default uses standard table (Table 1.3); only enable decayed table when confirmed "decayed ice" per operational assessment

### Polar Code (IMO MSC.385(94)) – semantics for documentation
- Open water: sea ice concentration < 1/10 (0.1)
- FYI thickness ranges used for interpretation:
  - thin FYI: 0.3–0.7 m
  - medium FYI: 0.7–1.2 m
  - FYI general: 0.3–2.0 m

### Notes (ship-specific; keep configurable, do not claim universal)
- SWH / wave thresholds and speed reduction rules are ship/PWOM dependent
- fuel model parameters are ship-specific
""".strip()

    if block not in cur:
        cur = (cur.rstrip() + "\n\n" + block + "\n")
        DOC.write_text(cur, encoding="utf-8")

def write_regression_test() -> None:
    TEST.parent.mkdir(parents=True, exist_ok=True)
    TEST.write_text(
        r'''
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
'''.lstrip(),
        encoding="utf-8",
    )

def write_dump_script() -> None:
    DUMP.parent.mkdir(parents=True, exist_ok=True)
    DUMP.write_text(
        r'''
import json
from pathlib import Path
import yaml

def main():
    cfgp = Path("arcticroute/config/polar_rules.yaml")
    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8", errors="ignore")) or {}
    out = {
        "polaris": cfg.get("polaris", {}),
        "polar_code_definitions": cfg.get("polar_code_definitions", {}),
        "scope_notes": {
            "authoritative": [
                "POLARIS RIO thresholds + elevated speed limits",
                "Polar Code open water + FYI thickness ranges (semantics)",
            ],
            "ship_specific": [
                "wave thresholds (SWH) should come from PWOM/operational assessment",
                "fuel/eco model parameters are ship-specific",
            ],
        },
    }
    Path("reports").mkdir(exist_ok=True)
    outp = Path("reports/phase14_authoritative_rules.json")
    outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", outp)

if __name__ == "__main__":
    main()
'''.lstrip(),
        encoding="utf-8",
    )

def main():
    cfg = safe_read_yaml(CFG)
    cfg = upsert_authoritative_rules(cfg)
    safe_write_yaml(CFG, cfg)
    upsert_sources_doc()
    write_regression_test()
    write_dump_script()
    print("OK: patched yaml/docs/tests/scripts")

if __name__ == "__main__":
    main()