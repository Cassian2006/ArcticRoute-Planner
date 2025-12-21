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
