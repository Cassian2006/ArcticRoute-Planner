from __future__ import annotations

import json
import sys
from pathlib import Path

from arcticroute.io.static_assets import static_assets_summary

REQUIRED_IDS = [
    "bathymetry_ibcao_v4_200m_nc",
    "ports_world_port_index_geojson",
    "corridors_shipping_hydrography_geojson",
    "rules_pub150_pdf",
]

OPTIONAL_IDS = [
    "bathymetry_ibcao_v5_1_400m_tif",
]


def main() -> int:
    summary = static_assets_summary(required_ids=REQUIRED_IDS, optional_ids=OPTIONAL_IDS)

    Path("reports").mkdir(exist_ok=True)
    out_path = Path("reports/static_assets_doctor.json")
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[doctor] report -> {out_path}")

    missing_required = summary.get("missing_required", [])
    missing_optional = summary.get("missing_optional", [])
    unknown_assets = summary.get("unknown_assets", [])

    if unknown_assets:
        print("[doctor] warning: unknown assets in manifest:", ", ".join(unknown_assets))

    if missing_optional:
        print("[doctor] missing optional:", ", ".join(missing_optional))

    if missing_required:
        print("[doctor] missing required:", ", ".join(missing_required))
        return 2

    print("[doctor] all required static assets present")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
