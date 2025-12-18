"""验证 Phase 11 产物齐全。"""
from pathlib import Path

need = [
    "reports/demo_run_phase11/route.geojson",
    "reports/demo_run_phase11/cost_breakdown.json",
    "reports/demo_run_phase11/polaris_diagnostics.csv",
    "reports/demo_run_phase11/summary.txt",
]

missing = [f for f in need if not Path(f).exists()]

if missing:
    print("MISSING FILES:")
    for f in missing:
        print(f"  - {f}")
    raise SystemExit(1)
else:
    print("ALL FILES PRESENT:")
    for f in need:
        p = Path(f)
        print(f"  [OK] {f} ({p.stat().st_size} bytes)")
    print("\n[OK] Phase 11 acceptance check PASSED")
    raise SystemExit(0)

