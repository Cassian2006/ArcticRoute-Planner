"""
Phase 12: 对照实验产物验收脚本

验证每个场景是否产出了必需的文件：
- route.geojson
- cost_breakdown.json
- polaris_diagnostics.csv
- summary.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = [
    "route.geojson",
    "cost_breakdown.json",
    "polaris_diagnostics.csv",
    "summary.txt",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dir", required=True, help="Ablation output root dir, e.g., reports/ablation"
    )
    args = ap.parse_args()

    root = Path(args.dir)
    assert root.exists(), f"Missing dir: {root}"
    summary = root / "summary.csv"
    assert summary.exists(), f"Missing summary.csv: {summary}"

    # scenario subdirs: any folder with run_meta.json
    run_dirs = [p.parent for p in root.rglob("run_meta.json")]
    assert run_dirs, "No run_meta.json found; suite did not run?"
    missing_any = False

    print(f"{'='*60}")
    print(f"Verifying ablation outputs in: {root}")
    print(f"{'='*60}\n")

    # Step 1: 检查必需文件
    for d in sorted(set(run_dirs)):
        miss = [f for f in REQUIRED if not (d / f).exists()]
        if miss:
            missing_any = True
            print(f"[FAIL] {d.name}: missing {miss}")
        else:
            print(f"[OK]   {d.name}")

    # Step 2: 读取 run_meta.json 并检查对照差异
    print(f"\n{'='*60}")
    print("Checking ablation contrasts...")
    print(f"{'='*60}\n")
    
    meta_map = {}
    for d in sorted(set(run_dirs)):
        meta_file = d / "run_meta.json"
        if meta_file.exists():
            import json
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta_map[d.name] = meta
    
    # 检查 A vs B（demo 组）
    if "A_demo_no_polaris" in meta_map and "B_demo_polaris" in meta_map:
        a = meta_map["A_demo_no_polaris"]
        b = meta_map["B_demo_polaris"]
        
        # 检查 polaris_enabled 不同
        if a.get("polaris_enabled_requested") == b.get("polaris_enabled_requested"):
            print(f"[FAIL] A vs B: polaris_enabled should differ")
            missing_any = True
        else:
            print(f"[OK]   A vs B: polaris_enabled differs ({a.get('polaris_enabled_requested')} vs {b.get('polaris_enabled_requested')})")
        
        # 检查 polaris 指标差异
        a_points = a.get("polaris_points", 0)
        b_points = b.get("polaris_points", 0)
        a_special = a.get("polaris_special_fraction", 0)
        b_special = b.get("polaris_special_fraction", 0)
        
        if a_points == b_points and a_special == b_special:
            print(f"[FAIL] A vs B: polaris metrics should differ (points={a_points} vs {b_points}, special={a_special} vs {b_special})")
            missing_any = True
        else:
            print(f"[OK]   A vs B: polaris metrics differ (points={a_points} vs {b_points}, special={a_special:.3f} vs {b_special:.3f})")
    
    # 检查 C vs D（cmems 组）
    if "C_cmems_no_polaris" in meta_map and "D_cmems_polaris" in meta_map:
        c = meta_map["C_cmems_no_polaris"]
        d = meta_map["D_cmems_polaris"]
        
        # 检查 polaris_enabled 不同
        if c.get("polaris_enabled_requested") == d.get("polaris_enabled_requested"):
            print(f"[FAIL] C vs D: polaris_enabled should differ")
            missing_any = True
        else:
            print(f"[OK]   C vs D: polaris_enabled differs ({c.get('polaris_enabled_requested')} vs {d.get('polaris_enabled_requested')})")
        
        # 检查 polaris 指标差异
        c_points = c.get("polaris_points", 0)
        d_points = d.get("polaris_points", 0)
        c_special = c.get("polaris_special_fraction", 0)
        d_special = d.get("polaris_special_fraction", 0)
        
        if c_points == d_points and c_special == d_special:
            print(f"[FAIL] C vs D: polaris metrics should differ (points={c_points} vs {d_points}, special={c_special} vs {d_special})")
            missing_any = True
        else:
            print(f"[OK]   C vs D: polaris metrics differ (points={c_points} vs {d_points}, special={c_special:.3f} vs {d_special:.3f})")
    
    # 检查 demo 组不应探测 CMEMS
    for key in ["A_demo_no_polaris", "B_demo_polaris"]:
        if key in meta_map:
            meta = meta_map[key]
            fallback = meta.get("fallback_reason", "")
            if "nextsim" in fallback.lower() or "l4" in fallback.lower():
                print(f"[FAIL] {key}: demo should not probe CMEMS (fallback_reason={fallback})")
                missing_any = True
            else:
                print(f"[OK]   {key}: no CMEMS probing detected")

    print(f"\n{'='*60}")
    if missing_any:
        print("[FAIL] Some scenarios failed verification")
        print(f"{'='*60}")
        raise SystemExit(1)
    else:
        print("[OK] All scenarios passed verification")
        print(f"{'='*60}")
        raise SystemExit(0)


if __name__ == "__main__":
    main()

