#!/usr/bin/env python
"""
生成 CMEMS describe JSON 文件（用于解析变量）
"""
import subprocess
import json
from pathlib import Path

def main():
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # SIC describe
    print("[*] 生成 SIC describe...")
    try:
        res = subprocess.run(
            [
                "copernicusmarine",
                "describe",
                "--contains",
                "cmems_mod_arc_phy_anfc_nextsim_hm",
                "--return-fields",
                "all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        sic_path = reports_dir / "cmems_sic_describe.json"
        sic_path.write_text(res.stdout, encoding="utf-8")
        print(f"[OK] 已写入 {sic_path}")
    except Exception as e:
        print(f"[ERROR] SIC describe 失败: {e}")
        return 1
    
    # SWH describe
    print("[*] 生成 SWH describe...")
    try:
        res = subprocess.run(
            [
                "copernicusmarine",
                "describe",
                "--contains",
                "dataset-wam-arctic-1hr3km-be",
                "--return-fields",
                "all",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
        swh_path = reports_dir / "cmems_swh_describe.json"
        swh_path.write_text(res.stdout, encoding="utf-8")
        print(f"[OK] 已写入 {swh_path}")
    except Exception as e:
        print(f"[ERROR] SWH describe 失败: {e}")
        return 1
    
    print("[OK] 两个 describe JSON 已生成")
    return 0

if __name__ == "__main__":
    exit(main())

