"""
静态资源检查脚本。

检查项目所需的静态资源文件是否存在。
"""

from __future__ import annotations

import json
from pathlib import Path


def check_static_assets():
    """检查静态资源文件。"""
    
    # 定义必需的资源
    required_assets = [
        # 暂时没有必需的静态资源
    ]
    
    # 定义可选的资源
    optional_assets = [
        "data_real/ais/derived/ais_density_2024_demo.nc",
        "data_real/ais/derived/ais_density_2024_real.nc",
        "data_real/bathymetry/ibcao_v4_400m_subset.nc",
        "data_real/landmask/landmask_demo.nc",
        "data_real/grid/grid_demo.nc",
    ]
    
    missing_required = []
    missing_optional = []
    
    # 检查必需资源
    for asset in required_assets:
        asset_path = Path(asset)
        if not asset_path.exists():
            missing_required.append(asset)
    
    # 检查可选资源
    for asset in optional_assets:
        asset_path = Path(asset)
        if not asset_path.exists():
            missing_optional.append(asset)
    
    # 生成报告
    report = {
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "all_ok": len(missing_required) == 0,
    }
    
    # 保存到 reports 目录
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    report_file = reports_dir / "static_assets_doctor.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # 打印结果
    print("=== Static Assets Doctor ===\n")
    print(f"Missing Required: {len(missing_required)}")
    if missing_required:
        for asset in missing_required:
            print(f"  - {asset}")
    else:
        print("  (none)")
    
    print(f"\nMissing Optional: {len(missing_optional)}")
    if missing_optional:
        for asset in missing_optional:
            print(f"  - {asset}")
    else:
        print("  (none)")
    
    print(f"\nAll Required Assets OK: {report['all_ok']}")
    print(f"\nReport saved to: {report_file}")
    
    return report


def main():
    report = check_static_assets()
    
    # 如果有缺失的必需资源，返回非零退出码
    if not report["all_ok"]:
        exit(1)


if __name__ == "__main__":
    main()

