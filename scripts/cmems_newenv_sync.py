"""
CMEMS 新环境同步脚本。

根据下载的数据生成可用的环境数据索引。
"""

from __future__ import annotations

import json
from pathlib import Path


def scan_cmems_data() -> dict:
    """
    扫描 CMEMS 数据目录，生成环境索引。
    
    Returns:
        环境数据索引
    """
    index = {
        "nextsim": [],
        "l4": [],
    }
    
    # 扫描 nextsim 数据
    nextsim_dir = Path("data_real/cmems_nextsim")
    if nextsim_dir.exists():
        for nc_file in nextsim_dir.glob("*.nc"):
            index["nextsim"].append({
                "path": str(nc_file),
                "size": nc_file.stat().st_size,
                "mtime": nc_file.stat().st_mtime,
            })
    
    # 扫描 L4 数据
    l4_dir = Path("data_real/cmems_l4")
    if l4_dir.exists():
        for nc_file in l4_dir.glob("*.nc"):
            index["l4"].append({
                "path": str(nc_file),
                "size": nc_file.stat().st_size,
                "mtime": nc_file.stat().st_mtime,
            })
    
    return index


def main() -> None:
    """主函数。"""
    print("=" * 60)
    print("CMEMS New Environment Sync")
    print("=" * 60)
    
    # 扫描数据
    print("\nScanning CMEMS data directories...")
    index = scan_cmems_data()
    
    print(f"\nFound:")
    print(f"  nextsim files: {len(index['nextsim'])}")
    print(f"  L4 files: {len(index['l4'])}")
    
    # 保存索引
    output_path = Path("reports/cmems_newenv_index.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    
    print(f"\n[OK] Environment index saved to: {output_path}")
    
    # 显示详情
    if index["nextsim"]:
        print("\nnextsim files:")
        for item in index["nextsim"]:
            print(f"  - {item['path']} ({item['size']} bytes)")
    
    if index["l4"]:
        print("\nL4 files:")
        for item in index["l4"]:
            print(f"  - {item['path']} ({item['size']} bytes)")


if __name__ == "__main__":
    main()

