#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PolarRoute 集成交付验证脚本

验证所有交付物是否已准备完毕
"""

import json
import sys
from pathlib import Path

# 设置输出编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def check_file(path: str) -> tuple[bool, int]:
    """检查文件是否存在并返回大小"""
    p = Path(path)
    if p.exists():
        return True, p.stat().st_size
    return False, 0

def main():
    print("=" * 70)
    print("PolarRoute 集成 - 交付验证报告")
    print("=" * 70)
    print()

    files = {
        "核心文件": [
            "data_sample/polarroute/vessel_mesh_empty.json",
            "data_sample/polarroute/config_empty.json",
            "data_sample/polarroute/waypoints_example.json",
        ],
        "演示输出": [
            "data_sample/polarroute/vessel_mesh_demo.json",
            "data_sample/polarroute/routes_demo.geojson",
        ],
        "脚本文件": [
            "scripts/integrate_polarroute.py",
            "scripts/demo_polarroute_simple.py",
            "scripts/test_polarroute_integration.py",
        ],
        "文档文件": [
            "POLARROUTE_QUICK_START.md",
            "POLARROUTE_INTEGRATION_GUIDE.md",
            "POLARROUTE_DELIVERY_SUMMARY.md",
            "POLARROUTE_CHECKLIST.md",
            "POLARROUTE_README.md",
        ],
    }

    total_files = 0
    total_size = 0
    all_exist = True

    for category, file_list in files.items():
        print(f"{category}:")
        for f in file_list:
            exists, size = check_file(f)
            if exists:
                total_files += 1
                total_size += size
                print(f"  [OK] {Path(f).name:40} {size:>8} bytes")
            else:
                print(f"  [FAIL] {Path(f).name:40} NOT FOUND")
                all_exist = False
        print()

    print("=" * 70)
    print(f"总计: {total_files} 个文件, {total_size:,} 字节")
    print("=" * 70)
    print()

    if all_exist:
        print("[SUCCESS] All deliverables are ready!")
        print()
        print("快速开始:")
        print("  python scripts/demo_polarroute_simple.py")
        print()
        print("查看文档:")
        print("  - POLARROUTE_README.md (项目总结)")
        print("  - POLARROUTE_QUICK_START.md (5分钟快速开始)")
        print("  - POLARROUTE_INTEGRATION_GUIDE.md (详细指南)")
        print()
        print("验证 vessel_mesh.json 结构:")
        with open("data_sample/polarroute/vessel_mesh_empty.json") as f:
            mesh = json.load(f)
            print(f"  - 版本: {mesh['metadata']['version']}")
            print(f"  - 网格类型: {mesh['grid']['type']}")
            print(f"  - 分辨率: {mesh['grid']['resolution_degrees']}°")
            print(f"  - 环境层: {len(mesh['environmental_layers'])} 个")
            print(f"  - 船舶容器: {len(mesh['vehicles'])} 个")
            print(f"  - 路由容器: {len(mesh['routes'])} 个")
        print()
        print("验证配置文件:")
        with open("data_sample/polarroute/config_empty.json") as f:
            config = json.load(f)
            print(f"  - 算法: {config['routing']['algorithm']}")
            print(f"  - 优化方法: {config['routing']['optimization_method']}")
            print(f"  - 冰级: {config['vessel_defaults']['ice_class']}")
            weights = config['environmental_weights']
            total_weight = sum(weights.values())
            print(f"  - 权重和: {total_weight:.2f}")
        print()
        print("=" * 70)
        print("Status: [COMPLETE] Ready to use")
        print("=" * 70)
        return 0
    else:
        print("[ERROR] Some files are missing!")
        return 1

if __name__ == "__main__":
    exit(main())

