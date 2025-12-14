#!/usr/bin/env python3
"""
扫描并检查所有可用的 AIS 密度候选文件。

输出：
- 候选文件列表
- 每个文件的形状、变量名、坐标名
- 是否与当前 grid 兼容
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from arcticroute.core.cost import discover_ais_density_candidates, list_available_ais_density_files
from arcticroute.core.grid import load_grid_with_landmask, make_demo_grid


def inspect_ais_density_file(path: Path) -> dict:
    """
    检查单个 AIS 密度文件。
    
    Returns:
        {
            "path": str,
            "exists": bool,
            "shape": tuple or None,
            "variables": list,
            "coords": list,
            "error": str or None,
        }
    """
    result = {
        "path": str(path),
        "exists": path.exists(),
        "shape": None,
        "variables": [],
        "coords": [],
        "error": None,
    }
    
    if not path.exists():
        result["error"] = "File not found"
        return result
    
    try:
        import xarray as xr
        
        ds = xr.open_dataset(path, decode_times=False)
        
        # 获取变量
        result["variables"] = list(ds.data_vars.keys())
        result["coords"] = list(ds.coords.keys())
        
        # 获取第一个数据变量的形状
        if ds.data_vars:
            first_var = list(ds.data_vars.values())[0]
            result["shape"] = tuple(first_var.shape)
        
        ds.close()
    except Exception as e:
        result["error"] = str(e)
    
    return result


def main():
    """主函数。"""
    print("=" * 80)
    print("AIS 密度候选文件扫描")
    print("=" * 80)
    
    # 1) 使用 discover_ais_density_candidates() 扫描
    print("\n[1] 使用 discover_ais_density_candidates() 扫描：")
    candidates = discover_ais_density_candidates()
    if candidates:
        for cand in candidates:
            path = cand["path"]
            label = cand["label"]
            print(f"  - {label}: {path}")
            info = inspect_ais_density_file(Path(path))
            if info["error"]:
                print(f"    ERROR: {info['error']}")
            else:
                print(f"    Shape: {info['shape']}")
                print(f"    Variables: {info['variables']}")
                print(f"    Coords: {info['coords']}")
    else:
        print("  (no candidates found)")
    
    # 2) 使用 list_available_ais_density_files() 扫描
    print("\n[2] 使用 list_available_ais_density_files() 扫描：")
    available = list_available_ais_density_files()
    if available:
        for label, path in available.items():
            print(f"  - {label}: {path}")
            info = inspect_ais_density_file(path)
            if info["error"]:
                print(f"    ERROR: {info['error']}")
            else:
                print(f"    Shape: {info['shape']}")
                print(f"    Variables: {info['variables']}")
                print(f"    Coords: {info['coords']}")
    else:
        print("  (no files found)")
    
    # 3) 加载当前 grid 并检查兼容性
    print("\n[3] 当前 Grid 信息：")
    grid, land_mask, meta = load_grid_with_landmask(prefer_real=True)
    print(f"  Grid shape: {grid.shape()}")
    print(f"  Grid source: {meta.get('source')}")
    print(f"  Land mask shape: {land_mask.shape}")
    
    # 4) 检查与 demo grid 的兼容性
    print("\n[4] Demo Grid 信息：")
    demo_grid, demo_mask = make_demo_grid()
    print(f"  Demo grid shape: {demo_grid.shape()}")
    
    # 5) 总结
    print("\n" + "=" * 80)
    print("总结：")
    print(f"  Real grid shape: {grid.shape()}")
    print(f"  Demo grid shape: {demo_grid.shape()}")
    print(f"  Available AIS density files: {len(available)}")
    if available:
        for label, path in available.items():
            info = inspect_ais_density_file(path)
            if info["shape"]:
                match_real = info["shape"] == grid.shape()
                match_demo = info["shape"] == demo_grid.shape()
                status = []
                if match_real:
                    status.append("✓ matches real grid")
                if match_demo:
                    status.append("✓ matches demo grid")
                if not status:
                    status.append("✗ no match")
                print(f"  - {label}: {info['shape']} {', '.join(status)}")
    print("=" * 80)


if __name__ == "__main__":
    main()




