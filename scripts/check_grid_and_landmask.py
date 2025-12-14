"""
检查网格与陆地掩码的脚本。

用法: python -m scripts.check_grid_and_landmask

输出：
- 网格来源（real 或 demo）
- 网格形状
- 陆地掩码来源路径
- 陆地/海洋比例
- 网格范围（角落坐标）
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from arcticroute.core.grid import load_real_grid_from_nc, load_grid_with_landmask
from arcticroute.core.landmask import load_real_landmask_from_nc, load_landmask, _scan_landmask_candidates


def main() -> None:
    """
    主函数：加载并显示网格与陆地掩码的基本信息。

    首先尝试加载真实网格和 landmask，如果失败则回退到 demo。
    """
    print("=" * 80)
    print("网格与陆地掩码检查")
    print("=" * 80)
    
    print("\n[1] 扫描可用的 landmask 候选文件：")
    candidates = _scan_landmask_candidates()
    if candidates:
        for path, desc in candidates:
            print(f"  - {desc}: {path}")
    else:
        print("  (未找到候选文件)")
    
    print("\n[2] 加载网格与 landmask...")
    
    # 使用统一的加载函数
    grid, land_mask, meta = load_grid_with_landmask(prefer_real=True)
    
    source = meta.get("source", "unknown")
    data_root = meta.get("data_root", "unknown")
    
    print(f"  Grid source: {source}")
    print(f"  Data root: {data_root}")
    
    # 显示信息
    ny, nx = grid.shape()
    print(f"\n[3] 网格信息：")
    print(f"  Shape: {ny} x {nx}")
    print(f"  Lat range: [{grid.lat2d.min():.3f}, {grid.lat2d.max():.3f}]")
    print(f"  Lon range: [{grid.lon2d.min():.3f}, {grid.lon2d.max():.3f}]")
    
    # 计算陆地比例
    frac_land = float(land_mask.sum()) / land_mask.size if land_mask.size > 0 else 0.0
    frac_ocean = 1.0 - frac_land
    print(f"\n[4] 陆地掩码统计：")
    print(f"  Land fraction: {frac_land:.6f} ({int(land_mask.sum())} cells)")
    print(f"  Ocean fraction: {frac_ocean:.6f} ({int((~land_mask).sum())} cells)")
    
    # 显示角落坐标
    lat0 = grid.lat2d[0, 0]
    lon0 = grid.lon2d[0, 0]
    lat1 = grid.lat2d[-1, -1]
    lon1 = grid.lon2d[-1, -1]
    print(f"\n[5] 网格范围：")
    print(f"  Top-left: ({lat0:.3f}°N, {lon0:.3f}°E)")
    print(f"  Bottom-right: ({lat1:.3f}°N, {lon1:.3f}°E)")
    
    print("\n" + "=" * 80)
    if source == "real":
        print("[OK] Successfully loaded real grid and landmask")
    else:
        print("[WARN] Using demo grid and landmask (real data not available)")
    print("=" * 80)


if __name__ == "__main__":
    main()

