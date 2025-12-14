"""
测试 load_real_env_for_grid 函数。
"""

from pathlib import Path
import numpy as np
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.env_real import load_real_env_for_grid

# 创建一个 demo grid
grid, _ = make_demo_grid()

print(f"Grid shape: {grid.shape()}")
print(f"Grid lat range: [{grid.lat2d.min():.2f}, {grid.lat2d.max():.2f}]")
print(f"Grid lon range: [{grid.lon2d.min():.2f}, {grid.lon2d.max():.2f}]")

# 尝试加载真实环境数据
print("\n" + "=" * 80)
print("Testing load_real_env_for_grid with ym='202412'")
print("=" * 80)

env = load_real_env_for_grid(grid, ym="202412")

if env is not None:
    print(f"\n✓ Successfully loaded environment data")
    print(f"  SIC: {env.sic is not None} (shape: {env.sic.shape if env.sic is not None else 'N/A'})")
    print(f"  Wave: {env.wave_swh is not None} (shape: {env.wave_swh.shape if env.wave_swh is not None else 'N/A'})")
    print(f"  Ice Thickness: {env.ice_thickness_m is not None} (shape: {env.ice_thickness_m.shape if env.ice_thickness_m is not None else 'N/A'})")
else:
    print(f"\n✗ Failed to load environment data")

