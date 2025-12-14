"""
测试 load_real_env_for_grid 函数 - 带详细调试。
"""

from pathlib import Path
import os
import numpy as np
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.env_real import REAL_ENV_DIR, DATA_BACKUP_ROOT

# 创建一个 demo grid
grid, _ = make_demo_grid()

print(f"Grid shape: {grid.shape()}")

# 检查路径常量
print(f"\n[PATH CONSTANTS]")
print(f"  DATA_BACKUP_ROOT: {DATA_BACKUP_ROOT}")
print(f"  REAL_ENV_DIR: {REAL_ENV_DIR}")
print(f"  REAL_ENV_DIR exists: {REAL_ENV_DIR.exists()}")

# 检查候选文件
ym = "202412"
print(f"\n[CANDIDATE FILES for ym={ym}]")

candidates = [
    Path(__file__).resolve().parents[0] / "data_real" / ym / f"sic_{ym}.nc",
    REAL_ENV_DIR / f"sic_{ym}.nc",
    REAL_ENV_DIR / "ice_copernicus_sic.nc",
]

for candidate in candidates:
    exists = candidate.exists()
    status = "✓" if exists else "✗"
    print(f"  {status} {candidate}")

# 检查 REAL_ENV_DIR 中的文件
print(f"\n[FILES IN REAL_ENV_DIR]")
if REAL_ENV_DIR.exists():
    for f in sorted(REAL_ENV_DIR.glob("*.nc")):
        print(f"  - {f.name}")
else:
    print(f"  Directory does not exist")

# 现在尝试加载
print(f"\n" + "=" * 80)
print("Testing load_real_env_for_grid")
print("=" * 80)

from arcticroute.core.env_real import load_real_env_for_grid

env = load_real_env_for_grid(grid, ym=ym)

if env is not None:
    print(f"\n✓ Successfully loaded environment data")
    print(f"  SIC: {env.sic is not None}")
    print(f"  Wave: {env.wave_swh is not None}")
    print(f"  Ice Thickness: {env.ice_thickness_m is not None}")
else:
    print(f"\n✗ Failed to load environment data")

