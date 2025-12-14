"""
调试脚本：检查真实环境数据文件的存在性。

用法：
    python -m scripts.debug_real_env_paths --ym 202412
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# ============================================================================
# 路径常量（与 env_real.py 同步）
# ============================================================================

DATA_BACKUP_ROOT = Path(
    os.environ.get(
        "ARCTICROUTE_DATA_BACKUP",
        r"C:\Users\sgddsf\Desktop\ArcticRoute_data_backup",
    )
)

REAL_ENV_DIR = DATA_BACKUP_ROOT / "data_processed" / "newenv"

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_REAL_DIR = PROJECT_ROOT / "data_real"


def check_file_exists(path: Path, description: str = "") -> None:
    """
    检查文件是否存在并打印结果。
    
    Args:
        path: 文件路径
        description: 文件描述
    """
    exists = path.exists()
    status = "✓ FOUND" if exists else "✗ NOT FOUND"
    print(f"  {status}: {description}")
    print(f"         {path}")


def debug_real_env_paths(ym: str = "202412") -> None:
    """
    检查所有真实环境数据文件的存在性。
    
    Args:
        ym: 年月字符串（格式 "YYYYMM"）
    """
    print("\n" + "=" * 100)
    print(f"DEBUG REAL ENVIRONMENT DATA PATHS - YM={ym}")
    print("=" * 100)
    
    # 打印配置信息
    print(f"\n[CONFIG]")
    print(f"  DATA_BACKUP_ROOT: {DATA_BACKUP_ROOT}")
    print(f"  REAL_ENV_DIR: {REAL_ENV_DIR}")
    print(f"  PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"  DATA_REAL_DIR: {DATA_REAL_DIR}")
    
    # 检查根目录是否存在
    print(f"\n[DIRECTORY CHECKS]")
    print(f"  DATA_BACKUP_ROOT exists: {DATA_BACKUP_ROOT.exists()}")
    print(f"  REAL_ENV_DIR exists: {REAL_ENV_DIR.exists()}")
    print(f"  DATA_REAL_DIR exists: {DATA_REAL_DIR.exists()}")
    
    # 检查 REAL_ENV_DIR 中的文件
    print(f"\n[GRID FILES in REAL_ENV_DIR]")
    check_file_exists(
        REAL_ENV_DIR / "env_clean.nc",
        "env_clean.nc (grid file)"
    )
    check_file_exists(
        REAL_ENV_DIR / "grid_spec.nc",
        "grid_spec.nc (grid file)"
    )
    check_file_exists(
        REAL_ENV_DIR / "land_mask_gebco.nc",
        "land_mask_gebco.nc (landmask file)"
    )
    
    # 检查 SIC 文件
    print(f"\n[SIC FILES]")
    check_file_exists(
        REAL_ENV_DIR / f"sic_{ym}.nc",
        f"sic_{ym}.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        REAL_ENV_DIR / "ice_copernicus_sic.nc",
        "ice_copernicus_sic.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        DATA_REAL_DIR / ym / f"sic_{ym}.nc",
        f"sic_{ym}.nc (from DATA_REAL_DIR/{ym})"
    )
    
    # 检查 wave 文件
    print(f"\n[WAVE FILES]")
    check_file_exists(
        REAL_ENV_DIR / f"wave_{ym}.nc",
        f"wave_{ym}.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        REAL_ENV_DIR / "wave_swh.nc",
        "wave_swh.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        DATA_REAL_DIR / ym / f"wave_{ym}.nc",
        f"wave_{ym}.nc (from DATA_REAL_DIR/{ym})"
    )
    
    # 检查 ice_thickness 文件
    print(f"\n[ICE THICKNESS FILES]")
    check_file_exists(
        REAL_ENV_DIR / f"ice_thickness_{ym}.nc",
        f"ice_thickness_{ym}.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        REAL_ENV_DIR / "ice_thickness.nc",
        "ice_thickness.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        DATA_REAL_DIR / ym / f"ice_thickness_{ym}.nc",
        f"ice_thickness_{ym}.nc (from DATA_REAL_DIR/{ym})"
    )
    
    # 检查 landmask 文件
    print(f"\n[LANDMASK FILES]")
    check_file_exists(
        REAL_ENV_DIR / "land_mask_gebco.nc",
        "land_mask_gebco.nc (from REAL_ENV_DIR)"
    )
    check_file_exists(
        DATA_REAL_DIR / ym / "land_mask_gebco.nc",
        f"land_mask_gebco.nc (from DATA_REAL_DIR/{ym})"
    )
    
    # 列出 REAL_ENV_DIR 中的所有文件
    print(f"\n[FILES IN REAL_ENV_DIR]")
    if REAL_ENV_DIR.exists():
        nc_files = list(REAL_ENV_DIR.glob("*.nc"))
        if nc_files:
            print(f"  Found {len(nc_files)} NetCDF files:")
            for f in sorted(nc_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"  No NetCDF files found in {REAL_ENV_DIR}")
    else:
        print(f"  Directory does not exist: {REAL_ENV_DIR}")
    
    # 列出 DATA_REAL_DIR/{ym} 中的所有文件
    print(f"\n[FILES IN DATA_REAL_DIR/{ym}]")
    ym_dir = DATA_REAL_DIR / ym
    if ym_dir.exists():
        nc_files = list(ym_dir.glob("*.nc"))
        if nc_files:
            print(f"  Found {len(nc_files)} NetCDF files:")
            for f in sorted(nc_files):
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"    - {f.name} ({size_mb:.2f} MB)")
        else:
            print(f"  No NetCDF files found in {ym_dir}")
    else:
        print(f"  Directory does not exist: {ym_dir}")
    
    print("\n" + "=" * 100)
    print("END OF DEBUG REPORT")
    print("=" * 100 + "\n")


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="Debug script to check real environment data file existence"
    )
    parser.add_argument(
        "--ym",
        default="202412",
        help="Year-month string in format YYYYMM (default: 202412)",
    )
    
    args = parser.parse_args()
    
    debug_real_env_paths(ym=args.ym)


if __name__ == "__main__":
    main()

