"""
检查网格与陆地掩码的脚本。

用法: python -m scripts.check_grid_and_landmask

输出：
- 网格来源（real 或 demo）
- 网格形状
- 陆地掩码候选列表（含 signature/shape/varname）
- 最终采用的 landmask 路径
- 陆地/海洋比例
- 是否重采样
- 缺失时的修复指引
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from arcticroute.core.grid import load_real_grid_from_nc, load_grid_with_landmask
from arcticroute.core.landmask import load_real_landmask_from_nc, load_landmask
from arcticroute.core.landmask_select import scan_landmask_candidates
from arcticroute.core.env_real import get_data_root


def main() -> None:
    """
    主函数：加载并显示网格与陆地掩码的基本信息。

    首先尝试加载真实网格和 landmask，如果失败则回退到 demo。
    """
    print("=" * 80)
    print("网格与陆地掩码检查")
    print("=" * 80)

    # 显示数据根目录
    data_root = get_data_root()
    print(f"\n[0] 数据根目录配置：")
    print(f"  ARCTICROUTE_DATA_ROOT: {os.getenv('ARCTICROUTE_DATA_ROOT', '(未设置)')}")
    print(f"  实际使用: {data_root}")

    # 扫描候选文件
    print(f"\n[1] 扫描可用的 landmask 候选文件：")
    candidates = scan_landmask_candidates()
    if candidates:
        print(f"  找到 {len(candidates)} 个候选：")
        for i, cand in enumerate(candidates, 1):
            sig_str = f"sig={cand.grid_signature}" if cand.grid_signature else "sig=None"
            shape_str = f"shape={cand.shape}" if cand.shape else "shape=None"
            var_str = f"var={cand.varname}" if cand.varname else "var=None"
            note_str = f" [{cand.note}]" if cand.note else ""
            print(f"    {i}. {cand.path}")
            print(f"       {sig_str}, {shape_str}, {var_str}{note_str}")
    else:
        print(f"  (未找到候选文件)")
        print(f"  建议搜索目录：")
        print(f"    - {data_root / 'data_real' / 'landmask'}")
        print(f"    - {data_root / 'data_real' / 'env'}")
        print(f"    - {data_root / 'data_real'}")

    print(f"\n[2] 加载网格与 landmask...")

    # 使用统一的加载函数
    grid, land_mask, meta = load_grid_with_landmask(prefer_real=True)

    source = meta.get("source", "unknown")
    data_root_str = meta.get("data_root", "unknown")

    print(f"  Grid source: {source}")
    print(f"  Data root: {data_root_str}")

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

    # 显示 landmask 加载详情
    print(f"\n[5] Landmask 加载详情：")
    landmask_path = meta.get("landmask_path", "unknown")
    landmask_resampled = meta.get("landmask_resampled", False)
    landmask_note = meta.get("landmask_note", "unknown")
    print(f"  Path: {landmask_path}")
    print(f"  Resampled: {landmask_resampled}")
    print(f"  Note: {landmask_note}")

    # 显示角落坐标
    lat0 = grid.lat2d[0, 0]
    lon0 = grid.lon2d[0, 0]
    lat1 = grid.lat2d[-1, -1]
    lon1 = grid.lon2d[-1, -1]
    print(f"\n[6] 网格范围：")
    print(f"  Top-left: ({lat0:.3f}°N, {lon0:.3f}°E)")
    print(f"  Bottom-right: ({lat1:.3f}°N, {lon1:.3f}°E)")

    # 修复指引
    print(f"\n[7] 修复指引：")
    if "fallback" in str(landmask_path).lower() or "demo" in str(landmask_path).lower():
        print(f"  [WARN] 当前使用 demo landmask（真实数据未找到）")
        print(f"  ")
        print(f"  要使用真实 landmask，请：")
        print(f"  1. 将 landmask 文件放到以下目录之一：")
        print(f"     - {data_root / 'data_real' / 'landmask'} (推荐)")
        print(f"     - {data_root / 'data_real' / 'env'}")
        print(f"     - {data_root / 'data_real'}")
        print(f"  ")
        print(f"  2. 文件名应包含 'landmask' 或 'land_mask'")
        print(f"  ")
        print(f"  3. 文件格式：NetCDF (.nc)")
        print(f"     - 应包含 land_mask/landmask/mask/lsm/land/is_land 变量")
        print(f"     - 可选：grid_signature 属性用于精确匹配")
        print(f"  ")
        print(f"  4. 重新运行此脚本验证")
    else:
        print(f"  [OK] 成功加载真实 landmask")

    print("\n" + "=" * 80)
    if "fallback" not in str(landmask_path).lower() and "demo" not in str(landmask_path).lower():
        print("[OK] Successfully loaded real grid and landmask")
    else:
        print("[WARN] Using demo grid and/or landmask (real data not fully available)")
    print("=" * 80)


if __name__ == "__main__":
    main()

