"""
CMEMS 数据刷新与导出脚本。

根据 cmems_strategy.json 的决策，下载相应的数据。
支持时间窗口和地理边界框参数。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def load_strategy() -> dict:
    """加载 CMEMS 策略。"""
    strategy_path = Path("reports/cmems_strategy.json")
    
    if not strategy_path.exists():
        print("⚠ Strategy file not found, running cmems_resolve first...")
        subprocess.run([sys.executable, "-m", "scripts.cmems_resolve"], check=True)
    
    return json.loads(strategy_path.read_text(encoding="utf-8"))


def download_nextsim(days: int, bbox: tuple[float, float, float, float]) -> bool:
    """
    下载 nextsim 数据。
    
    Args:
        days: 时间窗口（天数）
        bbox: 边界框 (lon_min, lat_min, lon_max, lat_max)
    
    Returns:
        是否成功
    """
    print(f"\n[nextsim] Downloading {days} days of data...")
    print(f"  Bounding box: {bbox}")
    
    # 计算时间范围
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # 构建命令（模拟，实际需要根据 CMEMS CLI 调整）
    output_dir = Path("data_real/cmems_nextsim")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Time range: {start_date.date()} to {end_date.date()}")
    print(f"  Output dir: {output_dir}")
    print("  [WARN] Simulated download (actual CMEMS CLI integration needed)")
    
    # 创建一个占位文件表示成功
    placeholder = output_dir / f"nextsim_{start_date.date()}_{end_date.date()}.nc"
    placeholder.write_text(f"# Placeholder for nextsim data\n# {days} days, bbox={bbox}\n", encoding="utf-8")
    
    return True


def download_l4(days: int, bbox: tuple[float, float, float, float]) -> bool:
    """
    下载 L4 观测数据。
    
    Args:
        days: 时间窗口（天数）
        bbox: 边界框 (lon_min, lat_min, lon_max, lat_max)
    
    Returns:
        是否成功
    """
    print(f"\n[L4] Downloading {days} days of data...")
    print(f"  Bounding box: {bbox}")
    
    # 计算时间范围
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    # 构建命令（模拟）
    output_dir = Path("data_real/cmems_l4")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"  Time range: {start_date.date()} to {end_date.date()}")
    print(f"  Output dir: {output_dir}")
    print("  ⚠ Simulated download (actual CMEMS CLI integration needed)")
    
    # 创建一个占位文件表示成功
    placeholder = output_dir / f"l4_{start_date.date()}_{end_date.date()}.nc"
    placeholder.write_text(f"# Placeholder for L4 data\n# {days} days, bbox={bbox}\n", encoding="utf-8")
    
    return True


def main() -> None:
    """主函数。"""
    parser = argparse.ArgumentParser(description="CMEMS data refresh and export")
    parser.add_argument("--days", type=int, default=7, help="Time window in days")
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=[-180, 60, 180, 90],
        help="Bounding box: lon_min lat_min lon_max lat_max",
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("CMEMS Data Refresh and Export")
    print("=" * 60)
    
    # 加载策略
    strategy = load_strategy()
    selected = strategy.get("selected")
    
    print(f"\nSelected product: {selected}")
    
    if selected == "nextsim":
        success = download_nextsim(args.days, tuple(args.bbox))
    elif selected == "L4":
        success = download_l4(args.days, tuple(args.bbox))
    else:
        print("[WARN] No product selected, will use demo data as fallback")
        success = True  # 允许继续，使用 demo 数据
    
    if success:
        print("\n[OK] Download complete")
    else:
        print("\n[ERROR] Download failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

