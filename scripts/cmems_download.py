#!/usr/bin/env python
"""
CMEMS 近实时数据下载脚本

使用 copernicusmarine subset 命令下载海冰浓度和波浪数据。
支持自动滚动更新（可重复执行）。
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

def load_resolved_config():
    """加载已解析的 dataset-id 和变量配置"""
    config_path = Path("reports/cmems_resolved.json")
    if not config_path.exists():
        print(f"[ERROR] {config_path} 不存在，请先运行 cmems_resolve.py")
        return None
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_subset(dataset_id: str, variable: str, start_date: str, end_date: str, 
               output_dir: str, output_filename: str, bbox: dict):
    """
    执行 copernicusmarine subset 命令
    
    Args:
        dataset_id: 数据集 ID
        variable: 变量名
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
        output_dir: 输出目录
        output_filename: 输出文件名
        bbox: 边界框 {min_lon, max_lon, min_lat, max_lat}
    """
    cmd = [
        "copernicusmarine",
        "subset",
        "--dataset-id", dataset_id,
        "--variable", variable,
        "--start-datetime", start_date,
        "--end-datetime", end_date,
        "--minimum-longitude", str(bbox["min_lon"]),
        "--maximum-longitude", str(bbox["max_lon"]),
        "--minimum-latitude", str(bbox["min_lat"]),
        "--maximum-latitude", str(bbox["max_lat"]),
        "--output-directory", output_dir,
        "--output-filename", output_filename,
    ]
    
    print(f"[INFO] 执行命令: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"[OK] 下载完成: {output_filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 下载失败: {e}")
        return False

def main():
    # 加载配置
    config = load_resolved_config()
    if not config:
        return
    
    # 创建输出目录
    output_dir = Path("data/cmems_cache")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义时间范围（近两天）
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=2)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"[INFO] 下载时间范围: {start_date_str} 到 {end_date_str}")
    
    # 北极 bbox
    bbox = {
        "min_lon": -40,
        "max_lon": 60,
        "min_lat": 65,
        "max_lat": 85,
    }
    
    # 下载海冰浓度数据
    print("\n=== 下载海冰浓度数据 (SIC) ===")
    sic_config = config.get("sic")
    if sic_config:
        sic_dataset_id = sic_config["dataset_id"]
        sic_variable = sic_config["variables"][0]  # 使用第一个变量 (sic)
        
        run_subset(
            dataset_id=sic_dataset_id,
            variable=sic_variable,
            start_date=start_date_str,
            end_date=end_date_str,
            output_dir=str(output_dir),
            output_filename="sic_latest.nc",
            bbox=bbox,
        )
    else:
        print("[WARN] 未找到 SIC 配置")
    
    # 下载波浪数据
    print("\n=== 下载波浪数据 (SWH) ===")
    wav_config = config.get("wav")
    if wav_config:
        wav_dataset_id = wav_config["dataset_id"]
        # 查找有效波高变量
        swh_variable = None
        for var in wav_config["variables"]:
            if "significant_height" in var.lower():
                swh_variable = var
                break
        
        if swh_variable:
            run_subset(
                dataset_id=wav_dataset_id,
                variable=swh_variable,
                start_date=start_date_str,
                end_date=end_date_str,
                output_dir=str(output_dir),
                output_filename="swh_latest.nc",
                bbox=bbox,
            )
        else:
            print("[WARN] 未找到有效波高变量")
    else:
        print("[WARN] 未找到 WAV 配置")
    
    print("\n[OK] 下载流程完成")

if __name__ == "__main__":
    main()

