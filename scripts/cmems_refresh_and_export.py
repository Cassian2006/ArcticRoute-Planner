#!/usr/bin/env python
"""
CMEMS 数据刷新和导出脚本

自动运行 copernicusmarine subset 下载最新数据，
并生成带时间戳的输出文件和元数据记录。

使用方式:
    python scripts/cmems_refresh_and_export.py [--days N] [--output-dir DIR]
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_resolved_config() -> dict:
    """加载已解析的 CMEMS 配置"""
    config_path = Path("reports/cmems_resolved.json")
    if not config_path.exists():
        raise FileNotFoundError(
            f"配置文件不存在: {config_path}\n"
            "请先运行: python scripts/cmems_resolve.py"
        )
    
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_subset(
    dataset_id: str,
    variable: str,
    start_date: str,
    end_date: str,
    output_dir: str,
    output_filename: str,
    bbox: dict,
) -> bool:
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
    
    Returns:
        True 如果成功，False 如果失败
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
    
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"下载成功: {output_filename}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"下载失败: {e}")
        logger.error(f"stdout: {e.stdout}")
        logger.error(f"stderr: {e.stderr}")
        return False


def _safe_atomic_write_text(target_path: Path, content: str, min_bytes: int = 1000) -> bool:
    """安全写入：仅当内容长度达到阈值时，原子替换目标文件。
    
    返回 True 表示已成功替换；False 表示未替换（内容过短）。
    """
    data = content.encode("utf-8")
    if len(data) < min_bytes:
        return False
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # 在同目录创建临时文件，确保 os.replace 原子性
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target_path.parent), prefix=target_path.name+".") as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        temp_name = tf.name
    os.replace(temp_name, target_path)
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="CMEMS 数据刷新和导出脚本"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=2,
        help="下载最近 N 天的数据（默认 2）；若提供 --start/--end 则忽略此参数",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/cmems_cache",
        help="输出目录（默认 data/cmems_cache）",
    )
    # 数据集参数
    parser.add_argument(
        "--sic-dataset-id",
        type=str,
        default="cmems_mod_arc_phy_anfc_nextsim_hm",
        help="SIC 数据集 ID（默认 cmems_mod_arc_phy_anfc_nextsim_hm）",
    )
    parser.add_argument(
        "--swh-dataset-id",
        type=str,
        default="dataset-wam-arctic-1hr3km-be",
        help="SWH 数据集 ID（默认 dataset-wam-arctic-1hr3km-be）",
    )
    # 时间参数
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="开始时间 (YYYY-MM-DD 或 ISO8601)，不提供则按 --days 计算",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="结束时间 (YYYY-MM-DD 或 ISO8601)，不提供则为今天 UTC",
    )
    # bbox 合并参数
    parser.add_argument(
        "--bbox",
        nargs="+",
        type=str,
        default=None,
        help="bbox 支持两种格式：1) 单个参数 'minlon,maxlon,minlat,maxlat'；2) 四个参数 -40 60 65 85",
    )
    parser.add_argument(
        "--bbox-min-lon",
        type=float,
        default=-40,
        help="边界框最小经度（默认 -40）",
    )
    parser.add_argument(
        "--bbox-max-lon",
        type=float,
        default=60,
        help="边界框最大经度（默认 60）",
    )
    parser.add_argument(
        "--bbox-min-lat",
        type=float,
        default=65,
        help="边界框最小纬度（默认 65）",
    )
    parser.add_argument(
        "--bbox-max-lat",
        type=float,
        default=85,
        help="边界框最大纬度（默认 85）",
    )
    # 仅探测变量模式
    parser.add_argument(
        "--describe-only",
        action="store_true",
        help="仅执行 describe，输出到 reports/cmems_*_describe.json 后退出",
    )
    
    args = parser.parse_args()
    
    # 仅探测变量（describe-only）
    if args.describe_only:
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        sic_desc = reports_dir / "cmems_sic_describe.json"
        swh_desc = reports_dir / "cmems_swh_describe.json"
        
        # 改进：捕获 stdout+stderr 和 exit code
        sic_ok = False
        try:
            res1 = subprocess.run(
                [
                    "copernicusmarine",
                    "describe",
                    "--contains",
                    args.sic_dataset_id,
                    "--return-fields",
                    "all",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            # 记录 exit code
            sic_exitcode_file = reports_dir / "cmems_sic_describe.exitcode.txt"
            sic_exitcode_file.write_text(str(res1.returncode), encoding="utf-8")
            
            # 记录 stderr（如果有）
            if res1.stderr:
                sic_stderr_file = reports_dir / "cmems_sic_describe.stderr.txt"
                sic_stderr_file.write_text(res1.stderr, encoding="utf-8")
                logger.warning(f"SIC describe stderr: {res1.stderr[:200]}")
            
            # 使用安全写入：仅当输出 >= 1000 字节才替换
            if _safe_atomic_write_text(sic_desc, res1.stdout, min_bytes=1000):
                logger.info(f"SIC describe 已安全写入: {sic_desc} (exit code: {res1.returncode})")
                sic_ok = True
            else:
                logger.warning(f"SIC describe 输出过短（<1000字节），保留旧文件: {sic_desc} (exit code: {res1.returncode})")
        except subprocess.TimeoutExpired as e:
            logger.error(f"SIC describe 超时（60秒）")
            (reports_dir / "cmems_sic_describe.exitcode.txt").write_text("-1", encoding="utf-8")
        except Exception as e:
            logger.error(f"SIC describe 异常: {e}")
            (reports_dir / "cmems_sic_describe.exitcode.txt").write_text("-2", encoding="utf-8")
        
        swh_ok = False
        try:
            res2 = subprocess.run(
                [
                    "copernicusmarine",
                    "describe",
                    "--contains",
                    args.swh_dataset_id,
                    "--return-fields",
                    "all",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            # 记录 exit code
            swh_exitcode_file = reports_dir / "cmems_swh_describe.exitcode.txt"
            swh_exitcode_file.write_text(str(res2.returncode), encoding="utf-8")
            
            # 记录 stderr（如果有）
            if res2.stderr:
                swh_stderr_file = reports_dir / "cmems_swh_describe.stderr.txt"
                swh_stderr_file.write_text(res2.stderr, encoding="utf-8")
                logger.warning(f"SWH describe stderr: {res2.stderr[:200]}")
            
            # 使用安全写入
            if _safe_atomic_write_text(swh_desc, res2.stdout, min_bytes=1000):
                logger.info(f"SWH describe 已安全写入: {swh_desc} (exit code: {res2.returncode})")
                swh_ok = True
            else:
                logger.warning(f"SWH describe 输出过短（<1000字节），保留旧文件: {swh_desc} (exit code: {res2.returncode})")
        except subprocess.TimeoutExpired as e:
            logger.error(f"SWH describe 超时（60秒）")
            (reports_dir / "cmems_swh_describe.exitcode.txt").write_text("-1", encoding="utf-8")
        except Exception as e:
            logger.error(f"SWH describe 异常: {e}")
            (reports_dir / "cmems_swh_describe.exitcode.txt").write_text("-2", encoding="utf-8")
        
        # describe-only 模式：如果任何一个失败或输出过短，非零退出
        if not (sic_ok and swh_ok):
            logger.error("describe-only 模式：至少一个 describe 失败或输出过短")
            logger.info("诊断文件已保存到 reports/ 目录：")
            logger.info("  - cmems_sic_describe.exitcode.txt")
            logger.info("  - cmems_sic_describe.stderr.txt (如果有错误)")
            logger.info("  - cmems_swh_describe.exitcode.txt")
            logger.info("  - cmems_swh_describe.stderr.txt (如果有错误)")
            return 1
        
        logger.info(f"已安全写入 {sic_desc} 与 {swh_desc}")
        return 0

    # 加载配置（优先）
    try:
        config = load_resolved_config()
    except FileNotFoundError:
        logger.warning("未找到 reports/cmems_resolved.json，将尝试使用默认变量名")
        config = {}
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 计算时间范围
    if args.start:
        start_date_str = args.start.split("T")[0]
    else:
        start_date_str = (datetime.utcnow().date() - timedelta(days=args.days)).strftime("%Y-%m-%d")
    if args.end:
        end_date_str = args.end.split("T")[0]
    else:
        end_date_str = datetime.utcnow().date().strftime("%Y-%m-%d")
    
    logger.info(f"下载时间范围: {start_date_str} 到 {end_date_str}")
    
    # 边界框
    if args.bbox:
        try:
            if isinstance(args.bbox, list):
                if len(args.bbox) == 1 and "," in args.bbox[0]:
                    parts = [float(x) for x in args.bbox[0].split(",")]
                elif len(args.bbox) == 4:
                    parts = [float(x) for x in args.bbox]
                else:
                    raise ValueError
            else:
                parts = [float(x) for x in str(args.bbox).split(",")]
            assert len(parts) == 4
            args.bbox_min_lon, args.bbox_max_lon, args.bbox_min_lat, args.bbox_max_lat = parts
        except Exception:
            logger.error(f"--bbox 参数无效：{args.bbox}，应为 'minlon,maxlon,minlat,maxlat' 或四个独立参数 -40 60 65 85")
            return 1
    bbox = {
        "min_lon": args.bbox_min_lon,
        "max_lon": args.bbox_max_lon,
        "min_lat": args.bbox_min_lat,
        "max_lat": args.bbox_max_lat,
    }
    
    # 记录元数据
    refresh_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "start_date": start_date_str,
        "end_date": end_date_str,
        "bbox": bbox,
        "downloads": {},
    }
    
    # 下载海冰数据
    logger.info("=" * 60)
    logger.info("下载海冰浓度数据 (SIC)")
    logger.info("=" * 60)
    
    sic_config = config.get("sic")
    if sic_config:
        sic_dataset_id = sic_config["dataset_id"]
        sic_variable = sic_config["variables"][0]  # 使用第一个变量
        
        # 生成带时间戳的文件名
        timestamp = datetime.utcnow().strftime("%Y%m%d")
        sic_filename = f"sic_{timestamp}.nc"
        
        success = run_subset(
            dataset_id=sic_dataset_id,
            variable=sic_variable,
            start_date=start_date_str,
            end_date=end_date_str,
            output_dir=str(output_dir),
            output_filename=sic_filename,
            bbox=bbox,
        )
        
        if success:
            sic_path = output_dir / sic_filename
            refresh_record["downloads"]["sic"] = {
                "dataset_id": sic_dataset_id,
                "variable": sic_variable,
                "filename": sic_filename,
                "path": str(sic_path),
                "timestamp": datetime.utcnow().isoformat(),
                "success": True,
            }
        else:
            refresh_record["downloads"]["sic"] = {
                "dataset_id": sic_dataset_id,
                "variable": sic_variable,
                "success": False,
                "error": "下载失败",
            }
    else:
        logger.warning("未找到 SIC 配置")
    
    # 下载波浪数据
    logger.info("=" * 60)
    logger.info("下载波浪数据 (SWH)")
    logger.info("=" * 60)
    
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
            # 生成带时间戳的文件名（包含小时）
            timestamp = datetime.utcnow().strftime("%Y%m%d%H")
            swh_filename = f"swh_{timestamp}.nc"
            
            success = run_subset(
                dataset_id=wav_dataset_id,
                variable=swh_variable,
                start_date=start_date_str,
                end_date=end_date_str,
                output_dir=str(output_dir),
                output_filename=swh_filename,
                bbox=bbox,
            )
            
            if success:
                swh_path = output_dir / swh_filename
                refresh_record["downloads"]["swh"] = {
                    "dataset_id": wav_dataset_id,
                    "variable": swh_variable,
                    "filename": swh_filename,
                    "path": str(swh_path),
                    "timestamp": datetime.utcnow().isoformat(),
                    "success": True,
                }
            else:
                refresh_record["downloads"]["swh"] = {
                    "dataset_id": wav_dataset_id,
                    "variable": swh_variable,
                    "success": False,
                    "error": "下载失败",
                }
        else:
            logger.warning("未找到有效波高变量")
    else:
        logger.warning("未找到 WAV 配置")
    
    # 保存元数据记录
    logger.info("=" * 60)
    logger.info("保存元数据记录")
    logger.info("=" * 60)
    
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    refresh_json = reports_dir / "cmems_refresh_last.json"
    with open(refresh_json, "w", encoding="utf-8") as f:
        json.dump(refresh_record, f, indent=2, ensure_ascii=False)
    
    logger.info(f"元数据已保存: {refresh_json}")
    
    # 打印总结
    logger.info("=" * 60)
    logger.info("下载完成总结")
    logger.info("=" * 60)
    
    sic_success = refresh_record["downloads"].get("sic", {}).get("success", False)
    swh_success = refresh_record["downloads"].get("swh", {}).get("success", False)
    
    logger.info(f"SIC 下载: {'✅ 成功' if sic_success else '❌ 失败'}")
    logger.info(f"SWH 下载: {'✅ 成功' if swh_success else '❌ 失败'}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"元数据: {refresh_json}")
    
    return 0 if (sic_success or swh_success) else 1


if __name__ == "__main__":
    sys.exit(main())

