"""
拟合环境阻力函数的幂指数 (p, q)。

使用 AIS 航速数据，通过去船型的速度比拟合：
  - 海冰阻力：f_ice = sic ** p
  - 海况阻力：f_wave = wave_swh ** q

输出：
  - reports/fitted_speed_exponents_{ym}.json
  - reports/fitted_speed_exponents_{ym}.csv (可选)

用法：
  python -m scripts.fit_speed_exponents --ym 202412 --max_points 200000
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ============================================================================
# 数据类与配置
# ============================================================================

@dataclass
class FitResult:
    """拟合结果数据类。"""
    ym: str
    p_sic: float
    q_wave: float
    b0: float
    b1: float
    b2: float
    rmse_train: float
    rmse_holdout: float
    r2_holdout: float
    n_samples_used: int
    n_mmsi_used: int
    n_bad_lines: int
    n_nan_dropped: int
    n_sog_filtered: int
    timestamp_utc: str
    notes: str

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# AIS 流式读取
# ============================================================================

def iter_ais_records_from_dir(
    ais_dir: Path | str,
    ym: str,
    max_records: int = 200000,
) -> tuple[List[Dict], int, int]:
    """
    从 AIS 原始目录流式读取 JSON/JSONL 文件。
    
    Args:
        ais_dir: AIS 原始数据目录
        ym: 目标月份，格式 YYYYMM (e.g., "202412")
        max_records: 最大读取记录数
    
    Returns:
        (records, n_bad_lines, n_total_files)
        其中 records 是 list of dict，包含 timestamp, lat, lon, sog, mmsi
    """
    ais_dir = Path(ais_dir)
    if not ais_dir.exists():
        logger.warning(f"AIS 目录不存在: {ais_dir}")
        return [], 0, 0
    
    # 解析目标月份
    try:
        year = int(ym[:4])
        month = int(ym[4:6])
    except (ValueError, IndexError):
        raise ValueError(f"Invalid ym format: {ym}, expected YYYYMM")
    
    records: List[Dict] = []
    n_bad_lines = 0
    n_total_files = 0
    
    # 遍历所有 JSON/JSONL 文件
    for json_file in sorted(ais_dir.glob("*.json")) + sorted(ais_dir.glob("*.jsonl")):
        n_total_files += 1
        logger.info(f"读取 AIS 文件: {json_file.name}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                for line_no, line in enumerate(f, 1):
                    if len(records) >= max_records:
                        logger.info(f"已达到最大记录数 {max_records}")
                        return records, n_bad_lines, n_total_files
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        # 尝试解析 JSON
                        if line.startswith('{'):
                            obj = json.loads(line)
                        else:
                            # 可能是 GeoJSON feature
                            obj = json.loads(line)
                        
                        # 处理 GeoJSON feature
                        if isinstance(obj, dict) and 'properties' in obj:
                            props = obj['properties']
                            coords = obj.get('geometry', {}).get('coordinates', [])
                            if coords and len(coords) >= 2:
                                obj = {**props, 'lon': coords[0], 'lat': coords[1]}
                        
                        # 提取必要字段
                        record = _extract_ais_fields(obj)
                        if record is None:
                            continue
                        
                        # 检查时间戳是否在目标月份
                        if not _is_in_month(record['timestamp'], year, month):
                            continue
                        
                        records.append(record)
                    
                    except (json.JSONDecodeError, KeyError, ValueError) as e:
                        n_bad_lines += 1
                        if n_bad_lines <= 5:  # 只打印前 5 个错误
                            logger.debug(f"坏行 {json_file.name}:{line_no}: {e}")
        
        except Exception as e:
            logger.error(f"读取文件失败 {json_file}: {e}")
            n_bad_lines += 1
    
    logger.info(f"读取完成: {len(records)} 条记录, {n_bad_lines} 条坏行, {n_total_files} 个文件")
    return records, n_bad_lines, n_total_files


def _extract_ais_fields(obj: Dict) -> Optional[Dict]:
    """从 AIS 对象提取必要字段。"""
    try:
        # 列名别名
        aliases = {
            'timestamp': ['timestamp', 'time', 'datetime', 'basedatetime', 'utc'],
            'lat': ['lat', 'latitude'],
            'lon': ['lon', 'longitude'],
            'sog': ['sog', 'speed', 'speedoverground'],
            'mmsi': ['mmsi'],
        }
        
        record = {}
        for std_name, candidates in aliases.items():
            value = None
            for cand in candidates:
                if cand in obj:
                    value = obj[cand]
                    break
            
            if value is None:
                return None  # 缺少必要字段
            
            record[std_name] = value
        
        # 验证数据类型
        record['lat'] = float(record['lat'])
        record['lon'] = float(record['lon'])
        record['sog'] = float(record['sog'])
        record['mmsi'] = str(record['mmsi'])
        
        # 验证范围
        if not (-90 <= record['lat'] <= 90):
            return None
        if not (-180 <= record['lon'] <= 180):
            return None
        if record['sog'] < 0:
            return None
        
        return record
    
    except (KeyError, ValueError, TypeError):
        return None


def _is_in_month(timestamp_str: str, year: int, month: int) -> bool:
    """检查时间戳是否在指定月份。"""
    try:
        # 尝试多种时间戳格式
        for fmt in [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%dT%H:%M:%SZ',
            '%Y/%m/%d %H:%M:%S',
        ]:
            try:
                dt = datetime.strptime(timestamp_str, fmt)
                return dt.year == year and dt.month == month
            except ValueError:
                continue
        
        # 如果都失败，尝试从字符串前缀提取
        if len(timestamp_str) >= 7:
            ym_str = timestamp_str[:7].replace('-', '').replace('/', '')
            if ym_str == f"{year}{month:02d}":
                return True
        
        return False
    except Exception:
        return False


# ============================================================================
# 速度标签构造（去船型）
# ============================================================================

def compute_speed_ratios(
    records: List[Dict],
    baseline_percentile: float = 80.0,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    为每条记录计算去船型的速度比。
    
    步骤：
    1. 按 MMSI 分组，计算每个船的 baseline_speed = P{percentile}(sog)
    2. 计算 speed_ratio = clip(sog / baseline_speed, 0.05, 1.2)
    3. 计算 y = log(speed_ratio)
    
    Args:
        records: AIS 记录列表
        baseline_percentile: 用于计算 baseline 的百分位数
    
    Returns:
        (df, mmsi_baselines)
        df: 包含 timestamp, lat, lon, sog, mmsi, baseline_speed, speed_ratio, y 的 DataFrame
        mmsi_baselines: {mmsi: baseline_speed} 映射
    """
    df = pd.DataFrame(records)
    
    # 按 MMSI 计算 baseline
    mmsi_baselines: Dict[str, float] = {}
    for mmsi, group in df.groupby('mmsi'):
        baseline = np.percentile(group['sog'].values, baseline_percentile)
        mmsi_baselines[mmsi] = baseline
    
    # 计算 speed_ratio 和 y
    df['baseline_speed'] = df['mmsi'].map(mmsi_baselines)
    df['speed_ratio'] = np.clip(df['sog'] / df['baseline_speed'], 0.05, 1.2)
    df['y'] = np.log(df['speed_ratio'])
    
    logger.info(f"计算速度比: {len(df)} 条记录, {len(mmsi_baselines)} 个 MMSI")
    
    return df, mmsi_baselines


# ============================================================================
# 环境场采样
# ============================================================================

def load_env_grids(
    data_root: Path | str,
) -> Tuple[xr.Dataset, xr.Dataset]:
    """
    加载 SIC 和 Wave 数据集。
    
    Args:
        data_root: 数据根目录
    
    Returns:
        (sic_ds, wave_ds)
    """
    data_root = Path(data_root)
    
    sic_path = data_root / "data_processed" / "newenv" / "ice_copernicus_sic.nc"
    wave_path = data_root / "data_processed" / "newenv" / "wave_swh.nc"
    
    logger.info(f"加载 SIC: {sic_path}")
    sic_ds = xr.open_dataset(sic_path)
    
    logger.info(f"加载 Wave: {wave_path}")
    wave_ds = xr.open_dataset(wave_path)
    
    return sic_ds, wave_ds


def sample_env_at_points(
    df: pd.DataFrame,
    sic_ds: xr.Dataset,
    wave_ds: xr.Dataset,
) -> Tuple[pd.DataFrame, int, int]:
    """
    在 AIS 点位采样环境场。
    
    Args:
        df: 包含 lat, lon 的 DataFrame
        sic_ds: SIC 数据集
        wave_ds: Wave 数据集
    
    Returns:
        (df_with_env, n_nan_sic, n_nan_wave)
        df_with_env: 添加了 sic, wave_swh 列的 DataFrame
    """
    # 提取坐标轴
    lat_axis = sic_ds['latitude'].values
    lon_axis = sic_ds['longitude'].values
    sic_var = sic_ds['sic'].values  # shape: (time, lat, lon)
    
    wave_lat_axis = wave_ds['latitude'].values
    wave_lon_axis = wave_ds['longitude'].values
    wave_var = wave_ds['wave_swh'].values  # shape: (time, lat, lon)
    
    # 假设只有一个时间步（或使用第一个时间步）
    if sic_var.ndim == 3:
        sic_var = sic_var[0]  # 取第一个时间步
    if wave_var.ndim == 3:
        wave_var = wave_var[0]
    
    sic_values = []
    wave_values = []
    n_nan_sic = 0
    n_nan_wave = 0
    
    for lat, lon in zip(df['lat'].values, df['lon'].values):
        # 采样 SIC
        i = np.searchsorted(lat_axis, lat)
        i = np.clip(i, 0, len(lat_axis) - 1)
        j = np.searchsorted(lon_axis, lon)
        j = np.clip(j, 0, len(lon_axis) - 1)
        
        sic_val = float(sic_var[i, j]) if not np.isnan(sic_var[i, j]) else np.nan
        if np.isnan(sic_val):
            n_nan_sic += 1
        sic_values.append(sic_val)
        
        # 采样 Wave
        i_w = np.searchsorted(wave_lat_axis, lat)
        i_w = np.clip(i_w, 0, len(wave_lat_axis) - 1)
        j_w = np.searchsorted(wave_lon_axis, lon)
        j_w = np.clip(j_w, 0, len(wave_lon_axis) - 1)
        
        wave_val = float(wave_var[i_w, j_w]) if not np.isnan(wave_var[i_w, j_w]) else np.nan
        if np.isnan(wave_val):
            n_nan_wave += 1
        wave_values.append(wave_val)
    
    df['sic'] = sic_values
    df['wave_swh'] = wave_values
    
    logger.info(f"环境采样: {n_nan_sic} 个 NaN SIC, {n_nan_wave} 个 NaN Wave")
    
    return df, n_nan_sic, n_nan_wave


# ============================================================================
# 网格搜索拟合
# ============================================================================

def grid_search_fit(
    df: pd.DataFrame,
    p_min: float = 0.8,
    p_max: float = 2.6,
    q_min: float = 0.8,
    q_max: float = 3.0,
    coarse_step: float = 0.2,
    fine_step: float = 0.05,
    holdout_ratio: float = 0.2,
    random_seed: int = 42,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """
    两阶段网格搜索拟合 p, q。
    
    Args:
        df: 包含 sic, wave_swh, y 的 DataFrame
        p_min, p_max: p 的搜索范围
        q_min, q_max: q 的搜索范围
        coarse_step: 粗搜步长
        fine_step: 细搜步长
        holdout_ratio: 验证集比例
        random_seed: 随机种子
    
    Returns:
        (p_best, q_best, b0, b1, b2, rmse_train, rmse_holdout, r2_holdout)
    """
    # 移除 NaN 行
    df_clean = df.dropna(subset=['sic', 'wave_swh', 'y'])
    logger.info(f"清洁数据: {len(df_clean)} 条记录")
    
    # 分割训练/验证集
    np.random.seed(random_seed)
    n_samples = len(df_clean)
    n_holdout = int(n_samples * holdout_ratio)
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    train_idx = indices[n_holdout:]
    holdout_idx = indices[:n_holdout]
    
    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_holdout = df_clean.iloc[holdout_idx].reset_index(drop=True)
    
    logger.info(f"训练集: {len(df_train)}, 验证集: {len(df_holdout)}")
    
    # 第一阶段：粗搜
    logger.info("第一阶段：粗搜...")
    p_range = np.arange(p_min, p_max + coarse_step, coarse_step)
    q_range = np.arange(q_min, q_max + coarse_step, coarse_step)
    
    best_rmse = np.inf
    best_p = p_min
    best_q = q_min
    
    for p in p_range:
        for q in q_range:
            rmse = _compute_rmse(df_train, p, q)
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = p
                best_q = q
    
    logger.info(f"粗搜最优: p={best_p:.2f}, q={best_q:.2f}, rmse={best_rmse:.6f}")
    
    # 第二阶段：细搜
    logger.info("第二阶段：细搜...")
    p_range_fine = np.arange(
        max(p_min, best_p - 2 * coarse_step),
        min(p_max, best_p + 2 * coarse_step) + fine_step,
        fine_step
    )
    q_range_fine = np.arange(
        max(q_min, best_q - 2 * coarse_step),
        min(q_max, best_q + 2 * coarse_step) + fine_step,
        fine_step
    )
    
    best_rmse = np.inf
    best_p = p_min
    best_q = q_min
    
    for p in p_range_fine:
        for q in q_range_fine:
            rmse = _compute_rmse(df_train, p, q)
            if rmse < best_rmse:
                best_rmse = rmse
                best_p = p
                best_q = q
    
    logger.info(f"细搜最优: p={best_p:.4f}, q={best_q:.4f}, rmse={best_rmse:.6f}")
    
    # 拟合最终模型
    b0, b1, b2, rmse_train, rmse_holdout, r2_holdout = _fit_linear_model(
        df_train, df_holdout, best_p, best_q
    )
    
    return best_p, best_q, b0, b1, b2, rmse_train, rmse_holdout, r2_holdout


def _compute_rmse(df: pd.DataFrame, p: float, q: float) -> float:
    """计算给定 p, q 的 RMSE。"""
    try:
        x1 = np.clip(df['sic'].values ** p, -1e10, 1e10)
        x2 = np.clip(df['wave_swh'].values ** q, -1e10, 1e10)
        y = df['y'].values
        
        # 检查是否有 NaN 或 inf
        if np.any(~np.isfinite(x1)) or np.any(~np.isfinite(x2)) or np.any(~np.isfinite(y)):
            return np.inf
        
        # 最小二乘拟合
        X = np.column_stack([np.ones(len(df)), x1, x2])
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
        
        if not np.all(np.isfinite(coef)):
            return np.inf
        
        y_pred = X @ coef
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))
        return rmse if np.isfinite(rmse) else np.inf
    except Exception:
        return np.inf


def _fit_linear_model(
    df_train: pd.DataFrame,
    df_holdout: pd.DataFrame,
    p: float,
    q: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    拟合线性模型 y = b0 + b1*x1 + b2*x2。
    
    Returns:
        (b0, b1, b2, rmse_train, rmse_holdout, r2_holdout)
    """
    try:
        # 训练集
        x1_train = np.clip(df_train['sic'].values ** p, -1e10, 1e10)
        x2_train = np.clip(df_train['wave_swh'].values ** q, -1e10, 1e10)
        y_train = df_train['y'].values
        
        X_train = np.column_stack([np.ones(len(df_train)), x1_train, x2_train])
        coef = np.linalg.lstsq(X_train, y_train, rcond=None)[0]
        b0, b1, b2 = coef
        
        y_pred_train = X_train @ coef
        rmse_train = np.sqrt(np.mean((y_train - y_pred_train) ** 2))
        
        # 验证集
        x1_holdout = np.clip(df_holdout['sic'].values ** p, -1e10, 1e10)
        x2_holdout = np.clip(df_holdout['wave_swh'].values ** q, -1e10, 1e10)
        y_holdout = df_holdout['y'].values
        
        X_holdout = np.column_stack([np.ones(len(df_holdout)), x1_holdout, x2_holdout])
        y_pred_holdout = X_holdout @ coef
        rmse_holdout = np.sqrt(np.mean((y_holdout - y_pred_holdout) ** 2))
        
        # R2
        ss_res = np.sum((y_holdout - y_pred_holdout) ** 2)
        ss_tot = np.sum((y_holdout - np.mean(y_holdout)) ** 2)
        r2_holdout = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        logger.info(f"模型系数: b0={b0:.6f}, b1={b1:.6f}, b2={b2:.6f}")
        logger.info(f"训练 RMSE: {rmse_train:.6f}")
        logger.info(f"验证 RMSE: {rmse_holdout:.6f}, R2: {r2_holdout:.6f}")
        
        return b0, b1, b2, rmse_train, rmse_holdout, r2_holdout
    except Exception as e:
        logger.error(f"拟合失败: {e}")
        return 0.0, -0.5, -0.3, np.inf, np.inf, 0.0


# ============================================================================
# 输出文件
# ============================================================================

def write_results(
    result: FitResult,
    output_dir: Path | str = "reports",
) -> Tuple[Path, Path]:
    """
    写入拟合结果到 JSON 和 CSV。
    
    Returns:
        (json_path, csv_path)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # JSON 文件
    json_path = output_dir / f"fitted_speed_exponents_{result.ym}.json"
    with open(json_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    logger.info(f"写入 JSON: {json_path}")
    
    # CSV 文件（可选）
    csv_path = output_dir / f"fitted_speed_exponents_{result.ym}.csv"
    csv_df = pd.DataFrame([result.to_dict()])
    csv_df.to_csv(csv_path, index=False)
    logger.info(f"写入 CSV: {csv_path}")
    
    return json_path, csv_path


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="拟合环境阻力函数的幂指数 (p, q)"
    )
    parser.add_argument("--ym", type=str, required=True, help="目标月份 (YYYYMM)")
    parser.add_argument("--max_points", type=int, default=200000, help="最大读取记录数")
    parser.add_argument("--holdout", type=float, default=0.2, help="验证集比例")
    parser.add_argument("--p_min", type=float, default=0.8, help="p 最小值")
    parser.add_argument("--p_max", type=float, default=2.6, help="p 最大值")
    parser.add_argument("--q_min", type=float, default=0.8, help="q 最小值")
    parser.add_argument("--q_max", type=float, default=3.0, help="q 最大值")
    parser.add_argument("--coarse_step", type=float, default=0.2, help="粗搜步长")
    parser.add_argument("--fine_step", type=float, default=0.05, help="细搜步长")
    parser.add_argument("--data_root", type=str, default=None, help="数据根目录")
    parser.add_argument("--output_dir", type=str, default="reports", help="输出目录")
    
    args = parser.parse_args()
    
    # 确定数据根目录
    if args.data_root is None:
        args.data_root = os.getenv("ARCTICROUTE_DATA_ROOT")
    if args.data_root is None:
        args.data_root = Path(__file__).resolve().parents[2] / "data_real"
    
    data_root = Path(args.data_root)
    logger.info(f"数据根目录: {data_root}")
    
    # 步骤 1: 读取 AIS 数据
    ais_dir = data_root / "ais" / "raw"
    if not ais_dir.exists():
        logger.warning(f"AIS 目录不存在: {ais_dir}")
        logger.info("使用合成数据进行演示...")
        # 生成合成数据用于演示
        records = _generate_synthetic_ais_data(args.ym, args.max_points)
    else:
        records, n_bad_lines, n_files = iter_ais_records_from_dir(
            ais_dir, args.ym, args.max_points
        )
    
    if len(records) == 0:
        logger.error("没有读取到 AIS 数据")
        return
    
    logger.info(f"读取 AIS 记录: {len(records)}")
    
    # 步骤 2: 计算速度比
    df, mmsi_baselines = compute_speed_ratios(records)
    n_sog_filtered = len(records) - len(df)
    
    # 步骤 3: 加载环境场
    try:
        sic_ds, wave_ds = load_env_grids(data_root)
    except Exception as e:
        logger.error(f"加载环境场失败: {e}")
        return
    
    # 步骤 4: 采样环境场
    df, n_nan_sic, n_nan_wave = sample_env_at_points(df, sic_ds, wave_ds)
    
    # 步骤 5: 网格搜索拟合
    p_best, q_best, b0, b1, b2, rmse_train, rmse_holdout, r2_holdout = grid_search_fit(
        df,
        p_min=args.p_min,
        p_max=args.p_max,
        q_min=args.q_min,
        q_max=args.q_max,
        coarse_step=args.coarse_step,
        fine_step=args.fine_step,
        holdout_ratio=args.holdout,
    )
    
    # 步骤 6: 生成结果
    result = FitResult(
        ym=args.ym,
        p_sic=p_best,
        q_wave=q_best,
        b0=b0,
        b1=b1,
        b2=b2,
        rmse_train=rmse_train,
        rmse_holdout=rmse_holdout,
        r2_holdout=r2_holdout,
        n_samples_used=len(df.dropna(subset=['sic', 'wave_swh', 'y'])),
        n_mmsi_used=len(mmsi_baselines),
        n_bad_lines=0,  # 从流式读取中获取
        n_nan_dropped=n_nan_sic + n_nan_wave,
        n_sog_filtered=n_sog_filtered,
        timestamp_utc=datetime.utcnow().isoformat(),
        notes=f"baseline=P80 per MMSI; holdout_ratio={args.holdout}",
    )
    
    # 步骤 7: 写入结果
    json_path, csv_path = write_results(result, args.output_dir)
    
    logger.info("=" * 60)
    logger.info(f"拟合完成！")
    logger.info(f"  p_sic = {p_best:.4f}")
    logger.info(f"  q_wave = {q_best:.4f}")
    logger.info(f"  holdout RMSE = {rmse_holdout:.6f}")
    logger.info(f"  holdout R2 = {r2_holdout:.6f}")
    logger.info(f"输出文件: {json_path}")
    logger.info("=" * 60)


def _generate_synthetic_ais_data(ym: str, n_records: int) -> List[Dict]:
    """生成合成 AIS 数据用于演示。"""
    logger.info(f"生成 {n_records} 条合成 AIS 数据...")
    
    np.random.seed(42)
    year, month = int(ym[:4]), int(ym[4:6])
    
    records = []
    for i in range(n_records):
        timestamp = f"{year}-{month:02d}-{(i % 28) + 1:02d}T{(i % 24):02d}:00:00"
        lat = 70 + np.random.uniform(-5, 5)
        lon = -30 + np.random.uniform(-30, 30)
        sog = np.random.uniform(5, 20)
        mmsi = str(200000000 + (i % 100))
        
        records.append({
            'timestamp': timestamp,
            'lat': lat,
            'lon': lon,
            'sog': sog,
            'mmsi': mmsi,
        })
    
    return records


if __name__ == "__main__":
    main()

