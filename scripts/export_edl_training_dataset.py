from __future__ import annotations

"""
EDL 训练数据导出脚本（E0.2）

功能：
- 读取真实网格（同 planner 用的）、Copernicus SIC / wave、AIS density（.nc），可选冰厚
- 在网格上按点采样生成训练样本（每个有效海洋网格点 ≥1 样本）
- 生成特征列与标签（safe/risky），并控制最大样本数（默认 500k）
- 输出到 data_real/edl_training/edl_train.parquet
- 打印数据概览（行数、类别比例、特征范围）

可作为脚本运行，也可在单元测试中 import 使用：
    from scripts.export_edl_training_dataset import export_edl_training_dataset
"""

import os
from pathlib import Path
from typing import Optional, Sequence, Tuple, Dict

import numpy as np
import pandas as pd

# 依赖 xarray 读取 NetCDF
try:
    import xarray as xr
except Exception:  # pragma: no cover - 在少数环境中可能缺失
    xr = None

# 项目内工具
from arcticroute.core.env_real import load_real_env_for_grid, get_data_root
from arcticroute.core.grid import Grid2D, get_project_root


# ---------------------------- 常量与默认阈值 ---------------------------- #
DEFAULT_OUTPUT_PATH = Path("data_real/edl_training/edl_train.parquet")

# 安全阈值（与 E0.1 文档一致）
SAFE_THRESHOLDS = {
    "sic_safe": 0.30,      # sic < 0.30 视为安全（注意是 0-1）
    "sic_risky": 0.70,     # sic >= 0.70 视为高风险
    "ice_safe": 1.0,       # m
    "ice_risky": 2.0,      # m
    "wave_safe": 4.0,      # m
    "wave_risky": 5.0,     # m
    "ais_safe": 0.10,      # 0-1 归一化
    "ais_risky": 0.02,     # 非常低
}

AIS_VAR_CANDIDATES: Tuple[str, ...] = (
    "ais_density", "AIS_density", "ais", "density", "ais_dens"
)


# ---------------------------- 工具函数 ---------------------------- #

def _load_ais_density_nc(
    grid: Grid2D,
    nc_path: Path,
    var_candidates: Tuple[str, ...] = AIS_VAR_CANDIDATES,
    time_index: int = 0,
) -> Optional[np.ndarray]:
    """从 NetCDF 读取 AIS 密度，并与 grid 形状一致。

    返回 2D np.ndarray（ny, nx），范围裁剪到 [0,1]；失败返回 None。
    """
    if xr is None:
        print("[EDL_EXPORT] xarray not available, cannot read AIS density")
        return None
    if not nc_path.exists():
        print(f"[EDL_EXPORT] AIS density file not found: {nc_path}")
        return None

    try:
        ds = xr.open_dataset(nc_path, decode_times=False)
    except Exception as e:
        print(f"[EDL_EXPORT] failed to open AIS density {nc_path}: {e}")
        return None

    try:
        da = None
        for name in var_candidates:
            if name in ds:
                da = ds[name]
                break
        if da is None:
            # 尝试 data_vars 的第一个变量
            if len(ds.data_vars) > 0:
                first_key = list(ds.data_vars)[0]
                da = ds[first_key]
                print(f"[EDL_EXPORT] AIS density var not found, fallback to '{first_key}'")
            else:
                print(f"[EDL_EXPORT] AIS density var not found in {nc_path}")
                return None

        vals = da.values
        if vals.ndim == 3:
            vals = vals[min(time_index, vals.shape[0]-1), :, :]
        if vals.ndim != 2:
            print(f"[EDL_EXPORT] unexpected AIS density dims: {vals.shape}")
            return None

        ny, nx = grid.shape()
        if vals.shape != (ny, nx):
            print(f"[EDL_EXPORT] AIS density shape {vals.shape} != grid {ny,nx}")
            return None

        vals = np.asarray(vals, dtype=float)
        # 规范化到 0-1
        if np.nanmax(vals) > 1.0:
            vmax = np.nanmax(vals)
            if vmax > 0:
                vals = vals / vmax
        vals = np.clip(vals, 0.0, 1.0)
        return vals
    except Exception as e:
        print(f"[EDL_EXPORT] error reading AIS density: {e}")
        return None
    finally:
        try:
            ds.close()
        except Exception:
            pass


def _safe_minmax(x: np.ndarray | pd.Series) -> Tuple[float, float]:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.any():
        return float(np.nanmin(x[finite])), float(np.nanmax(x[finite]))
    return float("nan"), float("nan")


# ---------------------------- 主导出函数 ---------------------------- #

def export_edl_training_dataset(
    *,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    max_samples: int = 500_000,
    # 可选：提供覆盖的数据路径（用于单元测试或自定义数据）
    nc_sic_path: Optional[Path | str] = None,
    nc_wave_path: Optional[Path | str] = None,
    nc_ice_thickness_path: Optional[Path | str] = None,
    nc_ais_density_path: Optional[Path | str] = None,
    time_index: int = 0,
    random_seed: int = 42,
    thresholds: Dict[str, float] = SAFE_THRESHOLDS,
    replicate_by_vessel: bool = False,
) -> pd.DataFrame:
    """构建并导出 EDL 训练数据集。

    返回生成的 DataFrame（也会写 parquet）。
    """
    np.random.seed(random_seed)

    # 1) 加载真实环境（网格 + sic + wave + landmask）
    env = load_real_env_for_grid(
        grid=None,
        nc_sic_path=Path(nc_sic_path) if nc_sic_path else None,
        nc_wave_path=Path(nc_wave_path) if nc_wave_path else None,
        nc_ice_thickness_path=Path(nc_ice_thickness_path) if nc_ice_thickness_path else None,
        time_index=time_index,
        ym=None,
    )
    if env is None or env.grid is None:
        raise RuntimeError("[EDL_EXPORT] Failed to load real environment grid/layers")

    grid = env.grid
    ny, nx = grid.shape()

    # 2) AIS density（更智能：依次在多个目录中查找 *.nc）
    if nc_ais_density_path is None:
        data_root = get_data_root()
        prj_root = get_project_root()
        cwd = Path.cwd()

        candidate_dirs = [
            data_root / "ais" / "derived",                     # 首选：外部数据根
            prj_root / "data_real" / "ais" / "derived",      # 回退：项目内 data_real
            cwd / "data_real" / "ais" / "derived",           # 回退：当前工作目录下 data_real
        ]
        tested: list[str] = []
        found: Optional[Path] = None
        for d in candidate_dirs:
            tested.append(str(d))
            if d.exists():
                nc_list = sorted(d.glob("*.nc"))
                if nc_list:
                    found = nc_list[0]
                    break
        if found is not None:
            nc_ais_density_path = found
            print(f"[EDL_EXPORT] using AIS density: {nc_ais_density_path}")
        else:
            print("[EDL_EXPORT] no AIS density file found in any of:")
            for t in tested:
                print(f"  - {t}")
            print("[EDL_EXPORT] will set ais_density=0 (all zeros)")
            nc_ais_density_path = None

    if nc_ais_density_path is not None:
        ais_density = _load_ais_density_nc(grid, Path(nc_ais_density_path), time_index=time_index)
    else:
        ais_density = None

    # 3) 取出数据层
    sic = env.sic  # 0-1
    wave = env.wave_swh
    land = env.land_mask  # 1=land, 0=ocean（若可用）

    # 冰厚（目前可能没有）
    ice_thk = None
    if nc_ice_thickness_path is not None:
        # 简单尝试读取（变量名常见候选）
        if xr is not None and Path(nc_ice_thickness_path).exists():
            try:
                with xr.open_dataset(Path(nc_ice_thickness_path), decode_times=False) as ds_ice:
                    da = None
                    for nm in ("sithick", "sit", "ice_thickness", "ice_thk"):
                        if nm in ds_ice:
                            da = ds_ice[nm]
                            break
                    if da is not None:
                        vals = da.values
                        if vals.ndim == 3:
                            vals = vals[min(time_index, vals.shape[0]-1), :, :]
                        if vals.ndim == 2 and vals.shape == (ny, nx):
                            ice_thk = np.asarray(vals, dtype=float)
            except Exception as e:
                print(f"[EDL_EXPORT] read ice thickness failed: {e}")

    # 缺失处理：若为空则置零（并单独记录）
    if sic is None:
        sic = np.zeros((ny, nx), dtype=float)
    if wave is None:
        wave = np.zeros((ny, nx), dtype=float)
    if ais_density is None:
        ais_density = np.zeros((ny, nx), dtype=float)
    if ice_thk is None:
        ice_thk = np.zeros((ny, nx), dtype=float)

    # 4) 构建有效海洋掩膜
    if land is not None:
        ocean_mask = (land == 0)
    else:
        # 若无陆地掩膜，则全部有效
        ocean_mask = np.ones((ny, nx), dtype=bool)

    # 丢弃 NaN/Inf
    finite_mask = (
        np.isfinite(sic) & np.isfinite(wave) & np.isfinite(ais_density) & np.isfinite(ice_thk)
    )
    valid_mask = ocean_mask & finite_mask

    # 5) 生成 DataFrame（每个有效点一条记录，可选按不同船型复制）
    lat2d, lon2d = grid.lat2d, grid.lon2d
    yy, xx = np.where(valid_mask)

    def _gather_at(idxs: Tuple[np.ndarray, np.ndarray]) -> Dict[str, np.ndarray]:
        iy, ix = idxs
        return {
            "lat": lat2d[iy, ix].astype(float),
            "lon": lon2d[iy, ix].astype(float),
            "sic": sic[iy, ix].astype(float),
            "ice_thickness_m": ice_thk[iy, ix].astype(float),
            "wave_swh": wave[iy, ix].astype(float),
            "ais_density": ais_density[iy, ix].astype(float),
        }

    base = _gather_at((yy, xx))

    # 时间特征：尝试从 SIC 或 WAVE 文件读取 time；失败则固定一个日期
    month = np.full(len(yy), 8, dtype=np.int16)  # 默认 8 月
    dayofyear = np.full(len(yy), 220, dtype=np.int16)  # 默认第 220 天

    # 船舶类别：默认 1(Panamax)
    vessel = np.full(len(yy), 1, dtype=np.int8)

    data = {
        **base,
        "month": month,
        "dayofyear": dayofyear,
        "vessel_class_id": vessel,
    }

    df = pd.DataFrame(data)

    if replicate_by_vessel:
        # 可选：为不同船级复制样本（0/1/2）
        dfs = []
        for vid in [0, 1, 2]:
            dfi = df.copy()
            dfi["vessel_class_id"] = vid
            dfs.append(dfi)
        df = pd.concat(dfs, ignore_index=True)

    # 6) 生成标签（简化规则）
    th = thresholds

    safe_mask = (
        (df["ais_density"] > th["ais_safe"]) &
        (df["sic"] < th["sic_safe"]) &
        ((df["ice_thickness_m"] < th["ice_safe"]) | (df["ice_thickness_m"].isna())) &
        (df["wave_swh"] < th["wave_safe"]) 
    )

    risky_mask = (
        (df["ais_density"] < th["ais_risky"]) &
        (
            (df["ice_thickness_m"] >= th["ice_risky"]) |
            (df["sic"] >= th["sic_risky"]) |
            (df["wave_swh"] >= th["wave_risky"]) 
        )
    )

    # 默认边界：其余标记为 risky（保守）
    labels = np.where(safe_mask, 1, np.where(risky_mask, 0, 0)).astype(np.int8)
    df["label_safe_risky"] = labels

    # 7) 控制样本数量（最多 max_samples）
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=random_seed).reset_index(drop=True)

    # 8) 强制列类型并排序列
    df = df.astype({
        "lat": "float32",
        "lon": "float32",
        "month": "int16",
        "dayofyear": "int16",
        "sic": "float32",
        "ice_thickness_m": "float32",
        "wave_swh": "float32",
        "ais_density": "float32",
        "vessel_class_id": "int8",
        "label_safe_risky": "int8",
    })

    # 9) 导出 Parquet
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression="snappy", index=False)

    # 10) 打印统计
    n = len(df)
    n_safe = int((df["label_safe_risky"] == 1).sum())
    n_risky = n - n_safe
    print(f"[EDL_EXPORT] rows={n}  safe={n_safe} ({n_safe/n:.1%})  risky={n_risky} ({n_risky/n:.1%})")

    for col in [
        "sic", "ice_thickness_m", "wave_swh", "ais_density",
    ]:
        vmin, vmax = _safe_minmax(df[col].values)
        print(f"[EDL_EXPORT] {col}: min={vmin:.4f}  max={vmax:.4f}")

    return df


# ---------------------------- CLI ---------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export EDL training dataset (E0.2)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="输出 parquet 路径")
    parser.add_argument("--max-samples", type=int, default=500_000, help="最大样本数")
    parser.add_argument("--sic", type=str, default=None, help="SIC nc 覆盖路径")
    parser.add_argument("--wave", type=str, default=None, help="Wave nc 覆盖路径")
    parser.add_argument("--ice", type=str, default=None, help="Ice thickness nc 覆盖路径")
    parser.add_argument("--ais", type=str, default=None, help="AIS density nc 路径")
    parser.add_argument("--time-index", type=int, default=0, help="time 维索引")
    parser.add_argument("--replicate-by-vessel", action="store_true", help="按船级复制样本（0/1/2）")

    args = parser.parse_args()

    export_edl_training_dataset(
        output_path=Path(args.output),
        max_samples=args.max_samples,
        nc_sic_path=Path(args.sic) if args.sic else None,
        nc_wave_path=Path(args.wave) if args.wave else None,
        nc_ice_thickness_path=Path(args.ice) if args.ice else None,
        nc_ais_density_path=Path(args.ais) if args.ais else None,
        time_index=args.time_index,
        replicate_by_vessel=args.replicate_by_vessel,
    )

