"""
Preprocess AIS raw data (prefer JSON) into AIS density NetCDF.

Supports demo grid (40x80) and real grid (~500x5333).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import xarray as xr

from arcticroute.core.ais_ingest import load_ais_from_raw_dir, rasterize_ais_density_to_grid
from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc

MAX_RECORDS_PER_FILE = 50_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess AIS to density NetCDF.")
    parser.add_argument(
        "--grid-mode",
        choices=["demo", "real"],
        default="demo",
        help="选择栅格模式：demo=40x80 演示网格；real=真实成本网格。",
    )
    return parser.parse_args()


def load_grid_by_mode(grid_mode: str):
    if grid_mode == "demo":
        grid, _ = make_demo_grid()
        return grid
    if grid_mode == "real":
        grid = load_real_grid_from_nc()
        if grid is None:
            raise RuntimeError("[AIS] failed to load real grid (env_clean.nc / grid_spec.nc not found)")
        return grid
    raise ValueError(f"unsupported grid_mode={grid_mode}")


def build_density_dataset(grid, df, grid_mode: str = "demo") -> xr.Dataset:
    """
    构建 AIS 密度数据集，包含坐标信息以支持后续重采样。
    
    任务 C2：添加网格元信息到 NetCDF 属性，以便后续能够验证和重采样
    """
    import numpy as np
    
    density_da = rasterize_ais_density_to_grid(
        lat_points=df["lat"].to_numpy(),
        lon_points=df["lon"].to_numpy(),
        grid_lat2d=grid.lat2d,
        grid_lon2d=grid.lon2d,
    )
    
    # 提取 1D 坐标（假设网格是规则的）
    # 尝试从 2D 网格推断 1D 坐标
    lat_1d = None
    lon_1d = None
    
    # 检查是否是规则网格（纬度在列中相同，经度在行中相同）
    if np.allclose(grid.lat2d[:, 0], grid.lat2d[:, -1]):
        # 纬度沿列相同，可以提取 1D 纬度
        lat_1d = grid.lat2d[:, 0]
    
    if np.allclose(grid.lon2d[0, :], grid.lon2d[-1, :]):
        # 经度沿行相同，可以提取 1D 经度
        lon_1d = grid.lon2d[0, :]
    
    # 如果成功提取 1D 坐标，添加到数据集
    if lat_1d is not None and lon_1d is not None:
        density_da = density_da.assign_coords({
            "latitude": (("y",), lat_1d),
            "longitude": (("x",), lon_1d),
        })
    
    # 创建数据集，包含坐标信息
    ds = xr.Dataset(
        {"ais_density": density_da},
        coords={
            "latitude": (("y",), lat_1d) if lat_1d is not None else None,
            "longitude": (("x",), lon_1d) if lon_1d is not None else None,
        }
    )
    
    # 移除 None 坐标
    ds = ds.drop_vars([v for v in ds.coords if ds.coords[v] is None], errors="ignore")
    
    # ====================================================================
    # 任务 C2：添加网格元信息到 NetCDF 属性
    # 这样后续可以根据文件属性判断是否与当前网格匹配
    # ====================================================================
    grid_shape = density_da.shape
    grid_source = "demo" if grid_mode == "demo" else "env_clean"
    
    ds.attrs['grid_shape'] = f"{grid_shape[0]}x{grid_shape[1]}"
    ds.attrs['grid_source'] = grid_source
    ds.attrs['grid_lat_name'] = 'latitude'
    ds.attrs['grid_lon_name'] = 'longitude'
    ds.attrs['description'] = f'AIS density for {grid_source} grid ({grid_shape[0]}x{grid_shape[1]})'
    
    # 为数据变量也添加属性
    ds['ais_density'].attrs['grid_shape'] = f"{grid_shape[0]}x{grid_shape[1]}"
    ds['ais_density'].attrs['grid_source'] = grid_source
    
    return ds


def main() -> None:
    args = parse_args()
    grid_mode = args.grid_mode

    root = Path("data_real/ais/raw")
    print(f"[AIS] reading from {root.resolve()}")

    df = load_ais_from_raw_dir(
        root,
        prefer_json=True,
        max_records_per_file=MAX_RECORDS_PER_FILE,
    )
    print(f"[AIS] total rows after cleaning: {len(df)}")
    if df.empty:
        return

    grid = load_grid_by_mode(grid_mode)
    ds = build_density_dataset(grid, df, grid_mode=grid_mode)
    density_da = ds["ais_density"]
    print("[AIS] density stats:", float(density_da.min()), float(density_da.max()))
    print(f"[AIS] grid shape: {density_da.shape}")

    out_dir = Path("data_real/ais/derived")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 任务 C2：使用包含网格信息的文件名
    grid_shape = density_da.shape
    if grid_mode == "demo":
        out_name = f"ais_density_2024_grid_{grid_shape[0]}x{grid_shape[1]}_demo.nc"
    else:
        out_name = f"ais_density_2024_grid_{grid_shape[0]}x{grid_shape[1]}_env_clean.nc"
    
    out_path = out_dir / out_name
    ds.to_netcdf(out_path)
    print(f"[AIS] written density to {out_path}")
    print(f"[AIS] grid metadata: shape={grid_shape}, source={ds.attrs.get('grid_source', 'unknown')}")


if __name__ == "__main__":
    main()
