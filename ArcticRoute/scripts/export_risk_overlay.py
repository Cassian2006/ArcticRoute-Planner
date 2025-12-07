#!/usr/bin/env python3
"""Export composite risk overlays (e.g., for map viewers or docs).

@role: pipeline
"""

"""
导出 risk_env 底图 PNG + 边界，供前端叠加。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV = PROJECT_ROOT / "data_processed" / "env_clean.nc"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "overlays"


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (PROJECT_ROOT.parent / path).resolve()


def export_overlay(env_path: Path, tidx: int, output_dir: Path) -> None:
    ds = xr.open_dataset(env_path)
    try:
        if "risk_env" not in ds:
            raise KeyError("risk_env variable missing")
        risk = ds["risk_env"]
        if "time" in risk.dims:
            if tidx < 0 or tidx >= risk.sizes["time"]:
                raise IndexError(f"tidx {tidx} out of range (0..{risk.sizes['time'] - 1})")
            slice_da = risk.isel(time=tidx)
        else:
            slice_da = risk

        data = slice_da.values.astype("float32")
        finite = np.isfinite(data)
        if not finite.any():
            raise ValueError("risk slice contains no finite values")
        data = np.clip(data, 0.0, 1.0)

        lats = slice_da.coords["latitude"].values
        lons = slice_da.coords["longitude"].values
    finally:
        ds.close()

    lat_min = float(np.min(lats))
    lat_max = float(np.max(lats))
    lon_min = float(np.min(lons))
    lon_max = float(np.max(lons))
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]

    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"risk_t{tidx}.png"
    json_path = output_dir / f"risk_t{tidx}_bounds.json"

    plt.figure(figsize=(10, 6), dpi=200)
    plt.axis("off")
    plt.imshow(data, cmap="viridis", origin="lower", alpha=0.4)
    plt.tight_layout(pad=0)
    plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()

    json_path.write_text(json.dumps({"bounds": bounds}), encoding="utf-8")

    print(
        "[导出完成] "
        f"shape={data.shape}, "
        f"bounds={bounds}, "
        f"min={float(np.min(data)):.3f}, "
        f"max={float(np.max(data)):.3f}, "
        f"mean={float(np.mean(data)):.3f}, "
        f"png={png_path}, json={json_path}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Export risk_env overlay PNG/bounds")
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV, help="env_clean.nc 路径")
    parser.add_argument("--tidx", type=int, default=0, help="时间索引")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR, help="输出目录")
    args = parser.parse_args()

    env_path = resolve_path(args.env)
    output_dir = resolve_path(args.output_dir)
    export_overlay(env_path, args.tidx, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
