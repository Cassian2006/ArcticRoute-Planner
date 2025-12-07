#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick visualization helpers for environment/risk grids.

@role: analysis
"""

# 用法：
#   python scripts/visualize_env.py --in data_processed/env_clean.nc --var risk_env --tidx 0 --out docs/risk_env_t0.png
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def main(args):
    ds = xr.open_dataset(args.infile)
    if args.var not in ds:
        raise RuntimeError(f"变量 {args.var} 不在文件中，可选：{list(ds.data_vars)}")
    da = ds[args.var].isel(time=args.tidx)

    fig, ax = plt.subplots(figsize=(8, 6))
    lon = ds.longitude.values
    lat = ds.latitude.values
    mesh = ax.pcolormesh(lon, lat, np.squeeze(da.values), shading="auto")
    plt.colorbar(mesh, ax=ax, label=args.var)
    ax.set_title(f"{args.var} at time index {args.tidx}")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    Path(args.outfile).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    plt.savefig(args.outfile, dpi=150)
    print(f"✅ 已保存图片：{args.outfile}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--var", dest="var", default="risk_env")
    ap.add_argument("--tidx", dest="tidx", type=int, default=0)
    ap.add_argument("--out", dest="outfile", default="docs/preview.png")
    main(ap.parse_args())
