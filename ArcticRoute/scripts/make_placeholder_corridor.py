# -*- coding: utf-8 -*-
"""Generate a placeholder corridor layer for illustrative experiments.

@role: analysis
"""

"""Generate a placeholder corridor_prob.nc aligned with env_clean.nc."""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Iterable

import numpy as np


def _write_with_xarray(env_path: pathlib.Path, output_path: pathlib.Path) -> bool:
    try:
        import xarray as xr  # type: ignore
    except Exception:
        return False

    with xr.open_dataset(env_path) as ds:
        data_vars: Iterable[str] = list(ds.data_vars)
        if not data_vars:
            raise RuntimeError("env dataset has no data variables to infer dimensions from")
        key = data_vars[0]
        zeros = xr.zeros_like(ds[key]).astype("float32")
        ds_out = zeros.to_dataset(name="corridor_prob")
        ds_out.to_netcdf(output_path)
    return True


def _write_with_netcdf4(env_path: pathlib.Path, output_path: pathlib.Path) -> None:
    try:
        from netCDF4 import Dataset  # type: ignore
    except Exception as exc:  # pragma: no cover - fallback path
        raise RuntimeError("xarray/netCDF4 unavailable, cannot create placeholder corridor") from exc

    with Dataset(env_path, "r") as src, Dataset(output_path, "w") as dst:
        for name, dim in src.dimensions.items():
            dst.createDimension(name, len(dim) if not dim.isunlimited() else None)

        for name, var in src.variables.items():
            if len(var.dimensions) == 1:
                out_var = dst.createVariable(name, var.datatype, var.dimensions)
                out_var.setncatts({attr: var.getncattr(attr) for attr in var.ncattrs()})
                out_var[:] = var[:]

        template = None
        for name, var in src.variables.items():
            if len(var.dimensions) >= 1 and name not in dst.variables:
                template = var
                break

        if template is None:
            raise RuntimeError("Unable to infer corridor dimensions from env_clean.nc")

        corr = dst.createVariable("corridor_prob", "f4", template.dimensions, zlib=False)
        corr.setncatts(
            {
                "long_name": "Corridor probability (placeholder)",
                "description": "Auto-generated zero corridor for AI demo self-heal",
            }
        )
        corr[:] = np.zeros(template.shape, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create placeholder corridor_prob.nc")
    parser.add_argument("--env-path", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    env_path = args.env_path.resolve()
    output_path = args.output.resolve()
    if not env_path.exists():
        raise FileNotFoundError(f"env file not found: {env_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _write_with_xarray(env_path, output_path):
        _write_with_netcdf4(env_path, output_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - runtime safeguard
        print(f"[placeholder] failed: {exc}", file=sys.stderr)
        sys.exit(1)
