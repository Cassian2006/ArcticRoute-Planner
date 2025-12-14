#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick health check for core datasets and minimal route computation.

@role: core
"""

"""
ArcticRoute 快速健康检查：
- 打印 Python / xarray / numpy 版本信息
- 读取 runtime 配置并校验风险场 / 走廊 / 事故数据是否就绪
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

import numpy as np
import xarray as xr
import yaml

from ArcticRoute.config.schema import (
    ValidationError as SchemaValidationError,
    format_validation_error,
    model_dump,
    validate_runtime_config,
)


def _load_config(project_root: Path, cfg_path: Path) -> tuple[dict, Path]:
    if not cfg_path.is_absolute():
        cfg_path = project_root / cfg_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    try:
        cfg_model = validate_runtime_config(raw)
    except SchemaValidationError as err:
        raise ValueError(format_validation_error(err, str(cfg_path))) from err
    return model_dump(cfg_model), cfg_path


def _check_env(env_path: Path) -> None:
    if not env_path.exists():
        raise FileNotFoundError(f"Missing risk field file: {env_path}")
    with xr.open_dataset(env_path) as ds:
        if "risk_env" not in ds:
            raise KeyError("risk_env variable missing in risk dataset")
        sample = ds["risk_env"].isel(time=0)
        print(f"[OK] Risk slice shape={sample.shape}, dtype={sample.dtype}")


def _check_corridor(project_root: Path, path_token: str) -> None:
    path = Path(path_token)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"缺少走廊文件: {path}")
    with xr.open_dataset(path) as ds:
        if "corridor_prob" not in ds:
            raise KeyError("走廊文件缺少 corridor_prob 变量")
        da = ds["corridor_prob"]
        sample = da.isel(time=0) if "time" in da.dims else da
        mean_val = float(np.nanmean(sample.values))
        print(f"[OK] Corridor grid shape={sample.shape}, mean={mean_val:.3f}")


def _check_accident(project_root: Path, path_token: str) -> None:
    path = Path(path_token)
    if not path.is_absolute():
        path = project_root / path
    if not path.exists():
        raise FileNotFoundError(f"缺少事故文件: {path}")
    with xr.open_dataset(path) as ds:
        if "accident_density" not in ds:
            raise KeyError("事故文件缺少 accident_density 变量")
        da = ds["accident_density"]
        sample = da.isel(time=0) if "time" in da.dims else da
        max_val = float(np.nanmax(sample.values))
        print(f"[OK] Accident grid shape={sample.shape}, max={max_val:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="ArcticRoute quick healthcheck")
    parser.add_argument("--cfg", default="config/runtime.yaml", help="运行配置 YAML 路径")
    args = parser.parse_args()

    project_root = PROJECT_ROOT

    print(f"Python: {sys.version.split()[0]}")
    print(f"xarray: {xr.__version__}")
    print(f"numpy: {np.__version__}")

    try:
        cfg, cfg_path = _load_config(project_root, Path(args.cfg))
        print(f"[OK] Loaded config: {cfg_path}")

        env_path = Path(cfg["data"]["env_nc"])
        if not env_path.is_absolute():
            env_path = project_root / env_path
        _check_env(env_path)

        corridor_path = cfg["behavior"].get("corridor_path")
        if corridor_path:
            _check_corridor(project_root, corridor_path)
        else:
            print("[INFO] corridor_path not set; skipping corridor check")

        accident_path = cfg["behavior"].get("accident_path")
        if accident_path:
            _check_accident(project_root, accident_path)
        else:
            print("[INFO] accident_path not set; skipping accident check")
        cv_ok = False
        try:
            from ArcticRoute.core.predictors.cv_sat import SatCVPredictor
            from ArcticRoute.core.predictors.dl_ice import DLIcePredictor

            run_cfg = cfg.get("run", {})
            var_name = run_cfg.get("var", "risk_env")
            tidx = int(run_cfg.get("tidx", 0))
            sat_ds = SatCVPredictor(env_path, var_name).prepare(tidx)
            ice_ds = DLIcePredictor(env_path, var_name).prepare(tidx)
            cv_ok = "sat_dummy" in sat_ds and "ice_prob" in ice_ds
        except Exception as err:
            print(f"[WARN] CV placeholder predictor check failed: {err}")
            cv_ok = False
        print(f"CV 占位预测器可用：{'OK' if cv_ok else 'FAIL'}")

    except Exception as err:
        print(f"[ERR] Health check failed: {err}")
        return 1

    print("[OK] Health check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
