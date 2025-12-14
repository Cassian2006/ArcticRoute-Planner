#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stub pipeline to mock CV mosaic outputs during development.

@role: pipeline
"""

"""Create placeholder satellite mosaics aligned to the env_nc grid."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(path: Path) -> Dict[str, Any]:
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration missing: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate placeholder satellite mosaic")
    parser.add_argument("--cfg", default="config/runtime.yaml", help="Runtime config path")
    parser.add_argument("--tidx", type=int, default=0, help="Time index to annotate")
    parser.add_argument("--mission", default="S2", help="Mission label (e.g., S1/S2)")
    args = parser.parse_args(argv)

    cfg = _load_config(Path(args.cfg))
    data_cfg = cfg.get("data") or {}
    env_token = data_cfg.get("env_nc")
    if not env_token:
        raise ValueError("config.data.env_nc missing")
    env_path = Path(env_token)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path

    sat_cache_token = data_cfg.get("sat_cache_dir", "data_processed/sat_cache")
    sat_cache_dir = Path(sat_cache_token)
    if not sat_cache_dir.is_absolute():
        sat_cache_dir = PROJECT_ROOT / sat_cache_dir
    sat_cache_dir.mkdir(parents=True, exist_ok=True)

    target_name = f"sat_mosaic_t{args.tidx}_{args.mission}.tif"
    target_path = sat_cache_dir / target_name

    from ArcticRoute.io.stac_ingest import stub_mosaic_to_grid

    stub_mosaic_to_grid(env_path, target_path, fill_value=0.5)
    print(
        "[INFO] Placeholder mosaic generated (constant 0.5). "
        "Use real STAC ingestion once the CV pipeline is enabled."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
