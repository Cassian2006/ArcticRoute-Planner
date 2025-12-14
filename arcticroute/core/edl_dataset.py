"""Helpers to build lightweight EDL training tables from existing scenarios and real env data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .cost import _normalize_ais_density_array, _regrid_ais_density_to_grid, load_ais_density_for_grid
from .env_real import load_real_env_for_grid
from .grid import Grid2D, make_demo_grid
from .eco.vessel_profiles import get_default_profiles
from .scenarios import ALLOWED_GRID_MODES, load_all_scenarios


@dataclass
class EDLSampleConfig:
    feature_columns: list[str]
    target_column: str
    sample_weight_column: str | None
    max_positive_per_scenario: int
    max_negative_per_scenario: int
    ais_density_threshold: float
    ocean_mask_min_fraction: float
    ym: str
    output_dir: Path
    filename_pattern: str


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config must be a mapping at top level: {path}")
    return payload


def _validate_list(obj: Any, name: str) -> list[str]:
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"{name} must be a non-empty list")
    vals: list[str] = []
    for item in obj:
        if not isinstance(item, str):
            raise ValueError(f"{name} entries must be strings")
        vals.append(item)
    return vals


def load_edl_dataset_config(path: Path | str = Path("configs/edl_dataset.yaml")) -> EDLSampleConfig:
    """Read YAML config and perform minimal validation."""
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"EDL dataset config not found: {cfg_path}")

    payload = _read_yaml(cfg_path)
    schema = payload.get("schema") or {}
    sampling = payload.get("sampling") or {}
    grid_cfg = payload.get("grid") or {}
    output_cfg = payload.get("output") or {}

    feature_columns = _validate_list(schema.get("feature_columns"), "schema.feature_columns")
    target_column = schema.get("target_column")
    if not isinstance(target_column, str) or not target_column:
        raise ValueError("schema.target_column must be a non-empty string")
    sample_weight_column = schema.get("sample_weight_column")
    if sample_weight_column is not None and not isinstance(sample_weight_column, str):
        raise ValueError("schema.sample_weight_column must be a string or null")

    def _ensure_int(val: Any, name: str) -> int:
        try:
            return int(val)
        except Exception as exc:
            raise ValueError(f"{name} must be an integer") from exc

    max_positive = _ensure_int(sampling.get("max_positive_per_scenario"), "sampling.max_positive_per_scenario")
    max_negative = _ensure_int(sampling.get("max_negative_per_scenario"), "sampling.max_negative_per_scenario")
    ais_density_threshold = float(sampling.get("ais_density_threshold", 0.05))
    ocean_mask_min_fraction = float(sampling.get("ocean_mask_min_fraction", 0.0))

    grid_mode = str(grid_cfg.get("mode", "auto")).lower()
    if grid_mode not in {"auto", *ALLOWED_GRID_MODES}:
        raise ValueError(f"grid.mode must be one of ['auto'] + {sorted(ALLOWED_GRID_MODES)}")
    ym = str(grid_cfg.get("ym", "")).strip()
    if not ym:
        raise ValueError("grid.ym is required")

    output_dir = Path(output_cfg.get("dir") or "data_real/edl/training")
    filename_pattern = output_cfg.get("filename_pattern") or "edl_dataset_{scenario_id}.parquet"

    return EDLSampleConfig(
        feature_columns=feature_columns,
        target_column=target_column,
        sample_weight_column=sample_weight_column,
        max_positive_per_scenario=max_positive,
        max_negative_per_scenario=max_negative,
        ais_density_threshold=ais_density_threshold,
        ocean_mask_min_fraction=ocean_mask_min_fraction,
        ym=ym,
        output_dir=output_dir,
        filename_pattern=filename_pattern,
    )


def _resolve_scenario(scenario_id: str):
    scenarios = load_all_scenarios()
    chosen_id = scenario_id
    if scenario_id not in scenarios and f"{scenario_id}_edl" in scenarios:
        chosen_id = f"{scenario_id}_edl"
    if chosen_id not in scenarios:
        raise ValueError(f"Scenario '{scenario_id}' not found in configs/scenarios.yaml")
    return scenarios[chosen_id]


def _prepare_grid(grid_mode: str, ym: str, existing_grid: Grid2D | None = None):
    if grid_mode == "demo":
        grid, land_mask = make_demo_grid()
        return grid, land_mask, None
    env = load_real_env_for_grid(grid=existing_grid, ym=ym)
    if env is not None and env.grid is not None:
        land_mask = env.land_mask if env.land_mask is not None else np.zeros_like(env.grid.lat2d, dtype=bool)
        return env.grid, land_mask, env
    grid, land_mask = make_demo_grid()
    return grid, land_mask, None


def _align_ais_density(grid: Grid2D, prefer_real: bool) -> np.ndarray:
    ais_da = load_ais_density_for_grid(grid, prefer_real=prefer_real)
    if ais_da is None:
        return np.zeros(grid.shape(), dtype=float)
    aligned = None
    try:
        if hasattr(ais_da, "shape") and ais_da.shape == grid.shape():
            aligned = np.asarray(getattr(ais_da, "values", ais_da), dtype=float)
        else:
            aligned = _regrid_ais_density_to_grid(ais_da, grid)
    except Exception:
        aligned = None
    if aligned is None:
        return np.zeros(grid.shape(), dtype=float)
    return _normalize_ais_density_array(aligned)


def _build_frame(
    indices: np.ndarray,
    grid: Grid2D,
    month: int,
    sic: np.ndarray | None,
    wave: np.ndarray | None,
    ice_thickness: np.ndarray | None,
    ais_density: np.ndarray,
    vessel_dwt: float | None,
    vessel_max_ice: float | None,
    label: int,
    sample_weight_column: str | None,
    target_column: str,
) -> pd.DataFrame:
    lat_flat = grid.lat2d.ravel()[indices]
    lon_flat = grid.lon2d.ravel()[indices]
    sic_flat = sic.ravel()[indices] if sic is not None else np.full_like(lat_flat, np.nan, dtype=float)
    wave_flat = wave.ravel()[indices] if wave is not None else np.full_like(lat_flat, np.nan, dtype=float)
    ice_flat = (
        ice_thickness.ravel()[indices] if ice_thickness is not None else np.full_like(lat_flat, np.nan, dtype=float)
    )
    ais_flat = ais_density.ravel()[indices]

    data: dict[str, Any] = {
        "lat": lat_flat,
        "lon": lon_flat,
        "month": np.full_like(lat_flat, month, dtype=int),
        "sic": sic_flat,
        "wave_swh": wave_flat,
        "ice_thickness": ice_flat,
        "ais_density": ais_flat,
        "vessel_dwt": np.full_like(lat_flat, vessel_dwt if vessel_dwt is not None else np.nan, dtype=float),
        "vessel_max_ice_thickness": np.full_like(
            lat_flat, vessel_max_ice if vessel_max_ice is not None else np.nan, dtype=float
        ),
        target_column: np.full_like(lat_flat, label, dtype=int),
    }

    if sample_weight_column:
        data[sample_weight_column] = np.ones_like(lat_flat, dtype=float)

    return pd.DataFrame(data)


def build_edl_training_table(
    scenario_id: str,
    cfg: EDLSampleConfig,
    grid_mode: str = "auto",
) -> pd.DataFrame:
    """
    Construct a grid-level table for EDL training and write it to parquet.
    """
    scenario = _resolve_scenario(scenario_id)
    chosen_grid_mode = scenario.grid_mode if grid_mode == "auto" else str(grid_mode).lower()
    if chosen_grid_mode not in ALLOWED_GRID_MODES:
        raise ValueError(f"grid_mode must be one of {sorted(ALLOWED_GRID_MODES)} or 'auto'")

    # Prefer scenario YM; fall back to config YM if absent.
    ym = scenario.ym or cfg.ym
    try:
        month = int(str(ym)[4:6])
    except Exception:
        month = 1

    profiles = get_default_profiles()
    vessel = profiles.get(scenario.vessel)
    vessel_dwt = vessel.dwt if vessel is not None else None
    vessel_max_ice = vessel.get_effective_max_ice_thickness() if vessel is not None else None

    grid, land_mask, env = _prepare_grid(chosen_grid_mode, ym, None)

    sic = env.sic if env is not None else None
    wave = env.wave_swh if env is not None else None
    ice_thickness = env.ice_thickness_m if env is not None else None

    # Ensure land_mask exists
    if land_mask is None:
        land_mask = np.zeros(grid.shape(), dtype=bool)
    ocean_mask = ~land_mask

    ais_density = _align_ais_density(grid, prefer_real=chosen_grid_mode == "real")

    ocean_fraction = ocean_mask.astype(float).ravel()
    valid_ocean = ocean_fraction >= cfg.ocean_mask_min_fraction
    # For single-cell sampling, ocean_mask_min_fraction reduces to requiring ocean cells only.
    positive_mask = (ais_density.ravel() > cfg.ais_density_threshold) & valid_ocean & np.isfinite(ais_density.ravel())
    negative_mask = (ais_density.ravel() == 0) & valid_ocean & np.isfinite(ais_density.ravel())

    pos_indices = np.where(positive_mask)[0]
    neg_indices = np.where(negative_mask)[0]

    if pos_indices.size > cfg.max_positive_per_scenario:
        pos_indices = np.random.default_rng(42).choice(pos_indices, cfg.max_positive_per_scenario, replace=False)
    if neg_indices.size > cfg.max_negative_per_scenario:
        neg_indices = np.random.default_rng(42).choice(neg_indices, cfg.max_negative_per_scenario, replace=False)

    pos_df = _build_frame(
        pos_indices,
        grid,
        month,
        sic,
        wave,
        ice_thickness,
        ais_density,
        vessel_dwt,
        vessel_max_ice,
        label=1,
        sample_weight_column=cfg.sample_weight_column,
        target_column=cfg.target_column,
    )
    neg_df = _build_frame(
        neg_indices,
        grid,
        month,
        sic,
        wave,
        ice_thickness,
        ais_density,
        vessel_dwt,
        vessel_max_ice,
        label=0,
        sample_weight_column=cfg.sample_weight_column,
        target_column=cfg.target_column,
    )

    df = pd.concat([pos_df, neg_df], ignore_index=True)
    if not df.empty:
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    output_dir = (cfg.output_dir or Path("data_real/edl/training")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / cfg.filename_pattern.format(scenario_id=scenario_id)
    df.to_parquet(out_path, index=False)
    return df
