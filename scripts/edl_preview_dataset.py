from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from arcticroute.core.edl_dataset import build_edl_training_table, load_edl_dataset_config


def _describe_feature(df: pd.DataFrame, col: str) -> str:
    series = pd.to_numeric(df[col], errors="coerce")
    finite = series.replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return f"{col}: no finite values"
    return f"{col}: min={finite.min():.4f}, max={finite.max():.4f}, mean={finite.mean():.4f}"


def preview_dataset(df: pd.DataFrame, target_column: str) -> None:
    total_rows, total_cols = df.shape
    print(f"[EDL_PREVIEW] rows={total_rows}, cols={total_cols}")

    counts = df[target_column].value_counts()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    total = pos + neg if (pos + neg) > 0 else 1
    print(f"[EDL_PREVIEW] positives={pos} ({pos/total:.2%}), negatives={neg} ({neg/total:.2%})")

    feature_candidates = [
        "sic",
        "wave_swh",
        "ice_thickness",
        "ais_density",
        "vessel_dwt",
        "vessel_max_ice_thickness",
    ]
    for col in feature_candidates:
        if col in df.columns:
            print("  - " + _describe_feature(df, col))


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview EDL training dataset for a scenario.")
    parser.add_argument("--scenario", required=True, help="Scenario ID, e.g., barents_to_chukchi")
    parser.add_argument("--grid-mode", default="auto", help="Override grid mode (auto/demo/real)")
    parser.add_argument("--reuse", action="store_true", help="Reuse existing parquet if present")
    args = parser.parse_args()

    cfg = load_edl_dataset_config()
    scenario_id = args.scenario
    out_path = Path(cfg.output_dir) / cfg.filename_pattern.format(scenario_id=scenario_id)

    if args.reuse and out_path.exists():
        print(f"[EDL_PREVIEW] Reusing dataset at {out_path}")
        df = pd.read_parquet(out_path)
    else:
        df = build_edl_training_table(scenario_id, cfg, grid_mode=args.grid_mode)
        print(f"[EDL_PREVIEW] Built dataset at {out_path}")

    preview_dataset(df, cfg.target_column)


if __name__ == "__main__":
    main()
