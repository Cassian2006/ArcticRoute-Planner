"""Multi-scenario evaluation script for comparing EDL modes.

This script compares the performance of efficient, edl_safe, and edl_robust modes
across multiple scenarios. It computes delta metrics (distance, cost, risk reduction)
and outputs both a CSV report and a terminal summary.

Typical usage:
    # 1. Run scenario suite first (if not already done)
    python -m scripts.run_scenario_suite

    # 2. Evaluate mode comparison
    python -m scripts.eval_scenario_results \\
        --input reports/scenario_suite_results.csv \\
        --output reports/eval_mode_comparison.csv

    # 3. Terminal will print per-scenario comparison and global summary
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate and compare EDL modes across scenarios."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("reports") / "scenario_suite_results.csv",
        help="Input CSV with scenario results (default: reports/scenario_suite_results.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "eval_mode_comparison.csv",
        help="Output CSV for mode comparison (default: reports/eval_mode_comparison.csv)",
    )
    parser.add_argument(
        "--pretty-print",
        dest="pretty_print",
        action="store_true",
        default=True,
        help="Print aligned text table to terminal (default: True)",
    )
    parser.add_argument(
        "--no-pretty-print",
        dest="pretty_print",
        action="store_false",
        help="Disable pretty printing to terminal",
    )
    return parser.parse_args()


def evaluate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate and compare EDL modes against efficient baseline.

    Args:
        df: Input DataFrame with columns:
            - scenario_id
            - mode (efficient, edl_safe, edl_robust)
            - reachable (bool or 0/1)
            - distance_km
            - total_cost
            - edl_risk_cost (or similar)
            - edl_uncertainty_cost (or similar)

    Returns:
        DataFrame with comparison metrics for edl_safe and edl_robust vs efficient.
    """
    # Ensure reachable is boolean
    df["reachable"] = df["reachable"].astype(bool)

    # Rename edl cost columns if they exist with different names
    if "edl_risk_cost" not in df.columns and "edl_risk" in df.columns:
        df = df.rename(columns={"edl_risk": "edl_risk_cost"})
    if "edl_uncertainty_cost" not in df.columns and "edl_uncertainty" in df.columns:
        df = df.rename(columns={"edl_uncertainty": "edl_uncertainty_cost"})

    # Ensure these columns exist with default 0 if missing
    for col in ["edl_risk_cost", "edl_uncertainty_cost"]:
        if col not in df.columns:
            df[col] = 0.0

    eval_rows = []
    scenarios = df["scenario_id"].unique()

    for scenario_id in scenarios:
        scen_df = df[df["scenario_id"] == scenario_id]

        # Filter to reachable routes only
        reachable_df = scen_df[scen_df["reachable"] == True]

        if len(reachable_df) == 0:
            logger.warning(f"Scenario '{scenario_id}': no reachable routes found")
            continue

        # Get efficient baseline
        eff_rows = reachable_df[reachable_df["mode"] == "efficient"]
        if len(eff_rows) == 0:
            logger.warning(
                f"Scenario '{scenario_id}': no 'efficient' mode found, skipping"
            )
            continue

        eff_row = eff_rows.iloc[0]
        eff_dist = float(eff_row["distance_km"])
        eff_cost = float(eff_row["total_cost"])
        eff_risk = float(eff_row.get("edl_risk_cost", 0.0))
        eff_unc = float(eff_row.get("edl_uncertainty_cost", 0.0))

        # Avoid division by zero
        if eff_dist < 1e-6:
            logger.warning(
                f"Scenario '{scenario_id}': efficient distance is ~0, skipping"
            )
            continue

        # Compare edl_safe and edl_robust
        for mode in ["edl_safe", "edl_robust"]:
            mode_rows = reachable_df[reachable_df["mode"] == mode]
            if len(mode_rows) == 0:
                continue

            mode_row = mode_rows.iloc[0]
            mode_dist = float(mode_row["distance_km"])
            mode_cost = float(mode_row["total_cost"])
            mode_risk = float(mode_row.get("edl_risk_cost", 0.0))
            mode_unc = float(mode_row.get("edl_uncertainty_cost", 0.0))

            # Compute deltas
            delta_dist_km = mode_dist - eff_dist
            rel_dist_pct = 100.0 * delta_dist_km / eff_dist if eff_dist > 0 else 0.0

            delta_cost = mode_cost - eff_cost
            rel_cost_pct = 100.0 * delta_cost / eff_cost if eff_cost > 0 else 0.0

            delta_edl_risk = mode_risk - eff_risk

            # Risk reduction percentage
            if eff_risk > 1e-6:
                risk_reduction_pct = 100.0 * (eff_risk - mode_risk) / eff_risk
            else:
                risk_reduction_pct = np.nan

            delta_edl_unc = mode_unc - eff_unc

            eval_rows.append(
                {
                    "scenario_id": scenario_id,
                    "mode": mode,
                    "delta_dist_km": delta_dist_km,
                    "rel_dist_pct": rel_dist_pct,
                    "delta_cost": delta_cost,
                    "rel_cost_pct": rel_cost_pct,
                    "delta_edl_risk": delta_edl_risk,
                    "risk_reduction_pct": risk_reduction_pct,
                    "delta_edl_unc": delta_edl_unc,
                }
            )

    eval_df = pd.DataFrame(eval_rows)
    return eval_df


def print_pretty_summary(eval_df: pd.DataFrame) -> None:
    """Print aligned text table of evaluation results."""
    if eval_df.empty:
        print("[INFO] No evaluation results to display.")
        return

    print("\n" + "=" * 100)
    print("SCENARIO-BY-SCENARIO COMPARISON")
    print("=" * 100)

    scenarios = sorted(eval_df["scenario_id"].unique())
    for scenario_id in scenarios:
        scen_df = eval_df[eval_df["scenario_id"] == scenario_id]
        print(f"\n[{scenario_id}]")
        print(
            f"{'Mode':<12} {'Δdist(km)':>12} {'Δdist(%)':>10} {'Δcost':>10} "
            f"{'Δcost(%)':>10} {'risk_red(%)':>12}"
        )
        print("-" * 80)
        for _, row in scen_df.iterrows():
            mode = row["mode"]
            delta_dist = row["delta_dist_km"]
            rel_dist = row["rel_dist_pct"]
            delta_cost = row["delta_cost"]
            rel_cost = row["rel_cost_pct"]
            risk_red = row["risk_reduction_pct"]

            risk_red_str = (
                f"{risk_red:>12.2f}" if pd.notna(risk_red) else f"{'NaN':>12}"
            )
            print(
                f"{mode:<12} {delta_dist:>12.2f} {rel_dist:>10.2f} {delta_cost:>10.2f} "
                f"{rel_cost:>10.2f} {risk_red_str}"
            )

    # Global summary
    print("\n" + "=" * 100)
    print("GLOBAL SUMMARY")
    print("=" * 100)

    for mode in ["edl_safe", "edl_robust"]:
        mode_df = eval_df[eval_df["mode"] == mode]
        if mode_df.empty:
            continue

        # Only consider rows where risk_reduction_pct is not NaN
        risk_valid = mode_df[pd.notna(mode_df["risk_reduction_pct"])]

        if len(risk_valid) > 0:
            avg_risk_red = risk_valid["risk_reduction_pct"].mean()
            avg_rel_dist = mode_df["rel_dist_pct"].mean()

            count_better_risk = (risk_valid["risk_reduction_pct"] > 0).sum()
            count_better_risk_small_detour = (
                (risk_valid["risk_reduction_pct"] > 0)
                & (mode_df["rel_dist_pct"] <= 5.0)
            ).sum()

            print(f"\n{mode.upper()}:")
            print(f"  Avg risk reduction:          {avg_risk_red:>8.2f}%")
            print(f"  Avg distance increase:       {avg_rel_dist:>8.2f}%")
            print(f"  Scenarios with better risk:  {count_better_risk:>8d}")
            print(
                f"  Better risk + small detour:  {count_better_risk_small_detour:>8d}"
            )


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Load input CSV
    input_path: Path = args.input
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)

    logger.info(f"Loading input from {input_path}")
    df = pd.read_csv(input_path)

    # Validate required columns
    required_cols = ["scenario_id", "mode", "reachable", "distance_km", "total_cost"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        sys.exit(1)

    # Run evaluation
    logger.info("Evaluating scenarios...")
    eval_df = evaluate(df)

    if eval_df.empty:
        logger.warning("No evaluation results generated.")
        sys.exit(0)

    # Write output CSV
    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    eval_df.to_csv(output_path, index=False)
    logger.info(f"Wrote {len(eval_df)} rows to {output_path}")

    # Print pretty summary
    if args.pretty_print:
        print_pretty_summary(eval_df)

    print(f"\n[INFO] Evaluation complete.")


if __name__ == "__main__":
    main()

