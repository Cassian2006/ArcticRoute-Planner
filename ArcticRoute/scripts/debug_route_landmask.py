"""
Visual sanity check: plot cost/land mask and the A* route in grid (i,j) space.

Usage:
    python ArcticRoute/scripts/debug_route_landmask.py

This is a developer-only diagnostic script; it does NOT affect Planner or core APIs.
"""

from __future__ import annotations

import sys
from pathlib import Path

# -- Ensure repo root (the parent of 'ArcticRoute') is on sys.path --
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
# -- End of path setup --

import numpy as np
import matplotlib.pyplot as plt

from ArcticRoute.core import planner_service

try:
    import xarray as xr  # optional, only used for type checks
except Exception:  # pragma: no cover
    xr = None  # type: ignore

# ------------------------
# Demo parameters (align with example scenario)
# ------------------------
YM = "202412"
SCENARIO = "default"  # placeholder, not used in this debug script

START_IJ = (60, 150)
GOAL_IJ = (60, 1000)

W_ICE = 1.0
W_ACCIDENT = 0.0
PRIOR_WEIGHT = 0.0
ALLOW_DIAGONAL = True
HEURISTIC = "euclidean"  # or "manhattan"


def get_authoritative_land_mask(env):
    """
    仅使用 env.land_mask_da（约定：1=land, 0=ocean）。
    若为 None，返回 None，不做任何启发式/回退。
    """
    lm_da = getattr(env, "land_mask_da", None)
    if lm_da is None:
        return None
    lm_vals = lm_da.values if hasattr(lm_da, "values") else np.asarray(lm_da)
    lm01 = np.zeros_like(lm_vals, dtype=np.uint8)
    try:
        finite = np.isfinite(lm_vals)
        thr = 0.5
        lm01[finite] = (lm_vals[finite] > thr).astype(np.uint8)
        lm01[~finite] = 1  # 非有限视为陆地
    except Exception:
        lm01 = (np.asarray(lm_vals).astype(float) > 0.5).astype(np.uint8)
    return lm01


def main():
    # 1) load env
    env_ctx = planner_service.load_environment(
        ym=YM,
        w_ice=W_ICE,
        w_accident=W_ACCIDENT,
        prior_weight=PRIOR_WEIGHT,
    )

    # 2) authoritative land mask
    lm = get_authoritative_land_mask(env_ctx)
    if lm is None:
        print("[LMASK] env_ctx.land_mask_da is None; nothing to debug")
        sys.exit(0)

    # 3) compute route (to evaluate land crossings)
    route_result = planner_service.compute_route(
        env_ctx,
        start_ij=START_IJ,
        goal_ij=GOAL_IJ,
        allow_diagonal=ALLOW_DIAGONAL,
        heuristic=HEURISTIC,
    )

    if not getattr(route_result, "reachable", True):
        print("[INFO] Route not reachable. Skipping mask stats and plot.")
        return

    path_ij = route_result.path_ij

    # mask stats
    try:
        finite = np.isfinite(lm)
        vals = lm[finite]
        vmin = float(vals.min()) if vals.size else float("nan")
        vmax = float(vals.max()) if vals.size else float("nan")
        n0 = int((lm == 0).sum())
        n1 = int((lm == 1).sum())
        tot = int(lm.size)
        print(f"[LMASK] stats: min={vmin:.3f}, max={vmax:.3f}, zeros(ocean)={n0}, ones(land)={n1}, total={tot}")
    except Exception:
        pass

    # route-land stats (1=land)
    h, w = lm.shape
    violations = sum(1 for (i, j) in path_ij if 0 <= i < h and 0 <= j < w and lm[i, j] == 1)
    ratio = (violations / len(path_ij)) if path_ij else 0.0
    print(f"[ROUTE_LAND] total_steps={len(path_ij)}, land_hits={violations}, ratio={ratio:.4f}")

    # plot
    rows = [p[0] for p in path_ij]
    cols = [p[1] for p in path_ij]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(1 - lm, origin="lower", cmap="gray")  # white=sea(1), black=land(0)
    if rows and cols:
        ax.plot(cols, rows, color="red", linewidth=1.5, label="route (i,j)")
        ax.legend()

    ax.set_title("Land mask (1=land,0=ocean) with route overlay in (i,j) grid")
    ax.set_xlabel("j (x index)")
    ax.set_ylabel("i (y index)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

