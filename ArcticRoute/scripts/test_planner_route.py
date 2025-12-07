"""Quick smoke test for Planner: runs a single A* route and prints a summary.

@role: core
"""

"""
Minimal test script for planner_service.compute_route() to allow reproducible debugging
without the Streamlit UI.
"""

import sys
from pathlib import Path

# -- Add project root to sys.path --
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# -- End of path setup --

import numpy as np
from ArcticRoute.core import planner_service

def run_test_case():
    """Defines and runs a single, representative test case for route planning."""
    print("--- Running Planner Service Test Case ---")

    # 1. Define test parameters
    ym = "202412"
    start_ij = (60, 150)
    end_ij = (60, 1000)

    # 2. Load environment
    print("\n[Step 1] Loading environment...")
    try:
        env_ctx = planner_service.load_environment(ym=ym, w_ice=0.7, w_accident=0.2)
        if env_ctx.cost_da is None:
            raise ValueError("Cost data is None.")
        print("  -> Success. Environment loaded.")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Compute route
    print("\n[Step 2] Computing route...")
    try:
        route_result = planner_service.compute_route(env_ctx, start_ij, end_ij, True, "manhattan")
        print("  -> Success. Computation finished.")
    except Exception as e:
        print(f"  -> FAILED: {e}")
        import traceback
        traceback.print_exc()
        return

    # 4. Print summary + ECO compare
    print("\n[Step 3] Route Result Summary")
    if route_result and route_result.reachable:
        summary = planner_service.summarize_route(route_result)
        print(f"[ROUTE] steps={summary['steps']}, distance_km={summary['distance_km']}, cost_sum={summary['cost_sum']}")

        # Eco OFF (simple estimate)
        eco_simple = planner_service.estimate_eco_simple(route_result)
        print(f"[ECO simple] fuel={eco_simple.fuel_total_t} t, co2={eco_simple.co2_total_t} t")

        # Eco ON (model evaluation)
        eco_model = planner_service.evaluate_route_eco(route_result, env_ctx)
        mode_tag = "model" if eco_model.details.get("ok", False) else "fallback_simple"
        if not eco_model.details.get("ok", False):
            reason = eco_model.details.get("reason") or eco_model.details.get("error", "")
            print(f"[INFO] evaluate_route_eco 未生效，已回退到 estimate_eco_simple: reason={reason}")
        print(f"[ECO model]  fuel={eco_model.fuel_total_t} t, co2={eco_model.co2_total_t} t, mode={mode_tag}")

        # Delta
        dfuel = round(eco_model.fuel_total_t - eco_simple.fuel_total_t, 2)
        dco2 = round(eco_model.co2_total_t - eco_simple.co2_total_t, 2)
        print(f"[Δ] fuel_diff={dfuel} t, co2_diff={dco2} t")
    else:
        print("  - Status: Path NOT FOUND.")

    print("\n--- Test Case Finished ---")

if __name__ == "__main__":
    run_test_case()
