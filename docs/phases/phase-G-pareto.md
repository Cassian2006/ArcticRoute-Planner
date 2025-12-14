# Phase G | Multi‑Objective Routing — Weight Scan & Pareto Selection

## 0) Context & Goal
- **Context.** We have a working A* planner (`core/planners/astar_grid_time.py`) and fused risk layers via `core/risk/fusion.py`. We now need systematic, reproducible **weight scanning** and **Pareto selection** across objectives:
  - Total risk integral
  - Total distance
  - (Optional) Congestion/interaction penalty and PriorPenalty
- **Goal.** Provide a CLI + small library layer to:
  1) Run **grid scans** over route weights and (optionally) risk aggregation mode
  2) Compute route‑level metrics
  3) Extract a **Pareto front** and **three representative routes**: _Safe_, _Efficient_, _Balanced_
  4) Emit JSON + HTML report, and GeoJSON/PNG exports for selected candidates  
  **No breaking changes** to existing inputs/outputs.

---

## 1) Scope / Non‑Goals
- **In scope**
  - Offline grid scan over `{w_r, w_d, w_p, w_c}` (and optional `risk_agg`, `alpha`)
  - Pareto non‑dominated sorting & representative selection
  - Minimal UI/CLI wiring and report generation
- **Out of scope**
  - Replacing A* or changing its local step cost model
  - Real‑time multi‑ship deconfliction (handled in Phase F/G+ if needed)

---

## 2) Artifacts & Contracts
- **Inputs**
  - `data_processed/risk/risk_fused_<ym>.nc` (var: `risk ∈ [0,1]`)
  - `data_processed/prior/prior_corridor_selected_<ym>.nc` (var: `prior_penalty ∈ [0,1]`)  
    _Fallback_: `prior_transformer_<ym>.nc` + compute `prior_penalty = 1.0 - P_prior`
  - (Optional) `data_processed/risk/R_interact_<ym>.nc` / `data_processed/features/ais_density_<ym>.nc`
- **Outputs (per scenario)**
  - `data_processed/routes/route_<ym>_<scenario>_<label>.geojson`  
    where `<label> ∈ {safe, balanced, efficient}` (or `candNN`)
  - `reports/d_stage/phaseG/pareto_front_<ym>_<scenario>.json`  
    (`points[]`: metrics + weights + path ids)
  - `reports/d_stage/phaseG/summary_<ym>_<scenario>.json`  
    (chosen representatives + metrics + references)
  - `reports/d_stage/phaseG/pareto_<ym>_<scenario>.html`  
    (scatter/parallel plots; comparison table; quick download links)
- **CLI**
  - `route.scan` (Alpha) — run scans and compute Pareto
  - `route` (Alpha) — single‑run (unchanged, if present)
  - `report.build` (Alpha) — render HTML report (Pareto + comparisons)
- **UI** (optional in this phase)
  - Add a simple “Compare routes (Pareto)” tab to load JSON and display 3 reps

---

## 3) Data & Interface Alignment
- **Planner:** `core/planners/astar_grid_time.py :: AStarGridTimePlanner.plan(...)`
- **Cost provider:** `core/cost/env_risk_cost.py :: compute(...)`  
  The scan passes weights to this provider; no change to the A* inner loop.
- **Scenarios:** use `config/scenarios.yaml` (already present).  
  Minimal schema (extend if needed):

```yaml
scenarios:
  - id: "nsr_wbound"
    ym: "202412"
    start: [69.0, 33.0]    # [lat, lon]
    goal: [70.5, 170.0]
    dep_time: "2024-12-10T00:00:00Z"  # optional
    grid_id: "EPSG:3413"
    weights:
      presets:
        safe:      { w_r: 1.0, w_d: 0.25, w_p: 0.5, w_c: 0.25 }
        efficient: { w_r: 0.7, w_d: 1.0,  w_p: 0.2, w_c: 0.10 }
        balanced:  { w_r: 1.0, w_d: 0.5,  w_p: 0.4, w_c: 0.20 }
      grid:
        w_r: [0.8, 1.0]
        w_d: [0.2, 0.4, 0.8, 1.0]
        w_p: [0.2, 0.4, 0.6]
        w_c: [0.0, 0.2, 0.4]
      risk_agg: "mean"    # or "cvar"
      alpha: 0.95
If risk_agg="cvar" is not implemented in‑planner, precompute a risk surface via offline CVaR approximation (Phase F/G note), and pass it as the risk layer.

4) Tasks (execution order)
 G‑01 Weight grid & presets

New module core/route/weights.py

iter_weight_grid(spec: dict) -> Iterator[dict] (expand grid and include presets)

Validate bounds; normalize or keep raw? (Keep raw and rely on planner)

 G‑02 Route metrics

New module core/route/metrics.py

compute_distance_km(path) -> float (reuse existing haversine_m)

integrate_field_along_path(field_da, path) -> float (risk and prior_penalty integrals: ∑ distance_segment * field_at_segment)

summarize_route(...) -> dict returning:

json
复制代码
{ "distance_km": ..., "risk_integral": ..., "prior_integral": ..., "congest_integral": ... }
 G‑03 Pareto extraction & representatives

New module core/route/pareto.py

nondominated(points, keys=("risk_integral","distance_km","congest_integral")) -> list[idx]

pick_representatives(points) -> {safe, efficient, balanced}

safe = argmin risk_integral

efficient = argmin distance_km

balanced = closest to normalized origin in z‑scores (or knee point detection)

 G‑04 CLI wiring

In api/cli.py add:

route.scan --scenario <id> --ym <YYYYMM> --risk-source {fused|ice} --risk-agg {mean|cvar} --alpha 0.95 --grid config/scenarios.yaml --export 3 --out reports/d_stage/phaseG/

report.build --ym <YYYYMM> --include pareto --scenario <id>

 G‑05 Outputs

Save all candidate routes (candNN) and 3 representatives (safe/balanced/efficient) to data_processed/routes/*.geojson

Save pareto_front_*.json and summary_*.json; generate pareto_*.html

 G‑06 Tests

Unit: pareto sorting, metrics monotonicity

E2E: tiny grid, 6–12 weight combos, ensure front size ≥ 2 and 3 reps exported

5) Definition of Done (Acceptance)
 route.scan runs with a provided scenario and produces:

pareto_front_<ym>_<scenario>.json with ≥ 5 distinct points and non‑dominated set ≥ 2

Three representatives exported as route_<ym>_<scenario>_{safe,balanced,efficient}.geojson

 report.build --include pareto renders an HTML page containing:

Scatter plot(s): distance vs risk (+ color by w_c or prior integral)

Parallel coordinates for weights and metrics

Comparison table for the 3 selected routes

 Runtime ≤ 10 minutes on 3070Ti for ≤ 36 weight combos (single scenario)

 No breaking changes in existing CLI/UI; all artifacts include .meta.json

6) Risk & Fallback
Performance blow‑up on large grids and many weights → cap grid sizes; offer --max-combos and/or early termination when front stabilizes.

Unstable metrics due to NaNs or missing layers → ensure np.nan_to_num; log warnings in meta and skip problem routes.

No CVaR support yet → default to risk_agg=mean; allow later offline risk_cvar precomputation.

7) TL;DR for Codex (Execution Checklist)
You must read docs/_index.md and this file end‑to‑end before coding.
Then implement in the following order:

Create modules

ArcticRoute/core/route/weights.py — grid expander & presets

ArcticRoute/core/route/metrics.py — distance & field integrals; summarize_route

ArcticRoute/core/route/pareto.py — non‑domination + representative picks

Wire CLI

Edit ArcticRoute/api/cli.py and add:

route.scan command (Alpha) — parses scenario + runs all weight combos; calls planner; saves cand routes; writes front/summary JSON

report.build --include pareto — renders Pareto HTML for one scenario using the JSON

Write artifacts

GeoJSON for each candidate; plus the 3 representatives

reports/d_stage/phaseG/pareto_front_<ym>_<scenario>.json

reports/d_stage/phaseG/summary_<ym>_<scenario>.json

reports/d_stage/phaseG/pareto_<ym>_<scenario>.html

Each with adjacent .meta.json containing logical_id, inputs, run_id, git_sha, etc.

Smoke run (example)

bash
复制代码
python -m ArcticRoute.api.cli route.scan \
    --scenario nsr_wbound --ym 202412 \
    --risk-source fused --risk-agg mean \
    --grid config/scenarios.yaml --export 3 \
    --out reports/d_stage/phaseG/

python -m ArcticRoute.api.cli report.build \
    --ym 202412 --include pareto --scenario nsr_wbound
Minimal tests

Add unit tests for Pareto and metrics monotonicity under tests/

Provide a tiny scenario for CI smoke (≤ 12 combos)

8) Report Layout (for report.build --include pareto)
Charts

Scatter: distance_km (x) vs risk_integral (y). Color by w_c or prior integral; mark Pareto set.

Parallel coordinates: {w_r,w_d,w_p,w_c,alpha} → {distance,risk,prior,congest}

Tables

Top‑N Pareto candidates (N ≤ 12)

The 3 representatives (Safe/Balanced/Efficient), with download links

Metadata

ym, scenario, run_id, input logical IDs, generation time

File locations

HTML: reports/d_stage/phaseG/pareto_<ym>_<scenario>.html

JSON: reports/d_stage/phaseG/{pareto_front,summary}_<ym>_<scenario>.json