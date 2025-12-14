Phase M | Eco-routing & Emissions (Switchable Module)
0) Context & Goal

Context. Phases K–L delivered multimodal fusion and domain adaptation. We now add a switchable green-sailing module that estimates fuel and CO₂ along routes and optionally optimizes for it, without breaking any existing contracts.

Goal.

Implement a simple, explainable fuel/CO₂ model driven by vessel class and environmental penalties (ice/wave);

Make eco as a toggle in CLI/UI and as an objective term (w_e·CO2_norm) and a Pareto dimension;

Provide exports and reports (totals + segment attribution).

No breaking changes when --eco off.

1) Scope / Non-Goals

In scope: fuel & CO₂ estimation per segment; CLI/UI toggle; route objective term & Pareto; reports; tests.

Out of scope: high-fidelity engine simulation; speed optimization; regulatory calculators. All coefficients are provided via config and can be calibrated later.

2) Artifacts & Contracts

Inputs

Vessel class mapping: io/vessel_class.py (MMSI → class), and new config/eco.yaml

Layers already available: R_ice_eff_<ym>.nc, R_wave_<ym>.nc (optional), route geometry

(Optional) Escort flag already reflected in R_ice_eff

Outputs

data_processed/eco/eco_route_<ym>_<scenario>_<label>.json

{ fuel_total_t, co2_total_t, per_segment: [...] }

Normalized eco surface (optional cache): data_processed/eco/eco_cost_<ym>.nc (eco_cost_norm ∈ [0,1])

Reports:

reports/d_stage/phaseM/eco_summary_<ym>_<scenario>.json

reports/d_stage/phaseM/eco_summary_<ym>_<scenario>.html (tables + stacked bars)

Meta: adjacent .meta.json with logical_id, inputs, run_id, git_sha, config_hash, metrics, warnings

CLI/UI (no breakage)

route ... --eco {off|on} --w_e <float> (default off)

route.scan adds co2_total_t as a Pareto dimension when eco is on

UI sidebar: checkbox “Green Sailing (CO₂)” + slider w_e

3) Model (simple & explainable)

For each path segment i (length d_i_nm in nautical miles):

Base fuel per nautical mile from vessel class:

fuel_per_nm_base [t/nm], v_ref [kn]


Environmental multipliers (unitless, ≥1):

phi_ice(i)  = 1 + α_ice  · R_ice_eff(i)
phi_wave(i) = 1 + α_wave · R_wave(i)            # optional
phi_class   = ice_class_factor(class)           # e.g., 0.8 for strong ice class, 1.2 for weak


Segment fuel & CO₂:

fuel_i_t = fuel_per_nm_base · phi_class · phi_ice(i) · phi_wave(i) · d_i_nm
co2_i_t  = fuel_i_t · EF_CO2                      # EF_CO2 in tCO2 / t fuel (from config)


Route totals:

fuel_total_t = Σ_i fuel_i_t
co2_total_t  = Σ_i co2_i_t


Routing objective term (when eco is on):

Build a grid eco_cost_norm ∈ [0,1] by projecting (per-cell) fuel_per_nm penalties and quantile-normalizing, analogous to Risk.

Extended objective:

J = w_r·Risk + w_d·Dist + w_p·PriorPenalty (+ w_c·Congest) (+ w_e·CO2_norm)


Notes

All coefficients live in config/eco.yaml. Nothing is hard-coded.

This model is linear-in-penalties and easy to explain; we can later upgrade to speed–power ~ v^3 if needed.

4) Config (config/eco.yaml, example)
eco:
  enabled_default: false
  ef_co2_t_per_t_fuel: 3.114  # configurable; default value here; can be adjusted
  alpha_ice: 0.8              # ice penalty slope
  alpha_wave: 0.3             # wave penalty slope (optional)
  vessel_classes:
    icebreaker:
      fuel_per_nm_base: 0.020   # t per nm at v_ref
      v_ref_kn: 12
      ice_class_factor: 0.6
    cargo_iceclass:
      fuel_per_nm_base: 0.015
      v_ref_kn: 12
      ice_class_factor: 0.8
    cargo_standard:
      fuel_per_nm_base: 0.012
      v_ref_kn: 12
      ice_class_factor: 1.0
    fishing_small:
      fuel_per_nm_base: 0.006
      v_ref_kn: 10
      ice_class_factor: 1.1

5) Tasks (execution order)

 M-01 Fuel/CO₂ core

New: ArcticRoute/core/eco/fuel.py

fuel_per_nm_map(ym, vessel_class, alpha_ice, alpha_wave) -> xr.DataArray[eco_cost_nm_t]

quantile-normalize to [0,1] → eco_cost_norm

New: ArcticRoute/core/eco/route_eval.py

eval_route_eco(route_geojson, eco_cost_nm_t, ef_co2) -> {fuel_total_t, co2_total_t, per_segment}

 M-02 CLI wiring

Edit ArcticRoute/api/cli.py:

eco.preview --ym <YYYYMM> --scenario <id> --class <name> # dumps totals for a sample route

Extend route & route.scan:

add --eco {off|on} --w_e <float> --class <name>

when --eco on, include eco_cost_norm in objective and co2_total_t in metrics

 M-03 UI

apps/route_params.py / apps/app_min.py:

Checkbox “Green Sailing (CO₂)”

Slider w_e (0..1)

Dropdown Vessel Class (from config/eco.yaml)

 M-04 Reports

New: ArcticRoute/core/reporting/eco.py

Build JSON + HTML with totals and segment stacked bars (fuel/CO₂ contribution vs distance)

CLI: report.build --ym <YYYYMM> --include eco --scenario <id>

 M-05 Tests

Unit: monotonicity (fuel_total increases with α_ice and with average R_ice_eff)

Unit: normalization bounds [0,1] for eco_cost_norm

E2E: --eco on vs off produces different route metrics; objective decreases when w_e grows (toward eco-friendly path)

6) CLI (smoke)
# 1) Build eco cost map (implicit in preview)
python -m ArcticRoute.api.cli eco.preview --ym 202412 --scenario nsr_wbound --class cargo_iceclass

# 2) Route WITHOUT eco (baseline)
python -m ArcticRoute.api.cli route --ym 202412 --risk-source fused --eco off --w_e 0.0

# 3) Route WITH eco (optimize for CO₂ as well)
python -m ArcticRoute.api.cli route --ym 202412 --risk-source fused --eco on --w_e 0.5 --class cargo_iceclass

# 4) Pareto scan with CO₂ as a dimension
python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound --ym 202412 \
  --risk-source fused --eco on --grid config/scenarios.yaml --export 3

# 5) Report
python -m ArcticRoute.api.cli report.build --ym 202412 --include eco --scenario nsr_wbound

7) Definition of Done (Acceptance)

 --eco off reproduces previous results exactly (bit-for-bit where applicable)

 --eco on adds co2_total_t to metrics, and w_e>0 shifts the route toward lower-CO₂ corridors

 Segment-wise eco attribution is exported (JSON + HTML)

 UI toggle & vessel class dropdown function correctly

 All artifacts have .meta.json and comply with contracts

8) Risks & Mitigations

Coefficient uncertainty → keep all coefficients in config/eco.yaml; document sources; allow easy tuning

Scaling mismatch between CO₂_norm and Risk → use quantile normalization & sanity tests

Escort/ice interactions → rely on R_ice_eff (already escort-adjusted); provide ice_class_factor per vessel