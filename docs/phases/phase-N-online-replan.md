Phase N | Online Assimilation & Real-time Replanning
0) Context & Goal

Context. Phases K–M delivered multimodal fusion, domain adaptation, and switchable eco-routing. We now aim to ingest near-real-time (NRT) updates (ice/wave/interaction) and replan routes when conditions change, with smooth handover and no breaking changes.

Goal.

Add nowcasting/assimilation of updated layers (risk deltas, confidence-weighted blending);

Introduce event/period triggers to replan from the vessel’s current position, with waypoint locks and hysteresis to avoid oscillation;

Provide Live mode in CLI/UI and a small watcher service; keep offline mode unchanged.

1) Scope / Non-Goals

In scope: NRT pulls; risk update blending; incremental replanning; triggers; UI Live toggle; reports & logs.

Out of scope: Full 4D variational DA or heavy CFD; AIS provider integration beyond simple polling (use existing ingesters).

2) Artifacts & Contracts

Inputs

Updated layers for the same grid: R_ice_eff_<ym>D<dd>.nc, R_wave_<ym>D<dd>.nc, R_interact_<ym>D<dd>.nc (optional), risk_fused_<ym>.nc (baseline), prior_penalty (unchanged)

(Optional) Vessel live position or “progress along route” estimate

Outputs

Updated fused surface (live cache): data_processed/risk/risk_fused_live_<ts>.nc (var: risk; same shape/attrs)

Replanned route(s):
data_processed/routes/live/route_<scenario>_<ts>_vNN.geojson (and ..._robust.geojson if enabled)

Live state & logs: outputs/live/<scenario>/{state.json, events.log}

Reports:
reports/d_stage/phaseN/replan_summary_<scenario>_<ts>.json and .html (what changed, why)

Meta: adjacent .meta.json everywhere, consistent with prior contracts

Compatibility

When Live mode off, planner behaves exactly as before.

3) Minimal Data & Latency Targets

NRT ice/wave layers at daily or sub-daily cadence (same CRS/grid)

Optional updated interaction/congestion layers (coarser cadence is fine)

Targets: blend+plan end-to-end ≤ 60 s per scenario on 3070Ti for medium grids; watcher interval ≥ 5 min by default.

4) Architecture & Algorithms
4.1 Assimilation / Blending (risk update)

For an updated component C_new with confidence w_now ∈ [0,1] and baseline C_base:

C_blend = normalize( w_now · C_new + (1 - w_now) · C_base )


Apply per component (R_ice_eff, R_wave, R_interact) then run the same fusion plugin used in Phase K (or a fast fallback, e.g., stacking). Confidence can depend on data age (exp(-Δt/τ)) and known source reliability.

4.2 Replanning strategy (incremental)

Start point: project current vessel position onto last route polyline; lock passed waypoints.

Triggers (any):

Periodic: every T_period (e.g., 30–60 min)

Risk jump ahead: moving window along the next L_ahead NM shows mean(risk) > τ_hi or Δrisk > τ_delta

Interaction surge: R_interact increment > τ_interact

Eco change (if --eco on): projected co2_total_t increases by > τ_co2%

Hysteresis: cool-down T_cool (no replan soon after a replan); route-change threshold Δroute_len and Δrisk to avoid flipping.

Safe handover: stitch new route at a handover point H ahead by H_dist NM; keep path segment before H frozen.

4.3 Robust option (if Phase I enabled)

Sample K fused surfaces (evidential) → choose route minimizing Expected Shortfall (CVaR@α); output _robust.geojson.

5) Tasks (Execution Order)

 N-01 Live cache & blending

New: ArcticRoute/core/online/blend.py

blend_components(components: dict, conf: dict, norm="quantile") -> dict

fuse_live(components_blend, method="stacking|unetformer") -> xr.DataArray[risk]

 N-02 Replan engine

New: ArcticRoute/core/route/replan.py

should_replan(state, deltas, cfg) -> bool, reason

stitch_and_plan(current_pos, route_old, risk, prior_penalty, params) -> route_new

 N-03 Live state & watcher

New: ArcticRoute/ops/watchers/replan_watcher.py

Polls NRT updates, computes deltas, calls CLI route.replan when triggered; writes outputs/live/...

 N-04 CLI wiring

Extend ArcticRoute/api/cli.py:

risk.nowcast --ym <YYYYMM> --since <ISO8601> --conf <0..1> → downloads/loads updates, runs blend -> fuse_live

route.replan --scenario <id> --live --eco {off|on} --risk-agg {mean|q|cvar} --alpha 0.95

watch.run --scenario <id> --interval 300 --rules configs/replan.yaml

 N-05 UI Live mode

apps/app_min.py: Live tab

Toggle Live Mode, show last update time, “Replan now” button, reason for last replan, diff vs previous route

 N-06 Reports

New: ArcticRoute/core/reporting/replan.py

Summaries: trigger reason, risk/eco deltas, distance/risk/CO₂ changes, map overlays; HTML export

 N-07 Tests

Unit: blending bounds; trigger logic hysteresis; stitch correctness (no backward segments)

E2E: simulate a risk jump → watcher triggers → route version increments → report generated

6) Config Samples
6.1 Replan rules (configs/replan.yaml)
replan:
  period_sec: 1800          # periodic check
  lookahead_nm: 50
  risk_threshold: 0.55
  risk_delta: 0.15
  interact_delta: 0.10
  eco_delta_pct: 5          # only when --eco on
  cool_down_sec: 900
  handover_nm: 8
  min_change_nm: 3

6.2 Live confidence (configs/runtime.yaml addition)
live:
  conf_ice_tau_hours: 24
  conf_wave_tau_hours: 12
  default_conf: 0.7

7) CLI (Smoke)
# 1) Pull & blend updates into a live fused surface
python -m ArcticRoute.api.cli risk.nowcast --ym 202412 --since -P6H --conf 0.8

# 2) Replan once from current state (mean or CVaR)
python -m ArcticRoute.api.cli route.replan --scenario nsr_wbound --live --risk-agg mean

# 3) Start watcher (every 5 minutes, using replan rules)
python -m ArcticRoute.api.cli watch.run --scenario nsr_wbound --interval 300 --rules configs/replan.yaml

8) Definition of Done (Acceptance)

 risk.nowcast produces risk_fused_live_<ts>.nc with valid shape/attrs and .meta.json

 route.replan writes a new versioned route and a diff summary; distance increase ≤ 10% unless risk jump is severe

 Watcher triggers only when rules are met; cool-down and min-change prevent oscillation

 UI Live tab displays last update time, reason, and route diff; “Live off” reproduces offline behavior exactly

 Reports include trigger reason, metric deltas, and overlays

9) Risks & Mitigations

Noisy updates → route flapping: hysteresis, cool-down, handover distance, min-change thresholds

Latency spikes: fast stacking fallback; pre-tile fusion; cache prior computations

Data gaps: confidence fallback to baseline; warn in meta & UI; skip replan if confidence too low