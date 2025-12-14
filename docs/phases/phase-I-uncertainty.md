# Phase I | Uncertainty-Aware Fusion & Robust Routing

## 0) Context & Goal
**Context.** Phases F–H added convoy/interaction layers, Pareto scans, and explainability. Fusion still outputs a single risk surface. We now want **uncertainty-aware fusion** and **robust routing** so that routes are optimized not only for expected risk but also for downside risk (quantiles / CVaR).

**Goal.** Deliver:
1) **Evidential fusion** (Beta/Dirichlet evidence) + **Product-of-Experts (PoE)** as pluggable fusion heads;
2) Route-side **risk aggregation**: mean / quantile / **CVaR@α** using evidential distributions;
3) **Robust routes** via sampling / ensemble (majority or expected shortfall);
4) **Uncertainty report** (maps, reliability of uncertainty, coverage vs. confidence), wired to CLI/UI and Catalog.

No breaking changes to existing CLI/UI; all artifacts keep the same shapes and naming contracts.

---

## 1) Scope / Non-Goals
- **In scope**
  - New fusion plugins: `evidential`, `poe`
  - Distribution outputs: mean, variance (and optional raw evidence parameters)
  - Risk aggregators: `mean | q | cvar` in route planner (exact or sampled)
  - Robust routing by sampling K risk realizations
  - Reporting & UI toggles for uncertainty
- **Out of scope**
  - Retraining prior Transformer (Phase E)
  - Heavy Bayesian deep nets (we use lightweight evidential heads)

---

## 2) Artifacts & Contracts
- **Inputs**
  - Same as Phase F/G/H: `risk_fused_<ym>.nc` inputs and component layers (ice/wave/acc/prior/congest/R_interact)
  - Weak labels for calibration: AIS corridor mask (positive), incidents/closed-ice mask (negative)
  - (Optional) Extreme-month sets for stress tests
- **Fusion Outputs**
  - `data_processed/risk/risk_fused_<ym>.nc`
    - var: `risk ∈ [0,1]` (mean)
    - attrs (new): `risk_var` (if embedded), or separate dataset fields
  - Optional companion: `risk_fused_params_<ym>.nc` (e.g., Beta α,β or PoE log-weights)
- **Routing Outputs**
  - Same GeoJSON/PNG/HTML as Phase G; add route-level metrics for **CVaR** and **variance**
  - Robust routes: `route_<ym>_<scenario>_<label>_robust.geojson`
- **Reports**
  - `reports/d_stage/phaseI/uncertainty_<ym>.json` (metrics)  
  - `reports/d_stage/phaseI/uncertainty_<ym>.png` (maps/plots)  
  - `reports/d_stage/phaseI/robust_<ym>_<scenario>.html` (ensemble vs mean)
- **Meta**
  - Adjacent `.meta.json` for every artifact with `logical_id, inputs, run_id, git_sha, config_hash, metrics, warnings`.

---

## 3) Minimal Data Checklist
- ✅ Component layers for the month (ice/wave/acc/prior/etc.) aligned to EPSG:3413
- ✅ Weak labels: AIS density → positives; incidents/closed-ice → negatives
- ➕ (Recommended) **Extreme months** (e.g., heavy ice or high incident months) for stress calibration
- ➕ (Optional) Any external reliability set (held-out period) to verify calibration

---

## 4) Tasks (Execution Order)
- [ ] **I-01 Evidential fusion head**
  - New: `core/fusion_adv/evidential.py` (FusionPlugin)
    - Inputs: multi-channel raster stack (e.g., `R_ice_eff, R_wave, R_acc, PriorPenalty, R_interact, congest, edges`)
    - Output: `risk_mean ∈ [0,1]`, `risk_var ≥ 0`; optionally Beta params (α, β)
    - Training: weak labels (0/1/NaN) with evidential loss (e.g., NLL + evidence regularization)
    - Save/Load: serialize weights and config
- [ ] **I-02 PoE fusion head**
  - New: `core/fusion_adv/poe.py`
    - Combine component “expert” probabilities via normalized product (in log-space for stability)
    - Optional temperature / per-expert weights, calibrated on weak labels
- [ ] **I-03 Route-side aggregators**
  - New/extend: `core/cost/aggregators.py`
    - `aggregate_risk(risk_mean, risk_var=None, mode="mean|q|cvar", alpha=0.95)`
      - If Beta params available: compute `quantile`/`CVaR` analytically or numerically
      - Else: fall back to high-quantile mean (Phase F placeholder)
  - Wire to planner CLI: `--risk-agg {mean,q,cvar} --alpha 0.95`
- [ ] **I-04 Robust route via sampling**
  - New: `route.robust` (CLI)
    - Sample K risk surfaces from evidential distribution (Beta)
    - Compute K routes; select **least expected shortfall** or **majority-vote path**
    - Output robust route + dispersion statistics (per-segment variance)
- [ ] **I-05 Calibration & uncertainty report**
  - New: `core/reporting/uncertainty.py`
    - Metrics: NLL, Brier, ECE, **Var-calibration** (risk variance vs. error), **Sharpness** (avg std)
    - Plots: reliability, uncertainty histograms, risk-variance scatter
  - CLI: `report.build --ym <YYYYMM> --include uncertainty`
- [ ] **I-06 UI toggles**
  - `apps/layers_risk.py` and `route_params.py`
    - Add “Aggregation: mean / q / CVaR (α)” controls
    - Optional “Show uncertainty map” overlay
- [ ] **I-07 Tests**
  - Unit: evidential head returns bounded mean & non-negative var; PoE monotonicity
  - Unit: aggregator produces `mean ≤ q ≤ cvar` (for α≥0.5)
  - E2E: robust route diverges from mean in high-variance corridors

---

## 5) CLI (Smoke)
```bash
# 1) Train/apply evidential fusion
python -m ArcticRoute.api.cli risk.fuse --ym 202412 --method evidential --config config/risk_fuse_202412.yaml

# 2) Route with CVaR aggregation
python -m ArcticRoute.api.cli route --ym 202412 --risk-source fused --risk-agg cvar --alpha 0.9

# 3) Robust route (sampling K=16)
python -m ArcticRoute.api.cli route.robust --ym 202412 --risk-source fused --samples 16 --alpha 0.9 --out reports/d_stage/phaseI/

# 4) Uncertainty report
python -m ArcticRoute.api.cli report.build --ym 202412 --include uncertainty
6) Definition of Done (Acceptance)
 risk.fuse --method evidential produces risk_fused_<ym>.nc with valid mean in [0,1] and attached variance (attr or companion dataset)

 Route with --risk-agg cvar increases detour rate in high-variance waters while keeping distance overhead ≤ 8% vs. mean for the same scenario

 route.robust exports a robust route and dispersion stats; ensemble disagreement > 0 in at least one hotspot region

 report.build --include uncertainty renders metrics + plots; ECE improves or remains acceptable; uncertainty is well-calibrated on weak labels

 All artifacts have .meta.json; no breaking changes to existing paths/UI

7) Risks & Mitigations
Over-conservatism (routes too long) → cap α, offer hybrid objective: mean + λ·std; expose slider

Poor calibration on weak labels → add isotonic/temperature calibration post-head; document in meta

Computation (robust K samples) → start with small K (e.g., 8–16); cache sampled surfaces; parallelize if available

Numerical instability in PoE → operate in log-space; floor probabilities to ε

