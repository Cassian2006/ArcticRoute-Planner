# Phase H | Explainability & Reporting

## 0) Context & Goal
**Context.** ArcticRoute has end‑to‑end data → fusion → routing → UI. We now need first‑class **explainability** and **reproducible reporting** so that results can be trusted, audited, and presented.

**Goal.** Deliver:
1) **Path‑segment attribution** (stacked contribution bars of `R_ice / R_wave / R_acc / PriorPenalty / Congest / R_interact` along the selected route);
2) **Calibration & reliability** (Brier/ECE + reliability diagram for raster‑level weak labels);
3) **Animations** (risk / prior / fused layers over time; optional side‑by‑side route overlays);
4) **Audit page** (run_id, git_sha, config_hash, inputs, metrics, warnings) generated automatically from meta files.

No breaking changes to existing CLI/UI/artifacts. Everything is **Docs‑as‑Tasks** compatible.

---

## 1) Scope / Non‑Goals
- **In scope**
  - Per‑route segment attribution and CSV/JSON export
  - Calibration metrics & plots from weak supervision
  - GIF/MP4 timeline animations (risk/prior/fused; optional route overlays)
  - Audit HTML (reads `.meta.json` files + run log)
- **Out of scope**
  - Model retraining or new fusion methods (covered by Phases F/G)
  - Real‑time SHAP for heavy CNN/Transformer models (we provide lightweight occlusion/sensitivity alternatives)

---

## 2) Artifacts & Contracts
- **Inputs**
  - `data_processed/risk/risk_fused_<ym>.nc` (var: `risk`)
  - `data_processed/prior/prior_corridor_selected_<ym>.nc` or `prior_transformer_<ym>.nc` (var: `prior_penalty` or `P_prior`)
  - (Optional) `data_processed/risk/R_interact_<ym>.nc`, `data_processed/features/ais_density_<ym>.nc`
  - (Optional for animation) Time‑stacked layers for the same `<ym>`, e.g., `risk_fused_<ym>_D{dd}.nc`
  - Weak labels (any of): AIS corridor mask (positive), incidents mask (negative), frozen areas (negative)
- **Outputs**
  - **Attribution per route:**
    - `data_processed/routes/route_attr_<ym>_<scenario>_<label>.json` (per‑segment contributions)
    - `reports/d_stage/phaseH/route_attr_<ym>_<scenario>_<label>.png` (stacked bar)
  - **Calibration:**
    - `reports/d_stage/phaseH/calibration_<ym>.json` (ECE/Brier/reliability bins)
    - `reports/d_stage/phaseH/calibration_<ym>.png` (reliability diagram)
  - **Animations:**
    - `reports/d_stage/phaseH/anim_<ym>_<layer>.gif/mp4` (e.g., `layer ∈ {risk, prior_penalty, fused}`)
  - **Audit:**
    - `reports/d_stage/phaseH/audit_<ym>_<scenario>.html` (meta summary)
- **Meta**
  - Every output writes adjacent `.meta.json` with `logical_id`, `inputs`, `run_id`, `git_sha`, `config_hash`, `metrics`, `warnings`.

---

## 3) Minimal Data Checklist (what you likely already have)
- ✅ `risk_fused_<ym>.nc` and `prior_*_<ym>.nc` (aligned to EPSG:3413; dims `(time?, y, x)`)
- ✅ **Weak labels** (any subset available):
  - `data_processed/features/ais_density_<ym>.nc` → positive corridor labels (threshold configurable)
  - `data_processed/incidents/*.parquet|*.csv` → negative labels
  - Sea‑ice derived mask (closed ice → negative)
- ✅ Route GeoJSONs from previous phases for selected scenarios
- ➕ (Optional, for better animations) daily or sub‑monthly risk/prior stacks for the same `<ym>`

If any optional time‑stack is missing, the animation step will gracefully fall back to **monthly snapshots**.

---

## 4) Tasks (execution order)
- [ ] **H‑01 Segment attribution core**
  - New: `core/route/attribution.py`
    - `sample_field_along_path(field: xr.DataArray, path: LineString|GeoJSON, spacing_m: float) -> np.ndarray`
    - `segment_contributions(path, fields: Dict[str, xr.DataArray]) -> pd.DataFrame`  
      Returns per‑segment distances and per‑source cost contributions:
      \[
        \Delta J = \sum_s w_s \cdot \frac{(f_s(i) + f_s(i+1))}{2} \cdot \Delta d_i
      \]
      where `s ∈ {risk components, prior_penalty, congest, r_interact}`.
- [ ] **H‑02 CLI for route explanation**
  - Edit `api/cli.py` add:
    - `route.explain --route FILE.geojson --ym <YYYYMM> --out reports/d_stage/phaseH/`
      - Loads fused layers; computes per‑segment contributions;
      - Writes JSON + a stacked bar PNG.
- [ ] **H‑03 Calibration & reliability**
  - New: `core/reporting/calibration.py`
    - `build_weak_labels(ais_density, incidents, ice_mask, cfg) -> xr.DataArray{0/1/NaN}`
    - `reliability_curve(pred: xr.DataArray, labels: xr.DataArray, n_bins=15) -> dict`
    - `metrics(pred, labels) -> {"ece":..., "brier":..., "auc":...}`
  - CLI:
    - `report.build --ym <YYYYMM> --include calibration`
- [ ] **H‑04 Animation**
  - New: `core/reporting/animate.py`
    - `animate_layers(layers: List[Path], out: Path, fps=4, side_by_side: bool=False, overlay_routes: Optional[List[Path]]=None)`
  - CLI:
    - `report.animate --ym <YYYYMM> --layers risk,prior,fused --fps 4 --out reports/d_stage/phaseH/`
- [ ] **H‑05 Audit page**
  - New: `core/reporting/audit.py`
    - `collect_meta(paths: List[Path]) -> dict` (merge all `.meta.json`)
    - `render_audit_html(meta: dict, out_html: Path)` (uses a minimal template; lists inputs, hashes, metrics, warnings)
  - CLI:
    - `report.build --ym <YYYYMM> --include audit --scenario <id>`
- [ ] **H‑06 UI (optional for this phase)**
  - `apps/app_min.py` add tab “Explain”
    - Upload or pick a recent route → render stacked bar & download CSV/PNG
    - Toggle show/hide per‑source layers for visual confirmation

---

## 5) Definitions & Formulas

### 5.1 Segment‑wise contribution
For a polyline `p0..pN`, define the incremental cost for source `s` along segment `i → i+1` as:
\[
\Delta J_{s,i} = w_s \cdot \frac{f_s(p_i) + f_s(p_{i+1})}{2} \cdot d(p_i, p_{i+1})
\]
- `f_s` is the (interpolated) field value at segment endpoints (bilinear in grid space).
- `w_s` are the current route weights (read from scenario or CLI).
- Sum across segments and sources for totals and percentages.

### 5.2 Reliability & calibration
- **Brier score** on weak labels (0/1).
- **Expected Calibration Error (ECE)** with equal‑width bins.
- **Reliability diagram**: empirical pos. rate vs predicted risk/probability.

---

## 6) CLI Examples (smoke)
```bash
# 1) Explain one route (path contributions)
python -m ArcticRoute.api.cli route.explain \
  --route data_processed/routes/route_202412_nsr_wbound_balanced.geojson \
  --ym 202412 --out reports/d_stage/phaseH/

# 2) Calibration with weak labels
python -m ArcticRoute.api.cli report.build \
  --ym 202412 --include calibration

# 3) Animations (monthly snapshots or time stack)
python -m ArcticRoute.api.cli report.animate \
  --ym 202412 --layers risk,fused --fps 4 --out reports/d_stage/phaseH/

# 4) Audit summary (collect meta around the month/scenario)
python -m ArcticRoute.api.cli report.build \
  --ym 202412 --include audit --scenario nsr_wbound
7) Definition of Done (Acceptance)
 route.explain produces {json,png} with per‑source contributions; values sum to the planner’s reported objective within ≤ 2% tolerance.

 report.build --include calibration outputs ECE/Brier/ROC‑AUC JSON and a reliability diagram PNG.

 report.animate creates at least one GIF/MP4 (risk or fused), with correct georeferenced overlays.

 report.build --include audit renders HTML listing run_id, git_sha, config_hash, inputs, metrics, warnings.

 No breaking changes; all outputs have adjacent .meta.json.

8) Risks & Mitigations
Interpolation drift between grid and route line → use consistent CRS (EPSG:3413) and a stable haversine_m distance; verify with unit tests.

Label noise in weak supervision → report metrics with CI intervals; expose min_density threshold; allow excluding ambiguous bins.

Large GIFs → cap frames or use MP4; compress with imagemagick/imageio-ffmpeg if available.

Missing time‑stack → degrade gracefully to monthly snapshot animation.

