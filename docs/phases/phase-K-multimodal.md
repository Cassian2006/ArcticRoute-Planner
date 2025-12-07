# Phase K | Multimodal Fusion Training (CV + Accident + Transformer Prior)

## 0) Context & Goal
**Context.** Phases C–E built risk components (ice/wave/accident) and a Transformer-based prior; Phases F–J added interaction, Pareto, explainability, uncertainty, and deployability. Fusion still relies partly on linear weighting.  
**Goal.** Train a **learned fusion** model that ingests **Accident (R_acc)** and **CV features** (ice-edge/lead) together with existing layers, while **retaining the Transformer prior** as a stable input and a routing penalty. Keep outputs and CLI/UI contracts unchanged.

**Deliverables.**
1) Offline builders for `R_acc` and CV rasters (`edge_dist`, `lead_prob`) with shared grid/attrs.
2) A lightweight **UNet-Former** fusion plugin (2D; optional temporal window later).
3) Weak-supervised training + calibration (Brier/ECE) and evaluation.
4) Compatible inference: `risk_fused_<ym>.nc` (var: `risk ∈ [0,1]`) + calibration report.

---

## 1) Scope / Non-Goals
- **In scope:** Accident & CV channels, dataset builder, UNet-Former training/inference, isotonic/logistic calibration, calibration report, CLI/UI wiring, tests.
- **Out of scope:** Rewriting A*, retraining the prior Transformer, heavy video-level CV models (keep CV light).

---

## 2) Artifacts & Contracts
**Inputs (aligned to EPSG:3413; dims `(time?, y, x)`):**
- Ice/Wave/Accident baseline rasters (existing): `R_ice_<ym>.nc` (or `R_ice_eff_<ym>.nc`), `R_wave_<ym>.nc`, `R_acc_<ym>.nc` (rebuilt below).
- Prior: `prior_corridor_selected_<ym>.nc` (`prior_penalty`) or `prior_transformer_<ym>.nc` (`P_prior` → derive `prior_penalty = 1 − P_prior`).
- CV features (new): `edge_dist_<ym>.nc`, `lead_prob_<ym>.nc`.
- Weak labels (any subset): `ais_density_<ym>.nc` (positives above threshold), incidents mask / closed-ice mask (negatives).

**Outputs:**
- `data_processed/risk/risk_fused_<ym>.nc`  (var: `risk`; same path as before; soft-link ok)
- `reports/d_stage/phaseK/calibration_<ym>.{json,png}` (ECE/Brier/reliability)
- Training snapshots: `outputs/phaseK/fusion_unetformer/<run_id>/{ckpt, logs}`

**Meta:** Adjacent `.meta.json` for every new artifact (`logical_id, inputs, run_id, git_sha, config_hash, metrics, warnings`).

---

## 3) Minimal Data Checklist
- ✅ Incidents table `{lat, lon, ts[, type]}` per month (CSV/Parquet).
- ✅ CV imagery (COG/GeoTIFF or mosaics) already present in `data_processed/cog` / `stac_cache` / `sat_mosaic_*.tif`.
- ✅ `ais_density_<ym>.nc` (for weak positives) and base risk layers (ice/wave).
- ➕ (Recommended) 1–2 **extreme months** for robust evaluation.

---

## 4) Tasks (Execution Order)
- [ ] **K-01 Accident layer (REBUILD/REUSE).**  
  Module: `ArcticRoute/core/risk/accident.py`  
  CLI: `risk.accident.build --ym <YYYYMM>`  
  Logic: KDE over incidents with exposure correction using AIS density; normalize to `[0,1]`; write `R_acc_<ym>.nc` with complete attrs/meta.

- [ ] **K-02 CV features (edge/lead).**  
  Modules: `ArcticRoute/core/cv/edge.py`, `ArcticRoute/core/cv/lead.py`  
  CLI:  
  `cv.edge.build --ym <YYYYMM>` → `edge_dist_<ym>.nc` (edge strength or distance-to-edge; Sobel/Scharr + morphology; normalized to [0,1])  
  `cv.lead.build --ym <YYYYMM>` → `lead_prob_<ym>.nc` (open-water cracks; simple spectral/texture threshold or tiny U-Net if available)  
  Ensure exact grid alignment; record `source`, `norm`, `grid_id` attrs.

- [ ] **K-03 Fusion dataset builder.**  
  Module: `ArcticRoute/core/fusion_adv/dataset.py`  
  Functions:  
  `make_weak_labels(ais_density, incidents_mask, ice_mask, cfg) -> xr.DataArray{0/1/NaN}`  
  `build_patches(channels: Dict[str, xr.DataArray], labels, tile=256, stride=128, aug=True) -> torch.Dataset`  
  Channels include: `R_ice_eff/R_ice`, `R_wave`, `R_acc`, `prior_penalty`, `edge_dist`, `lead_prob`, optional `R_interact`, `congest`.

- [ ] **K-04 UNet-Former fusion plugin (training).**  
  Module: `ArcticRoute/core/fusion_adv/unetformer.py` (≤ 5–8M params; bf16/AMP; SDPA)  
  Loss: masked BCE (ignore NaN labels) ± focal; L2 regularizer; optional calibration loss.  
  Early stop on val ECE/Brier. Save `best.ckpt`.

- [ ] **K-05 Calibration head / stacking baseline.**  
  Module: `ArcticRoute/core/fusion_adv/calibrate.py`  
  Methods: logistic regression and isotonic (scikit-learn).  
  CLI option to enable post-hoc calibration; write curves to report.

- [ ] **K-06 Inference & write fused risk.**  
  CLI: `risk.fuse --ym <YYYYMM> --method unetformer [--calibrated]`  
  Loads ckpt + channels → `risk_fused_<ym>.nc` (var: `risk`), with attrs `source='unetformer'`, `calibrated=true/false`.

- [ ] **K-07 CLI wiring.**  
  Edit `ArcticRoute/api/cli.py` to add:  
  `risk.accident.build`, `cv.edge.build`, `cv.lead.build`,  
  `risk.fuse.train --ym <YYYYMM> --method unetformer --inputs R_ice_eff,R_wave,R_acc,prior_penalty,edge_dist,lead_prob`,  
  `risk.fuse --ym <YYYYMM> --method unetformer [--calibrated]`,  
  `report.build --ym <YYYYMM> --include calibration`.

- [ ] **K-08 UI updates (optional this phase).**  
  `apps/layers_risk.py`, `apps/app_min.py`: toggles to overlay `edge_dist` / `lead_prob`; display calibration summary.

- [ ] **K-09 Tests.**  
  Unit: channel stacking order & normalization, label masking, calibration monotonicity.  
  E2E: tiny grid → train for a few steps → run inference → produce `risk_fused_<ym>.nc` and calibration figure.

- [ ] **K-10 Docs & Meta.**  
  Ensure every output writes `.meta.json` and add short README snippet in report.

---

## 5) Model & Training Details
- **Channels (typical):** `[R_ice_eff, R_wave, R_acc, prior_penalty, edge_dist, lead_prob, (optional R_interact, congest)]`
- **Normalization:** per-channel z-score or min-max; record policy in `attrs['norm']`.
- **Augmentation:** random flips/rotations; mild intensity jitter; no spatial scaling (to preserve meter scale).
- **Weak labels:**  
  - Positives: `ais_density ≥ τ_pos` (configurable quantile)  
  - Negatives: incidents mask + closed-ice mask  
  - Ignore elsewhere (NaN)  
- **Validation metrics:** AUC, AP, Brier, ECE; report all; choose ckpt by **ECE** or **Brier**.
- **Compute:** single 3070Ti; use bf16/AMP; gradient accumulation if memory-bound.

---

## 6) CLI Examples (Smoke)
```bash
# 1) Build accident & CV layers
python -m ArcticRoute.api.cli risk.accident.build --ym 202412
python -m ArcticRoute.api.cli cv.edge.build --ym 202412
python -m ArcticRoute.api.cli cv.lead.build --ym 202412

# 2) Train fusion (UNet-Former)
python -m ArcticRoute.api.cli risk.fuse.train \
  --ym 202412 --method unetformer \
  --inputs R_ice_eff,R_wave,R_acc,prior_penalty,edge_dist,lead_prob \
  --epochs 10 --batch 8 --tile 256 --stride 128

# 3) Inference & calibration
python -m ArcticRoute.api.cli risk.fuse --ym 202412 --method unetformer --calibrated
python -m ArcticRoute.api.cli report.build --ym 202412 --include calibration
7) Definition of Done (Acceptance)
 risk_fuse.train produces best.ckpt and logs; validation ECE ≤ 0.08 or ↓10% vs linear baseline.

 risk.fuse --method unetformer writes risk_fused_<ym>.nc with risk ∈ [0,1], aligned dims/coords, complete attrs/meta.

 Calibration report contains ECE/Brier + reliability plot; post-hoc calibration improves or matches raw model.

 (Optional) Overlay of CV layers in UI; no breaking changes elsewhere.

8) Risks & Mitigations
Label noise → higher ignore ratio; robust loss (focal); threshold sweep for ais_density.

CV artifacts → gradient smoothing; clamp to [0,1]; optional median filter.

Overfitting → early stopping on ECE; lightweight model; limited epochs.

Performance → tiling with stride; gradient accumulation; bf16/AMP.

9) Dependencies
Add (if missing) to requirements.txt:
scikit-learn, einops, scikit-image (for edges/morphology).
Keep torch, xarray, hdbscan as previously planned.