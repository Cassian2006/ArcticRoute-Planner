# Phase L | Domain Adaptation & Vessel-Type Specialization

## 0) Context & Goal
**Context.** Phase K delivered a learned multimodal fusion (CV + Accident + Prior). However, polar routing differs by **region** (e.g., NSR vs. NWP), **season** (summer vs. winter), and **vessel type** (ice-class cargo vs. fishing).  
**Goal.** Adapt fusion and routing to domains without breaking existing contracts:
1) **Bucketing & calibration** by region/season/vessel-type;
2) **Mixture-of-Experts (MoE) gate** using Transformer prior embeddings + context;
3) Lightweight **finetuning/adapters** per bucket to improve reliability (ECE/Brier) and path quality;
4) Scenario-aware routing defaults (weights/presets) for each domain.

---

## 1) Scope / Non-Goals
- **In scope:** domain buckets, gate, per-bucket calibration/finetune, scenario presets, evaluation & reports.
- **Out of scope:** re-training the Phase E prior from scratch; large-scale distributed training.

---

## 2) Artifacts & Contracts
**Inputs**
- Phase K fused model checkpoint(s): `outputs/phaseK/fusion_unetformer/<run_id>/best.ckpt`
- Prior assets: `embeddings_<ym>.parquet`, `prior_corridor_selected_<ym>.nc` or `prior_transformer_<ym>.nc`
- Metadata: vessel class map (`io/vessel_class.py`), region polygons (GeoJSON), season rules
- Existing channels: `R_ice_eff/R_ice, R_wave, R_acc, prior_penalty/P_prior, R_interact, congest, edge_dist, lead_prob`

**Outputs**
- Gate and experts:
  - `outputs/phaseL/moe/<run_id>/{gate.ckpt, expert_{k}.ckpt}` (or adapters)
  - `data_processed/risk/risk_fused_<ym>.nc` stays **identical in path & var name** (`risk`)
- Calibrations per bucket:
  - `reports/d_stage/phaseL/calibration_<bucket>_<ym>.{json,png}`
- Scenario presets:
  - `config/scenarios.yaml` extended with `presets_by_bucket` and defaults
- Meta for all artifacts: adjacent `.meta.json` with logical_id/inputs/run_id/git_sha/config_hash/metrics/warnings

**No breaking changes** in CLI/UI or file shapes.

---

## 3) Bucketing Definition
- **Region**: e.g., `NSR`, `NWP`, `Barents`, using a GeoJSON polygon set.
- **Season**: rules based on month (e.g., DJF, MAM, JJA, SON) or custom “navigable window”.
- **Vessel type**: from AIS/vessel registry (e.g., `ice_class`, `cargo`, `tanker`, `fishing`).

`bucket_id = f(region, season, vessel_type)` with fallback hierarchy (if unknown, backoff to broader bucket).

---

## 4) Tasks (Execution Order)
- [ ] **L-01 Context features & bucketer**
  - New: `ArcticRoute/core/domain/bucketer.py`
    - `infer_bucket(latlon, ts, vessel_type) -> bucket_id`
    - Utilities: point-in-region (vectorized), month→season, vessel map
  - Extend `config/runtime.yaml` with `buckets`, `region_geojson`, `season_rules`.

- [ ] **L-02 Gate using prior embeddings (REUSE Phase E)**
  - New: `ArcticRoute/core/fusion_adv/gate.py`
    - Inputs: `prior embedding (from embeddings_<ym>.parquet)`, context (`region one-hot`, `season one-hot`, `vessel type one-hot`)
    - Output: soft weights over K experts (K=2~4)
    - Train with weak labels by bucket (minimize ECE/Brier per bucket)

- [ ] **L-03 Experts / adapters per bucket**
  - Option A (lightweight): **adapters/LoRA** layers on top of the Phase K UNet-Former; one adapter per bucket
  - Option B: separate tiny experts initialized from Phase K checkpoint (freeze most layers)
  - Save each expert as `expert_<bucket>.ckpt` or adapter weights

- [ ] **L-04 Calibration per bucket**
  - Extend `core/fusion_adv/calibrate.py`
    - Fit isotonic/logistic per bucket; write curves and reliability diagrams
  - CLI: `risk.fuse.calibrate --ym <YYYYMM> --by-bucket`

- [ ] **L-05 Inference wiring**
  - `risk.fuse --ym <YYYYMM> --method unetformer --moe --by-bucket`
    - Pipeline: build channels → infer bucket grid → gate → expert forward → (optional) per-bucket calibration → write `risk_fused_<ym>.nc`
  - Fallback: if gate confidence < τ, use global expert (Phase K)

- [ ] **L-06 Scenario presets for routing**
  - Extend `config/scenarios.yaml` with `presets_by_bucket` (weights & risk_agg)
  - `route.scan` reads the bucket of the scenario start→goal line to auto-select presets

- [ ] **L-07 Evaluation & report**
  - `reports/d_stage/phaseL/`: ECE/Brier per bucket, path metrics grouped by bucket
  - Compare vs Phase K global model (A/B)

- [ ] **L-08 Tests**
  - Bucketer correctness (region/season/vessel splits)
  - Gate outputs sum-to-1; smoothness under small context changes
  - Per-bucket calibration strictly improves or not worse than global

---

## 5) Model Details
- **Gate**: small MLP over concatenated features: `[prior_embed_reduced(UMAP/PCA), region1h, season1h, vessel1h] → softmax(K)`
- **Experts**: share backbone; adapters/LoRA to limit memory; freeze backbone if GPU-limited
- **Training target**: weak labels by bucket; optimize ECE/Brier; early stop on bucket-val ECE
- **Inference**: per-pixel gate weight × expert outputs, or hard-assign if argmax>τ

---

## 6) CLI Examples (Smoke)
```bash
# 1) Calibrate by bucket
python -m ArcticRoute.api.cli risk.fuse.calibrate --ym 202412 --by-bucket

# 2) Inference with MoE gate
python -m ArcticRoute.api.cli risk.fuse --ym 202412 --method unetformer --moe --by-bucket

# 3) Route with bucket presets
python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound --ym 202412 --risk-source fused --risk-agg mean --export 3
7) Definition of Done (Acceptance)
 risk_fuse --moe --by-bucket writes risk_fused_<ym>.nc with same shape/attrs as before

 Per-bucket ECE improves by ≥10% vs global; Brier not worse

 route.scan using bucket presets yields better risk integral with ≤5% distance overhead (vs global)

 All outputs have .meta.json; UI unchanged except preset auto-selection

8) Risks & Mitigations
Sparse buckets → merge to parent bucket; apply temperature scaling instead of adapters

Gate instability → smooth with entropy penalty; floor/ceil weights; fallback to global expert

Complexity creep → keep K small (≤4); adapters instead of full experts; reuse Phase K backbone

9) Dependencies
umap-learn or scikit-learn (PCA) for embedding reduction (optional)

Reuse existing: scikit-learn (calibration), torch, xarray

