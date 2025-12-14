# Phase P | Reproducible Paper/Competition Package

## 0) Context & Goal
**Context.** Phases K–O delivered multimodal fusion, domain specialization, eco-routing, online replanning, and human-in-the-loop. We now need a **reproducible package** for papers/competitions: one-command experiments, auto-generated figures/tables/videos, and a minimal dataset bundle with documentation.

**Goal.**
1) One-command **reproduction scripts** (end-to-end or ablated);
2) Auto **figures/tables** (calibration, Pareto, attribution, uncertainty, eco, ablation);
3) Short **videos** (time-lapse layers, route comparisons);
4) **Dataset card** + **Method card** + licensing/citation;
5) Exportable **zip bundle** for submission or reviewers.

_No breaking changes to existing CLI/UI; this is a wrapper that reuses existing artifacts and contracts._

---

## 1) Scope / Non-Goals
- **In scope:** scripts, figure/table/video generators, export bundler, docs, minimal dataset builder.
- **Out of scope:** new ML models or UI redesigns; heavy DVC/MLflow migrations (we reuse Catalog meta).

---

## 2) Artifacts & Contracts
**Inputs (reused)**
- Cataloged artifacts (`*.meta.json`) for selected months/scenarios
- Reports from previous phases (G/H/I/M/N/O)

**Outputs**
- **Repro scripts:** `scripts/repro/` (bash + powershell), `scripts/repro/reproduce_<profile>.sh|ps1`
- **Figures/Tables:** `reports/paper/figures/*.png`, `reports/paper/tables/*.csv|.md`
- **Videos:** `reports/paper/videos/*.mp4|.gif`
- **Docs:** 
  - `reports/paper/paper.md` (or `paper.tex`) with placeholders auto-filled
  - `reports/paper/dataset_card.md`, `reports/paper/method_card.md`
  - `CITATION.cff`, `LICENSE` (template)
- **Bundle:** `outputs/release/arcticroute_repro_<tag>.zip`
- **Meta:** each generated asset gets an adjacent `.meta.json` referencing inputs and run_id

**CLI**
- `paper.build` (figures/tables)
- `paper.video` (animations/overlays)
- `paper.bundle` (zip with checksums)
- `paper.check` (verify hashes, environment, versions)

---

## 3) Repro Profiles
Define 2–3 preset **profiles** under `config/paper_profiles.yaml`:
- `quick`: smallest grid, 1 month, few weight combos (runtime < 10 min)
- `full`: 2–3 months, full figures/tables
- `ablation`: toggles (no-CV, no-accident, no-prior, linear vs unetformer, mean vs CVaR)

Example:
```yaml
profiles:
  quick:
    months: [202412]
    scenarios: ["nsr_wbound"]
    figures: [calibration, pareto, attribution, uncertainty, eco]
    videos: [risk_tl, fused_tl, route_compare]
    ablations: []
  full:
    months: [202411, 202412, 202501]
    scenarios: ["nsr_wbound","nwp_eastbound"]
    figures: [calibration, pareto, attribution, uncertainty, eco, domain_bucket]
    videos: [risk_tl, fused_tl, route_compare]
    ablations: [no_cv, no_accident, no_prior, linear_fuse, cvar_vs_mean]
  ablation:
    months: [202412]
    scenarios: ["nsr_wbound"]
    figures: [ablation_grid]
    videos: []
    ablations: [no_cv, no_accident, no_prior, linear_fuse]
4) Tasks (Execution Order)
 P-01 Repro runner

New: ArcticRoute/paper/repro.py

run_profile(profile_id): orchestrates CLI calls (fuse/route/scan/report) from previous phases using LayerGraph/Recipe if available

Records run_id, git_sha, config_hash into a master repro_log.json

Shell wrappers: scripts/repro/reproduce_quick.sh|ps1, reproduce_full.sh|ps1

 P-02 Figure builder

New: ArcticRoute/paper/figures.py

fig_calibration, fig_pareto, fig_attribution, fig_uncertainty, fig_eco, fig_domain_bucket, fig_ablation_grid

Inputs: existing JSONs/CSVs from reports (H/I/G/M/L)

Outputs: reports/paper/figures/*.png (+ .meta.json)

 P-03 Table builder

New: ArcticRoute/paper/tables.py

tab_metrics_summary (distance/risk/CO₂ by scenario)

tab_ablation (deltas vs baseline)

Output CSV/Markdown under reports/paper/tables/

 P-04 Video pipeline

New: ArcticRoute/paper/video.py

make_timeline(layer_paths, out, fps=4) (risk/fused time-lapse)

make_route_compare(baseline, candidate, out) (side-by-side)

Reuse report.animate helpers if present (# REUSE)

 P-05 Docs & cards

Templates under docs/paper_templates/:

paper.md.tpl with placeholders for figures/tables auto-inserts

dataset_card.md.tpl, method_card.md.tpl

New: ArcticRoute/paper/render.py (Jinja2) → render to reports/paper/

 P-06 Bundle & checks

New: ArcticRoute/paper/bundle.py

Collect selected artifacts + docs into outputs/release/*.zip

Write SHA256SUMS.txt and MANIFEST.json (with logical_id listing)

paper.check verifies hashes, environment (python --version, torch, CUDA), and mandatory files

 P-07 CI smoke (optional)

GitHub Actions (or local script) to run paper.build --profile quick on push to release/*

5) Document Templates (key fields)
Dataset Card

Title, Languages/CRS (EPSG:3413), Spatial/Temporal coverage, Sources, Processing steps, Known issues, Licenses, Ethics

Method Card

Inputs/Outputs schema, Training signals (weak labels), Calibration metrics, Uncertainty, Routing objective, Limitations

6) CLI (Smoke)
bash
复制代码
# 1) Quick profile reproduction (figures + tables)
python -m ArcticRoute.api.cli paper.build --profile quick

# 2) Videos
python -m ArcticRoute.api.cli paper.video --profile quick

# 3) Bundle for reviewers
python -m ArcticRoute.api.cli paper.bundle --profile quick --tag v2.0-quick

# 4) Sanity check bundle
python -m ArcticRoute.api.cli paper.check --bundle outputs/release/arcticroute_repro_v2.0-quick.zip
7) Definition of Done (Acceptance)
 paper.build --profile quick produces ≥ 5 figures and ≥ 2 tables with correct meta and references

 paper.video writes ≥ 1 timeline and ≥ 1 route-compare video

 paper.bundle zips reproducible assets + docs + checksums; paper.check passes

 Docs (paper.md, dataset/method cards) are rendered with populated numbers and linked figures/tables

 No changes to existing pipelines; failures are isolated to paper/*

8) Risks & Mitigations
Missing inputs: figure builders verify presence; fall back with clear warnings and skip specific panels

Large bundles: include only curated artifacts; compress videos (MP4 preferred)

Environment drift: paper.check captures versions/hashes; recommend pinned requirements.txt

9) Minimal Data Checklist
At least one month (e.g., 202412) with full chain artifacts (risk/prior/routes/reports)

Optional: second scenario for route-compare video

No new external data required