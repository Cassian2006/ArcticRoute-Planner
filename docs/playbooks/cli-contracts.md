```markdown
# CLI / Artifacts / Naming Contracts

> Defines CLI interfaces, file naming, metadata, and compatibility rules for all ArcticRoute commands.

---

## 1. Common Parameters & Exit Codes
- `--ym YYYYMM` — target month (required)
- `--config PATH` — YAML config override
- `--out-dir PATH` — default `data_processed/...`
- `--run-id STR` — defaults to `YYYYMMDD-HHMMSS-<sha7>`
- Exit codes: `0` success, `>0` failure (with human-readable advice)
- Every successful run must write `<file>.meta.json`

---

## 2. Existing Stable Commands
prior.ds.prepare
prior.train
prior.embed
prior.cluster
prior.centerline
prior.eval
prior.export
prior.select

yaml
复制代码

---

## 3. New / Planned Commands
> **Alpha** = may change; **Beta** = interface frozen.

### 3.1 Risk & Fusion
risk.fuse --ym --method {linear,stacking,unetformer,poe,evidential,crf} [--config]
risk.interact.build --ym [--method dcpa-tcpa]
risk.ice.apply-escort --ym --eta 0.3

markdown
复制代码
- Inputs auto-discovered (`risk/*.nc`, `prior/*.nc`)
- Output: `data_processed/risk/risk_fused_<ym>.nc`
- Intermediate files: `R_interact_<ym>.nc`, `R_ice_eff_<ym>.nc`

### 3.2 Routing & Aggregation
route --ym --risk-source {ice|fused}
[--risk-agg {mean,q,cvar}] [--alpha 0.95]
[--w_r --w_d --w_p --w_c]

markdown
复制代码
- Outputs: GeoJSON / HTML / PNG / metrics JSON

### 3.3 Orchestration & Reporting
recipe run -f configs/recipes/<name>.yaml
catalog ls|show <logical_id>
report.build --ym --include metrics,calibration,pareto,attribution

yaml
复制代码

---

## 4. Naming Rules
- Path: `data_processed/<domain>/<name>_<ym>.nc`
- Variable names:
  - Fused risk: `risk`
  - Prior penalty: `prior_penalty`
  - Interaction: `risk` (filename carries `R_interact`)
- Always write a `.meta.json` next to each data file.

---

## 5. Meta JSON Schema
```json
{
  "logical_id": "ar://risk/fused@202412?grid=3413",
  "path": "data_processed/risk/risk_fused_202412.nc",
  "inputs": ["ar://risk/ice@202412","ar://risk/wave@202412"],
  "created_at": "2025-01-15T12:34:56Z",
  "run_id": "20250115-123456-a1b2c3d",
  "git_sha": "a1b2c3d",
  "config_hash": "sha1:...",
  "metrics": {"auc": 0.78, "ece": 0.06},
  "ui_tags": ["phaseF","fused","adv"],
  "warnings": [],
  "attrs": {"grid_id": "EPSG:3413","var_name": "risk","units": "1"}
}
6. Validation Checklist
Dimensions (time?, y, x); coords monotonic

Values ∈ [0, 1]

attrs include grid_id, var_name, source

.meta.json includes all required fields above

7. Compatibility Policy
Advanced outputs may use _adv_ suffix, then soft-link to legacy path.

New CLI params must have safe defaults; no breaking changes.