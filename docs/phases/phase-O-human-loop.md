# Phase O | Human-in-the-Loop Review, Constraints & Field Validation

## 0) Context & Goal
**Context.** Phases K–N delivered multimodal fusion, domain adaptation, eco-routing, and online replanning. We now introduce a **review/approval loop** where experts can annotate routes, add **constraints** (no-go areas, minimum clearances, locked waypoints), and trigger constrained re-plans. All changes are auditable and reversible. When disabled, the system behaves exactly as before.

**Goal.**
1) **Review UI & schema** for annotations and approvals;
2) **Constraints engine** (no-go polygons, corridor locks, min-distance to ice edge/coast);
3) **Feedback → plan**: translate annotations into cost/constraints, replan, and produce A/B comparisons;
4) **Approved artifacts** with signatures and audit trail.

_No breaking changes to existing CLI/UI when the feature is off._

---

## 1) Scope / Non-Goals
- **In scope:** annotations, constraints, stitching with locked segments, A/B evaluation, approval artifacts, reports.
- **Out of scope:** multi-user auth/permissions, server-side comment threads; we store review data locally as JSON/JSONL.

---

## 2) Artifacts & Contracts
**Inputs**
- Existing layers: `risk_fused_<ym>.nc`, `prior_corridor_selected_<ym>.nc`, `R_interact_<ym>.nc` (optional), eco config if enabled
- Route candidates from Phase G/N: `data_processed/routes/route_<ym>_<scenario>_*.geojson`

**Outputs**
- **Review pack**: `reports/d_stage/phaseO/review_<scenario>_<ts>.zip` (maps, metrics, route JSON, empty feedback template)
- **Feedback file**: `data_processed/review/feedback_<scenario>_<ts>.jsonl`
  - Each line: `{route_id, segment_idx?, tag, severity, note, geometry?}`
- **Constraints** (derived):
  - `data_processed/constraints/no_go_<scenario>_<ts>.geojson`
  - `data_processed/constraints/locks_<scenario>_<ts>.geojson` (locked waypoints/segments)
- **Constrained route(s)**:
  - `data_processed/routes/route_<scenario>_<ts>_constrained.geojson`
- **Approval**:
  - `data_processed/routes/approved/route_<scenario>_<ts>_approved.geojson`
  - Adjacent `.meta.json` includes `approved_by`, `approved_at`, `feedback_digest`, `violations_resolved=true/false`
- **Reports**:
  - `reports/d_stage/phaseO/review_summary_<scenario>_<ts>.{json,html}` (what changed, why; A/B tables; violation list)

**Compatibility**
- Shapes/vars unchanged; only adds new optional artifacts.

---

## 3) Tag Taxonomy (default, configurable)
- `avoid_ice_edge`, `avoid_shallow`, `avoid_traffic_hotspot`, `avoid_mpa` (marine protected area)
- `lock_corridor` (do not deviate beyond X nm)
- `no_go_polygon` (polygon geometry required)
- `min_clearance_km`: parameterized numeric tag
- `prefer_prior` / `prefer_smoother` (soft preferences)
- `comment` (free-text; no routing effect)

Defined in `config/review.yaml`.

---

## 4) Tasks (Execution Order)
- [ ] **O-01 Feedback schema & ingest**
  - New: `ArcticRoute/core/feedback/schema.py`  (# REUSE utils if present)
    - Validate JSONL: tags, severities, optional geometries (WKT/GeoJSON)
  - New: `ArcticRoute/io/feedback_ingest.py`
    - Merge multiple reviewers; deduplicate; compute a summary digest

- [ ] **O-02 Constraints engine**
  - New: `ArcticRoute/core/constraints/engine.py`
    - Inputs: feedback digest, `config/constraints.yaml`
    - Build:
      - **No-go mask** from polygons/buffers (coastline, MPA, ice edge offset)
      - **Min-distance** raster to ice edge/coastline: enforce `dist >= d_min`
      - **Corridor locks**: band buffer around selected polyline(s)
    - Output:
      - `constraints_mask` (hard constraints)
      - `constraints_soft_cost` (penalty field, normalized [0,1])

- [ ] **O-03 Locked-waypoints stitching**
  - New: `ArcticRoute/core/route/locks.py`
    - Split old route at lock points; A* only optimizes between locks; forbid backward edges
    - Ensure continuity and no self-crossings

- [ ] **O-04 Review & approve CLI**
  - Edit `ArcticRoute/api/cli.py`:
    - `route.review --scenario <id> --ym <YYYYMM> --out ...` → emit review pack & empty feedback template
    - `route.apply.feedback --scenario <id> --ym <YYYYMM> --feedback FILE.jsonl [--locks FILE.geojson] [--no-go FILE.geojson]`
      - Build constraints → constrained replan → write constrained route + summary
    - `route.approve --route FILE.geojson --by "Name <email>" --note "..."` → move to approved/ + meta

- [ ] **O-05 UI Review tab**
  - `apps/app_min.py`:
    - Load candidate route → add pins (segment-level), draw polygon (no-go), set min clearance, toggle “lock corridor”
    - Export feedback JSONL & optional polygons; button to “Apply & Replan”
    - Diff viewer: before vs after; list violations resolved

- [ ] **O-06 Violation checker**
  - New: `ArcticRoute/core/constraints/checker.py`
    - `check(route, constraints) -> {violations[], stats}`
  - CLI: `constraints.check --route FILE --ym <YYYYMM>`

- [ ] **O-07 A/B evaluation**
  - New: `ArcticRoute/core/route/abtest.py`
    - Compare baseline vs constrained: distance/risk/CO₂, #violations, #locks satisfied
  - Report builder extension: add A/B tables to review summary

- [ ] **O-08 Tests**
  - Unit: mask application, buffer distances, lock stitching (no backward), checker catches violations
  - E2E: import feedback → rebuild constraints → replan → fewer violations; meta & report written

---

## 5) Config Samples

### 5.1 Review tags (`config/review.yaml`)
```yaml
tags:
  avoid_ice_edge: {severity: [low, med, high], buffer_km_default: 3}
  avoid_traffic_hotspot: {severity: [med, high]}
  no_go_polygon: {}
  lock_corridor: {band_km_default: 5}
  min_clearance_km: {min: 1, max: 20, default: 3}
  prefer_prior: {}
  prefer_smoother: {}
5.2 Constraints (config/constraints.yaml)
yaml
复制代码
constraints:
  min_clearance_km_default: 3
  coast_buffer_km: 1
  ice_edge_source: data_processed/features/edge_<ym>.nc
  mpa_geojson: data_processed/env/mpa.geojson   # optional
6) CLI (Smoke)
bash
复制代码
# 1) Create a review pack
python -m ArcticRoute.api.cli route.review --scenario nsr_wbound --ym 202412 --out reports/d_stage/phaseO/

# 2) Apply feedback and replan with constraints
python -m ArcticRoute.api.cli route.apply.feedback \
  --scenario nsr_wbound --ym 202412 \
  --feedback data_processed/review/feedback_nsr_wbound_202412.jsonl

# 3) Check violations on the new route
python -m ArcticRoute.api.cli constraints.check \
  --route data_processed/routes/route_nsr_wbound_202412_constrained.geojson --ym 202412

# 4) Approve the constrained route
python -m ArcticRoute.api.cli route.approve \
  --route data_processed/routes/route_nsr_wbound_202412_constrained.geojson \
  --by "Reviewer A <rev@example.com>" --note "Clears ice-edge hotspot"
7) Definition of Done (Acceptance)
 route.review produces a pack and an empty feedback template

 route.apply.feedback generates constraints and a constrained route; violations decrease vs baseline

 constraints.check passes with violations=0 or documented exceptions

 route.approve writes approved artifact with meta (approved_by, approved_at, digest)

 UI Review tab allows drawing polygons, setting clearances, locking segments, exporting feedback

 All artifacts write adjacent .meta.json; no breaking changes when feature is unused

8) Risks & Mitigations
Conflicting constraints → show checker output; allow per-constraint priority; fallback to soft penalties

Over-constraining → warn when feasible path cannot be found; suggest relaxing buffers

Manual errors → schema validation; undo history (keep old routes versioned)

Drift between UI & CLI → use the same feedback schema (schema.py)