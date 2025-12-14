# Testing & Quality Assurance

> Goal: Small, fast, deterministic tests covering the entire pipeline.

---

## 1. Tooling & Layers
- Framework: `pytest`
- Linters: `ruff` (or `flake8`), `black`, `isort`
- Type checker: `mypy`
- Test layers:
  - **unit** — single function (< 1 s)
  - **integration** — cross-module (< 30 s)
  - **e2e-smoke** — full CLI run (< 2 min)
  - **gpu** — optional, mark with `@pytest.mark.gpu`

---

## 2. Fixtures
- `tests/fixtures/grid.py`: small grid (64×64)
- `tests/fixtures/risk.py`: synthetic `R_ice, R_wave, R_acc`
- `tests/fixtures/ais.py`: 3–5 sample tracks + `segment_index.parquet`
- All fixtures must be **deterministic** (fixed seed).

---

## 3. Assertions
- Numerical tolerance: `np.allclose(..., rtol=1e-5, atol=1e-6)`
- Risk values ∈ [0, 1]; assert clip count
- Dimensions: must match `(time?, y, x)` and aligned coords

---

## 4. Critical Coverage
- `core/risk/fusion.py`: quantile normalization, missing-layer fallback
- `core/cost/env_risk_cost.py`: monotonic response to weights
- `io/align.py`: attrs propagation
- **Phase F new tests:**
  - `congest/encounter.py`: DCPA/TCPA symmetry, physical sanity
  - `risk/escort.py`: boundary cases `eta=0` and `0.3`

---

## 5. CLI Smoke Tests
```bash
python -m ArcticRoute.api.cli risk.ice.apply-escort --ym 202412 --eta 0.3
python -m ArcticRoute.api.cli risk.interact.build --ym 202412
python -m ArcticRoute.api.cli risk.fuse --ym 202412 --method stacking
Assertions:

Output files exist

Corresponding .meta.json parses correctly

Required attrs present

6. Baseline & Regression
Metrics: AUC, Brier, ECE (+ path-level total risk / detour %)

Baselines stored in tests/baselines/*.json

CI fails if metrics fall below baseline

7. Marks & Skips
Slow tests: @pytest.mark.slow

GPU: @pytest.mark.gpu

Skip when optional deps missing (pytest.skip("requires sklearn"))

8. Coverage & Quality Gates
Overall coverage ≥ 70 %; core modules ≥ 85 %

Lint errors = 0; black/isort formatting enforced

9. Benchmarks (optional)
tests/benchmarks/ may include micro-benchmarks

Record run time across CI builds (non-blocking)