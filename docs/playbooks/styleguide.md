# ArcticRoute Code Style & Engineering Conventions

> Goal: Maintain performance and reproducibility while maximizing readability and predictability.  
> Applies to all modules under `ArcticRoute/`.

---

## 1. Language & Version
- Python **3.10+** required.
- Full **type annotations** (PEP 484) on all public APIs, class methods, and functions.
- Use `@dataclass` for structured records; prefer `@dataclass(frozen=True)` for immutable configs.

---

## 2. Naming & Module Layout
| Element | Style | Example |
|----------|--------|----------|
| Module/File | `lower_snake_case.py` | `fusion_adv.py` |
| Class | `PascalCase` | `AStarGridTimePlanner` |
| Function/Variable | `lower_snake_case` | `aggregate_risk` |
| Constant | `UPPER_CASE` | `MAX_EPOCHS` |

- **Grid dimensions:** `time, y, x`; coordinates `lat, lon`.
- **Variable names:**  
  - Risk layer: `risk ∈ [0,1]`  
  - Prior penalty: `prior_penalty ∈ [0,1]`  
  - Congestion / interaction: `congest`, `r_interact`
- **Directory roles:**
  - `core/` — algorithmic core (risk, prior, planners, predictors, route, cost, congest, utils)
  - `io/` — ingestion, alignment, data access
  - `api/` — CLI entrypoints
  - `apps/` — Streamlit UI & visualization
  - `config/` — YAMLs & schemas
  - `data_processed/` — generated artifacts (read-only)
  - `reports/` — HTML/PNG/JSON outputs

---

## 3. Documentation & Comments
- Use **NumPy-style docstrings**.  
- Every public API must specify parameters, returns, exceptions, and examples.
- Always state coordinate system, units, and projection.

```python
def aggregate_risk(risk: xr.DataArray, mode: str = "mean", alpha: float = 0.95) -> xr.DataArray:
    """
    Aggregate risk into a scalar surface.

    Parameters
    ----------
    risk : xarray.DataArray
        Risk values in [0, 1], dims: (time?, y, x)
    mode : {"mean", "q", "cvar"}
        Aggregation mode.
    alpha : float
        Quantile / CVaR level.

    Returns
    -------
    xarray.DataArray
        Aggregated risk, same coords.
    """
4. IO / NetCDF / Catalog Rules
Compression: zlib=True, complevel=4; dtype float32; _FillValue=np.nan

Chunking: follow config/runtime.yaml or grid_spec.json

Global attrs: grid_id, created_at, run_id, git_sha, config_hash, source, units, var_name

Catalog meta: every output must have <file>.meta.json

Safe write: write to temp → atomic rename.

5. Numerical Conventions
Clamp probabilities: np.clip(arr, 0, 1, out=arr)

Always record normalization in attrs['norm'], e.g. "quantile@0.01-0.99".

Use reindex_like for alignment; never interpolate silently.

6. Logging & Error Handling
Use logging; no print() in library code.

CLI may use click.echo.

Each CLI run must log run_id / git_sha / config_hash / inputs → .meta.json.

Recoverable issues: use warnings.warn and record in meta.json.

Exceptions must include actionable next-step hints.

7. Config & Reproducibility
All configs in config/*.yaml; CLI params only override essentials.

Fix random seeds for training / sampling; store seed in meta.json.

8. Performance
Prefer xarray vectorization; avoid Python loops.

Use Dask only when necessary; call .compute() before writing.

If using Numba, wrap with @njit(cache=True) and provide pure-Python fallback.

9. PyTorch Guidelines
Call torch.set_float32_matmul_precision("high").

Support bf16/AMP toggles via CLI.

Pass device and precision from caller; never hard-code.

10. Commit Conventions
Follow Conventional Commits:

vbnet
复制代码
feat: add risk.interact build (dcpa/tcpa)
fix: fusion quantile-norm edge-case with NaNs
refactor: move precheck to LayerGraph.align
docs: add phase-F-convoy playbook
test: add aggregator cvar smoke
11. Imports & Dependencies
Prefer relative imports within package; absolute for cross-domain modules.

Delay heavy optional imports (sklearn, etc.) inside functions.