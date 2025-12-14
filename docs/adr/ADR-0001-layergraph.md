# ADR-0001: Introduce LayerGraph + Catalog + Plugins Architecture

- **Status:** Accepted  
- **Date:** 2025-11-09  
- **Owner:** ArcticRoute Core Team  
- **Related Phases:** F (Convoy), G (Pareto), H (Explainability)

---

## Context
Current state:
- `prior.*` pipeline is complete, but `risk.fuse` still relies on **linear weighting**, limiting multimodal methods (UNet, Transformer, Evidential, PoE, CRF).
- Artifact tracking is implicit; no standardized metadata.
- UI/CLI must remain backward-compatible while enabling new fusion and routing schemes (quantile/CVaR).

---

## Decision
Adopt a **LayerGraph + Catalog + Plugins** lightweight architecture.

1. **LayerGraph Runner** — YAML-based recipe describing `load → align → fuse → route → report`; handles alignment, caching, orchestration.  
2. **Catalog** — every artifact writes a `.meta.json` containing logical ID, inputs, hashes, metrics; supports `catalog ls/show/verify`.  
3. **Fusion Plugins** — unified API wrapping `stacking / unetformer / poe / evidential / crf`; training + inference consistent; outputs `risk ∈ [0,1]` (+ uncertainty).  
4. **CostChannel** — route-side aggregator (`mean / quantile / CVaR`), independent of A* core; integrates evidential variance when available.

Interfaces remain unchanged: outputs still written to `data_processed/risk/risk_fused_<ym>.nc`; UI reuses the existing "Risk" layer.

---

## Architecture Diagram

```mermaid
graph TD
  SRC[SOURCES<br/>AIS/Ice/Wave/Accident/Prior/Congest] --> IO[IO & Features<br/>ingest/align/feature]
  IO --> LAYERS[Layers<br/>R_ice/R_wave/R_acc/PriorPenalty/R_interact]
  LAYERS --> GRAPH[LayerGraph Runner<br/>Recipe/Align/Cache]
  GRAPH --> FUSION[Fusion Plugins<br/>stacking/unetformer/poe/evidential/crf]
  FUSION --> RISK[risk_fused_<ym>.nc]
  GRAPH --> ROUTE[Routing<br/>A* + CostChannel(mean/q/cvar)]
  RISK --> ROUTE
  ROUTE --> REPORT[Reports/UI]
  GRAPH --> CATALOG[Catalog<br/>*.meta.json]
  RISK --> CATALOG
  ROUTE --> CATALOG
  REPORT --> CATALOG
Alternatives Considered
Option	Verdict	Reason
Keep script-based pipelines	❌	No traceability; poor modularity
Heavy schedulers (Airflow / Prefect / Luigi)	❌	Overkill for research scale
Full MLflow/DVC migration	⚠️	High coupling, limited short-term benefit

Consequences
✅ Pluggable fusion methods, reproducible experiments, unified reporting

⚠️ Initial cost: new catalog.py, recipe.py, and plugin skeleton

🚧 Potential regression risk → mitigated via soft-link fallback + A/B metrics + UI toggle

Migration Plan
Add minimal Catalog + Runner wrapping existing risk.fuse.precheck.

Replace linear weighting with stacking (calibrated).

Integrate unetformer + crf post-processing; add calibration plots.

Introduce evidential / poe and route-side CVaR@α.

YAML-Recipe support for Phase G (Pareto scans) and Phase H (Attribution / Animation).

Plugin Interface
python
复制代码
class FusionPlugin:
    name: str = "base"
    def fit(self, inputs: dict[str, xr.DataArray],
            labels: Optional[xr.DataArray] = None, **kw) -> None: ...
    def transform(self, inputs: dict[str, xr.DataArray], **kw) -> xr.DataArray: ...
    def save(self, path: str) -> None: ...
    @classmethod
    def load(cls, path: str) -> "FusionPlugin": ...
Open Questions
Should grid_spec.json become a formal spec (chunks/compression/CRS)?

Mapping of logical_id → remote object storage?

Is online CVaR approximation within A* worthwhile, or keep offline aggregation?

Review Schedule
Re-evaluate after Phases F & G

Metrics: ECE improvement, path-level risk reduction, reproducibility