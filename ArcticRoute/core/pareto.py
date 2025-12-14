from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

import math

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None


@dataclass(frozen=True)
class ParetoSolution:
    key: str
    objectives: Dict[str, float]          # e.g. distance_km, total_cost, edl_risk, edl_uncertainty
    route: List[tuple[float, float]]      # (lat, lon)
    component_totals: Dict[str, float]    # e.g. ice_risk, wave_risk, ais_corridor_reward ...
    meta: Dict[str, Any]                  # weights / toggles / notes


def dominates(a: ParetoSolution, b: ParetoSolution, fields: Sequence[str]) -> bool:
    """a dominates b if a is no worse in all fields and better in at least one (minimization)."""
    better = False
    for f in fields:
        av = float(a.objectives.get(f, 0.0))
        bv = float(b.objectives.get(f, 0.0))
        if math.isnan(av) and not math.isnan(bv):
            return False
        if av > bv:
            return False
        if av < bv:
            better = True
    return better


def pareto_front(cands: Sequence[ParetoSolution], fields: Sequence[str]) -> List[ParetoSolution]:
    out: List[ParetoSolution] = []
    for i, a in enumerate(cands):
        dominated = False
        for j, b in enumerate(cands):
            if i == j:
                continue
            if dominates(b, a, fields):
                dominated = True
                break
        if not dominated:
            out.append(a)
    return out


def extract_objectives_from_breakdown(breakdown: Any) -> Dict[str, float]:
    """
    breakdown is RouteCostBreakdown (see arcticroute.core.analysis.compute_route_cost_breakdown)
    Fields are best-effort: missing -> 0.0
    """
    obj: Dict[str, float] = {}
    # distance_km: prefer s_km[-1]
    dist = 0.0
    try:
        s = getattr(breakdown, "s_km", None)
        if s:
            dist = float(s[-1])
    except Exception:
        dist = 0.0
    obj["distance_km"] = dist

    # total_cost
    obj["total_cost"] = float(getattr(breakdown, "total_cost", 0.0) or 0.0)

    # edl components (best-effort)
    ct = getattr(breakdown, "component_totals", {}) or {}
    obj["edl_risk"] = float(ct.get("edl_risk", 0.0) or 0.0)

    # uncertainty may appear under different keys
    obj["edl_uncertainty"] = float(
        ct.get("edl_uncertainty_penalty", ct.get("edl_uncertainty", 0.0)) or 0.0
    )
    return obj


def solutions_to_dataframe(solutions: Sequence[ParetoSolution]):
    if pd is None:
        raise RuntimeError("pandas is required for solutions_to_dataframe()")

    rows = []
    for s in solutions:
        row = {"key": s.key}
        row.update(s.objectives)
        # include common components if present
        for k in [
            "ice_risk", "wave_risk", "ais_risk", "ais_density", "ais_corridor_reward",
            "base_distance",
        ]:
            if k in s.component_totals:
                row[k] = float(s.component_totals.get(k, 0.0) or 0.0)
        rows.append(row)

    df = pd.DataFrame(rows)
    # ensure stable columns
    for col in ["distance_km", "total_cost", "edl_risk", "edl_uncertainty"]:
        if col not in df.columns:
            df[col] = 0.0
    return df
