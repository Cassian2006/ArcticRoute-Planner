from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple
import math


def nondominated(points: Sequence[Dict], keys: Tuple[str, ...] = ("risk_integral", "distance_km", "congest_integral")) -> List[int]:
    """Return indices of non-dominated points (minimize all keys). Missing keys are treated as +inf.
    """
    n = len(points)
    nd: List[int] = []
    def val(p: Dict, k: str) -> float:
        v = p.get(k)
        try:
            return float(v)
        except Exception:
            return float("inf")
    for i in range(n):
        pi = points[i]
        dominated = False
        for j in range(n):
            if j == i:
                continue
            pj = points[j]
            better_or_equal = True
            strictly_better = False
            for k in keys:
                vi = val(pi, k)
                vj = val(pj, k)
                if vj > vi:
                    better_or_equal = False
                    break
                if vj < vi:
                    strictly_better = True
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            nd.append(i)
    return nd


def pick_representatives(points: Sequence[Dict]) -> Dict[str, int]:
    """Pick indices for safe (min risk), efficient (min distance), balanced (closest to z-score origin)."""
    if not points:
        return {"safe": -1, "efficient": -1, "balanced": -1}
    # safe
    safe = min(range(len(points)), key=lambda i: float(points[i].get("risk_integral", float("inf"))))
    # efficient
    efficient = min(range(len(points)), key=lambda i: float(points[i].get("distance_km", float("inf"))))
    # balanced: z-score normalization of [risk, distance, congest]
    vals = {k: [] for k in ("risk_integral", "distance_km", "congest_integral")}
    for p in points:
        for k in vals:
            vals[k].append(float(p.get(k, float("inf"))))
    means = {k: (sum(v)/len(v) if v else 0.0) for k, v in vals.items()}
    stds = {k: (math.sqrt(sum((x-means[k])**2 for x in v)/len(v)) if v else 1.0) for k, v in vals.items()}
    def zsum(i: int) -> float:
        s = 0.0
        for k in vals:
            x = float(points[i].get(k, float("inf")))
            s += ((x - means[k]) / (stds[k] or 1.0))**2
        return s
    balanced = min(range(len(points)), key=zsum)
    return {"safe": safe, "efficient": efficient, "balanced": balanced}


__all__ = ["nondominated", "pick_representatives"]

