from __future__ import annotations

"""
Weight grid & presets expansion for Phase G.

iter_weight_grid(spec: dict) -> Iterator[dict]
- spec example:
  { "presets": {safe:{...}, efficient:{...}, balanced:{...}},
    "grid": {"w_r":[...],"w_d":[...],"w_p":[...],"w_c":[...]}}
- yields dicts with keys among {w_r,w_d,w_p,w_c,label}
"""
from itertools import product
from typing import Dict, Iterator, List, Optional


def _clean_weight(w: Optional[float], default: float = 1.0) -> float:
    try:
        v = float(w) if w is not None else default
        if v < 0:
            v = 0.0
        return v
    except Exception:
        return default


def iter_weight_grid(spec: Dict) -> Iterator[Dict[str, float]]:
    presets = (spec or {}).get("presets", {}) or {}
    grid = (spec or {}).get("grid", {}) or {}

    # yield presets first
    for label, ws in presets.items():
        yield {
            "label": str(label),
            "w_r": _clean_weight(ws.get("w_r"), 1.0),
            "w_d": _clean_weight(ws.get("w_d"), 1.0),
            "w_p": _clean_weight(ws.get("w_p"), 0.0),
            "w_c": _clean_weight(ws.get("w_c"), 0.0),
        }

    # expand grid
    wr = [float(x) for x in grid.get("w_r", [1.0])]
    wd = [float(x) for x in grid.get("w_d", [1.0])]
    wp = [float(x) for x in grid.get("w_p", [0.0])]
    wc = [float(x) for x in grid.get("w_c", [0.0])]

    for a, b, c, d in product(wr, wd, wp, wc):
        yield {"w_r": float(a), "w_d": float(b), "w_p": float(c), "w_c": float(d)}


__all__ = ["iter_weight_grid"]

