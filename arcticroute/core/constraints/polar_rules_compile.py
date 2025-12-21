from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8", errors="ignore")) or {}

def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    # b overwrites a
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def _merge_sources(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    ai = (a.get("sources") or {}).get("items") or []
    bi = (b.get("sources") or {}).get("items") or []
    merged = {}
    for it in ai + bi:
        if isinstance(it, dict) and it.get("id"):
            merged[it["id"]] = it
    out = dict(a.get("sources") or {})
    out["items"] = list(merged.values())
    return out

def compile_authoritative_rules(
    template_path: str,
    override_path: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    tpl_p = Path(template_path)
    ov_p = Path(override_path) if override_path else None

    tpl = _load_yaml(tpl_p)
    ov = _load_yaml(ov_p) if ov_p else {}

    compiled = _deep_merge(tpl, ov)
    compiled["sources"] = _merge_sources(tpl, ov)

    # count missing threshold values
    thresholds = (compiled.get("thresholds") or {})
    missing = 0
    filled = 0
    for k, spec in thresholds.items():
        if not isinstance(spec, dict):
            continue
        if spec.get("value") is None:
            missing += 1
        else:
            filled += 1

    meta = {
        "ruleset_id": (compiled.get("meta") or {}).get("ruleset_id"),
        "template_path": str(tpl_p),
        "override_path": str(ov_p) if ov_p else None,
        "missing_count": missing,
        "filled_count": filled,
    }
    compiled.setdefault("meta", {})
    compiled["meta"]["compiled_from"] = "template+overrides"
    compiled["meta"]["missing_count"] = missing
    compiled["meta"]["filled_count"] = filled
    return compiled, meta
