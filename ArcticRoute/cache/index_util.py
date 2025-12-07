"""Cache index read/write utilities (Phase A minimal).

- register_artifact(run_id, kind, path, attrs): append/update artifact entry into cache/index/cache_index.json
- find_artifacts(kind, month=None): return list of entries filtered by kind (and optional month key in attrs)

Notes:
- JSON schema: {"artifacts": [ {"run_id": str, "kind": str, "path": str, "attrs": dict, "ts": "YYYYMMDDTHHMMSS"} ], "version": 1}
- File location: <REPO_ROOT>/cache/index/cache_index.json
- Safe to import when file missing (lazy create on first register)
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = REPO_ROOT / "cache" / "index"
INDEX_FILE = INDEX_DIR / "cache_index.json"


def _load() -> Dict[str, Any]:
    if not INDEX_FILE.exists():
        return {"version": 1, "artifacts": []}
    try:
        return json.loads(INDEX_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "artifacts": []}


def _save(payload: Dict[str, Any]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def register_artifact(run_id: str, kind: str, path: str, attrs: Dict[str, Any] | None = None) -> None:
    """Register or update an artifact entry in the cache index.

    - run_id: execution id (e.g., 20251106T080000)
    - kind: logical artifact type, e.g., "risk_grid", "prior_model", "report_html"
    - path: filesystem path (as given)
    - attrs: free-form metadata; recommended keys include: month (YYYYMM), ym, hash, size, notes
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id required")
    if not isinstance(kind, str) or not kind:
        raise ValueError("kind required")
    if not isinstance(path, str) or not path:
        raise ValueError("path required")
    data = _load()
    arts: List[Dict[str, Any]] = list(data.get("artifacts") or [])
    ts = time.strftime("%Y%m%dT%H%M%S")
    entry = {"run_id": run_id, "kind": kind, "path": path, "attrs": dict(attrs or {}), "ts": ts}

    # Upsert: replace if same run_id & kind & path
    replaced = False
    for i, it in enumerate(arts):
        if it.get("run_id") == run_id and it.get("kind") == kind and it.get("path") == path:
            arts[i] = entry
            replaced = True
            break
    if not replaced:
        arts.append(entry)
    data["artifacts"] = arts
    _save(data)


def find_artifacts(kind: str, month: Optional[str] = None) -> List[Dict[str, Any]]:
    """Find artifacts by kind, optionally filtered by month (attrs.month or attrs.ym).

    Returns a list of entries (most recent first).
    """
    if not isinstance(kind, str) or not kind:
        return []
    data = _load()
    arts: List[Dict[str, Any]] = list(data.get("artifacts") or [])
    out: List[Dict[str, Any]] = []
    for it in arts:
        if it.get("kind") != kind:
            continue
        if month:
            a = it.get("attrs") or {}
            ym = a.get("month") or a.get("ym")
            if str(ym) != str(month):
                continue
        out.append(it)
    # sort by ts desc
    out.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
    return out


__all__ = ["register_artifact", "find_artifacts", "INDEX_FILE"]
























