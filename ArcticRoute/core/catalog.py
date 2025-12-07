from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# REUSE cache index utilities
try:
    from ArcticRoute.cache.index_util import find_artifacts, INDEX_FILE  # type: ignore
except Exception:  # pragma: no cover
    find_artifacts = None  # type: ignore
    INDEX_FILE = Path(__file__).resolve().parents[2] / "cache" / "index" / "cache_index.json"  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class CatalogEntry:
    path: str
    kind: str
    ym: Optional[str] = None
    run_id: Optional[str] = None
    size: Optional[int] = None
    hash: Optional[str] = None


def _month_key(ym: Optional[str]) -> Tuple[int, int]:
    if not ym or not ym.isdigit() or len(ym) != 6:
        return (0, 0)
    return (int(ym[:4]), int(ym[4:]))


def _iter_index_entries(tags: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """Yield artifact entries from cache index; fall back to empty when missing.
    """
    if INDEX_FILE and Path(INDEX_FILE).exists():
        try:
            obj = json.loads(Path(INDEX_FILE).read_text(encoding="utf-8"))
        except Exception:
            obj = {"artifacts": []}
        arts: List[Dict[str, Any]] = list(obj.get("artifacts") or [])
        if tags:
            tagset = set(tags)
            arts = [e for e in arts if str(e.get("kind")) in tagset]
        return arts
    # 没有索引时返回空
    return []


def gc_list(keep_months: int = 6, tags: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Return GC candidates by month for given tags.

    - keep_months: 最近月份保留数
    - tags: 过滤 kind（如 risk_grid, prior_model, report_html 等）

    返回：{"kept": [...], "candidates": [...], "months": [...], "stats": {...}}
    每个 entry 至少包含 path/kind/ym/size/hash/run_id。
    """
    arts = _iter_index_entries(tags)
    # 归一
    norm: List[Dict[str, Any]] = []
    for e in arts:
        attrs = dict(e.get("attrs") or {})
        norm.append({
            "path": e.get("path"),
            "kind": e.get("kind"),
            "ym": attrs.get("month") or attrs.get("ym"),
            "size": attrs.get("size") or attrs.get("size_bytes"),
            "hash": attrs.get("hash"),
            "run_id": e.get("run_id"),
        })
    months = sorted({str(e.get("ym")) for e in norm if e.get("ym")}, key=_month_key)
    keep_set = set(months[-keep_months:]) if keep_months and months else set()

    kept: List[Dict[str, Any]] = []
    candidates: List[Dict[str, Any]] = []
    for e in norm:
        ym = e.get("ym")
        if ym and ym in keep_set:
            kept.append(e)
        else:
            candidates.append(e)
    stats = {
        "total": len(norm),
        "kept": len(kept),
        "candidates": len(candidates),
        "months": months,
    }
    return {"kept": kept, "candidates": candidates, "months": months, "stats": stats}


def _hash_file(path: Path, algo: str = "sha1", chunk: int = 1024 * 1024) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def verify(logical_id: str) -> Dict[str, Any]:
    """Verify artifact(s) by logical_id (kind or basename pattern).

    兼容策略：
    - 若 logical_id 对应于 kind（精确匹配），返回该 kind 最新项的状态
    - 否则，将其视为文件名片段（如 prior_transformer_202412.nc），在 REPO_ROOT 下搜索
    返回：{"ok": bool, "items": [{path, exists, size, hash?}], "hint": str}
    """
    items: List[Dict[str, Any]] = []
    ok = True
    hint = ""

    # 1) kind 精确匹配
    arts = _iter_index_entries(tags=[logical_id])
    if arts:
        # 取最近 ts 的一个
        arts.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
        e = arts[0]
        p = Path(str(e.get("path")))
        if not p.is_absolute():
            p = REPO_ROOT / p
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        item = {"path": str(p), "exists": exists, "size": size}
        if exists:
            try:
                item["hash"] = _hash_file(p)
            except Exception:
                pass
        items.append(item)
        ok = ok and exists
        hint = "kind"
        return {"ok": ok, "items": items, "hint": hint}

    # 2) 文件名片段搜索
    glob_pat = f"**/*{logical_id}*"
    for p in REPO_ROOT.glob(glob_pat):
        if p.is_file():
            try:
                size = p.stat().st_size
            except Exception:
                size = 0
            items.append({"path": str(p), "exists": True, "size": size})
    ok = ok and (len(items) > 0)
    hint = "glob" if items else "not-found"
    return {"ok": ok, "items": items, "hint": hint}


__all__ = ["gc_list", "verify", "CatalogEntry"]

