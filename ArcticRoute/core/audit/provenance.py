# REUSE: standalone audit helpers; does not modify existing pipeline
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

DENY_DEFAULT = ["mock","demo","sample","toy","synthetic"]

@dataclass
class Finding:
    path: str
    ok: bool
    reasons: List[str]
    meta: Optional[Dict[str, Any]] = None
    kind: Optional[str] = None


def _adjacent_meta(path: Path) -> Optional[Path]:
    # try foo.bar.meta.json and foo.meta.json
    cand1 = path.with_suffix(path.suffix + ".meta.json")
    if cand1.exists():
        return cand1
    cand2 = path.with_name(path.stem + ".meta.json")
    if cand2.exists():
        return cand2
    # try same name under reports/ or cache/index
    try:
        parent = path.parent
        alt = parent / (path.name + ".meta.json")
        if alt.exists():
            return alt
    except Exception:
        pass
    return None


def load_meta(path: str | Path) -> Optional[Dict[str, Any]]:
    p = Path(path)
    meta_p = _adjacent_meta(p)
    if not meta_p:
        return None
    try:
        return json.loads(meta_p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_kind_from_name(path: Path) -> Optional[str]:
    name = path.name.lower()
    if name.endswith('.parquet'):
        if 'track' in name:
            return 'parquet_tracks'
        return 'parquet'
    if name.endswith('.nc'):
        if name.startswith('risk_') or 'risk_fused' in name:
            return 'risk_nc'
        if 'prior' in name:
            return 'prior_nc'
        return 'nc'
    if name.endswith('.geojson'):
        return 'route_geojson'
    if name.endswith('.json'):
        return 'json'
    if name.endswith('.html'):
        return 'html'
    if name.endswith('.zip'):
        return 'zip'
    return None


def is_real(meta: Optional[Dict[str, Any]], deny_substrings: Iterable[str] = DENY_DEFAULT, require_method: Optional[str] = None) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if meta is None:
        return False, ["no_meta"]
    # deny substrings in source_tags, run tags, inputs
    deny = set(s.lower() for s in deny_substrings)
    def contains_deny(v: Any) -> bool:
        try:
            s = str(v).lower()
            return any(x in s for x in deny)
        except Exception:
            return False
    # scan known fields
    for key in ("source_tags","tags","attrs","inputs"):
        v = meta.get(key)
        if isinstance(v, dict):
            if any(contains_deny(k) or contains_deny(val) for k,val in v.items()):
                reasons.append("deny_tag")
        elif isinstance(v, (list, tuple)):
            if any(contains_deny(x) for x in v):
                reasons.append("deny_tag")
        elif v is not None:
            if contains_deny(v):
                reasons.append("deny_tag")
    # method check
    if require_method:
        m = meta.get('method') or meta.get('attrs',{}).get('method') or meta.get('fuse',{}).get('method')
        if str(m).lower() != str(require_method).lower():
            reasons.append(f"method_mismatch:{m}")
    # required provenance keys (warn level, not hard fail here)
    missing = [k for k in ("run_id","git_sha","config_hash","inputs") if meta.get(k) is None]
    if missing:
        reasons.append("warn_missing:"+",".join(missing))
    ok = True if (not reasons or all(r.startswith('warn_') for r in reasons)) else ("deny_tag" not in reasons and not any(r.startswith('method_mismatch') for r in reasons))
    # ok means no hard failures here; callers may upgrade warn to fail via flag
    return ok, reasons


def scan_catalog(allow_roots: Iterable[str], deny_substrings: Iterable[str]) -> List[Finding]:
    findings: List[Finding] = []
    for root in allow_roots:
        r = Path(root)
        if not r.exists():
            continue
        for p in r.rglob('*'):
            if not p.is_file():
                continue
            if str(p).endswith('.meta.json'):
                continue
            kind = _infer_kind_from_name(p) or ''
            if not kind:
                continue
            meta = load_meta(p)
            ok, reasons = is_real(meta, deny_substrings=deny_substrings, require_method=None)
            findings.append(Finding(path=str(p), ok=ok, reasons=reasons, meta=meta, kind=kind))
    return findings


def assert_real_artifact(path: str | Path, cfg: Dict[str, Any]) -> None:
    deny = cfg.get('deny_name_substrings') or DENY_DEFAULT
    req = cfg.get('require_method')
    p = Path(path)
    # filename quick deny
    low = p.name.lower()
    for frag in deny:
        if frag in low:
            raise AssertionError(f"deny_name:{frag}")
    meta = load_meta(p)
    ok, reasons = is_real(meta, deny_substrings=deny, require_method=req)
    if not ok:
        raise AssertionError(",".join(reasons))

