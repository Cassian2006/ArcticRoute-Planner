# REUSE: report builder for real-data audit
from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .provenance import Finding, scan_catalog, load_meta, is_real, _infer_kind_from_name  # type: ignore
from .data_sanity import size_checks, risk_stats, prior_stats, interact_corr  # type: ignore

DEFAULTS: Dict[str, Any] = {
    "allow_roots": [
        "ArcticRoute/data_processed/ais",
        "ArcticRoute/data_processed/risk",
        "ArcticRoute/data_processed/prior",
        "ArcticRoute/reports",
        "data_processed",
        "reports",
    ],
    "deny_name_substrings": ["mock","demo","sample","toy","synthetic"],
    "require_method": "unetformer",
    "min_size_bytes": {"parquet_tracks": 50_000_000, "risk_nc": 5_000_000, "prior_nc": 2_000_000},
    "sanity_thresholds": {
        "risk_nonzero_pct": [0.02, 0.95],
        "prior_penalty_std_min": 0.02,
        "interact_corr_min": 0.10,
    },
    "profiles": {
        "real.quick": {"months": ["202412"], "scenarios": ["nsr_wbound"], "require_real": True},
        "real.full": {"months": ["202411","202412","202501"], "scenarios": ["nsr_wbound","nwp_eastbound"], "require_real": True},
    },
}


def _merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def build_audit(roots: Iterable[str], cfg: Dict[str, Any], paths_filter: Optional[List[Path]] = None, fail_on_warn: bool = False, months: Optional[List[str]] = None) -> Tuple[Dict[str, Any], int]:
    deny = cfg.get("deny_name_substrings") or []
    min_bytes = cfg.get("min_size_bytes") or {}
    thr = cfg.get("sanity_thresholds") or {}

    # Collect candidate files
    findings: List[Finding] = []
    if paths_filter is not None:
        # direct list
        for p in paths_filter:
            if not p.is_file():
                continue
            kind = _infer_kind_from_name(p)
            if not kind:
                continue
            meta = load_meta(p)
            ok, reasons = is_real(meta, deny_substrings=deny, require_method=cfg.get("require_method"))
            findings.append(Finding(path=str(p), ok=ok, reasons=reasons, meta=meta, kind=kind))
    else:
        findings = scan_catalog(roots, deny)
        # 早期统一方法校验已取消，方法约束在后续仅对 risk_fused_* 执行

    # Optional month filter (restrict scope)
    if months:
        import re as _re
        ym_pat = _re.compile(r"(19|20|21)\d{4}")
        keep: List[Finding] = []
        for f in findings:
            name = Path(f.path).name
            m = ym_pat.search(name)
            if m:
                ym_token = m.group(0)
                if ym_token in set(months):
                    keep.append(f)
                else:
                    # 过滤与 profile 月份无关的产物
                    continue
            else:
                # 无法识别月份，保留
                keep.append(f)
        findings = keep

    # Enforce method only for risk layers
    req_method = cfg.get("require_method")
    if req_method:
        for i, f in enumerate(findings):
            if f.kind == "risk_nc":
                name = Path(f.path).name.lower()
                # 仅对 risk_fused_* 执行方法约束
                if ("risk_fused_" in name) or name.startswith("risk_fused_"):
                    okm, rms = is_real(f.meta, deny_substrings=deny, require_method=req_method)
                    findings[i].ok = okm
                    findings[i].reasons = list(set(f.reasons + rms))

    # Sanity per-file
    for f in findings:
        ok2, rs = size_checks(f.path, min_bytes, f.kind)
        if not ok2:
            f.ok = False
            f.reasons.extend(rs)
        # Type-specific stats checks
        if f.kind == "risk_nc":
            st = risk_stats(f.path)
            if st is not None:
                lo, hi = (thr.get("risk_nonzero_pct") or [0.0, 1.0])
                if not (lo <= st.get("nonzero_pct", 0.0) <= hi):
                    f.ok = False
                    f.reasons.append(f"risk_nonzero_pct_out:{st.get('nonzero_pct')}")
                if st.get("std", 0.0) <= float(thr.get("prior_penalty_std_min", 0.0)):
                    # reuse std_min as minimum variability guard for risk as well
                    f.ok = False
                    f.reasons.append(f"risk_std_low:{st.get('std')}")
        if f.kind == "prior_nc":
            st = prior_stats(f.path)
            if st is not None:
                if st.get("std", 0.0) <= float(thr.get("prior_penalty_std_min", 0.0)):
                    f.ok = False
                    f.reasons.append(f"prior_std_low:{st.get('std')}")

    # Global correlation checks per month when both layers available
    corr_stats: Dict[str, Any] = {}
    try:
        ais_dir = None
        risk_dir = None
        # resolve base from roots
        for r in roots:
            rp = Path(r)
            if (rp / "ArcticRoute").exists():
                # skip nested
                pass
        # gather by month
        from collections import defaultdict
        months: Dict[str, Dict[str, Path]] = defaultdict(dict)
        for f in findings:
            name = Path(f.path).name
            # ais_density_YYYYMM.nc
            if name.startswith("ais_density_") and name.endswith('.nc'):
                ym = name.split("_")[-1].replace('.nc','')
                months[ym]["ais"] = Path(f.path)
            # R_interact_YYYYMM.nc
            if (name.startswith("R_interact_") or name.startswith("r_interact_")) and name.endswith('.nc'):
                ym = name.split("_")[-1].replace('.nc','')
                months[ym]["interact"] = Path(f.path)
        thr_corr = float(thr.get("interact_corr_min", 0.0))
        for ym, m in months.items():
            if "ais" in m and "interact" in m:
                c = interact_corr(m["ais"], m["interact"])  # may be None
                corr_stats[ym] = {"pearson": c}
                if c is not None and c < thr_corr:
                    # mark related files as failed with reason
                    for f in findings:
                        if f.path.endswith(m["interact"].name) or f.path.endswith(m["ais"].name):
                            f.ok = False
                            f.reasons.append(f"interact_corr_low:{c}")
            else:
                corr_stats[ym] = {"pearson": None, "skipped": True}
    except Exception:
        pass

    # Summaries
    summary = {
        "checked": len(findings),
        "passed": sum(1 for f in findings if f.ok),
        "failed": sum(1 for f in findings if not f.ok),
        "warnings": sum(1 for f in findings for r in f.reasons if str(r).startswith("warn_")),
    }

    # Failure bucketization
    bucket: Dict[str, int] = {}
    for f in findings:
        for r in f.reasons:
            key = (
                "no_meta" if r == "no_meta" else
                "deny_tag" if r == "deny_tag" else
                "method_mismatch" if str(r).startswith("method_mismatch") else
                "size" if str(r).startswith("size_below_min") else
                "stats" if ("_low" in str(r) or "_out" in str(r)) else
                ("warn" if str(r).startswith("warn_") else "other")
            )
            bucket[key] = bucket.get(key, 0) + 1

    report = {
        "summary": summary,
        "fail_reasons": bucket,
        "corr": corr_stats,
        "findings": [
            {
                "path": f.path,
                "ok": f.ok,
                "reasons": f.reasons,
                "kind": f.kind,
            }
            for f in findings
        ],
    }

    exit_code = 0
    if summary["failed"] > 0:
        exit_code = 2
    elif fail_on_warn and summary["warnings"] > 0:
        exit_code = 3

    return report, exit_code


def write_reports(out_dir: Path, payload: Dict[str, Any]) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jpath = out_dir / "realdata_report.json"
    hpath = out_dir / "realdata_report.html"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # very small HTML renderer
    summary = payload.get("summary", {})
    rows = []
    for f in payload.get("findings", []):
        ok = "✅" if f.get("ok") else "❌"
        reasons = ", ".join(f.get("reasons", []))
        rows.append(f"<tr><td>{ok}</td><td>{f.get('kind','')}</td><td>{f.get('path','')}</td><td>{reasons}</td></tr>")
    html = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Real-Data Audit</title>
<style>body{{font-family:Arial,sans-serif}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body>
<h1>Real-Data & Method Audit</h1>
<p>Checked: {summary.get('checked',0)} | Passed: {summary.get('passed',0)} | Failed: {summary.get('failed',0)} | Warnings: {summary.get('warnings',0)}</p>
<h2>Findings</h2>
<table><thead><tr><th>OK</th><th>Kind</th><th>Path</th><th>Reasons</th></tr></thead><tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
    hpath.write_text(html, encoding="utf-8")
    return jpath, hpath

