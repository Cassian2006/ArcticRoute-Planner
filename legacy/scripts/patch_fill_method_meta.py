from __future__ import annotations
import json, sys, re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
report_path = REPO / "reports" / "audit" / "realdata_report.json"
if not report_path.exists():
    print("report not found:", report_path, file=sys.stderr)
    sys.exit(1)

js = json.loads(report_path.read_text(encoding="utf-8"))
findings = js.get("findings", [])

ROLE_MAP = [
    (re.compile(r"phaseG[/\\]pareto_.*\.html$", re.I), "pareto_report"),
    (re.compile(r"phaseG[/\\]pareto_front_.*\.json$", re.I), "pareto_front"),
    (re.compile(r"phaseG[/\\]summary_.*\.json$", re.I), "pareto_summary"),
    (re.compile(r"phaseI[/\\]route_.*_robust\.geojson$", re.I), "route_robust"),
    (re.compile(r"phaseH[/\\]calibration_.*\.json$", re.I), "calibration_report"),
    (re.compile(r"phaseH[/\\]audit_.*\.html$", re.I), "audit_report"),
]

def infer_method(path: str) -> str | None:
    for pat, val in ROLE_MAP:
        if pat.search(path):
            return val
    return None

written = []
for f in findings:
    path = f.get("path")
    reasons = [str(x) for x in f.get("reasons", [])]
    if not path or not any(r.startswith("method_mismatch") for r in reasons):
        continue
    m = infer_method(path)
    if not m:
        continue
    p = REPO / path if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        continue
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
    if meta.get("method") == m:
        continue
    meta["method"] = m
    if "run_id" not in meta:
        from time import gmtime, strftime
        meta["run_id"] = strftime("%Y-%m-%dT%H:%M:%SZ", gmtime())
    meta.setdefault("git_sha", "unknown")
    meta.setdefault("config_hash", "unknown")
    meta.setdefault("inputs", [])
    meta.setdefault("attrs", {})
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    written.append(str(meta_path))

print(json.dumps({"updated": written, "count": len(written)}, ensure_ascii=False))








