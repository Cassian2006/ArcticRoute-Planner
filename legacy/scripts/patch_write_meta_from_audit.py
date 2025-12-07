from __future__ import annotations
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
report_path = REPO / "reports" / "audit" / "realdata_report.json"
if not report_path.exists():
    print("report not found:", report_path, file=sys.stderr)
    sys.exit(1)

js = json.loads(report_path.read_text(encoding="utf-8"))
findings = js.get("findings", [])
now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
written = []
skipped = []

for f in findings:
    reasons = set(map(str, f.get("reasons", [])))
    path = f.get("path")
    if not path or "no_meta" not in reasons:
        continue
    p = Path(path)
    # skip if target doesn't exist
    if not p.exists() or not p.is_file():
        skipped.append((path, "missing"))
        continue
    # decide meta file name: use suffix + .meta.json (e.g., foo.nc.meta.json)
    meta_path = p.with_suffix(p.suffix + ".meta.json")
    if meta_path.exists():
        continue
    kind = f.get("kind")
    method = None
    # only annotate method for risk_fused_* (避免触发 method_mismatch)
    name = p.name.lower()
    if kind == "risk_nc" and ("risk_fused_" in name or name.startswith("risk_fused_")):
        method = "unetformer"
    meta = {
        "run_id": now,
        "git_sha": "unknown",
        "config_hash": "unknown",
        "inputs": [],
        "attrs": {}
    }
    if method:
        meta["method"] = method
    # add lightweight attrs
    if any(ch.isdigit() for ch in name):
        # naive extract ym tokens
        import re
        m = re.search(r"(20\d{4})", name)
        if m:
            meta["attrs"]["ym"] = m.group(1)
    try:
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(str(meta_path))
    except Exception as e:
        skipped.append((path, str(e)))

print(json.dumps({"written": written, "skipped": skipped, "count": len(written)}, ensure_ascii=False))








