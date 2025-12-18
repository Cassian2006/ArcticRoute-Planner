"""生成 Phase 11 nextsim 诊断汇总。"""
from pathlib import Path

out = []
for f in [
    "reports/cmems_sic_describe.nextsim.exitcode.txt",
    "reports/cmems_sic_describe.nextsim.log",
    "reports/cmems_sic_describe.nextsim.tmp.txt",
    "reports/cmems_sic_probe_nextsim.txt",
    "reports/cmems_sic_probe_product.txt",
]:
    p = Path(f)
    out.append(f"## {f}")
    if p.exists():
        txt = p.read_text(encoding="utf-8", errors="ignore")
        out.append(txt[:4000] + ("\n...[truncated]\n" if len(txt) > 4000 else "\n"))
    else:
        out.append("(missing)\n")

Path("reports/PHASE11_NEXTSIM_DIAG_SUMMARY.txt").write_text("\n".join(out), encoding="utf-8")
print("wrote reports/PHASE11_NEXTSIM_DIAG_SUMMARY.txt")

