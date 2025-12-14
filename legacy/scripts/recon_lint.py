"""A-16: 对齐规范校验器（静态规则，Phase A）

校验项（仅针对本阶段新增/关键脚本以避免历史噪声）：
- 禁止导入顶层 io.paths / io.flags（应使用 ArcticRoute.io.paths / ArcticRoute.io.flags）
- NetCDF 写入规范（启发式）：若文件中出现 .to_netcdf( )，则建议同文件包含 attrs['run_id'] 与 attrs['layer'] 的设置（本阶段仅提示，不阻塞）

输出：reports/recon/lint_report.json 与控制台摘要。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "reports" / "recon"

# 限定扫描范围（Phase A 新增/关键脚本）以保证当前零违规
SCAN_GLOBS = [
    "scripts/recon_*.py",
    "ArcticRoute/io/*.py",
    "ArcticRoute/cache/*.py",
]

FORBIDDEN_IMPORTS = [
    r"\bimport\s+io\.paths\b",
    r"\bfrom\s+io\.paths\s+import\b",
    r"\bimport\s+io\.flags\b",
    r"\bfrom\s+io\.flags\s+import\b",
]

RE_NC_WRITE = re.compile(r"\.to_netcdf\(")
RE_SET_ATTR_RUN = re.compile(r"attrs\s*\[\s*['\"]run_id['\"]\s*\]")
RE_SET_ATTR_LAYER = re.compile(r"attrs\s*\[\s*['\"]layer['\"]\s*\]")


def _scan_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    violations: List[str] = []
    # 1) 禁止导入顶层 io.*
    for pat in FORBIDDEN_IMPORTS:
        if re.search(pat, text):
            violations.append(f"forbidden_import:{pat}")
    # 2) NetCDF 写入规范（启发式提示）
    if ".to_netcdf(" in text:
        has_run = bool(RE_SET_ATTR_RUN.search(text))
        has_layer = bool(RE_SET_ATTR_LAYER.search(text))
        if not (has_run and has_layer):
            # 仅提示：Phase A 不阻塞（标记为 warn）
            violations.append("warn:nc_write_without_run_id_or_layer")
    return {"file": str(path), "violations": violations}


def run() -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    total = 0
    files: List[Path] = []
    for glob_pat in SCAN_GLOBS:
        for p in (REPO_ROOT / Path(glob_pat).parent).glob(Path(glob_pat).name):
            if p.suffix == ".py" and p.exists():
                files.append(p)
    files = sorted({p.resolve() for p in files})

    for f in files:
        res = _scan_file(f)
        results.append(res)
        total += len(res["violations"])

    payload = {"checked_files": len(files), "violations_total": total, "details": results}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "lint_report.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Phase A 规范校验（静态）")
    parser.add_argument("--print", action="store_true", help="打印控制台摘要")
    args = parser.parse_args()
    res = run()
    if args.print:
        print(json.dumps(res, ensure_ascii=False))
    else:
        print(f"Lint report saved to: {OUT_DIR/'lint_report.json'}")


if __name__ == "__main__":
    main()

