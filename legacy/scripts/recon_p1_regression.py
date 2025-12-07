"""A-19: P1 回归核验（只读）

以 --help / dry-run 探测关键 CLI（merge, cost, summarize, route, pipeline, report）
- 不执行重写操作，仅打印/探测
- 记录用时、stdout/stderr 摘要、返回码
- 输出 reports/recon/p1_regression.md
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "reports" / "recon" / "p1_regression.md"


def _run(cmd: List[str], cwd: Path) -> Tuple[int, float, str, str]:
    t0 = time.time()
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out, err = p.communicate()
    dt = time.time() - t0
    return p.returncode, dt, out, err


def _summarize(text: str, n: int = 8) -> str:
    lines = (text or "").splitlines()
    head = "\n".join(lines[:n])
    return head


def main():
    parser = argparse.ArgumentParser(description="P1 Regression (dry-run/help)")
    args = parser.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    # 以 --help 为主，不触发写入；pipeline 用 --dry-run
    cases = [
        ("merge --help", [py, "-m", "ArcticRoute.api.cli", "merge", "--help"]),
        ("cost --help", [py, "-m", "ArcticRoute.api.cli", "cost", "--help"]),
        ("summarize --help", [py, "-m", "ArcticRoute.api.cli", "summarize", "--help"]),
        ("route --help", [py, "-m", "ArcticRoute.api.cli", "route", "--help"]),
        ("report --help", [py, "-m", "ArcticRoute.api.cli", "report", "--help"]),
        ("pipeline --dry-run (single month)", [py, "-m", "ArcticRoute.api.cli", "pipeline", "--ym", "202412", "--dry-run"]),
    ]

    lines: List[str] = ["# P1 Regression (Phase A only-read)", ""]
    ok_all = True
    for name, cmd in cases:
        rc, dt, out, err = _run(cmd, REPO_ROOT)
        ok = (rc == 0)
        ok_all = ok_all and ok
        lines.append(f"## {name}")
        lines.append(f"- cmd: {' '.join(cmd)}")
        lines.append(f"- returncode: {rc}")
        lines.append(f"- elapsed: {dt:.3f}s")
        if out:
            lines.append("- stdout(head):\n`````\n" + _summarize(out) + "\n`````")
        if err:
            lines.append("- stderr(head):\n`````\n" + _summarize(err) + "\n`````")
        lines.append("")

    lines.append(f"\nOverall: {'PASS' if ok_all else 'WARN/FAIL (see above)'}\n")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Regression written: {OUT}")


if __name__ == "__main__":
    main()














