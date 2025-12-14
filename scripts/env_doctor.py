from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fail-on-contamination", action="store_true")
    args = ap.parse_args()

    project_root = Path(__file__).resolve().parents[1]

    print("=== env_doctor ===")
    print("python:", sys.executable)
    print("cwd:", os.getcwd())
    print("project_root:", project_root)
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", ""))

    bad_hits = []
    for p in sys.path:
        s = (p or "").lower()
        if "minimum" in s:
            bad_hits.append(p)

    def _try_import(name: str):
        try:
            m = __import__(name)
            print(f"import {name}: OK ->", getattr(m, "__file__", None))
        except Exception as e:
            print(f"import {name}: FAIL -> {e}")

    _try_import("arcticroute")
    _try_import("ArcticRoute")

    if bad_hits:
        print("\n[!] sys.path contains 'minimum' entries:")
        for p in bad_hits:
            print("  -", p)

    if args.fail_on_contamination and bad_hits:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

