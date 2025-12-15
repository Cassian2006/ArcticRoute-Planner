from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _is_bad_path(p: str) -> bool:
    s = (p or "").lower()
    # 关键：把 minimum 老工程污染从 sys.path 移除
    if "minimum" in s:
        return True
    return False


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

    # 1) 确保本仓库根目录排在 sys.path 最前
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    else:
        sys.path.remove(str(project_root))
        sys.path.insert(0, str(project_root))

    # 2) 清理污染路径
    sys.path[:] = [p for p in sys.path if not _is_bad_path(p)]

    # 3) 若 arcticroute/ArcticRoute 已被错误导入，强制踢掉让其重新从本仓库加载
    for mod in ["arcticroute", "ArcticRoute"]:
        if mod in sys.modules:
            try:
                f = getattr(sys.modules[mod], "__file__", "") or ""
                if f and str(project_root).lower() not in f.lower():
                    sys.modules.pop(mod, None)
            except Exception:
                sys.modules.pop(mod, None)

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

