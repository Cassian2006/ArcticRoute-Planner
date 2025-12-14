from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT).lower()


def _is_bad_path(p: str) -> bool:
    s = (p or "").lower()
    # 关键：把 minimum 老工程污染从 sys.path 移除
    if "minimum" in s:
        return True
    return False


def pytest_configure(config):
    # 1) 确保本仓库根目录排在 sys.path 最前
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    else:
        sys.path.remove(str(PROJECT_ROOT))
        sys.path.insert(0, str(PROJECT_ROOT))

    # 2) 清理污染路径
    sys.path[:] = [p for p in sys.path if not _is_bad_path(p)]

    # 3) 若 arcticroute/ArcticRoute 已被错误导入，强制踢掉让其重新从本仓库加载
    for mod in ["arcticroute", "ArcticRoute"]:
        if mod in sys.modules:
            try:
                f = getattr(sys.modules[mod], "__file__", "") or ""
                if f and str(PROJECT_ROOT).lower() not in f.lower():
                    sys.modules.pop(mod, None)
            except Exception:
                sys.modules.pop(mod, None)

