from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

import streamlit as st


def _get_git_info(repo_root: Path) -> Dict[str, str]:
    """Return short git hash and current branch; tolerate missing git."""
    info: Dict[str, str] = {"hash": "unknown", "branch": "unknown"}
    try:
        info["hash"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(repo_root),
                text=True,
            )
            .strip()
            or "unknown"
        )
    except Exception as e:  # pragma: no cover - best effort
        info["hash"] = f"error:{e}"
    try:
        info["branch"] = (
            subprocess.check_output(
                ["git", "branch", "--show-current"],
                cwd=str(repo_root),
                text=True,
            )
            .strip()
            or "unknown"
        )
    except Exception as e:  # pragma: no cover - best effort
        info["branch"] = f"error:{e}"
    return info


def render_build_banner(entry_file: str | Path | None = None, page: str | None = None, show_sidebar: bool = True) -> None:
    """
    在页面底部显示构建信息，确保能够判断运行入口与分支。

    显示信息：
      - __file__（入口文件绝对路径）
      - 当前工作目录
      - sys.executable
      - git hash（短）与当前 branch
    """
    entry_path = Path(entry_file).resolve() if entry_file else Path(__file__).resolve()
    repo_root = entry_path.parent
    git_info = _get_git_info(repo_root)

    caption_lines = [
        f"📄 file: {entry_path}",
        f"📂 cwd: {Path.cwd()}",
        f"🐍 python: {sys.executable}",
        f"🔀 git: {git_info['hash']} ({git_info['branch']})",
    ]
    if page:
        caption_lines.insert(0, f"📑 page: {page}")

    if show_sidebar:
        with st.sidebar:
            st.markdown("---")
            for line in caption_lines:
                st.caption(line)


