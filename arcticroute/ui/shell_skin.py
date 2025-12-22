from __future__ import annotations

from pathlib import Path
import re

import streamlit as st

_SHELL_PATH = Path(__file__).parent / "assets" / "arctic_ui_shell.html"


def _extract_style(html: str) -> str:
    match = re.search(r"<style[^>]*>(.*?)</style>", html, flags=re.S | re.I)
    return match.group(1).strip() if match else ""


def inject_shell_css() -> None:
    css = ""
    if _SHELL_PATH.exists():
        html = _SHELL_PATH.read_text(encoding="utf-8", errors="ignore")
        css = _extract_style(html)

    if not css:
        css = """
        :root { --bg:#0b1020; --panel:rgba(255,255,255,.06); --text:rgba(255,255,255,.92); }
        body { background: var(--bg); color: var(--text); }
        """

    streamlit_tweaks = """
    /* Keep Streamlit controls visually close to the shell. */
    .stApp { background: transparent; }
    section[data-testid="stSidebar"] { background: rgba(12,16,32,.72); backdrop-filter: blur(10px); }
    div[data-testid="stMetric"] { background: rgba(255,255,255,.06); border-radius: 14px; padding: 8px 10px; }
    div.stButton > button { border-radius: 14px; padding: 10px 12px; }
    div[data-testid="stExpander"] { border-radius: 16px; overflow: hidden; }
    """

    st.markdown(f"<style>\n{css}\n{streamlit_tweaks}\n</style>", unsafe_allow_html=True)
