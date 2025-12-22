from __future__ import annotations

import re
from pathlib import Path

import streamlit as st

_ASSET_PATH = Path(__file__).resolve().parent / "assets" / "arctic_ui_shell.html"

_STREAMLIT_OVERRIDES = """
/* Streamlit overrides (safe, layout-preserving) */
.stApp {
  background: var(--bg, #0f141a);
  color: var(--text, #e6edf3);
}
.stApp, .stApp * {
  font-family: system-ui, "Segoe UI", "Microsoft YaHei", "Noto Sans SC", "PingFang SC", sans-serif !important;
}
[data-testid="stSidebar"] {
  background: var(--panel, #141b22);
  border-right: 1px solid var(--border, rgba(255,255,255,0.08));
}
.main .block-container {
  padding-top: 2rem;
  padding-bottom: 2rem;
}
.stButton > button {
  background: var(--accent, #2fa9ff);
  color: var(--accent-contrast, #08131a);
  border: 1px solid transparent;
  border-radius: 999px;
  padding: 0.6rem 1.4rem;
  font-weight: 600;
  letter-spacing: 0.2px;
  transition: transform 120ms ease, box-shadow 120ms ease, background 120ms ease;
}
.stButton > button:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 24px rgba(0,0,0,0.22);
}
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea,
.stSlider [data-baseweb="slider"] {
  background: var(--panel-2, #111820);
  color: var(--text, #e6edf3);
  border: 1px solid var(--border, rgba(255,255,255,0.08));
  border-radius: 12px;
}
.stMarkdown a {
  color: var(--accent, #2fa9ff);
}
.material-icons,
.material-icons-outlined,
.material-symbols-outlined {
  font-family: inherit !important;
}
"""


def _read_shell_html() -> str:
    if not _ASSET_PATH.exists():
        raise FileNotFoundError(f"Shell HTML not found: {_ASSET_PATH}")
    return _ASSET_PATH.read_text(encoding="utf-8")


def _strip_external_fonts(css: str) -> str:
    css = re.sub(r"@import\\s+[^;]+;", "", css, flags=re.IGNORECASE)
    css = re.sub(r"@font-face\\s*{.*?}", "", css, flags=re.DOTALL | re.IGNORECASE)
    return css


def _filter_shell_css(css: str) -> str:
    blocks = re.split(r"}", css)
    keep_blocks: list[str] = []
    for block in blocks:
        if "{" not in block:
            continue
        selector, body = block.split("{", 1)
        selector_clean = selector.strip()
        if not selector_clean:
            continue
        selector_lower = selector_clean.lower()
        keep = False
        if any(token in selector_lower for token in [":root", "[data-theme", ".theme"]):
            keep = True
        elif any(token in selector_lower for token in [".card", ".cover", ".btn", ".button", ".badge", ".chip", ".pill", ".tag"]):
            keep = True
        if keep:
            keep_blocks.append(f"{selector_clean} {{{body.strip()}}}")
    return "\n".join(keep_blocks)


def extract_shell_css() -> str:
    html = _read_shell_html()
    style_blocks = re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.DOTALL | re.IGNORECASE)
    if not style_blocks:
        return _STREAMLIT_OVERRIDES.strip()
    raw_css = "\n".join(style_blocks)
    filtered_css = _filter_shell_css(_strip_external_fonts(raw_css))
    return f"{filtered_css}\n{_STREAMLIT_OVERRIDES}".strip()


def inject_shell_css() -> None:
    css = extract_shell_css()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
