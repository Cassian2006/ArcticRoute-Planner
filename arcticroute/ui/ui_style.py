from __future__ import annotations
import streamlit as st

FONT_STACK = """
/* Prefer CJK-friendly fonts first; keep emoji fonts at end */
:root {
  --ar-font: "Microsoft YaHei UI","Microsoft YaHei","PingFang SC","Noto Sans CJK SC",
             "Source Han Sans SC","Heiti SC","SimHei",
             system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif;
}
html, body, [class*="st-"], .stApp, .stMarkdown, .stText, .stButton, .stSelectbox, .stRadio, .stCheckbox, input, textarea {
  font-family: var(--ar-font) !important;
}
</style>
"""

def inject_global_style() -> None:
    st.markdown(f"<style>{FONT_STACK}", unsafe_allow_html=True)
