from __future__ import annotations
import streamlit as st

def render_page_header(
    icon: str,
    title: str,
    tip: str | None = None,
    badge_text: str | None = None,
    badge_type: str = "enabled",  # enabled | disabled | info
) -> None:
    color = {
        "enabled": "#10b981",
        "disabled": "#ef4444",
        "info": "#2b6cb0",
    }.get(badge_type, "#2b6cb0")
    badge_html = f'<span style="background:{color}1A;color:{color};padding:4px 8px;border:1px solid {color}33;border-radius:999px;font-size:12px;">{badge_text}</span>' if badge_text else ""
    st.markdown(
        f"""
<div class="section" style="display:flex;gap:12px;align-items:flex-start;">
  <div style="font-size:28px;line-height:1;">{icon}</div>
  <div style="flex:1;">
    <h2 style="margin:0 0 4px;">{title} {badge_html}</h2>
    <p style="margin:0;color:#334155;">{tip or ''}</p>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )






