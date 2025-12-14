from __future__ import annotations
import streamlit as st

ACCENT = "#2b6cb0"

_GLOBAL_CSS = f"""
:root {{
  --accent: {ACCENT};
  --accent-600: {ACCENT};
  --text-900: #0f172a;
  --text-700: #334155;
  --muted-500: #64748b;
  --bg-soft: #f8fafc;
  --card-bg: #ffffff;
  --card-border: #e5e7eb;
}}

/* 基础排版 */
h1, h2, h3, h4 {{
  letter-spacing: 0.2px;
}}

.block-container {{
  max-width: 1100px;
}}

/* 统一按钮：将 Streamlit 按钮作为 primary */
.stButton > button {{
  background: linear-gradient(180deg, var(--accent), #1e4e80);
  color: #fff;
  border: 1px solid #13406a;
  padding: 0.5rem 1rem;
  border-radius: 8px;
  box-shadow: 0 2px 6px rgba(17, 66, 112, 0.25);
}}
.stButton > button:hover {{
  filter: brightness(1.02);
  box-shadow: 0 4px 10px rgba(17, 66, 112, 0.28);
}}

/* 链接按钮样式（secondary） */
.btn {{
  display: inline-block;
  padding: 8px 14px;
  border-radius: 8px;
  text-decoration: none !important;
  transition: all .2s ease;
}}
.btn-primary {{
  background: linear-gradient(180deg, var(--accent), #1e4e80);
  color: #fff !important;
  border: 1px solid #13406a;
}}
.btn-secondary {{
  background: #eef2f7;
  border: 1px solid #d1d5db;
  color: var(--text-700) !important;
}}
.btn-secondary:hover {{
  background: #e6ebf2;
}}

/* 卡片与分区 */
.card-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 22px;
}}
.card {{
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  border-radius: 12px;
  padding: 16px;
  box-shadow: 0 6px 18px rgba(15, 23, 42, 0.06);
  transition: all .2s ease;
  position: relative;
}}
.card:hover {{
  transform: translateY(-2px);
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.12);
}}
.card.disabled {{
  filter: grayscale(0.2);
  opacity: 0.6;
}}
.card h3 {{
  margin: 6px 0 6px;
  color: #0f172a;
}}
.card.disabled h3 {{
  color: #94a3b8;
}}
.card p {{
  color: var(--muted-500);
  margin: 6px 0 12px;
  font-size: 0.95rem;
}}
.card .meta {{
  font-size: 12px; color: #6b7280;
}}
.card .status-badge {{
  position: absolute;
  top: 10px;
  right: 10px;
  padding: 3px 8px;
  border-radius: 999px;
  font-size: 12px;
  border: 1px solid rgba(0,0,0,0.06);
}}
.card .status-on {{
  background: #10b9811A; color: #0ea5a3; border-color: #10b98133;
}}
.card .status-off {{
  background: #e5e7eb66; color: #6b7280; border-color: #e5e7eb;
}}

/* Info bar */
.info-bar {{
  margin-top: 14px;
  padding: 10px 12px;
  border-radius: 10px;
  background: rgba(43,108,176,0.08);
  border: 1px solid #e5e7eb;
  color: var(--text-700);
}}

/* 按钮 hover 轻微放大 */
.btn-primary:hover {{
  transform: translateY(-1px) scale(1.01);
}}

.section {{
  background: linear-gradient(0deg, #f9fbff, #ffffff);
  border: 1px solid #e5e7eb;
  border-radius: 10px;
  padding: 16px 16px 10px;
  margin: 12px 0 6px;
}}
.section h2 {{
  margin: 0 0 6px;
}}

/* 顶部 Hero */
.hero {{
  background: linear-gradient(135deg, rgba(43,108,176,0.12), rgba(43,108,176,0.03));
  border: 1px solid #e5e7eb;
  border-radius: 14px;
  padding: 24px 20px;
  margin: 8px 0 16px;
}}
.hero h1 {{
  margin: 0 0 6px;
}}
.hero p {{
  margin: 0; color: var(--text-700);
}}

/* JSON 显示弱化颜色，小字号 */
.small-json pre {{
  font-size: 12px;
  line-height: 1.3;
  filter: grayscale(0.25) contrast(0.95);
}}
"""


def get_global_css(theme_modern: bool = True) -> str:
    return _GLOBAL_CSS if theme_modern else ""


def read_theme_flag() -> bool:
    try:
        from pathlib import Path
        import yaml  # type: ignore
        repo = Path(__file__).resolve().parents[2]
        yml = (repo / "ArcticRoute" / "config" / "runtime.yaml")
        data = yaml.safe_load(yml.read_text(encoding="utf-8")) if yml.exists() else {}
        ui = data.get("ui") or {}
        return bool(ui.get("theme_modern", True))
    except Exception:
        return True


def inject_theme(theme_modern: bool = True) -> None:
    if not theme_modern:
        return
    key = "_arcticroute_theme_injected"
    if st.session_state.get(key):
        return
    st.markdown(f"<style>{get_global_css(True)}</style>", unsafe_allow_html=True)
    st.session_state[key] = True

