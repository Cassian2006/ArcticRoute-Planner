from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import streamlit as st


SUPPORTED_LANGS = ("zh", "en")
DEFAULT_LANG = "zh"


_STRINGS_ZH: Dict[str, str] = {
    "app_title": "ArcticRoute 北极航线规划系统",
    "nav_home": "封面 / Home",
    "nav_planner": "规划 / Planner",
    "nav_data": "数据 / Data",
    "btn_enter_planner": "进入系统 / Start",
    "data_page_title": "数据与资产 / Data & Assets",
    "data_discovery_section": "数据发现 / Data Discovery",
    "btn_rescan_assets": "🔄 重新扫描数据资产",
    "toast_rescanned": "已重新扫描数据资产",
    "btn_sync_newenv": "同步 newenv (SIC/SWH)",
    "sync_success": "newenv 文件已同步到本地。",
    "sync_failed": "无法在数据根中找到缺失的 newenv 文件。",
    "vessel_select_label": "船型 / Vessel profile",
    "lang_label": "Language / 语言",
}


_STRINGS_EN: Dict[str, str] = {
    "app_title": "ArcticRoute Arctic Route Planner",
    "nav_home": "Home",
    "nav_planner": "Planner",
    "nav_data": "Data",
    "btn_enter_planner": "Enter system / Start",
    "data_page_title": "Data & Assets",
    "data_discovery_section": "Data Discovery",
    "btn_rescan_assets": "🔄 Rescan data assets",
    "toast_rescanned": "Data assets rescanned",
    "btn_sync_newenv": "Sync newenv (SIC/SWH)",
    "sync_success": "newenv files synchronized to local directory.",
    "sync_failed": "Failed to find missing newenv files in data roots.",
    "vessel_select_label": "Vessel profile",
    "lang_label": "Language",
}


def tr(key: str, lang: str | None = None) -> str:
    """最小多语言字典查询。"""
    if lang is None:
        lang = st.session_state.get("lang", DEFAULT_LANG)
    if lang not in SUPPORTED_LANGS:
        lang = DEFAULT_LANG

    table = _STRINGS_ZH if lang == "zh" else _STRINGS_EN
    return table.get(key, key)


def render_lang_toggle() -> str:
    """侧边栏语言切换：使用自定义装饰单选，支持 query param 同步。"""
    qp_lang = None
    try:
        qp_lang = st.query_params.get("lang")
    except Exception:
        qp_lang = None

    current = st.session_state.get("lang", DEFAULT_LANG)
    if qp_lang in SUPPORTED_LANGS:
        current = qp_lang

    # 记录回 session_state
    st.session_state["lang"] = current if current in SUPPORTED_LANGS else DEFAULT_LANG
    label = tr("lang_label", lang=current)

    # 生成隐藏字段以保留其他 query（如 page）
    hidden_inputs = []
    try:
        for k, v in st.query_params.items():
            if k == "lang":
                continue
            hidden_inputs.append(f'<input type="hidden" name="{k}" value="{v}">')
    except Exception:
        pass
    hidden_html = "\n".join(hidden_inputs)

    zh_checked = "checked" if current == "zh" else ""
    en_checked = "checked" if current == "en" else ""

    st.markdown(
        f"""
<style>
/* lang toggle - Xtenso style */
.filter-switch {{
  border: 2px solid #ffc000;
  border-radius: 30px;
  position: relative;
  display: flex;
  align-items: center;
  height: 50px;
  width: 180px;
  overflow: hidden;
  background: #0f172a;
}}
.filter-switch input {{
  display: none;
}}
.filter-switch label {{
  flex: 1;
  text-align: center;
  cursor: pointer;
  border: none;
  border-radius: 30px;
  position: relative;
  overflow: hidden;
  z-index: 1;
  transition: all 0.5s;
  font-weight: 700;
  font-size: 18px;
  color: #7d7d7d;
}}
.filter-switch .background {{
  position: absolute;
  width: 49%;
  height: 38px;
  background-color: #ffc000;
  top: 4px;
  left: 4px;
  border-radius: 30px;
  transition: left 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
}}
#option-en:checked ~ .background {{
  left: 50%;
}}
#option-zh:checked + label[for="option-zh"] {{
  color: #0f172a;
  font-weight: 800;
}}
#option-en:checked + label[for="option-en"] {{
  color: #0f172a;
  font-weight: 800;
}}
#option-zh:not(:checked) + label[for="option-zh"],
#option-en:not(:checked) + label[for="option-en"] {{
  color: #cbd5e1;
}}
</style>

<div style="margin-bottom: 0.5rem; font-weight: 700; color: #f8fafc;">{label}</div>
<form class="filter-switch" method="get" oninput="this.submit()">
  {hidden_html}
  <input id="option-zh" name="lang" type="radio" value="zh" {zh_checked}/>
  <label class="option" for="option-zh">中文</label>
  <input id="option-en" name="lang" type="radio" value="en" {en_checked}/>
  <label class="option" for="option-en">English</label>
  <span class="background"></span>
</form>
""",
        unsafe_allow_html=True,
    )

    # 表单提交后刷新，依据 query params 决定 current；此处直接返回 session 中值
    return st.session_state.get("lang", DEFAULT_LANG)

