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
    """侧边栏语言切换控件。返回当前语言代码。"""
    current = st.session_state.get("lang", DEFAULT_LANG)
    idx = SUPPORTED_LANGS.index(current) if current in SUPPORTED_LANGS else 0
    label = tr("lang_label", lang=current)
    choice = st.selectbox(label, SUPPORTED_LANGS, index=idx)
    st.session_state["lang"] = choice
    return choice

from __future__ import annotations
import streamlit as st

_LANGS = ("zh", "en")

# 最小可交付：先覆盖导航、首页/驾驶舱/数据/诊断、planner mode、常用按钮
_DICT: dict[str, dict[str, str]] = {
    "en": {
        "app_title": "ArcticRoute",
        "nav": "Navigation",
        "home": "Home",
        "planner": "Planner Cockpit",
        "data": "Data",
        "diag": "Diagnostics",
        "lang": "Language",
        "lang_zh": "中文",
        "lang_en": "English",

        "planner_engine": "Planner engine",
        "mode_auto": "Auto (best available)",
        "mode_astar": "A* (always available)",
        "mode_pipe": "PolarRoute (pipeline dir)",
        "mode_ext": "PolarRoute (external mesh/config)",
        "availability": "Availability",
        "fallback_reason": "Fallback reason",
        "pipeline_dir": "Pipeline directory",
        "mesh_path": "External vessel_mesh.json",
        "routecfg_path": "External route_config.json",
        "apply": "Apply",
    },
    "zh": {
        "app_title": "ArcticRoute",
        "nav": "导航",
        "home": "首页",
        "planner": "航线规划驾驶舱",
        "data": "数据",
        "diag": "诊断",
        "lang": "语言",
        "lang_zh": "中文",
        "lang_en": "English",

        "planner_engine": "规划内核",
        "mode_auto": "自动（优先可用）",
        "mode_astar": "A*（始终可用）",
        "mode_pipe": "PolarRoute（pipeline 目录）",
        "mode_ext": "PolarRoute（外部 mesh/config）",
        "availability": "可用性",
        "fallback_reason": "回退原因",
        "pipeline_dir": "Pipeline 目录",
        "mesh_path": "外部 vessel_mesh.json",
        "routecfg_path": "外部 route_config.json",
        "apply": "应用",
    },
}

def get_lang() -> str:
    try:
        lang = st.session_state.get("lang", None)
        if lang in _LANGS:
            return lang
        # 默认中文（你是中文 UI）
        st.session_state["lang"] = "zh"
        return "zh"
    except (RuntimeError, AttributeError):
        # 不在 streamlit runtime 中，返回默认语言
        return "zh"

def set_lang(lang: str) -> None:
    if lang in _LANGS:
        st.session_state["lang"] = lang

def t(key: str) -> str:
    lang = get_lang()
    return _DICT.get(lang, {}).get(key, _DICT["en"].get(key, key))

def render_lang_toggle() -> None:
    lang = get_lang()
    # 放在 sidebar 顶部即可
    choice = st.selectbox(
        t("lang"),
        options=["zh", "en"],
        format_func=lambda x: t("lang_zh") if x == "zh" else t("lang_en"),
        index=0 if lang == "zh" else 1,
    )
    if choice != lang:
        set_lang(choice)
        st.rerun()

