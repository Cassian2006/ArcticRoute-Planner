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

