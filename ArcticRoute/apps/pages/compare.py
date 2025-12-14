from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore
from ArcticRoute.apps.components.error_card import show_error_card  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("compare", False):
        st.info("Compare 页面已在配置中禁用（ui.pages.compare=false）")
        return
    render_page_header("⚖️", "Compare · 代表路线对比", "展示 safe/balanced/efficient 与指标表")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 先通过 route.scan 生成 Pareto 前沿，再在此对比安全/效率/折中三条路线。
        - CLI 示例：
          - python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound_smoke --ym 202412 --out ArcticRoute/reports/d_stage/phaseG/
        """)
    ym = st.session_state.get("ym") or st.text_input("月份 YYYYMM", value="202412")
    scen = st.text_input("Scenario", value="nsr_wbound_smoke")
    out_dir = _repo_root()/"ArcticRoute"/"reports"/"d_stage"/"phaseG"
    pf = out_dir/f"pareto_front_{ym}_{scen}.json"
    if not pf.exists():
        show_error_card("NO_PARETO", "未找到 Pareto 前沿 JSON", f"请先运行 route.scan 生成前沿；示例：python -m ArcticRoute.api.cli route.scan --scenario {scen} --ym {ym}", {"expected": str(pf)})
        return
    try:
        data = json.loads(pf.read_text(encoding="utf-8"))
        reps = data.get("representatives") or {}
        st.write({k: reps.get(k) for k in ("safe","balanced","efficient")})
        st.download_button("下载 Pareto JSON", data=pf.read_bytes(), file_name=pf.name, mime="application/json")
        # 统一记录 compare.show 动作
        try:
            from ArcticRoute.apps import state as ui_state  # type: ignore
            ui_state.write_action_meta2(action="compare.show", inputs={"ym": ym, "scenario": scen}, outputs={})
        except Exception:
            pass
    except Exception as e:
        st.error(str(e))

