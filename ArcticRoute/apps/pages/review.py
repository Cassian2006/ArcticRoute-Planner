from __future__ import annotations
import json
import subprocess
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps import state as ui_state  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore
from ArcticRoute.apps.components.error_card import show_error_card  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("review", False):
        st.info("Review 页面已在配置中禁用（ui.pages.review=false）")
        return
    render_page_header("📝", "Review · 人在回路", "上传 feedback.jsonl 或锁点，应用后重规划")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 步骤：加载/上传 feedback.jsonl（可选 locks）→ 一键 Apply & Replan → 检查新路线与 violations。
        - CLI 示例：
          - python -m ArcticRoute.api.cli route.review --scenario nsr_wbound_smoke --ym 202412
          - python -m ArcticRoute.api.cli route.apply.feedback --scenario nsr_wbound_smoke --ym 202412 --feedback ArcticRoute/data_processed/review/feedback.jsonl
        """)
    st.markdown('<div class="section"><h2>构建 Review 包</h2><p>选择场景与月份，生成包以供人工审核。</p></div>', unsafe_allow_html=True)
    scen = st.text_input("Scenario ID", value="nsr_wbound_smoke")
    ym = st.text_input("月份 YYYYMM", value=str(st.session_state.get("ym", "202412")))

    c1, c2 = st.columns(2)
    if c1.button("生成 Review 包", width='stretch'):
        try:
            import sys
            out_dir = _repo_root()/"ArcticRoute"/"reports"/"d_stage"/"phaseO"
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "route.review", "--scenario", scen, "--ym", ym, "--out", str(out_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())
            if res.returncode == 0:
                st.success("已生成 Review 包")
                st.code(res.stdout[-1600:], language="json")
            else:
                st.error(f"route.review 失败：{res.returncode}")
                st.code(res.stderr[-1600:])
        except Exception as e:
            st.error(str(e))

    st.markdown("**上传或编辑 feedback.jsonl**")
    up = st.file_uploader("上传反馈 JSONL（可选）", type=["jsonl","txt"], accept_multiple_files=False)
    buf_text = st.text_area("或直接粘贴 JSONL 文本", value="", height=140)
    locks_up = st.file_uploader("可选：锁点 GeoJSON", type=["geojson","json"], accept_multiple_files=False)

    if c2.button("应用反馈并重规划", width='stretch'):
        try:
            review_dir = _repo_root()/"ArcticRoute"/"data_processed"/"review"
            review_dir.mkdir(parents=True, exist_ok=True)
            fb_path = review_dir/f"feedback_{scen}_{ym}.jsonl"
            if up is not None:
                fb_path.write_bytes(up.read())
            elif buf_text.strip():
                fb_path.write_text(buf_text.strip()+"\n", encoding="utf-8")
            else:
                fb_path.write_text("# empty\n", encoding="utf-8")
            locks_path = None
            if locks_up is not None:
                locks_path = review_dir/f"locks_{scen}_{ym}.geojson"
                locks_path.write_bytes(locks_up.read())
            import sys
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "route.apply.feedback", "--scenario", scen, "--ym", ym, "--feedback", str(fb_path)]
            if locks_path:
                cmd += ["--locks", str(locks_path)]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())
            if res.returncode == 0:
                st.success("约束重规划完成")
                try:
                    ui_state.write_action_meta(
                        action="review.apply",
                        inputs={"scenario": scen, "ym": ym, "has_locks": bool(locks_path)}
                    )
                except Exception:
                    pass
                st.code(res.stdout[-1600:], language="json")
            else:
                st.error(f"route.apply.feedback 失败：{res.returncode}")
                st.code(res.stderr[-1600:])
        except Exception as e:
            st.error(str(e))

