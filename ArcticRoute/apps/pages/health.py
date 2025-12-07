from __future__ import annotations
import json
import subprocess
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("health", False):
        st.info("Health 页面已在配置中禁用（ui.pages.health=false）")
        return
    render_page_header("🩺", "Health · 系统健康检查", "调用 CLI 生成健康报告并展示摘要")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 点击“运行 health.check”以执行系统健康检查，结果会写入 reports/health/ 目录。
        - 返回状态说明：通过(OK)/警告(WARN)/失败(FAIL)。失败会列出失败项；警告表示可继续使用但建议关注。
        - CLI 示例：
          - python -m ArcticRoute.api.cli health.check --out reports/health/health_latest.json
        """)
    st.markdown('<div class="section"><h2>运行检查</h2><p>输出路径可覆盖，默认写入 reports/health/</p></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    out_override = c1.text_input("可选：输出 JSON 路径", value="reports/health/health_latest.json")
    if c2.button("运行 health.check", width='stretch'):
        try:
            import sys
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "health.check"]
            if out_override:
                cmd += ["--out", out_override]
            res = subprocess.run(cmd, capture_output=True, text=True, cwd=_repo_root())
            if res.returncode == 0:
                st.success("health.check 完成")
                try:
                    st.json(json.loads(res.stdout))
                except Exception:
                    st.code(res.stdout[-1600:])
                try:
                    from ArcticRoute.apps import state as ui_state  # type: ignore
                    ui_state.write_action_meta2(action="health.check", inputs={"out": out_override}, outputs={})
                except Exception:
                    pass
            else:
                st.error(f"health.check 失败：{res.returncode}")
                st.code(res.stderr[-1600:])
        except Exception as e:
            st.error(str(e))

    # 最近产物快捷查看
    health_dir = _repo_root()/"reports"/"health"
    if health_dir.exists():
        cands = sorted(health_dir.glob("health_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if cands:
            p = cands[0]
            try:
                st.caption(f"最新健康报告：{p.name}")
                st.json(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                st.code(p.read_text(encoding="utf-8")[-2000:])

