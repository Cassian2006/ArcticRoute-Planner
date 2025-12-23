"""Streamlit entrypoint for the ArcticRoute UI shell."""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pandas as pd
import streamlit as st

from arcticroute.ui import home, planner_minimal, eval_results


def inject_global_style() -> None:
    """Lightweight global styling for tighter layout and softer cards."""
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }
        .stDataFrame { font-size: 0.9rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_debug_footer() -> None:
    """
    在页面底部显示当前运行入口文件和 git HEAD，防止“跑错文件/分支”。
    """
    entry_path = Path(__file__).resolve()
    try:
        git_root = entry_path.parent  # 仓库根目录在上层，已足够让 git 自动发现
        git_head = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(git_root),
            text=True,
        ).strip()
    except Exception as e:  # pragma: no cover - best-effort helper
        git_head = f"<git error: {e}>"

    st.markdown("---")
    st.caption(f"UI entry file: {entry_path}")
    st.caption(f"Git HEAD: {git_head}")


def render_experiment_view() -> None:
    """Simple placeholder for scenario experiment results."""
    results_path = Path(__file__).resolve().parent / "reports" / "scenario_suite_results.csv"
    st.subheader("场景实验结果")
    if not results_path.exists():
        st.info("reports/scenario_suite_results.csv 未找到，后续可在此接入实验页面。")
        return

    df_results = pd.read_csv(results_path)
    st.dataframe(df_results, use_container_width=True)

    if {"distance_km", "total_cost"}.issubset(df_results.columns):
        st.caption("距离-成本散点概览")
        try:
            st.scatter_chart(df_results, x="distance_km", y="total_cost", color="mode")
        except Exception:
            pass


def _resolve_page() -> str:
    """
    统一页面路由：
      1) 优先使用 query params: ?page=...
      2) 否则读取 st.session_state['page']，默认 'home'
    """
    page_param = None
    try:
        page_param = st.query_params.get("page")
    except Exception:
        page_param = None

    if page_param in {"home", "planner", "data"}:
        page = page_param
    else:
        page = st.session_state.get("page", "home")

    st.session_state["page"] = page
    return page


def _nav_radio(page: str) -> str:
    """
    单一侧边栏导航控件。
    任意来源的“去规划/去数据页”按钮都应只做两件事：
      - st.query_params['page'] = ...
      - st.session_state['page'] = ...
      - st.rerun()
    此处的 radio 也遵循同样模式。
    """
    labels = {
        "home": "封面 / Home",
        "planner": "规划 / Planner",
        "data": "数据 / Data",
    }
    keys = list(labels.keys())
    texts = list(labels.values())
    idx = keys.index(page) if page in keys else 0

    with st.sidebar:
        choice_text = st.radio("页面导航 / Navigation", texts, index=idx)

    new_page = keys[texts.index(choice_text)]
    if new_page != page:
        try:
            qp = st.query_params
            qp["page"] = new_page
        except Exception:
            pass
        st.session_state["page"] = new_page
        st.rerun()

    return new_page


def main() -> None:
    st.set_page_config(
        page_title="ArcticRoute UI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_ar_page_config_set"] = True
    inject_global_style()

    # 统一解析 page（query params 优先）
    page = _resolve_page()
    # 单一侧边栏导航，写回 page + query params
    page = _nav_radio(page)

    if page == "home":
        home.render()
    elif page == "planner":
        planner_minimal.render()
    elif page == "data":
        # 目前复用评估结果页作为 Data 页
        eval_results.render()
    else:
        # 保底：未知 page 时回到封面
        home.render()

    # 页面渲染完毕后，统一在底部显示 entry + git hash
    _render_debug_footer()


if __name__ == "__main__":
    main()
