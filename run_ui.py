"""Streamlit entrypoint for the ArcticRoute UI shell.

启动方式（唯一入口）：
    streamlit run run_ui.py
不要直接运行 arcticroute/ui/planner_minimal.py 以避免导航/布局重复。
"""

from __future__ import annotations

from pathlib import Path
import os
import subprocess

import pandas as pd
import streamlit as st


def _build_fingerprint():
    try:
        head = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        br = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True).strip()
    except Exception:
        head, br = "nogit", "nogit"
    return br, head


BR, HEAD = _build_fingerprint()
st.set_page_config(page_title="ArcticRoute Planner", layout="wide")

# 运行时指纹（务必放最上面，任何页面都能看到）
st.sidebar.markdown("### 🔎 Runtime Fingerprint")
st.sidebar.code(
    f\"branch={BR}\\ncommit={HEAD}\\nrun_ui={__file__}\\n\"
    f\"cwd={os.getcwd()}\\nPYTHONPATH={os.environ.get('PYTHONPATH','')}\"
)

try:
    import arcticroute.ui.planner_minimal as _pm

    st.sidebar.code(f\"planner_minimal={_pm.__file__}\")
except Exception as e:
    st.sidebar.error(f\"planner_minimal import failed: {e}\")

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


def main() -> None:
    st.set_page_config(
        page_title="ArcticRoute UI",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.session_state["_ar_page_config_set"] = True
    inject_global_style()

    page = st.sidebar.radio(
        "页面导航",
        options=["总览", "航线规划驾驶舱", "场景实验结果", "EDL 评估结果"],
        index=0,
    )

    if "active_page" in st.session_state and st.session_state.active_page == "planner":
        page = "航线规划驾驶舱"
        st.session_state.pop("active_page")

    if page == "总览":
        home.render()
    elif page == "航线规划驾驶舱":
        planner_minimal.render()
    elif page == "场景实验结果":
        render_experiment_view()
    elif page == "EDL 评估结果":
        eval_results.render()


if __name__ == "__main__":
    main()
