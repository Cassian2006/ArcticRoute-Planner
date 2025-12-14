"""Streamlit entrypoint for the ArcticRoute UI shell."""

from __future__ import annotations

from pathlib import Path

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
