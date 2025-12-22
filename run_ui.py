"""Streamlit entrypoint for the ArcticRoute UI shell."""

from __future__ import annotations

import streamlit as st

from arcticroute.ui.planner_minimal import render_app


def main() -> None:
    """纯入口：只设置页面配置并调用新的统一 render_app()"""
    st.set_page_config(
        page_title="ArcticRoute",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    render_app()


if __name__ == "__main__":
    main()
