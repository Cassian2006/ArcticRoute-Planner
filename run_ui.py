"""Streamlit entrypoint for the ArcticRoute UI shell."""

from __future__ import annotations

from pathlib import Path
import os
import base64

import pandas as pd
import streamlit as st

from arcticroute.ui import home, planner_minimal, eval_results
from arcticroute.ui.data_discovery import render_data_discovery_panel
from arcticroute.ui.i18n import tr, render_lang_toggle
from arcticroute.ui.build_banner import render_build_banner


def inject_global_style() -> None:
    """Lightweight global styling for tighter layout and softer cards."""
    st.markdown(
        """
        <style>
        /* 全局深色主题 */
        .stApp, .stAppViewContainer, .main, .block-container {
            background: #0b1220 !important;
            color: #e2e8f0 !important;
        }
        .stSidebar, section[data-testid="stSidebar"], .css-1d391kg, .css-1cypcdb {
            background: #0d1324 !important;
        }
        .stMarkdown, .stText, .stHeader, .stSubheader, .stCaption, p, h1, h2, h3, h4, h5, h6, label {
            color: #e2e8f0 !important;
        }
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


def inject_rain_overlay() -> None:
    """全局雨点动画背景（不挡交互，固定铺满）。"""
    st.markdown(
        """
<style>
.rain-overlay {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
  opacity: 0.18;
}
.rain-overlay::after {
  content: "";
  position: absolute;
  inset: 0;
  z-index: 1;
  background-image: radial-gradient(
    ellipse 1.5px 2px at 1.5px 50%,
    #0000 0,
    #0000 90%,
    #000 100%
  );
  background-size: 25px 8px;
}
.rain-overlay {
  --c: #09f;
  background-color: #000;
  background-image: radial-gradient(4px 100px at 0px 235px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 235px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 117.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 252px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 252px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 126px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 150px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 150px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 75px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 253px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 253px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 126.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 204px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 204px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 102px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 134px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 134px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 67px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 179px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 179px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 89.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 299px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 299px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 149.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 215px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 215px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 107.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 281px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 281px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 140.5px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 158px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 158px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 79px, var(--c) 100%, #0000 150%),
    radial-gradient(4px 100px at 0px 210px, var(--c), #0000),
    radial-gradient(4px 100px at 300px 210px, var(--c), #0000),
    radial-gradient(1.5px 1.5px at 150px 105px, var(--c) 100%, #0000 150%);
  background-size:
    300px 235px,
    300px 235px,
    300px 235px,
    300px 252px,
    300px 252px,
    300px 252px,
    300px 150px,
    300px 150px,
    300px 150px,
    300px 253px,
    300px 253px,
    300px 253px,
    300px 204px,
    300px 204px,
    300px 204px,
    300px 134px,
    300px 134px,
    300px 134px,
    300px 179px,
    300px 179px,
    300px 179px,
    300px 299px,
    300px 299px,
    300px 299px,
    300px 215px,
    300px 215px,
    300px 215px,
    300px 281px,
    300px 281px,
    300px 281px,
    300px 158px,
    300px 158px,
    300px 158px,
    300px 210px,
    300px 210px,
    300px 210px;
  animation: hi 150s linear infinite;
}
@keyframes hi {
  0% {
    background-position:
      0px 220px, 3px 220px, 151.5px 337.5px,
      25px 24px, 28px 24px, 176.5px 150px,
      50px 16px, 53px 16px, 201.5px 91px,
      75px 224px, 78px 224px, 226.5px 350.5px,
      100px 19px, 103px 19px, 251.5px 121px,
      125px 120px, 128px 120px, 276.5px 187px,
      150px 31px, 153px 31px, 301.5px 120.5px,
      175px 235px, 178px 235px, 326.5px 384.5px,
      200px 121px, 203px 121px, 351.5px 228.5px,
      225px 224px, 228px 224px, 376.5px 364.5px,
      250px 26px, 253px 26px, 401.5px 105px,
      275px 75px, 278px 75px, 426.5px 180px;
  }
  100% {
    background-position:
      0px 6800px, 3px 6800px, 151.5px 6917.5px,
      25px 13632px, 28px 13632px, 176.5px 13758px,
      50px 5416px, 53px 5416px, 201.5px 5491px,
      75px 17175px, 78px 17175px, 226.5px 17301.5px,
      100px 5119px, 103px 5119px, 251.5px 5221px,
      125px 8428px, 128px 8428px, 276.5px 8495px,
      150px 9876px, 153px 9876px, 301.5px 9965.5px,
      175px 13391px, 178px 13391px, 326.5px 13540.5px,
      200px 14741px, 203px 14741px, 351.5px 14848.5px,
      225px 18770px, 228px 18770px, 376.5px 18910.5px,
      250px 5082px, 253px 5082px, 401.5px 5161px,
      275px 6375px, 278px 6375px, 426.5px 6480px;
  }
}
.main .block-container { position: relative; z-index: 2; }
.stSidebar, section[data-testid="stSidebar"] { position: relative; z-index: 3; }
</style>
<div class="rain-overlay"></div>
        """,
        unsafe_allow_html=True,
    )


def inject_planner_grid_overlay() -> None:
    """规划页背景：浅色线框网格，不影响交互。"""
    st.markdown(
        """
<style>
.planner-grid-bg {
  position: fixed;
  inset: 0;
  width: 100%;
  height: 100%;
  z-index: 0;
  pointer-events: none;
  opacity: 0.22;
  background: #000000;
  --gap: 5em;
  --line: 1px;
  --color: rgba(255, 255, 255, 0.2);
  background-image: linear-gradient(
      -90deg,
      transparent calc(var(--gap) - var(--line)),
      var(--color) calc(var(--gap) - var(--line) + 1px),
      var(--color) var(--gap)
    ),
    linear-gradient(
      0deg,
      transparent calc(var(--gap) - var(--line)),
      var(--color) calc(var(--gap) - var(--line) + 1px),
      var(--color) var(--gap)
    );
  background-size: var(--gap) var(--gap);
}
.main .block-container { position: relative; z-index: 2; }
.stSidebar, section[data-testid="stSidebar"] { position: relative; z-index: 3; }
</style>
<div class="planner-grid-bg"></div>
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

    if page_param in {"home", "planner", "data", "diagnostics", "about"}:
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
    with st.sidebar:
        render_lang_toggle()
        st.markdown(
            """
<style>
.nav-stack {
  display: flex;
  flex-direction: column;
  gap: 6px;
  width: 210px;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  border-radius: 10px;
  background: #0f172a;
  color: #f8fafc;
  text-decoration: none;
  position: relative;
  transition: all 0.2s ease;
  border: 1px solid transparent;
}
.nav-item:link,
.nav-item:visited,
.nav-item:hover,
.nav-item:active {
  text-decoration: none;
}
.nav-item:hover {
  background: #131d35;
  border-color: rgba(79, 195, 247, 0.25);
}
.nav-item.active {
  background: linear-gradient(135deg, #1e293b, #0ea5e9);
  color: #fff;
  border-color: #38bdf8;
  box-shadow: 0 4px 12px rgba(14, 165, 233, 0.25);
}
.nav-item svg {
  width: 16px;
  height: 16px;
  flex-shrink: 0;
  opacity: 0.9;
}
.nav-item .label {
  font-weight: 600;
  letter-spacing: 0.2px;
}
</style>
""",
            unsafe_allow_html=True,
        )

        def _btn(label: str, svg: str, target: str) -> str:
            return f"""
  <a class="nav-item {'active' if target == page else ''}" href="?page={target}">
    {svg}
    <span class="label">{label}</span>
  </a>"""

        icons = {
            "home": """<svg data-name="Layer 2" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><path d="m1.5 13v1a.5.5 0 0 0 .3379.4731 18.9718 18.9718 0 0 0 6.1621 1.0269 18.9629 18.9629 0 0 0 6.1621-1.0269.5.5 0 0 0 .3379-.4731v-1a6.5083 6.5083 0 0 0 -4.461-6.1676 3.5 3.5 0 1 0 -4.078 0 6.5083 6.5083 0 0 0 -4.461 6.1676zm4-9a2.5 2.5 0 1 1 2.5 2.5 2.5026 2.5026 0 0 1 -2.5-2.5zm2.5 3.5a5.5066 5.5066 0 0 1 5.5 5.5v.6392a18.08 18.08 0 0 1 -11 0v-.6392a5.5066 5.5066 0 0 1 5.5-5.5z" fill="#7D8590"></path></svg>""",
            "planner": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><path d="m17.074 30h-2.148c-1.038 0-1.914-.811-1.994-1.846l-.125-1.635c-.687-.208-1.351-.484-1.985-.824l-1.246 1.067c-.788.677-1.98.631-2.715-.104l-1.52-1.52c-.734-.734-.78-1.927-.104-2.715l1.067-1.246c-.34-.635-.616-1.299-.824-1.985l-1.634-.125c-1.035-.079-1.846-.955-1.846-1.993v-2.148c0-1.038.811-1.914 1.846-1.994l1.635-.125c.208-.687.484-1.351.824-1.985l-1.068-1.247c-.676-.788-.631-1.98.104-2.715l1.52-1.52c.734-.734 1.927-.779 2.715-.104l1.246 1.067c.635-.34 1.299-.616 1.985-.824l.125-1.634c.08-1.034.956-1.845 1.994-1.845h2.148c1.038 0 1.914.811 1.994 1.846l.125 1.635c.687.208 1.351.484 1.985.824l1.246-1.067c.787-.676 1.98-.631 2.715.104l1.52 1.52c.734.734.78 1.927.104 2.715l-1.067 1.246c.34.635.616 1.299.824 1.985l1.634.125c1.035.079 1.846.955 1.846 1.993v2.148c0 1.038-.811 1.914-1.846 1.994l-1.635.125c-.208.687-.484 1.351-.824 1.985l1.067 1.246c.677.788.631 1.98-.104 2.715l-1.52 1.52c-.734.734-1.928.78-2.715.104l-1.246-1.067c-.635.34-1.299.616-1.985.824l-.125 1.634c-.079 1.035-.955 1.846-1.993 1.846zm-5.835-6.373c.848.53 1.768.912 2.734 1.135.426.099.739.462.772.898l.18 2.341 2.149-.001.18-2.34c.033-.437.347-.8.772-.898.967-.223 1.887-.604 2.734-1.135.371-.232.849-.197 1.181.089l1.784 1.529 1.52-1.52-1.529-1.784c-.285-.332-.321-.811-.089-1.181.53-.848.912-1.768 1.135-2.734.099-.426.462-.739.898-.772l2.341-.18h-.001v-2.148l-2.34-.18c-.437-.033-.8-.347-.898-.772-.223-.967-.604-1.887-1.135-2.734-.232-.37-.196-.849.089-1.181l1.529-1.784-1.52-1.52-1.784 1.529c-.332.286-.81.321-1.181.089-.848-.53-1.768-.912-2.734-1.135-.426-.099-.739-.462-.772-.898l-.18-2.341-2.148.001-.18 2.34c-.033.437-.347.8-.772.898-.967.223-1.887.604-2.734 1.135-.37.232-.849.197-1.181-.089l-1.785-1.529-1.52 1.52 1.529 1.784c.285.332.321.811.089 1.181-.53.848-.912 1.768-1.135 2.734-.099.426-.462.739-.898.772l-2.341.18.002 2.148 2.34.18c.437.033.8.347.898.772.223.967.604 1.887 1.135 2.734.232.37.196.849-.089 1.181l-1.529 1.784 1.52 1.52 1.784-1.529c.332-.287.813-.32 1.18-.089z" fill="#7D8590"></path><path d="m16 23c-3.859 0-7-3.141-7-7s3.141-7 7-7 7 3.141 7 7-3.141 7-7 7zm0-12c-2.757 0-5 2.243-5 5s2.243 5 5 5 5-2.243 5-5-2.243-5-5-5z" fill="#7D8590"></path></svg>""",
            "data": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 128 128"><path d="m109.9 20.63a6.232 6.232 0 0 0 -8.588-.22l-57.463 51.843c-.012.011-.02.024-.031.035s-.023.017-.034.027l-4.721 4.722a1.749 1.749 0 0 0 0 2.475l.341.342-3.16 3.16a8 8 0 0 0 -1.424 1.967 11.382 11.382 0 0 0 -12.055 10.609c-.006.036-.011.074-.015.111a5.763 5.763 0 0 1 -4.928 5.41 1.75 1.75 0 0 0 -.844 3.14c4.844 3.619 9.4 4.915 13.338 4.915a17.14 17.14 0 0 0 11.738-4.545l.182-.167a11.354 11.354 0 0 0 3.348-8.081c0-.225-.02-.445-.032-.667a8.041 8.041 0 0 0 1.962-1.421l3.16-3.161.342.342a1.749 1.749 0 0 0 2.475 0l4.722-4.722c.011-.011.018-.025.029-.036s.023-.018.033-.029l51.844-57.46a6.236 6.236 0 0 0 -.219-8.589zm-70.1 81.311-.122.111c-.808.787-7.667 6.974-17.826 1.221a9.166 9.166 0 0 0 4.36-7.036 1.758 1.758 0 0 0 .036-.273 7.892 7.892 0 0 1 9.122-7.414c.017.005.031.014.048.019a1.717 1.717 0 0 0 .379.055 7.918 7.918 0 0 1 4 13.317zm5.239-10.131c-.093.093-.194.176-.293.26a11.459 11.459 0 0 0 -6.289-6.286c.084-.1.167-.2.261-.3l3.161-3.161 6.321 6.326zm7.214-4.057-9.479-9.479 2.247-2.247 9.479 9.479zm55.267-60.879-50.61 56.092-9.348-9.348 56.092-50.61a2.737 2.737 0 0 1 3.866 3.866z" fill="#7D8590"></path></svg>""",
            "diagnostics": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32"><g transform="translate(-33.022 -30.617)"><path d="m49.021 31.617c-2.673 0-4.861 2.188-4.861 4.861 0 1.606.798 3.081 1.873 3.834h-7.896c-1.7 0-3.098 1.401-3.098 3.1s1.399 3.098 3.098 3.098h4.377l.223 2.641s-1.764 8.565-1.764 8.566c-.438 1.642.55 3.355 2.191 3.795s3.327-.494 3.799-2.191l2.059-5.189 2.059 5.189c.44 1.643 2.157 2.631 3.799 2.191s2.63-2.153 2.191-3.795l-1.764-8.566.223-2.641h4.377c1.699 0 3.098-1.399 3.098-3.098s-1.397-3.1-3.098-3.1h-7.928c1.102-.771 1.904-2.228 1.904-3.834 0-2.672-2.189-4.861-4.862-4.861zm0 2c1.592 0 2.861 1.27 2.861 2.861 0 1.169-.705 2.214-1.789 2.652-.501.203-.75.767-.563 1.273l.463 1.254c.145.393.519.654.938.654h8.975c.626 0 1.098.473 1.098 1.1s-.471 1.098-1.098 1.098h-5.297c-.52 0-.952.398-.996.916l-.311 3.701c-.008.096-.002.191.018.285 0 0 1.813 8.802 1.816 8.82.162.604-.173 1.186-.777 1.348s-1.184-.173-1.346-.777c-.01-.037-3.063-7.76-3.063-7.76-.334-.842-1.525-.842-1.859 0 0 0-3.052 7.723-3.063 7.76-.162.604-.741.939-1.346.777s-.939-.743-.777-1.348c.004-.019 1.816-8.82 1.816-8.82.02-.094.025-.189.018-.285l-.311-3.701c-.044-.518-.477-.916-.996-.916h-5.297c-.627 0-1.098-.471-1.098-1.098s.472-1.1 1.098-1.1h8.975c.419 0 .793-.262.938-.654l.463-1.254c.188-.507-.062-1.07-.563-1.273-1.084-.438-1.789-1.483-1.789-2.652.001-1.591 1.271-2.861 2.862-2.861z" fill="#7D8590"></path></g></svg>""",
            "about": """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 25" fill="none"><path fill-rule="evenodd" fill="#7D8590" d="m11.9572 4.31201c-3.35401 0-6.00906 2.59741-6.00906 5.67742v3.29037c0 .1986-.05916.3927-.16992.5576l-1.62529 2.4193-.01077.0157c-.18701.2673-.16653.5113-.07001.6868.10031.1825.31959.3528.67282.3528h14.52603c.2546 0 .5013-.1515.6391-.3968.1315-.2343.1117-.4475-.0118-.6093-.0065-.0085-.0129-.0171-.0191-.0258l-1.7269-2.4194c-.121-.1695-.186-.3726-.186-.5809v-3.29037c0-1.54561-.6851-3.023-1.7072-4.00431-1.1617-1.01594-2.6545-1.67311-4.3019-1.67311zm-8.00906 5.67742c0-4.27483 3.64294-7.67742 8.00906-7.67742 2.2055 0 4.1606.88547 5.6378 2.18455.01.00877.0198.01774.0294.02691 1.408 1.34136 2.3419 3.34131 2.3419 5.46596v2.97007l1.5325 2.1471c.6775.8999.6054 1.9859.1552 2.7877-.4464.795-1.3171 1.4177-2.383 1.4177h-14.52603c-2.16218 0-3.55087-2.302-2.24739-4.1777l1.45056-2.1593zm4.05187 11.32257c0-.5523.44772-1 1-1h5.99999c.5523 0 1 .4477 1 1s-.4477 1-1 1h-5.99999c-.55228 0-1-.4477-1-1z" clip-rule="evenodd"></path></svg>""",
        }

        labels = {
            "home": tr("nav_home"),
            "planner": tr("nav_planner"),
            "data": tr("nav_data"),
            "diagnostics": "Diagnostics / 诊断",
            "about": "About / 关于",
        }

        buttons = "\n".join(
            _btn(labels[k], icons[k], k) for k in ["home", "planner", "data", "diagnostics", "about"]
        )

        st.markdown(
            f"""
<div class="nav-stack">
{buttons}
</div>
""",
            unsafe_allow_html=True,
        )

    # 直接返回当前 page；切换通过按钮的 URL 更新实现
    return page


def render_diagnostics_page() -> None:
    """轻量诊断页：展示环境变量与常用路径。"""
    st.header("诊断 / Diagnostics")
    st.write("用于快速确认当前运行入口、环境变量与工作目录。")

    st.subheader("环境变量")
    env_vars = {
        "ARCTICROUTE_DATA_ROOT": os.environ.get("ARCTICROUTE_DATA_ROOT"),
        "PYTHONPATH": os.environ.get("PYTHONPATH"),
    }
    for k, v in env_vars.items():
        st.code(f"{k}={v}", language="text")

    st.subheader("路径")
    st.code(f"cwd: {Path.cwd()}", language="text")
    st.code(f"entry: {Path(__file__).resolve()}", language="text")

    st.subheader("说明")
    st.caption("如 UI 看起来异常，可先确认是否跑错分支/入口文件，再查看底部 build banner。")


def render_about_page() -> None:
    """About 页面：保留矩阵背景，移除展示卡片。"""
    logo_path = Path(r"C:\Users\sgddsf\Desktop\roundLOGO.png")
    logo_uri = ""
    if logo_path.exists():
        try:
            logo_b64 = base64.b64encode(logo_path.read_bytes()).decode("utf-8")
            logo_uri = f"data:image/png;base64,{logo_b64}"
        except Exception:
            logo_uri = ""

    st.markdown(
        """
<style>
.matrix-container {
  position: fixed;
  inset: 0;
  overflow: hidden;
  background: #000;
  display: flex;
  z-index: 0;
  pointer-events: none;
}
.matrix-pattern {
  position: relative;
  width: 1000px;
  height: 100%;
  flex-shrink: 0;
}
.matrix-column {
  position: absolute;
  top: -100%;
  width: 20px;
  height: 100%;
  font-size: 16px;
  line-height: 18px;
  font-weight: bold;
  animation: fall linear infinite;
  animation-duration: 8s; /* 全局变慢 */
  white-space: nowrap;
}
.matrix-column::before {
  content: "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789";
  position: absolute;
  top: 0;
  left: 0;
  background: linear-gradient(
    to bottom,
    #ffffff 0%,
    #ffffff 5%,
    #00ff41 10%,
    #00ff41 20%,
    #00dd33 30%,
    #00bb22 40%,
    #009911 50%,
    #007700 60%,
    #005500 70%,
    #003300 80%,
    rgba(0, 255, 65, 0.5) 90%,
    transparent 100%
  );
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
  writing-mode: vertical-lr;
  letter-spacing: 1px;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.matrix-column:nth-child(odd)::before {
  content: "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン123456789";
}
.matrix-column:nth-child(even)::before {
  content: "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポヴァィゥェォャュョッABCDEFGHIJKLMNOPQRSTUVWXYZ";
}
.matrix-column:nth-child(3n)::before {
  content: "アカサタナハマヤラワイキシチニヒミリウクスツヌフムユルエケセテネヘメレオコソトノホモヨロヲン0987654321";
}
.matrix-column:nth-child(4n)::before {
  content: "ンヲロヨモホノトソコオレメヘネテセケエルユムフヌツスクウリミヒニチシキイワラヤマハナタサカア";
}
.matrix-column:nth-child(5n)::before {
  content: "ガザダバパギジヂビピグズヅブプゲゼデベペゴゾドボポヴァィゥェォャュョッ!@#$%^&*()_+-=[]{}|;:,.<>?";
}
.matrix-column:nth-child(1) { left: 0px; animation-delay: -2.5s; animation-duration: 6s; }
.matrix-column:nth-child(2) { left: 25px; animation-delay: -3.2s; animation-duration: 7s; }
.matrix-column:nth-child(3) { left: 50px; animation-delay: -1.8s; animation-duration: 5.5s; }
.matrix-column:nth-child(4) { left: 75px; animation-delay: -2.9s; animation-duration: 6.5s; }
.matrix-column:nth-child(5) { left: 100px; animation-delay: -1.5s; animation-duration: 6s; }
.matrix-column:nth-child(6) { left: 125px; animation-delay: -3.8s; animation-duration: 7.5s; }
.matrix-column:nth-child(7) { left: 150px; animation-delay: -2.1s; animation-duration: 5.8s; }
.matrix-column:nth-child(8) { left: 175px; animation-delay: -2.7s; animation-duration: 6.6s; }
.matrix-column:nth-child(9) { left: 200px; animation-delay: -3.4s; animation-duration: 7.2s; }
.matrix-column:nth-child(10) { left: 225px; animation-delay: -1.9s; animation-duration: 5.7s; }
.matrix-column:nth-child(11) { left: 250px; animation-delay: -3.6s; animation-duration: 7.4s; }
.matrix-column:nth-child(12) { left: 275px; animation-delay: -2.3s; animation-duration: 6.2s; }
.matrix-column:nth-child(13) { left: 300px; animation-delay: -3.1s; animation-duration: 6.8s; }
.matrix-column:nth-child(14) { left: 325px; animation-delay: -2.6s; animation-duration: 5.9s; }
.matrix-column:nth-child(15) { left: 350px; animation-delay: -3.7s; animation-duration: 7.6s; }
.matrix-column:nth-child(16) { left: 375px; animation-delay: -2.8s; animation-duration: 6.4s; }
.matrix-column:nth-child(17) { left: 400px; animation-delay: -3.3s; animation-duration: 7s; }
.matrix-column:nth-child(18) { left: 425px; animation-delay: -2.2s; animation-duration: 5.6s; }
.matrix-column:nth-child(19) { left: 450px; animation-delay: -3.9s; animation-duration: 7.8s; }
.matrix-column:nth-child(20) { left: 475px; animation-delay: -2.4s; animation-duration: 6.5s; }
.matrix-column:nth-child(21) { left: 500px; animation-delay: -1.7s; animation-duration: 5.4s; }
.matrix-column:nth-child(22) { left: 525px; animation-delay: -3.5s; animation-duration: 7.1s; }
.matrix-column:nth-child(23) { left: 550px; animation-delay: -2s; animation-duration: 6s; }
.matrix-column:nth-child(24) { left: 575px; animation-delay: -4s; animation-duration: 8s; }
.matrix-column:nth-child(25) { left: 600px; animation-delay: -1.6s; animation-duration: 5.2s; }
.matrix-column:nth-child(26) { left: 625px; animation-delay: -3s; animation-duration: 6.7s; }
.matrix-column:nth-child(27) { left: 650px; animation-delay: -3.8s; animation-duration: 7.3s; }
.matrix-column:nth-child(28) { left: 675px; animation-delay: -2.5s; animation-duration: 5.8s; }
.matrix-column:nth-child(29) { left: 700px; animation-delay: -3.2s; animation-duration: 6.9s; }
.matrix-column:nth-child(30) { left: 725px; animation-delay: -2.7s; animation-duration: 6.3s; }
.matrix-column:nth-child(31) { left: 750px; animation-delay: -1.8s; animation-duration: 5.6s; }
.matrix-column:nth-child(32) { left: 775px; animation-delay: -3.6s; animation-duration: 7.2s; }
.matrix-column:nth-child(33) { left: 800px; animation-delay: -2.1s; animation-duration: 6.1s; }
.matrix-column:nth-child(34) { left: 825px; animation-delay: -3.4s; animation-duration: 6.8s; }
.matrix-column:nth-child(35) { left: 850px; animation-delay: -2.8s; animation-duration: 5.9s; }
.matrix-column:nth-child(36) { left: 875px; animation-delay: -3.7s; animation-duration: 7.5s; }
.matrix-column:nth-child(37) { left: 900px; animation-delay: -2.3s; animation-duration: 6.2s; }
.matrix-column:nth-child(38) { left: 925px; animation-delay: -1.9s; animation-duration: 5.7s; }
.matrix-column:nth-child(39) { left: 950px; animation-delay: -3.5s; animation-duration: 7s; }
.matrix-column:nth-child(40) { left: 975px; animation-delay: -2.6s; animation-duration: 6.4s; }
@keyframes fall {
  0% { transform: translateY(-10%); opacity: 1; }
  100% { transform: translateY(200%); opacity: 0; }
}
@media (max-width: 768px) {
  .matrix-column { font-size: 14px; line-height: 16px; width: 18px; }
}
@media (max-width: 480px) {
  .matrix-column { font-size: 12px; line-height: 14px; width: 15px; }
}

.about-content {
  position: relative;
  z-index: 1;
  padding: 2rem 0;
  display: flex;
  justify-content: center;
}
.card {
  width: 230px;
  height: 320px;
  background: rgb(39, 39, 39);
  border-radius: 12px;
  box-shadow: 0px 0px 30px rgba(0, 0, 0, 0.123);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: flex-start;
  transition-duration: .5s;
}

.profileImage {
  background: linear-gradient(to right,rgb(54, 54, 54),rgb(32, 32, 32));
  margin-top: 20px;
  width: 170px;
  height: 170px;
  border-radius: 50%;
  box-shadow: 5px 10px 20px rgba(0, 0, 0, 0.329);
}

.textContainer {
  width: 100%;
  text-align: left;
  padding: 22px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.name {
  font-size: 0.9em;
  font-weight: 600;
  color: white;
  letter-spacing: 0.5px;
}

.profile {
  font-size: 0.84em;
  color: rgb(194, 194, 194);
  letter-spacing: 0.2px;
}

.card:hover {
  background-color: rgb(43, 43, 43);
  transition-duration: .5s;
}
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
<div class="matrix-container">
  """ + "\n".join(['<div class="matrix-pattern">' + "".join('<div class="matrix-column"></div>' for _ in range(40)) + '</div>' for _ in range(5)]) + """
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
<div class="about-content">
  <div class="card">
    <div class="profileImage">
      """ + (f'<img src="{logo_uri}" alt="logo" style="width:170px;height:170px;border-radius:50%;object-fit:cover;" />' if logo_uri else '<div style="width:170px;height:170px;border-radius:50%;background:#333;display:flex;align-items:center;justify-content:center;color:#fff;font-weight:700;">Logo</div>') + """
    </div>
    <div class="textContainer">
      <p class="name">CaiYuanQi</p>
      <p class="profile">github@Cassian2006</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


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
        inject_planner_grid_overlay()
        planner_minimal.render()
    elif page == "data":
        inject_rain_overlay()
        st.header(tr("data_page_title"))
        render_data_discovery_panel()
        st.markdown("---")
        eval_results.render()
    elif page == "diagnostics":
        inject_rain_overlay()
        render_diagnostics_page()
    elif page == "about":
        render_about_page()
    else:
        home.render()

    # 页面底部统一显示构建信息
    render_build_banner(entry_file=__file__, page=page)


if __name__ == "__main__":
    main()
