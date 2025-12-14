from __future__ import annotations
import streamlit as st
from ArcticRoute.apps.pages import planner_v2 as planner

# 在任何 Streamlit UI 调用之前插入 CSS：隐藏左侧页面导航（ui app / Planner 切换）
st.markdown(
    """
    <style>
    /* 隐藏左侧页面导航（ui app / Planner 切换） */
    section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Planner", layout="wide")

# 追加全局 CSS：深色背景 + 玻璃卡片（不覆盖已有样式）
st.markdown(
    """
    <style>
    /* 整体背景再统一一下：深色 + 轻渐变 */
    .main {
        background: radial-gradient(circle at top, #10192b 0, #050810 55%, #020409 100%);
    }

    /* 通用玻璃卡片 */
    .glass-card {
        background: rgba(10, 20, 40, 0.82);
        border-radius: 22px;
        border: 1px solid rgba(180, 255, 255, 0.06);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.7);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        padding: 20px 22px;
        margin-bottom: 18px;
    }

    /* 中间地图这块可以更宽一点的卡片 */
    .glass-card-wide {
        background: rgba(10, 20, 40, 0.9);
        border-radius: 26px;
        border: 1px solid rgba(120, 220, 255, 0.12);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        padding: 22px 24px 18px 24px;
        margin-bottom: 16px;
    }

    .glass-card h3, .glass-card h4, .glass-card-wide h3, .glass-card-wide h4 {
        margin-top: 0;
        margin-bottom: 0.6rem;
        font-weight: 600;
    }

    .section-caption {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.6);
        margin-bottom: 0.5rem;
    }

    /* 让主内容区域不要被挤到下面，去掉默认的大 padding */
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1420px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def inject_planner_glass():
    """为 Planner 页面注入玻璃拟态与标题区域样式，不改动业务逻辑。"""
    st.markdown(
        """
        <style>
        /* 隐藏默认渲染的首个标题与其后紧邻的一段描述（若存在） */
        .block-container h1:first-of-type,
        .block-container h1:first-of-type + p {
          display: none !important;
        }

        /* 整体背景更暗一点（Planner 页） */
        .stApp {
          background: radial-gradient(circle at top, #151b2f 0, #050712 40%, #04040a 100%);
        }

        /* 外层居中容器 */
        .planner-hero-wrap {
          display: flex;
          justify-content: center;
          margin-top: 0.8rem;
          margin-bottom: 0.6rem;
          padding: 0 0.5rem;
        }

        /* 玻璃拟态卡片 */
        .planner-hero {
          width: min(1100px, 96%);
          background: rgba(255, 255, 255, 0.06);
          backdrop-filter: blur(12px) saturate(120%);
          -webkit-backdrop-filter: blur(12px) saturate(120%);
          border-radius: 18px;
          border: 1px solid rgba(255, 255, 255, 0.18);
          box-shadow:
            0 10px 26px rgba(0, 0, 0, 0.28),
            inset 0 0 0 1px rgba(255, 255, 255, 0.06);
          padding: 1.2rem 1.4rem;
          position: relative;
        }

        /* 轻霓虹描边效果（通过伪元素实现边框光晕） */
        .planner-hero::before {
          content: "";
          position: absolute;
          inset: -2px;
          border-radius: 20px;
          background: linear-gradient(135deg, rgba(30,242,255,0.35), rgba(99,102,241,0.25));
          filter: blur(14px);
          z-index: -1;
        }

        /* 标题与副标题排版 */
        .ph-title {
          font-weight: 800;
          font-size: clamp(1.2rem, 2.4vw, 1.6rem);
          letter-spacing: 0.04em;
          color: #e6f7ff;
          text-shadow:
            0 0 8px rgba(30, 242, 255, 0.55),
            0 0 18px rgba(30, 242, 255, 0.35);
          margin-bottom: 0.35rem;
        }
        .ph-subtitle {
          font-size: clamp(0.92rem, 1.6vw, 1.02rem);
          color: #cfe9ff;
          opacity: 0.92;
          line-height: 1.55;
        }

        /* 三列卡片的统一样式 */
        .arc-card {
          border-radius: 24px;
          padding: 1.2rem 1.4rem;
          margin-bottom: 1.2rem;
          background: rgba(10, 20, 40, 0.68);
          border: 1px solid rgba(255, 255, 255, 0.08);
          box-shadow: 0 18px 45px rgba(0, 0, 0, 0.55), 0 0 40px rgba(0, 255, 213, 0.05);
          backdrop-filter: blur(22px);
          -webkit-backdrop-filter: blur(22px);
        }
        .arc-card-center { background: rgba(12, 24, 50, 0.82); }
        .arc-card-right-action button {
          border-radius: 999px !important;
          padding: 0.85rem 1.2rem !important;
          font-size: 1.0rem !important;
          font-weight: 600 !important;
          background: linear-gradient(135deg, #ff7a5c, #ffb86c);
          color: #ffffff;
          border: none;
          box-shadow: 0 0 24px rgba(255, 150, 100, 0.8);
        }
        .arc-card-right-action button:hover { transform: translateY(-1px); box-shadow: 0 0 32px rgba(255, 190, 140, 0.95); }
        .arc-card h3, .arc-card h4, .arc-card h2 { text-shadow: 0 0 10px rgba(0, 255, 213, 0.6); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_planner_hero():
    html = """
    <div class="planner-hero-wrap">
      <div class="planner-hero">
        <div class="ph-title">ArcticRoute Planner — 北极航线智能规划</div>
        <div class="ph-subtitle">
          选择时间范围和起止点，系统将基于海冰、风险成本等多模态数据，自动规划一条兼顾安全与效率的北极航线，并给出关键指标。
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# 注入样式 + 渲染玻璃拟态标题，然后进入原业务渲染
inject_planner_glass()
render_planner_hero()
# 紧贴标题渲染主内容
planner.render()
