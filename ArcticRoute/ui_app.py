from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from streamlit.components.v1 import html as st_html

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

# 首屏亮点（仅保留两条）
HIGHLIGHT_LINES = [
    "多源风险融合 · SIC / 事故 / 拥挤 / 历史主航线",
    "绿色航行 · Eco 模式驱动燃油与 CO₂ 成本协同优化",
]

# 单页入口：Planner（精简版）
# - 仅保留 Planner 主页面；其他页面从导航中下线（见 ArcticRoute/pages 目录已清理）
# - 页面开关与默认值仍由 ArcticRoute/apps/registry.py 管理（仅 planner:true 有效）


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path):
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
    except Exception:
        return {}


# 保留原有全局页面配置
st.set_page_config(page_title="ArcticRoute UI", layout="wide")

# 确保将项目根目录（minimum）加入 sys.path，避免导入 ArcticRoute.* 失败
import sys as _sys, pathlib as _pathlib
_ROOT = _pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps import state as ui_state   # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore

ureg = UIRegistry()
_theme_on = read_theme_flag()
inject_theme(_theme_on)
flags = json.loads(ureg.to_json())


# ========== Sidebar：仅显示“关于 / About” ==========
with st.sidebar:
    with st.expander("关于 ArcticRoute Planner", expanded=False):
        st.markdown(
            """
**Author**：蔡元祺  
**项目名称**：ArcticRoute Planner — 北极多模态智能航线规划平台  
**联系方式**：  
- Email：`caiyuanqi2006@outlook.com`  
- GitHub：`https://github.com/Cassian2006`  

**开发者声明**：  
- 本项目由作者独立完成，用于课程 / 科研 / 比赛演示。  
- 禁止抄袭、未授权转载或用于商业用途。  

**数据与模型来源**：  
- Copernicus Marine Service（海冰、海浪等再分析/预报产品）  
- NSIDC Sea Ice Index / CDR（海冰浓度）  
- GEBCO 2025（海底地形与海陆线）  

**使用提醒**：  
- 本系统仅用于技术演示与决策辅助，不构成实际航行导航建议。  
- 航运决策需结合更高精度资料与专业人员判断。
            """
        )


# ========== 首页主体（美化首屏 + 两条打字机字幕 + 居中拉长按钮） ==========

def render_home():
    # 1) 注入全局背景、Hero 标题样式、打字机条、按钮样式
    st.markdown(
        """
        <style>
        /* 深色星空背景 */
        .stApp {
          background: radial-gradient(circle at top, #071524 0, #020510 45%, #000000 100%);
        }

        /* 顶部 Hero 标题区 */
        .hero-shell { display:flex; align-items:center; justify-content:center; padding: 3rem 1.2rem 0.6rem; }
        .hero-inner { text-align:center; }
        .neon-title {
          font-size: clamp(2.6rem, 5vw, 3.6rem);
          font-weight: 800;
          letter-spacing: 0.28em;
          text-transform: uppercase;
          margin-bottom: 0.8rem;
          color: #1ef2ff;
          text-shadow: 0 0 8px rgba(30,242,255,.9), 0 0 22px rgba(30,242,255,.7), 0 0 42px rgba(30,242,255,.4);
        }
        .hero-subtitle { font-size: 1.1rem; letter-spacing: .18em; color: #d4f8ff; opacity: .92; }

        /* 打字机荧光灯条 */
        .arc-hero-typing-wrapper { margin-top: 2.5rem; display:flex; justify-content:center; }
        .arc-hero-typing {
          padding: 0.85rem 2.5rem;
          border-radius: 999px;
          font-size: 1.15rem;
          letter-spacing: 0.12em;
          color: #eaffff;
          background: rgba(0, 255, 213, 0.08);
          border: 1px solid rgba(0, 255, 213, 0.5);
          box-shadow: 0 0 18px rgba(0, 255, 213, 0.55);
          text-shadow: 0 0 12px rgba(0, 255, 213, 0.95);
          white-space: nowrap; overflow: hidden;
        }
        #arc-hero-typing::after {
          content: "▋"; margin-left: 0.2rem; animation: arc-caret 0.8s infinite;
        }
        @keyframes arc-caret { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }

        /* 中下部按钮：玻璃拟态 + 渐变霓虹 */
        div.stButton > button[kind="secondary"], div.stButton > button {
          border-radius: 999px !important;
          padding: 0.9rem 3.5rem !important;
          font-size: 1.05rem !important;
          letter-spacing: 0.15em; text-transform: uppercase;
          border: 0;
          background: linear-gradient(135deg, #ff7a5c, #ffb86c);
          color: #ffffff;
          box-shadow: 0 0 30px rgba(255, 140, 80, 0.75);
        }
        div.stButton > button:hover { box-shadow: 0 0 40px rgba(255, 180, 120, 0.9); transform: translateY(-1px); }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 2) Hero 标题 + 副标题
    st.markdown(
        """
        <div class="hero-shell">
          <div class="hero-inner">
            <div class="neon-title">ARCTICROUTE PLANNER</div>
            <div class="hero-subtitle">北极多模态智能航线规划平台</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3) 打字机字幕（两行轮播）：结构 + JS
    st.markdown(
        """
        <div class="arc-hero-typing-wrapper">
          <div id="arc-hero-typing" class="arc-hero-typing"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 使用组件注入 JS（st.markdown 的 <script> 在部分版本中不会执行，组件更可靠）
    st_html(
        f"""
        <script>
        const lines = {json.dumps(HIGHLIGHT_LINES, ensure_ascii=False)};
        let idx = 0; let charIdx = 0; let isDeleting = false;
        const el = window.parent.document.getElementById('arc-hero-typing');
        function typeLoop() {{
          if (!el) return;
          const fullText = lines[idx % lines.length];
          charIdx += (isDeleting ? -1 : 1);
          el.textContent = fullText.substring(0, Math.max(0, charIdx));
          if (!isDeleting && charIdx === fullText.length) {{ setTimeout(() => (isDeleting = true), 1200); }}
          else if (isDeleting && charIdx <= 0) {{ isDeleting = false; idx++; }}
          const speed = isDeleting ? 40 : 60; setTimeout(typeLoop, speed);
        }}
        typeLoop();
        </script>
        """,
        height=0,
    )

    # 4) 页面底部留白 + 居中宽按钮（进入 Planner）
    st.markdown("<div style='height: 6rem'></div>", unsafe_allow_html=True)
    # 为了向右微移，与上方文本视觉中心更贴齐：加大左列、缩小右列
    col_left, col_btn, col_right = st.columns([1.35, 2, 0.65])
    with col_btn:
        if st.button("进入 Planner 工作台", key="go_planner_main"):
            try:
                st.switch_page("pages/00_Planner.py")
            except Exception:
                st.markdown("""
                    <meta http-equiv='refresh' content='0; url=00_Planner' />
                """, unsafe_allow_html=True)


# 渲染首页
render_home()
