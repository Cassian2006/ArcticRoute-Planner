"""
封面页 - 应用首页
"""

from __future__ import annotations

import streamlit as st

from arcticroute.ui.app_router import PAGE_PLANNER, PAGE_DATA, create_page_button


def render_cover() -> None:
    """渲染封面页"""
    
    st.markdown(
        """
        <div class="cover-card">
            <div style="display:flex; flex-direction:column; gap:1.5rem;">
                <div>
                    <div style="font-size:0.9rem; letter-spacing:0.18em; text-transform:uppercase; color:var(--text-muted, #8aa0b2);">
                        ArcticRoute Mission Control
                    </div>
                    <h1 style="margin:0.3rem 0 0.6rem 0; font-size:2.4rem; color:var(--text-primary, #f9fafb);">
                         北极航线智能规划
                    </h1>
                    <p style="margin:0; max-width:640px; color:var(--text-muted, #9fb2c0); line-height:1.6;">
                        多源海冰与波浪情报叠加，快速生成安全/效率/稳健三种航线。
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    st.markdown("### 从这里开始")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # 使用路由器的按钮创建函数
        create_page_button("进入规划", PAGE_PLANNER)
        create_page_button("打开数据页", PAGE_DATA)
    
    # 项目亮点
    st.markdown("---")
    st.markdown("### 项目亮点")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div class="card">
                <h4 style="color:var(--text-primary, #f9fafb); margin-top[object Object]模态成本</h4>
                <p style="color:var(--text-secondary, #e5e7eb); font-size:0.9rem; line-height:1.5;">
                    海冰 SIC/SIT + 海浪 SWH + AIS 拥挤度 + 冰级约束 + POLARIS 规则
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            """
            <div class="card">
                <h4 style="color:var(--text-primary, #f9fafb); margin-top:0;"> EDL 风险评估</h4>
                <p style="color:var(--text-secondary, #e5e7eb); font-size:0.9rem; line-height:1.5;">
                    miles-guess / PyTorch 模型 · 风险 + 不确定性双重评估
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            """
            <div class="card">
                <h4 style="color:var(--text-primary, #f9fafb); margin-top:0;"> 智能规划</h4>
                <p style="color:var(--text-secondary, #e5e7eb); font-size:0.9rem; line-height:1.5;">
                    三种策略：效率优先 / 风险均衡 / 稳健安全
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    # 数据源说明
    st.markdown("---")
    st.markdown("### 数据源")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **环境数据**
        -  海冰浓度 (SIC): Copernicus CMEMS
        -  海冰厚度 (SIT): CMEMS
        -  有效波高 (SWH): CMEMS
        -  海冰漂移: CMEMS
        """)
    
    with col2:
        st.markdown("""
        **静态资产**
        -  AIS 拥挤度: 历史航迹密度
        -  主航道走廊: 高斯核密度估计
        -  浅水区: 水深数据
        -  港口: 全球港口数据库
        """)
    
    # 版本信息
    st.markdown("---")
    st.caption("ArcticRoute v2.0 | 基于 PolarRoute + EDL + 多模态成本场")

