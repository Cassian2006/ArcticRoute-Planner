#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试流动管线 UI 组件的演示脚本。
"""

import streamlit as st
from arcticroute.ui.components.pipeline_flow import PipeNode, render_pipeline_flow
from datetime import datetime
import time

st.set_page_config(page_title="流动管线 UI 测试", layout="wide")

st.title("🔄 流动管线 UI 测试")

st.markdown("""
这是一个演示脚本，展示流动管线 UI 的各种状态和动画效果。
""")

# 初始化 session state
if "test_nodes" not in st.session_state:
    st.session_state.test_nodes = [
        PipeNode(key="parse", label="① 解析场景/参数", status="done", seconds=0.5, detail="参数解析完成"),
        PipeNode(key="grid_landmask", label="② 加载网格与 landmask", status="done", seconds=0.3, detail="grid=500×5333"),
        PipeNode(key="env_layers", label="③ 加载环境层", status="running", detail="加载 SIC/Wave..."),
        PipeNode(key="ais_density", label="④ 加载 AIS 密度", status="pending"),
        PipeNode(key="cost_field", label="⑤ 构建成本场", status="pending"),
        PipeNode(key="astar", label="⑥ A* 规划", status="pending"),
        PipeNode(key="analysis", label="⑦ 分析与诊断", status="pending"),
        PipeNode(key="render", label="⑧ 渲染与导出", status="pending"),
    ]

if "test_step" not in st.session_state:
    st.session_state.test_step = 0

# 创建两列布局
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("流动管线演示")
    render_pipeline_flow(
        st.session_state.test_nodes,
        title="🔄 规划流程管线",
        expanded=True,
    )

with col2:
    st.subheader("控制面板")
    
    if st.button("▶️ 下一步"):
        step = st.session_state.test_step
        nodes = st.session_state.test_nodes
        
        if step == 0:
            # 完成第 3 个节点
            nodes[2].status = "done"
            nodes[2].seconds = 0.2
            nodes[2].detail = "SIC/Wave 已加载"
            # 启动第 4 个节点
            nodes[3].status = "running"
            nodes[3].detail = "加载 AIS..."
        elif step == 1:
            # 完成第 4 个节点
            nodes[3].status = "done"
            nodes[3].seconds = 0.4
            nodes[3].detail = "AIS=(500, 5333)"
            # 启动第 5 个节点
            nodes[4].status = "running"
            nodes[4].detail = "构建成本场..."
        elif step == 2:
            # 完成第 5 个节点
            nodes[4].status = "done"
            nodes[4].seconds = 0.6
            nodes[4].detail = "3 种成本场"
            # 启动第 6 个节点
            nodes[5].status = "running"
            nodes[5].detail = "规划路线..."
        elif step == 3:
            # 完成第 6 个节点
            nodes[5].status = "done"
            nodes[5].seconds = 0.8
            nodes[5].detail = "可达=3/3"
            # 启动第 7 个节点
            nodes[6].status = "running"
            nodes[6].detail = "分析成本..."
        elif step == 4:
            # 完成第 7 个节点
            nodes[6].status = "done"
            nodes[6].seconds = 0.3
            nodes[6].detail = "分析完成"
            # 启动第 8 个节点
            nodes[7].status = "running"
            nodes[7].detail = "渲染地图..."
        elif step == 5:
            # 完成第 8 个节点
            nodes[7].status = "done"
            nodes[7].seconds = 0.5
            nodes[7].detail = "渲染完成"
        
        st.session_state.test_step = step + 1
        st.rerun()
    
    if st.button("🔄 重置"):
        st.session_state.test_nodes = [
            PipeNode(key="parse", label="① 解析场景/参数", status="pending"),
            PipeNode(key="grid_landmask", label="② 加载网格与 landmask", status="pending"),
            PipeNode(key="env_layers", label="③ 加载环境层", status="pending"),
            PipeNode(key="ais_density", label="④ 加载 AIS 密度", status="pending"),
            PipeNode(key="cost_field", label="⑤ 构建成本场", status="pending"),
            PipeNode(key="astar", label="⑥ A* 规划", status="pending"),
            PipeNode(key="analysis", label="⑦ 分析与诊断", status="pending"),
            PipeNode(key="render", label="⑧ 渲染与导出", status="pending"),
        ]
        st.session_state.test_step = 0
        st.rerun()
    
    st.divider()
    st.caption(f"当前步骤: {st.session_state.test_step}")

st.divider()

st.subheader("📝 说明")
st.markdown("""
### 流动管线 UI 特性

1. **节点状态**：
   - ⏳ pending：等待执行
   - 🚧 running：正在执行（管道流动）
   - ✅ done：执行完成（绿色）
   - ❌ fail：执行失败（红色）

2. **管道动画**：
   - 当节点状态为 "running" 时，连接管道会显示流动动画
   - 完成的节点之间的管道变为绿色
   - 失败的管道变为红色

3. **节点信息**：
   - 每个节点显示标签、状态和详情
   - 耗时信息在完成后显示
   - 底部显示完成数量和总耗时

4. **响应式设计**：
   - 节点横排排列，支持水平滚动
   - 适配不同屏幕宽度
   - 深色主题，与 Streamlit 风格一致
""")

