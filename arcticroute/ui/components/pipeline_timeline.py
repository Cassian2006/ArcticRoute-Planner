"""
Pipeline Timeline 组件 - 用于展示规划流程的实时进度

支持：
- 节点状态管理（待执行、执行中、完成、失败）
- 实时耗时统计
- 额外信息展示（如网格大小、AIS 候选数等）
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import time
import streamlit as st


@dataclass
class PipelineStage:
    """流程管线的单个阶段"""
    key: str                          # 唯一标识符
    label: str                        # 显示标签
    status: str = "pending"           # pending / running / done / fail
    dt_s: float = 0.0                 # 耗时（秒）
    extra_info: str = ""              # 额外信息（如 "grid=500×5333"）
    fail_reason: str = ""             # 失败原因


class Pipeline:
    """流程管线管理器"""
    
    def __init__(self):
        self.stages: Dict[str, PipelineStage] = {}
        self.start_times: Dict[str, float] = {}
    
    def add_stage(self, key: str, label: str) -> None:
        """添加一个新的阶段"""
        if key not in self.stages:
            self.stages[key] = PipelineStage(key=key, label=label)
    
    def start(self, key: str) -> None:
        """标记阶段开始执行"""
        if key in self.stages:
            self.stages[key].status = "running"
            self.start_times[key] = time.time()
    
    def done(self, key: str, extra_info: str = "") -> None:
        """标记阶段完成"""
        if key in self.stages:
            self.stages[key].status = "done"
            if key in self.start_times:
                self.stages[key].dt_s = time.time() - self.start_times[key]
            if extra_info:
                self.stages[key].extra_info = extra_info
    
    def fail(self, key: str, fail_reason: str = "") -> None:
        """标记阶段失败"""
        if key in self.stages:
            self.stages[key].status = "fail"
            if key in self.start_times:
                self.stages[key].dt_s = time.time() - self.start_times[key]
            if fail_reason:
                self.stages[key].fail_reason = fail_reason
    
    def get_stages_list(self) -> List[PipelineStage]:
        """获取所有阶段（按添加顺序）"""
        return list(self.stages.values())


def render_pipeline(stages: List[PipelineStage], container) -> None:
    """
    渲染流程管线
    
    Args:
        stages: PipelineStage 列表
        container: Streamlit 容器（如 st.empty() 返回的对象）
    """
    with container.container():
        # 状态图标映射
        status_icons = {
            "pending": "",
            "running": "",
            "done": "",
            "fail": ""
        }
        
        # 计算列数：每个节点 1 列 + 每两个节点间的箭头 1 列
        # 例如 7 个节点：node arrow node arrow node arrow node arrow node arrow node arrow node
        # 总共 7 + 6 = 13 列
        num_stages = len(stages)
        total_cols = num_stages + (num_stages - 1)
        
        cols = st.columns(total_cols)
        
        col_idx = 0
        for stage_idx, stage in enumerate(stages):
            # 渲染节点
            with cols[col_idx]:
                icon = status_icons.get(stage.status, "")
                st.markdown(f"<div style='text-align: center;'>{icon}</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='text-align: center; font-size: 0.85em; font-weight: bold;'>{stage.label}</div>", unsafe_allow_html=True)
                
                # 显示耗时
                if stage.status == "done":
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: green;'>{stage.dt_s:.2f}s</div>", unsafe_allow_html=True)
                elif stage.status == "fail":
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: red;'>{stage.dt_s:.2f}s</div>", unsafe_allow_html=True)
                elif stage.status == "running":
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: orange;'>运行中...</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div style='text-align: center; font-size: 0.75em; color: gray;'>-</div>", unsafe_allow_html=True)
                
                # 显示额外信息
                if stage.extra_info:
                    st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: #666;'>{stage.extra_info}</div>", unsafe_allow_html=True)
                
                # 显示失败原因
                if stage.fail_reason:
                    st.markdown(f"<div style='text-align: center; font-size: 0.7em; color: red;'> {stage.fail_reason}</div>", unsafe_allow_html=True)
            
            col_idx += 1
            
            # 在节点间添加箭头（除了最后一个节点）
            if stage_idx < num_stages - 1:
                with cols[col_idx]:
                    st.markdown("<div style='text-align: center; font-size: 1.2em;'>→</div>", unsafe_allow_html=True)
                col_idx += 1


def init_pipeline_in_session() -> Pipeline:
    """
    在 session_state 中初始化管线对象
    
    Returns:
        Pipeline 对象
    """
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Pipeline()
    return st.session_state.pipeline


def get_pipeline() -> Pipeline:
    """获取当前 session 中的 Pipeline 对象"""
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = Pipeline()
    return st.session_state.pipeline
