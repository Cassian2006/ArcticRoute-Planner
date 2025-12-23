# -*- coding: utf-8 -*-
"""
流动管线 UI 组件 - 显示规划流程各节点，节点间用流动管道连接。

支持：
- 节点状态：pending / running / done / fail
- 管道流动动画（running 时）
- 完成色（done 时）
- 失败红色（fail 时）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import streamlit as st


@dataclass
class PipeNode:
    """管线节点数据类"""
    key: str
    label: str
    status: str  # "pending" | "running" | "done" | "fail"
    seconds: Optional[float] = None
    detail: Optional[str] = None


CSS = """
<style>
.pipeline-wrap {
  padding: 14px 14px 10px;
  border-radius: 14px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(255, 255, 255, 0.04);
}

.pipeline-row {
  display: flex;
  align-items: center;
  gap: 10px;
  flex-wrap: nowrap;
  overflow-x: auto;
  padding-bottom: 6px;
}

.pnode {
  min-width: 140px;
  max-width: 220px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid rgba(255, 255, 255, 0.12);
  background: rgba(0, 0, 0, 0.18);
  transition: all 0.3s ease;
}

.pnode .t {
  font-weight: 700;
  font-size: 14px;
  line-height: 1.2;
}

.pnode .s {
  font-size: 12px;
  opacity: 0.85;
  margin-top: 4px;
}

.pnode .d {
  font-size: 12px;
  opacity: 0.8;
  margin-top: 6px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.pnode.pending {
  opacity: 0.65;
}

.pnode.running {
  border-color: rgba(120, 190, 255, 0.55);
  box-shadow: 0 0 0 2px rgba(120, 190, 255, 0.12) inset;
}

.pnode.done {
  border-color: rgba(120, 255, 180, 0.45);
}

.pnode.fail {
  border-color: rgba(255, 120, 120, 0.55);
  box-shadow: 0 0 0 2px rgba(255, 120, 120, 0.12) inset;
}

.pipe {
  height: 10px;
  min-width: 64px;
  flex: 1;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.10);
  background: rgba(255, 255, 255, 0.06);
}

.pipe.done {
  background: rgba(120, 255, 180, 0.25);
  border-color: rgba(120, 255, 180, 0.35);
}

.pipe.fail {
  background: rgba(255, 120, 120, 0.25);
  border-color: rgba(255, 120, 120, 0.35);
}

.pipe.active {
  background: linear-gradient(
    90deg,
    rgba(120, 190, 255, 0.15) 0%,
    rgba(120, 190, 255, 0.45) 25%,
    rgba(255, 255, 255, 0.10) 50%,
    rgba(120, 190, 255, 0.45) 75%,
    rgba(120, 190, 255, 0.15) 100%
  );
  background-size: 200% 100%;
  animation: pipeflow 1.2s linear infinite;
  border-color: rgba(120, 190, 255, 0.40);
}

@keyframes pipeflow {
  0% {
    background-position: 0% 50%;
  }
  100% {
    background-position: 200% 50%;
  }
}

.pfoot {
  margin-top: 10px;
  font-size: 12px;
  opacity: 0.8;
  display: flex;
  justify-content: space-between;
  gap: 10px;
  flex-wrap: wrap;
}

.badge {
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: rgba(255, 255, 255, 0.06);
}

</style>
"""


def _status_text(n: PipeNode) -> str:
    """返回节点状态的友好文本"""
    if n.status == "pending":
        return "⏳ 等待"
    if n.status == "running":
        return " 运行中"
    if n.status == "done":
        if n.seconds is None:
            return " 完成"
        return f" 完成 · {n.seconds:.2f}s"
    if n.status == "fail":
        return " 失败"
    return n.status


def render_pipeline(
    nodes: List[PipeNode],
    title: str = "计算流程管线",
    expanded: bool = True
) -> None:
    """
    渲染流动管线 UI。
    
    Args:
        nodes: PipeNode 列表
        title: expander 标题
        expanded: 是否默认展开
    """
    with st.expander(title, expanded=expanded):
        st.markdown(CSS, unsafe_allow_html=True)

        # 构建行 HTML
        parts = ['<div class="pipeline-wrap"><div class="pipeline-row">']

        for i, n in enumerate(nodes):
            cls = f"pnode {n.status}"
            detail = (n.detail or "").replace("<", "&lt;").replace(">", "&gt;")

            parts.append(f'''
              <div class="{cls}">
                <div class="t">{n.label}</div>
                <div class="s">{_status_text(n)}</div>
                <div class="d">{detail}</div>
              </div>
            ''')

            # 在节点间插入管道
            if i < len(nodes) - 1:
                # 管道状态取决于当前节点和下一个节点
                pipe_cls = "pipe"
                if n.status == "running":
                    pipe_cls += " active"
                elif n.status == "fail":
                    pipe_cls += " fail"
                elif n.status == "done" and nodes[i + 1].status in ("done", "running", "pending"):
                    pipe_cls += " done"

                parts.append(f'<div class="{pipe_cls}"></div>')

        parts.append('</div>')

        # 底部统计信息
        done = sum(1 for n in nodes if n.status == "done")
        fail = sum(1 for n in nodes if n.status == "fail")
        total_seconds = sum(n.seconds or 0 for n in nodes if n.seconds is not None)

        parts.append(f'''
          <div class="pfoot">
            <div class="badge">已完成 {done}/{len(nodes)}</div>
            <div class="badge">{'失败 ' + str(fail) + ' 项' if fail else '无失败'}</div>
            <div class="badge">总耗时 {total_seconds:.2f}s</div>
          </div>
        ''')

        parts.append('</div>')

        st.markdown("".join(parts), unsafe_allow_html=True)


