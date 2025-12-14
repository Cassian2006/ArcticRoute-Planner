from __future__ import annotations
import streamlit as st
from typing import Dict, List, Optional

"""
通用任务面板组件（最小版）
- 用法：在页面或模块中组织后台任务（由调用方负责创建与轮询），将快照列表传入此组件统一渲染。
- 兼容现有 app_min 的字段命名（id/status/command/stdout/stderr/...）。
- 后续可扩展：自动轮询、取消、下载工件等。
"""


def render_task_panel(title: str, task_snapshots: List[Dict], *, panel_id: str = "panel") -> None:
    st.subheader(title)
    if not task_snapshots:
        st.info("尚无后台任务。")
        return

    status_order = ("queued", "running", "finished", "failed")
    cols = st.columns(len(status_order))
    groups = {k: [] for k in status_order}
    for snap in task_snapshots:
        groups.setdefault(snap.get("status", "unknown"), []).append(snap)
    for i, s in enumerate(status_order):
        cols[i].metric(s.capitalize(), len(groups.get(s, [])))

    options = [snap.get("id", "-") for snap in task_snapshots]
    selected = st.selectbox(
        "选择任务以查看详情",
        options,
        index=0,
        format_func=lambda tid: next((f"{s.get('id')} · {s.get('start_time')}" for s in task_snapshots if s.get("id") == tid), tid),
        key=f"select-{panel_id}"
    )
    if not selected:
        return
    info = next((s for s in task_snapshots if s.get("id") == selected), None)
    if not info:
        st.warning("任务不存在或已被移除。")
        return

    left, right = st.columns([2, 1])
    with left:
        st.write(f"**状态：** {info.get('status')}")
        st.write(f"**开始时间：** {info.get('start_time') or '—'}")
        st.write(f"**结束时间：** {info.get('end_time') or '—'}")
        st.write(f"**返回码：** {info.get('returncode') if info.get('returncode') is not None else '—'}")
        if info.get("error"):
            st.error(info.get("error"))
        cmd = info.get("command") or []
        if cmd:
            st.code(" ".join(cmd), language="bash")
    with right:
        st.text_area("stdout", info.get("stdout", ""), height=180, key=f"stdout-{panel_id}-{selected}")
        st.text_area("stderr", info.get("stderr", ""), height=120, key=f"stderr-{panel_id}-{selected}")
