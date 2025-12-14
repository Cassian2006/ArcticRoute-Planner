from __future__ import annotations
from typing import Any, Dict, Optional
import streamlit as st


def show_error_card(error_code: str, message: str, hint: str, details: Optional[Dict[str, Any]] = None) -> None:
    """统一错误样式卡片。

    参数:
    - error_code: 错误码（如 NO_RISK_DATA, NO_PARETO, NO_ROUTE_SELECTED, FEEDBACK_INVALID, HEALTH_FAIL 等）
    - message: 简短可读的错误描述
    - hint: 建议操作（可包含 CLI 示例或指引路径）
    - details: 可选的结构化信息（将以可折叠 JSON 展示）
    """
    cc = error_code.strip().upper()
    color = "#fdd" if cc not in ("WARN", "WARNING") else "#ffd"
    st.markdown(
        f"""
        <div style='border:1px solid #f99;background:{color};padding:12px;border-radius:8px;'>
            <div style='font-weight:600;'>错误码：<code>{cc}</code></div>
            <div style='margin-top:6px;'>{message}</div>
            <div style='margin-top:8px;color:#555;'>建议操作：{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if details:
        with st.expander("展开查看详情 Details"):
            try:
                st.json(details)
            except Exception:
                st.write(details)

