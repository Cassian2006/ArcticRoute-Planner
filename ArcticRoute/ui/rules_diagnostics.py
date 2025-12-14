"""
极地规则诊断 UI 组件

显示启用的规则、禁行格点、命中率统计等。
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, Optional


def render_rules_diagnostics(rules_meta: Optional[Dict[str, Any]]) -> None:
    """
    渲染规则诊断区。
    
    Args:
        rules_meta: 规则应用元数据（来自 cost_field.meta["rules"]）
    """
    if rules_meta is None or not rules_meta:
        st.info("未启用极地规则")
        return
    
    # 检查是否有错误
    if "error" in rules_meta:
        st.error(f"规则应用出错：{rules_meta['error']}")
        return
    
    # 显示规则启用状态
    st.markdown("#### 🔧 极地规则诊断")
    
    rules_enabled = rules_meta.get("rules_enabled", False)
    if not rules_enabled:
        st.info("极地规则已禁用")
        return
    
    # 显示应用的规则列表
    rules_applied = rules_meta.get("rules_applied", [])
    if rules_applied:
        st.markdown("**启用的规则：**")
        for rule in rules_applied:
            st.caption(f"✅ {rule}")
    else:
        st.info("未应用任何规则（可能所有阈值都缺失）")
    
    # 显示警告
    warnings = rules_meta.get("warnings", [])
    if warnings:
        st.markdown("**⚠️ 警告：**")
        for warning in warnings:
            st.warning(warning, icon="⚠️")
    
    # 显示禁行统计
    blocked_count = rules_meta.get("blocked_count", 0)
    total_cells = rules_meta.get("total_cells", 0)
    blocked_fraction = rules_meta.get("blocked_fraction", 0.0)
    
    if total_cells > 0:
        st.markdown("**禁行统计：**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("禁行格点数", f"{blocked_count:,}")
        
        with col2:
            st.metric("总格点数", f"{total_cells:,}")
        
        with col3:
            st.metric("禁行比例", f"{blocked_fraction:.2%}")
    
    # 显示各规则的命中数量
    st.markdown("**规则命中统计：**")
    
    rule_hits = {
        "wave": rules_meta.get("wave_blocked_count", 0),
        "sic": rules_meta.get("sic_blocked_count", 0),
        "ice_thickness": rules_meta.get("thickness_blocked_count", 0),
    }
    
    # 过滤掉为 0 的规则
    active_rules = {k: v for k, v in rule_hits.items() if v > 0}
    
    if active_rules:
        for rule_name, hit_count in active_rules.items():
            hit_fraction = hit_count / total_cells if total_cells > 0 else 0.0
            st.caption(f"  • {rule_name}: {hit_count:,} 格点 ({hit_fraction:.2%})")
    else:
        st.caption("  • 无规则命中（所有格点都在阈值范围内）")
    
    # 显示详细信息（可展开）
    with st.expander("详细信息"):
        st.json(rules_meta)


def render_rules_config_input() -> Optional[str]:
    """
    渲染规则配置文件路径输入框。
    
    Returns:
        规则配置文件路径（可能为 None）
    """
    rules_config_path = st.text_input(
        "极地规则配置文件路径 (可选)",
        value=st.session_state.get("rules_config_path", ""),
        placeholder="例: arcticroute/config/polar_rules.yaml",
        help="若指定，将应用硬约束禁行 mask；否则不启用规则。",
    )
    st.session_state["rules_config_path"] = rules_config_path
    
    return rules_config_path if rules_config_path else None

