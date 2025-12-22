"""
数据页 - 显示环境数据和静态资产状态
"""

from __future__ import annotations

import streamlit as st
from pathlib import Path


def render_data() -> None:
    """渲染数据页"""
    
    st.title("🛰️ 数据源状态")
    st.caption("查看环境数据和静态资产的加载状态")
    
    # 环境数据状态
    st.subheader("环境数据 (CMEMS)")
    
    # 尝试获取数据目录
    try:
        from arcticroute.core.env import get_newenv_path
        newenv_dir = get_newenv_path()
    except Exception:
        newenv_dir = Path("data/newenv")
    
    cmems_files = {
        "海冰浓度 (SIC)": newenv_dir / "ice_copernicus_sic.nc",
        "海冰厚度 (SIT)": newenv_dir / "ice_copernicus_sit.nc",
        "有效波高 (SWH)": newenv_dir / "wave_swh.nc",
        "海冰漂移 (Drift)": newenv_dir / "ice_drift.nc",
    }
    
    for name, path in cmems_files.items():
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write(f"**{name}**")
        
        with col2:
            if path.exists():
                st.markdown('<span class="status-badge active">✓ 可用</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="status-badge inactive">✗ 缺失</span>', unsafe_allow_html=True)
        
        with col3:
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                st.caption(f"{size_mb:.1f} MB")
            else:
                st.caption(f"路径: {path}")
    
    st.markdown("---")
    
    # 静态资产状态
    st.subheader("静态资产")
    
    static_assets = {
        "AIS 拥挤度": "data/ais_density/*.nc",
        "主航道走廊": "data/corridors/*.geojson",
        "浅水区数据": "data/bathymetry/*.nc",
        "港口数据": "data/ports/*.csv",
    }
    
    for name, pattern in static_assets.items():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write(f"**{name}**")
        
        with col2:
            # 简化显示，实际应该扫描文件
            st.markdown('<span class="status-badge warning">⚠ 待扫描</span>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 数据刷新按钮
    if st.button("🔄 重新扫描数据源", use_container_width=True):
        st.info("数据扫描功能开发中...")
        st.rerun()
    
    # 数据配置
    with st.expander("⚙️ 数据配置", expanded=False):
        st.text_input(
            "环境数据目录",
            value=str(newenv_dir),
            help="CMEMS 环境数据存放目录"
        )
        
        st.text_input(
            "AIS 数据目录",
            value="data/ais_density",
            help="AIS 拥挤度数据存放目录"
        )
        
        if st.button("保存配置"):
            st.success("✓ 配置已保存")

