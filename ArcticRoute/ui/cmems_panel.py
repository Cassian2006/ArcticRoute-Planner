#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMEMS 数据面板组件

提供 UI 控件用于：
1. 选择环境数据源（real_archive / cmems_latest / manual_nc）
2. 刷新 CMEMS 最新数据
3. 显示刷新状态和元数据
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Literal
from datetime import datetime

import streamlit as st


def render_env_source_selector() -> Literal["real_archive", "cmems_latest", "manual_nc"]:
    """
    渲染环境数据源选择器
    
    Returns:
        选择的数据源: "real_archive" | "cmems_latest" | "manual_nc"
    """
    st.subheader("📊 环境数据源")
    
    env_source_options = [
        ("real_archive", "真实归档数据 (real_archive)"),
        ("cmems_latest", "CMEMS 近实时数据 (cmems_latest)"),
        ("manual_nc", "手动指定 NC 文件 (manual_nc)"),
    ]
    
    default_source = st.session_state.get("env_source", "real_archive")
    
    selected_source = st.radio(
        "选择数据源",
        options=[opt[0] for opt in env_source_options],
        format_func=lambda x: next(opt[1] for opt in env_source_options if opt[0] == x),
        index=[opt[0] for opt in env_source_options].index(default_source),
        horizontal=False,
    )
    
    st.session_state["env_source"] = selected_source
    
    return selected_source


def render_cmems_panel() -> None:
    """
    渲染 CMEMS 数据刷新面板
    
    包含：
    - 刷新按钮
    - 刷新状态显示
    - 最后刷新记录
    """
    st.subheader("🔄 CMEMS 数据刷新")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        refresh_now = st.button(
            "🔄 立即刷新 CMEMS 数据",
            help="运行 cmems_refresh_and_export.py 下载最新的 SIC 和 SWH 数据",
        )
    
    with col2:
        refresh_days = st.number_input(
            "回溯天数",
            min_value=1,
            max_value=30,
            value=2,
            help="下载最近 N 天的数据",
        )
    
    if refresh_now:
        with st.spinner("正在刷新 CMEMS 数据..."):
            try:
                result = subprocess.run(
                    [
                        "python",
                        "-m",
                        "scripts.cmems_refresh_and_export",
                        "--days",
                        str(refresh_days),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=Path.cwd(),
                )
                
                if result.returncode == 0:
                    st.success("✅ CMEMS 数据刷新成功！")
                    st.session_state["cmems_refresh_time"] = datetime.utcnow().isoformat()
                else:
                    st.error(f"❌ 刷新失败:\n{result.stderr}")
            except subprocess.TimeoutExpired:
                st.error("❌ 刷新超时（>5分钟）")
            except Exception as e:
                st.error(f"❌ 刷新异常: {e}")
    
    # 显示最后刷新记录
    st.markdown("---")
    st.subheader("📋 最后刷新记录")
    
    refresh_record_path = Path("reports/cmems_refresh_last.json")
    if refresh_record_path.exists():
        try:
            with open(refresh_record_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            
            # 显示时间窗
            st.write(f"**时间范围**: {record.get('start_date')} 至 {record.get('end_date')}")
            
            # 显示 bbox
            bbox = record.get("bbox", {})
            st.write(
                f"**地理范围**: "
                f"[{bbox.get('min_lon', '?')}, {bbox.get('max_lon', '?')}] × "
                f"[{bbox.get('min_lat', '?')}, {bbox.get('max_lat', '?')}]"
            )
            
            # 显示下载结果
            downloads = record.get("downloads", {})
            
            col_sic, col_swh = st.columns(2)
            
            with col_sic:
                sic_info = downloads.get("sic", {})
                if sic_info.get("success"):
                    st.success(f"✅ SIC: {sic_info.get('filename', '?')}")
                    st.caption(f"变量: {sic_info.get('variable', '?')}")
                else:
                    st.error(f"❌ SIC: {sic_info.get('error', '未知错误')}")
            
            with col_swh:
                swh_info = downloads.get("swh", {})
                if swh_info.get("success"):
                    st.success(f"✅ SWH: {swh_info.get('filename', '?')}")
                    st.caption(f"变量: {swh_info.get('variable', '?')}")
                else:
                    st.error(f"❌ SWH: {swh_info.get('error', '未知错误')}")
            
            st.caption(f"刷新时间: {record.get('timestamp', '?')}")
        
        except Exception as e:
            st.warning(f"⚠️ 无法读取刷新记录: {e}")
    else:
        st.info("📌 尚未刷新过 CMEMS 数据，请点击上方按钮进行刷新。")


def render_manual_nc_selector() -> Optional[str]:
    """
    渲染手动 NC 文件选择器
    
    Returns:
        选择的 NC 文件路径，或 None
    """
    st.subheader("📁 手动选择 NC 文件")
    
    nc_path = st.text_input(
        "NC 文件路径",
        value=st.session_state.get("manual_nc_path", ""),
        placeholder="例: data/cmems_cache/sic_20241215.nc",
        help="输入 SIC 或 SWH 的 NC 文件完整路径",
    )
    
    st.session_state["manual_nc_path"] = nc_path
    
    if nc_path:
        nc_file = Path(nc_path)
        if nc_file.exists():
            st.success(f"✅ 文件存在: {nc_file.stat().st_size / 1024 / 1024:.1f} MB")
            return nc_path
        else:
            st.error(f"❌ 文件不存在: {nc_path}")
            return None
    
    return None


def get_env_source_config() -> dict:
    """
    获取当前环境数据源的配置
    
    Returns:
        配置字典，包含:
        - source: 数据源类型
        - sic_path: SIC nc 文件路径（如果可用）
        - swh_path: SWH nc 文件路径（如果可用）
    """
    from scripts.cmems_utils import find_latest_nc
    
    source = st.session_state.get("env_source", "real_archive")
    config = {"source": source}
    
    if source == "cmems_latest":
        # 查找最新的 CMEMS 缓存文件
        sic_latest = find_latest_nc("data/cmems_cache", "sic")
        swh_latest = find_latest_nc("data/cmems_cache", "swh")
        
        if sic_latest:
            config["sic_path"] = str(sic_latest)
        if swh_latest:
            config["swh_path"] = str(swh_latest)
    
    elif source == "manual_nc":
        # 使用手动指定的路径
        manual_path = st.session_state.get("manual_nc_path")
        if manual_path:
            config["manual_path"] = manual_path
    
    return config


if __name__ == "__main__":
    # 测试
    st.set_page_config(page_title="CMEMS Panel Test", layout="wide")
    
    st.title("CMEMS 面板测试")
    
    source = render_env_source_selector()
    st.write(f"选择的数据源: {source}")
    
    if source == "cmems_latest":
        render_cmems_panel()
    elif source == "manual_nc":
        render_manual_nc_selector()
    
    config = get_env_source_config()
    st.json(config)

