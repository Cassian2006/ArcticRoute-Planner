# -*- coding: utf-8 -*-
"""
AIS Density 选择/匹配 UI 面板组件

提供：
  - 扫描候选文件
  - 选择对齐方法
  - 显示候选列表
  - 提示重采样信息
"""

from pathlib import Path
from typing import Optional, List, Dict, Tuple
import numpy as np
import streamlit as st

from arcticroute.core.ais_density_select import (
    scan_ais_density_candidates,
    select_best_candidate,
    load_and_align_density,
    AISDensityCandidate,
)
from arcticroute.core.grid import Grid2D


def render_ais_density_panel(
    grid: Optional[Grid2D] = None,
    grid_signature: Optional[str] = None,
    ais_weights_enabled: bool = True,
) -> Tuple[Optional[str], Optional[np.ndarray], Dict]:
    """
    渲染 AIS density 选择/匹配面板
    
    Args:
        grid: 目标网格对象（用于对齐）
        grid_signature: 网格签名（用于候选匹配）
        ais_weights_enabled: AIS 权重是否启用
    
    Returns:
        (ais_density_path, ais_density_array, metadata)
        其中 ais_density_array 可能为 None（未加载或禁用）
    """
    
    ais_density_path = None
    ais_density_array = None
    metadata = {}
    
    with st.expander("[object Object]选择/匹配", expanded=False):
        
        if not ais_weights_enabled:
            st.info("⚠️ AIS 权重未启用（所有权重为 0），跳过密度加载")
            return None, None, {}
        
        # ====================================================================
        # 1. 扫描参数
        # ====================================================================
        st.subheader("扫描参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_dirs_input = st.text_input(
                "扫描目录（逗号分隔，可选）",
                value="data_real/ais/density,data_real/ais/derived",
                help="留空使用默认目录；多个目录用逗号分隔",
            )
        
        with col2:
            align_method = st.selectbox(
                "对齐方法",
                options=["linear", "nearest"],
                index=0,
                help="linear: 双线性插值（平滑）；nearest: 最近邻（快速）",
            )
        
        auto_select = st.checkbox(
            "未指定文件时自动选择最佳匹配",
            value=True,
            help="若勾选，会自动选择与当前网格最匹配的文件；否则禁用 AIS",
        )
        
        # ====================================================================
        # 2. 扫描按钮
        # ====================================================================
        st.subheader("候选文件")
        
        col_scan, col_clear = st.columns(2)
        
        with col_scan:
            do_scan = st.button("🔎 扫描候选", use_container_width=True)
        
        with col_clear:
            do_clear = st.button("🗑️ 清除缓存", use_container_width=True)
        
        if do_clear:
            st.session_state.pop("ais_candidates_cache", None)
            st.session_state.pop("ais_density_path_selected", None)
            st.success("缓存已清除")
        
        # ====================================================================
        # 3. 执行扫描
        # ====================================================================
        candidates = []
        
        if do_scan:
            # 解析扫描目录
            search_dirs = None
            if search_dirs_input.strip():
                search_dirs = [d.strip() for d in search_dirs_input.split(",")]
            
            try:
                with st.spinner("正在扫描候选文件..."):
                    candidates = scan_ais_density_candidates(search_dirs=search_dirs)
                
                st.session_state["ais_candidates_cache"] = [
                    {
                        "path": c.path,
                        "grid_signature": c.grid_signature,
                        "shape": c.shape,
                        "varname": c.varname,
                        "note": c.note,
                        "match_type": c.match_type,
                    }
                    for c in candidates
                ]
                
                if candidates:
                    st.success(f"✅ 找到 {len(candidates)} 个候选文件")
                else:
                    st.warning("⚠️ 未找到候选文件")
            
            except Exception as e:
                st.error(f"❌ 扫描失败: {e}")
                candidates = []
        
        else:
            # 使用缓存
            candidates_cache = st.session_state.get("ais_candidates_cache", [])
            if candidates_cache:
                candidates = [
                    AISDensityCandidate(
                        path=c["path"],
                        grid_signature=c.get("grid_signature"),
                        shape=c.get("shape"),
                        varname=c.get("varname"),
                        note=c.get("note", ""),
                        match_type=c.get("match_type", "generic"),
                    )
                    for c in candidates_cache
                ]
        
        # ====================================================================
        # 4. 显示候选列表
        # ====================================================================
        if candidates:
            st.subheader(f"候选列表（共 {len(candidates)} 个）")
            
            # 构建表格数据
            table_data = []
            for i, cand in enumerate(candidates):
                table_data.append({
                    "序号": i + 1,
                    "文件名": Path(cand.path).name,
                    "网格签名": cand.grid_signature or "未知",
                    "形状": f"{cand.shape[0]}×{cand.shape[1]}" if cand.shape else "未知",
                    "变量名": cand.varname or "ais_density",
                    "类型": cand.match_type,
                    "备注": cand.note,
                })
            
            st.dataframe(table_data, use_container_width=True)
        
        # ====================================================================
        # 5. 选择文件
        # ====================================================================
        st.subheader("文件选择")
        
        col_explicit, col_auto = st.columns(2)
        
        with col_explicit:
            explicit_path = st.text_input(
                "显式指定文件路径（可选）",
                value="",
                help="若填写，将使用此文件；否则按下方选项处理",
            )
        
        with col_auto:
            if candidates:
                selected_idx = st.selectbox(
                    "或从候选中选择",
                    options=range(len(candidates)),
                    format_func=lambda i: f"{i+1}. {Path(candidates[i].path).name} ({candidates[i].match_type})",
                    key="ais_candidate_select",
                )
                auto_selected_path = candidates[selected_idx].path
            else:
                auto_selected_path = None
                st.info("无候选文件，请先扫描")
        
        # ====================================================================
        # 6. 确定最终路径
        # ====================================================================
        if explicit_path.strip():
            ais_density_path = explicit_path.strip()
            selection_source = "显式指定"
        elif auto_selected_path and auto_select:
            ais_density_path = auto_selected_path
            selection_source = "自动选择"
        elif auto_select and candidates:
            # 自动选择最佳匹配
            best = select_best_candidate(
                candidates=candidates,
                prefer_path=None,
                grid_signature=grid_signature,
            )
            if best:
                ais_density_path = best.path
                selection_source = f"自动最佳匹配 ({best.match_type})"
            else:
                ais_density_path = None
                selection_source = "无最佳匹配"
        else:
            ais_density_path = None
            selection_source = "未选择"
        
        # ====================================================================
        # 7. 加载并对齐
        # ====================================================================
        if ais_density_path:
            st.subheader("加载与对齐")
            
            try:
                if grid is None:
                    st.warning("⚠️ 网格未加载，无法对齐密度数据")
                    ais_density_array = None
                else:
                    with st.spinner(f"正在加载并对齐 {Path(ais_density_path).name}..."):
                        result = load_and_align_density(
                            density_path=ais_density_path,
                            grid=grid,
                            method=align_method,
                        )
                        
                        if result is not None:
                            ais_density_array, metadata = result
                            
                            # 显示加载信息
                            st.success("✅ 加载成功")
                            
                            col_info1, col_info2 = st.columns(2)
                            with col_info1:
                                st.metric("来源文件", Path(ais_density_path).name)
                                st.metric("对齐方法", align_method)
                            
                            with col_info2:
                                if metadata.get("resampled"):
                                    orig_shape = metadata.get("original_shape", "?")
                                    target_shape = metadata.get("target_shape", "?")
                                    st.metric("重采样", f"{orig_shape} → {target_shape}")
                                else:
                                    st.metric("重采样", "否")
                                
                                cache_status = metadata.get("cache_hit", False)
                                st.metric("缓存", "命中 ✓" if cache_status else "未命中")
                            
                            # 显示数据统计
                            st.info(
                                f"📊 数据统计：\n"
                                f"  - 形状: {ais_density_array.shape}\n"
                                f"  - 范围: [{np.nanmin(ais_density_array):.3f}, {np.nanmax(ais_density_array):.3f}]\n"
                                f"  - NaN 比例: {(np.isnan(ais_density_array).sum() / ais_density_array.size * 100):.1f}%"
                            )
                        
                        else:
                            st.error("❌ 加载失败")
                            ais_density_array = None
            
            except Exception as e:
                st.error(f"❌ 加载异常: {e}")
                ais_density_array = None
        
        else:
            if ais_weights_enabled:
                st.warning(
                    f"⚠️ 未选择 AIS 密度文件（{selection_source}）\n\n"
                    "将禁用 AIS 走廊/拥堵成本，仅使用其他风险因素"
                )
    
    return ais_density_path, ais_density_array, metadata


def render_ais_density_summary(
    ais_density_path: Optional[str],
    ais_density_array: Optional[np.ndarray],
    metadata: Dict,
) -> None:
    """
    在规划结果中显示 AIS 密度摘要
    
    Args:
        ais_density_path: 密度文件路径
        ais_density_array: 密度数组
        metadata: 元数据
    """
    
    if ais_density_path is None:
        return
    
    with st.expander("📊 AIS 密度信息", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("来源文件", Path(ais_density_path).name)
            if metadata.get("resampled"):
                st.metric("重采样", "是 ✓")
            else:
                st.metric("重采样", "否")
        
        with col2:
            if metadata.get("cache_hit"):
                st.metric("缓存状态", "命中 ✓")
            else:
                st.metric("缓存状态", "未命中")
            
            if ais_density_array is not None:
                st.metric("数据点数", f"{ais_density_array.size:,}")
        
        if ais_density_array is not None:
            st.info(
                f"📈 统计信息：\n"
                f"  - 形状: {ais_density_array.shape}\n"
                f"  - 最小值: {np.nanmin(ais_density_array):.4f}\n"
                f"  - 最大值: {np.nanmax(ais_density_array):.4f}\n"
                f"  - 平均值: {np.nanmean(ais_density_array):.4f}\n"
                f"  - NaN 数量: {np.isnan(ais_density_array).sum()}"
            )


