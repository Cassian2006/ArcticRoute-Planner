# -*- coding: utf-8 -*-
"""
侧边栏配置模块 - Phase UI-1
将侧边栏组织为四大区块：数据源/约束/成本组件/规划器
"""

from __future__ import annotations
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional
from arcticroute.core.cost import discover_ais_density_candidates, compute_grid_signature, has_ais_density_data
from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.config import EDL_MODES, list_edl_modes
from arcticroute.core.scenarios import load_all_scenarios


def render_data_source_section() -> Dict[str, Any]:
    """
    渲染数据源区块
    返回: {
        'env_source': 'demo' | 'cmems_latest' | 'manual_nc',
        'grid_mode': 'demo' | 'real',
        'cost_mode': str,
        'ym': str,
        'grid_signature': str | None,
    }
    """
    st.subheader("📊 数据源配置")
    
    config = {}
    
    # 环境数据源选择
    env_source_options = ["demo", "cmems_latest", "manual_nc"]
    env_source = st.selectbox(
        "环境数据源",
        options=env_source_options,
        index=0,
        format_func=lambda x: {
            "demo": "演示数据 (内置)",
            "cmems_latest": "CMEMS 最新数据",
            "manual_nc": "手动指定 NC 文件"
        }.get(x, x),
        help="选择环境数据的来源"
    )
    config['env_source'] = env_source
    
    # CMEMS 数据层开关
    if env_source == "cmems_latest":
        st.markdown("**CMEMS 数据层**")
        col1, col2 = st.columns(2)
        with col1:
            enable_sic = st.checkbox("海冰浓度 (SIC)", value=True, key="enable_sic")
            enable_swh = st.checkbox("有效波高 (SWH)", value=True, key="enable_swh")
        with col2:
            enable_sit = st.checkbox("海冰厚度 (SIT)", value=False, key="enable_sit")
            enable_drift = st.checkbox("海冰漂移 (Drift)", value=False, key="enable_drift")
        
        config['cmems_layers'] = {
            'enable_sic': enable_sic,
            'enable_swh': enable_swh,
            'enable_sit': enable_sit,
            'enable_drift': enable_drift,
        }
    
    # newenv sync 功能
    with st.expander("🔄 NewEnv 同步", expanded=False):
        if st.button("一键同步到 newenv"):
            st.info("同步功能将在后续实现")
        st.caption("显示 newenv_index.json 状态")
    
    # 网格模式
    grid_mode_options = ["demo", "real"]
    grid_mode = st.radio(
        "网格模式",
        options=grid_mode_options,
        index=0,
        format_func=lambda s: "演示网格" if s == "demo" else "真实网格",
        horizontal=True,
    )
    config['grid_mode'] = grid_mode
    
    # 计算网格签名
    try:
        if grid_mode == "demo":
            current_grid, _ = make_demo_grid()
        else:
            ym = st.session_state.get("ym", "202401")
            current_grid = load_real_grid_from_nc(ym=ym)
        
        grid_signature = compute_grid_signature(current_grid)
        config['grid_signature'] = grid_signature
        # 注意：grid_signature 不是 widget，可以直接设置
        if 'grid_signature' not in st.session_state or st.session_state['grid_signature'] != grid_signature:
            st.session_state['grid_signature'] = grid_signature
    except Exception as e:
        config['grid_signature'] = None
        if 'grid_signature' not in st.session_state:
            st.session_state['grid_signature'] = None
    
    # 成本模式
    cost_mode_options = ["demo_icebelt", "real_sic_if_available"]
    cost_mode = st.selectbox(
        "成本模式",
        options=cost_mode_options,
        index=1 if grid_mode == "real" else 0,
        format_func=lambda s: "演示冰带" if s == "demo_icebelt" else "真实 SIC/波浪",
    )
    config['cost_mode'] = cost_mode
    
    return config


def render_constraints_section() -> Dict[str, Any]:
    """
    渲染约束区块
    返回: {
        'polaris_enabled': bool,
        'use_decayed_table': bool,
        'hard_block_level': int,
        'elevated_penalty_scale': float,
        'shallow_enabled': bool,
        'min_depth_m': float,
        'w_shallow': float,
    }
    """
    st.subheader("⚠️ 约束配置")
    
    config = {}
    
    # POLARIS 冰级约束
    with st.expander("🧊 POLARIS 冰级约束", expanded=False):
        # 从 session_state 获取默认值，避免重复设置
        default_polaris_enabled = st.session_state.get('polaris_enabled', False)
        polaris_enabled = st.checkbox("启用 POLARIS", value=default_polaris_enabled, key="polaris_enabled")
        config['polaris_enabled'] = polaris_enabled
        
        if polaris_enabled:
            default_use_decayed = st.session_state.get('use_decayed_table', False)
            default_hard_block = st.session_state.get('hard_block_level', 3)
            default_elevated = st.session_state.get('elevated_penalty_scale', 2.0)
            
            use_decayed_table = st.checkbox("使用衰减表", value=default_use_decayed, key="use_decayed_table")
            hard_block_level = st.slider("硬禁区等级", 0, 5, default_hard_block, key="hard_block_level")
            elevated_penalty_scale = st.slider("提升惩罚系数", 0.0, 10.0, default_elevated, 0.5, key="elevated_penalty_scale")
            
            config['use_decayed_table'] = use_decayed_table
            config['hard_block_level'] = hard_block_level
            config['elevated_penalty_scale'] = elevated_penalty_scale
    
    # 浅水约束
    with st.expander("🌊 浅水约束", expanded=False):
        default_shallow_enabled = st.session_state.get('shallow_enabled', False)
        shallow_enabled = st.checkbox("启用浅水约束", value=default_shallow_enabled, key="shallow_enabled")
        config['shallow_enabled'] = shallow_enabled
        
        if shallow_enabled:
            default_min_depth = st.session_state.get('min_depth_m', 10.0)
            default_w_shallow = st.session_state.get('w_shallow', 2.0)
            
            min_depth_m = st.number_input("最小水深 (m)", 0.0, 100.0, default_min_depth, 1.0, key="min_depth_m")
            w_shallow = st.slider("浅水惩罚权重", 0.0, 10.0, default_w_shallow, 0.5, key="w_shallow")
            
            config['min_depth_m'] = min_depth_m
            config['w_shallow'] = w_shallow
            
            st.caption("需要 bathymetry 数据可用")
    
    return config


def render_cost_components_section() -> Dict[str, Any]:
    """
    渲染成本组件区块
    返回: {
        'w_ais_corridor': float,
        'w_ais_congestion': float,
        'w_ais': float,
        'ais_density_path': Path | None,
        'wave_penalty': float,
        'w_edl': float,
        'edl_uncertainty_weight': float,
    }
    """
    st.subheader("💰 成本组件")
    
    config = {}
    
    # AIS 密度成本
    with st.expander("🚢 AIS 密度成本", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            w_ais_corridor = st.slider(
                "主航线偏好",
                0.0, 10.0,
                float(st.session_state.get("w_ais_corridor", 2.0)),
                0.5,
                key="w_ais_corridor_slider",
                help="越高越倾向于沿历史航道"
            )
        with col2:
            w_ais_congestion = st.slider(
                "拥挤惩罚",
                0.0, 10.0,
                float(st.session_state.get("w_ais_congestion", 1.0)),
                0.5,
                key="w_ais_congestion_slider",
                help="惩罚极端拥挤区域"
            )
        
        config['w_ais_corridor'] = w_ais_corridor
        config['w_ais_congestion'] = w_ais_congestion
        
        # 旧版兼容
        w_ais = st.slider(
            "旧版权重 (deprecated)",
            0.0, 10.0,
            float(st.session_state.get("w_ais", 0.0)),
            0.1,
            key="w_ais_slider"
        )
        config['w_ais'] = w_ais
        
        # AIS 密度文件选择
        grid_sig = st.session_state.get("grid_signature")
        ais_candidates = discover_ais_density_candidates(grid_signature=grid_sig)
        
        ais_options = ["自动选择"]
        ais_path_map = {"自动选择": None}
        
        for cand in ais_candidates:
            label = cand["label"]
            match_type = cand.get("match_type", "generic")
            
            if match_type == "exact":
                label_with_type = f"{label} ✓"
            elif match_type == "demo":
                label_with_type = f"{label} (demo)"
            else:
                label_with_type = label
            
            ais_options.append(label_with_type)
            ais_path_map[label_with_type] = cand["path"]
        
        ais_choice = st.selectbox(
            "AIS 密度文件",
            options=ais_options,
            key="ais_density_selector"
        )
        
        ais_density_path = ais_path_map.get(ais_choice)
        config['ais_density_path'] = ais_density_path
        # ais_density_path 不是 widget key，可以安全设置
        if 'ais_density_path' not in st.session_state or st.session_state['ais_density_path'] != ais_density_path:
            st.session_state['ais_density_path'] = ais_density_path
        
        # 显示当前选中的文件和 grid_signature
        if ais_density_path:
            st.caption(f"📁 {Path(ais_density_path).name}")
        if grid_sig:
            st.caption(f"🔖 Grid: {grid_sig[:30]}...")
    
    # 波浪成本
    wave_penalty = st.slider(
        "🌊 波浪权重",
        0.0, 10.0,
        float(st.session_state.get("wave_penalty", 2.0)),
        0.5,
        key="wave_penalty_slider"
    )
    config['wave_penalty'] = wave_penalty
    
    # EDL 成本
    with st.expander("🧠 EDL 风险成本", expanded=False):
        w_edl = st.slider(
            "EDL 权重",
            0.0, 10.0,
            float(st.session_state.get("w_edl", 0.0)),
            0.5,
            key="w_edl_slider"
        )
        edl_uncertainty_weight = st.slider(
            "不确定性权重",
            0.0, 10.0,
            float(st.session_state.get("edl_uncertainty_weight", 0.0)),
            0.5,
            key="edl_uncertainty_weight_slider"
        )
        
        config['w_edl'] = w_edl
        config['edl_uncertainty_weight'] = edl_uncertainty_weight
    
    return config


def render_planner_backend_section() -> Dict[str, Any]:
    """
    渲染规划器后端区块
    返回: {
        'planner_backend': 'auto' | 'astar' | 'polarroute_pipeline' | 'polarroute_external',
    }
    """
    st.subheader("🎯 规划器后端")
    
    config = {}
    
    planner_options = ["auto", "astar", "polarroute_pipeline", "polarroute_external"]
    planner_backend = st.selectbox(
        "选择规划器",
        options=planner_options,
        index=0,
        format_func=lambda x: {
            "auto": "自动选择",
            "astar": "A* (内置)",
            "polarroute_pipeline": "PolarRoute Pipeline",
            "polarroute_external": "PolarRoute External"
        }.get(x, x),
        key="planner_backend_selector"
    )
    
    config['planner_backend'] = planner_backend
    
    st.caption(f"当前: {planner_backend}")
    
    return config


def render_sidebar_unified() -> Dict[str, Any]:
    """
    统一渲染侧边栏四大区块
    返回所有配置的字典
    """
    with st.sidebar:
        st.header("🎛️ 规划参数配置")
        
        # 区块 1: 数据源
        data_source_config = render_data_source_section()
        st.markdown("---")
        
        # 区块 2: 约束
        constraints_config = render_constraints_section()
        st.markdown("---")
        
        # 区块 3: 成本组件
        cost_components_config = render_cost_components_section()
        st.markdown("---")
        
        # 区块 4: 规划器
        planner_config = render_planner_backend_section()
        
        # 合并所有配置
        all_config = {
            **data_source_config,
            **constraints_config,
            **cost_components_config,
            **planner_config,
        }
        
        return all_config


def render_run_summary_panel(cost_meta: Dict[str, Any], cost_breakdown: Optional[Any] = None) -> None:
    """
    渲染运行摘要面板
    
    Args:
        cost_meta: 成本元数据字典
        cost_breakdown: 成本分解对象 (可选)
    """
    with st.expander("📋 运行摘要面板", expanded=False):
        st.markdown("### 数据层状态")
        
        # 显示已加载的数据层
        layers_status = []
        for layer_name in ['sic', 'swh', 'sit', 'drift', 'ais_density', 'bathymetry']:
            loaded = cost_meta.get(f'{layer_name}_loaded', False)
            status_icon = "✅" if loaded else "❌"
            layers_status.append(f"{status_icon} {layer_name}")
        
        cols = st.columns(3)
        for i, status in enumerate(layers_status):
            cols[i % 3].markdown(status)
        
        # Fallback 原因
        st.markdown("### Fallback 信息")
        fallback_reason = cost_meta.get('fallback_reason', 'None')
        if fallback_reason and fallback_reason != 'None':
            st.warning(f"⚠️ {fallback_reason}")
        else:
            st.success("✅ 无 fallback")
        
        # 规划器使用
        st.markdown("### 规划器信息")
        planner_used = cost_meta.get('planner_used', 'unknown')
        polaris_enabled = cost_meta.get('polaris_enabled', False)
        st.info(f"规划器: {planner_used} | POLARIS: {'启用' if polaris_enabled else '禁用'}")
        
        # 成本组件统计
        if cost_breakdown and hasattr(cost_breakdown, 'component_totals'):
            st.markdown("### 成本组件统计")
            
            component_stats = []
            for key, value in cost_breakdown.component_totals.items():
                if value is not None:
                    is_zero = abs(value) < 1e-6
                    component_stats.append({
                        '组件': key,
                        '总值': f"{value:.4f}",
                        '均值': f"{value / len(cost_breakdown.s_km) if cost_breakdown.s_km else 0:.4f}",
                        '全零': '是' if is_zero else '否'
                    })
            
            if component_stats:
                df_stats = pd.DataFrame(component_stats)
                st.dataframe(df_stats, use_container_width=True)
        
        # 下载按钮
        st.markdown("### 导出数据")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 cost_breakdown.json"):
                import json
                json_data = json.dumps(cost_meta, indent=2, ensure_ascii=False)
                st.download_button(
                    "下载 JSON",
                    data=json_data,
                    file_name="cost_breakdown.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📥 summary.txt"):
                summary_text = f"""运行摘要
================
数据层: {', '.join([k for k, v in cost_meta.items() if k.endswith('_loaded') and v])}
Fallback: {fallback_reason}
规划器: {planner_used}
POLARIS: {'启用' if polaris_enabled else '禁用'}
"""
                st.download_button(
                    "下载 TXT",
                    data=summary_text,
                    file_name="summary.txt",
                    mime="text/plain"
                )
        
        with col3:
            # polaris_diagnostics.csv 如果存在
            polaris_diag_path = Path("reports/polaris_diagnostics.csv")
            if polaris_diag_path.exists():
                if st.button("📥 polaris_diagnostics.csv"):
                    with open(polaris_diag_path, 'rb') as f:
                        st.download_button(
                            "下载 CSV",
                            data=f.read(),
                            file_name="polaris_diagnostics.csv",
                            mime="text/csv"
                        )
            else:
                st.caption("polaris_diagnostics.csv 不存在")

