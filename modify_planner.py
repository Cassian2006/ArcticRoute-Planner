#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改 planner_minimal.py 以实现三个任务：
A. 修正管线顺序与 AIS 状态
B. 删除简化版本管线（如果存在）
C. 改进 AIS 维度匹配处理
"""

import re

def main():
    # 读取原文件
    with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # ========================================================================
    # 任务 A：修正 AIS 状态处理 - 确保 AIS 完成时不停留在 pending
    # ========================================================================
    
    # 找到并替换 AIS 加载逻辑
    # 原始块从 "if w_ais > 0:" 开始（在主规划逻辑中）
    
    ais_old_block = '''        if w_ais > 0:
            pipeline.start('ais')
            try:
                from arcticroute.core import cost as cost_core
                import xarray as xr
                from pathlib import Path

                prefer_real = (grid_mode == "real")
                ais_density_path_obj = Path(ais_density_path) if ais_density_path is not None else None
                if ais_density_path_obj is not None and ais_density_path_obj.exists():
                    try:
                        with xr.open_dataset(ais_density_path_obj) as ds:
                            if "ais_density" in ds:
                                ais_da_loaded = ds["ais_density"].load()
                            elif ds.data_vars:
                                ais_da_loaded = list(ds.data_vars.values())[0].load()
                    except Exception as e:
                        ais_info["error"] = str(e)
                        st.warning(f"⚠ 加载选定的 AIS density 失败：{e}")

                if ais_da_loaded is None:
                    ais_da_loaded = cost_core.load_ais_density_for_grid(grid, prefer_real=prefer_real)

                if ais_da_loaded is not None:
                    ais_density = ais_da_loaded.values if hasattr(ais_da_loaded, "values") else np.asarray(ais_da_loaded)
                    ais_info.update({
                        "loaded": True,
                        "shape": ais_density.shape,
                    })
                    pipeline.done('ais', extra_info=f'candidates={len(ais_density.flat)}')
                    render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)
                    st.info(f"✓ 已加载 AIS 拥挤度密度数据，栅格={ais_info['shape']}")
                else:
                    ais_info["error"] = "未找到 AIS 密度 NC 文件"
                    st.warning("⚠ 当前未选择 AIS density 文件，AIS 拥挤度成本将被禁用。")
                    w_ais = 0.0
            except Exception as e:
                ais_info["error"] = str(e)
                st.warning(f"⚠ 加载 AIS 密度数据失败：{e}，AIS 拥挤度成本将被禁用")
                w_ais = 0.0'''
    
    ais_new_block = '''        # ====================================================================
        # 任务 A：AIS 密度加载与状态管理
        # 确保 AIS 步骤完成时不停留在 pending（成功加载或跳过都标记为 done）
        # ====================================================================
        if w_ais <= 0:
            # 权重为 0，直接标记 AIS 为 done（skip）
            _update_pipeline_node(3, "done", "跳过：权重为 0", seconds=0.1)
        else:
            # w_ais > 0，尝试加载 AIS 密度
            _update_pipeline_node(3, "running", "正在加载 AIS 密度...")
            
            try:
                from arcticroute.core import cost as cost_core
                import xarray as xr
                from pathlib import Path

                prefer_real = (grid_mode == "real")
                ais_density_path_obj = Path(ais_density_path) if ais_density_path is not None else None
                
                # 情况 1：用户未选择 AIS 文件
                if ais_density_path_obj is None:
                    _update_pipeline_node(3, "done", "跳过：未选择文件", seconds=0.1)
                    st.info("ℹ️ AIS 权重 > 0 但未选择文件，已跳过 AIS 密度加载")
                    w_ais = 0.0
                
                # 情况 2：文件存在，尝试加载
                elif ais_density_path_obj.exists():
                    try:
                        with xr.open_dataset(ais_density_path_obj) as ds:
                            if "ais_density" in ds:
                                ais_da_loaded = ds["ais_density"].load()
                            elif ds.data_vars:
                                ais_da_loaded = list(ds.data_vars.values())[0].load()
                            else:
                                ais_da_loaded = None
                        
                        if ais_da_loaded is not None:
                            ais_density = ais_da_loaded.values if hasattr(ais_da_loaded, "values") else np.asarray(ais_da_loaded)
                            ais_info.update({
                                "loaded": True,
                                "shape": ais_density.shape,
                            })
                            # 成功加载，标记为 done
                            _update_pipeline_node(3, "done", f"AIS={ais_density.shape[0]}×{ais_density.shape[1]} source={ais_density_path_obj.name}", seconds=0.3)
                            st.success(f"✅ 已加载 AIS 拥挤度密度数据，栅格={ais_info['shape']}")
                        else:
                            # 文件无效
                            _update_pipeline_node(3, "done", "跳过：文件格式无效", seconds=0.1)
                            st.warning("⚠️ AIS 密度文件格式无效，已跳过")
                            w_ais = 0.0
                    
                    except Exception as e:
                        # 加载失败
                        _update_pipeline_node(3, "fail", f"加载失败：{str(e)[:50]}", seconds=0.2)
                        st.error(f"❌ 加载 AIS 密度失败：{e}")
                        w_ais = 0.0
                
                # 情况 3：文件不存在
                else:
                    _update_pipeline_node(3, "done", f"跳过：文件不存在", seconds=0.1)
                    st.warning(f"⚠️ AIS 密度文件不存在：{ais_density_path_obj}")
                    w_ais = 0.0
            
            except Exception as e:
                # 意外错误
                _update_pipeline_node(3, "fail", f"异常：{str(e)[:50]}", seconds=0.2)
                st.error(f"❌ AIS 加载异常：{e}")
                w_ais = 0.0
        
        # 更新流动管线显示
        if "pipeline_flow_placeholder" in st.session_state:
            try:
                st.session_state.pipeline_flow_placeholder.empty()
                with st.session_state.pipeline_flow_placeholder.container():
                    render_pipeline_flow(
                        st.session_state.pipeline_flow_nodes,
                        title="🔧 规划流程管线",
                        expanded=st.session_state.get("pipeline_flow_expanded", True),
                    )
            except Exception:
                pass'''
    
    if ais_old_block in content:
        content = content.replace(ais_old_block, ais_new_block)
        print("✅ 任务 A 完成：修正 AIS 状态处理")
    else:
        print("⚠️ 未找到原始 AIS 加载块，跳过任务 A")
    
    # ========================================================================
    # 任务 B：检查并删除简化版本管线（如果存在）
    # ========================================================================
    
    # 搜索可能的简化版本管线代码
    simplified_patterns = [
        r'# .*简化.*管线.*\n.*?(?=\n    # |\n    if |\nif )',
        r'# .*Simplified.*pipeline.*\n.*?(?=\n    # |\n    if |\nif )',
    ]
    
    simplified_found = False
    for pattern in simplified_patterns:
        matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            print(f"⚠️ 发现可能的简化版本管线：{match.group(0)[:100]}")
            simplified_found = True
    
    if not simplified_found:
        print("✅ 任务 B 完成：未发现简化版本管线代码")
    
    # ========================================================================
    # 任务 C1：改进 AIS 选择器 - 按网格过滤 + 自动清空旧选择
    # ========================================================================
    
    # 找到 grid_signature 相关的代码并增强
    grid_sig_pattern = r'(grid_sig = st\.session_state\.get\("grid_signature", "N/A"\))'
    
    grid_sig_enhancement = '''# 任务 C1：检查网格是否变化，若变化则清空 AIS 选择
        current_grid_sig = compute_grid_signature(grid_mode=grid_mode, grid=None)
        previous_grid_sig = st.session_state.get("previous_grid_signature", None)
        
        if previous_grid_sig is not None and current_grid_sig != previous_grid_sig:
            # 网格已切换，清空 AIS 密度选择
            st.session_state["ais_density_path"] = None
            st.session_state["ais_density_path_selected"] = None
            st.session_state["ais_density_cache_key"] = None
            st.info(f"🔄 网格已切换（{previous_grid_sig[:20]}... → {current_grid_sig[:20]}...），已清空 AIS 密度选择以避免维度错配")
        
        st.session_state["previous_grid_signature"] = current_grid_sig
        grid_sig = current_grid_sig'''
    
    if re.search(grid_sig_pattern, content):
        content = re.sub(grid_sig_pattern, grid_sig_enhancement, content)
        print("✅ 任务 C1 部分完成：添加网格变化检测")
    
    # ========================================================================
    # 保存修改后的文件
    # ========================================================================
    
    if content != original_content:
        with open('arcticroute/ui/planner_minimal.py', 'w', encoding='utf-8') as f:
            f.write(content)
        print("\n✅ 所有修改已保存到 planner_minimal.py")
    else:
        print("\n⚠️ 未进行任何修改")

if __name__ == "__main__":
    main()
