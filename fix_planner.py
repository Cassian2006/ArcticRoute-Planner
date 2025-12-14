#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
修改 planner_minimal.py 以实现三个任务：
A. 修正管线顺序与 AIS 状态
B. 删除简化版本管线（如果存在）
C. 改进 AIS 维度匹配处理
"""

def main():
    # 读取原文件
    with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ========================================================================
    # 任务 A：找到并替换 AIS 加载块
    # ========================================================================
    
    # 找到 "ais_info = {" 这一行
    ais_info_line_idx = -1
    for i, line in enumerate(lines):
        if 'ais_info = {"loaded": False' in line and i > 1100:
            ais_info_line_idx = i
            print(f"✅ 找到 ais_info 初始化行：{i+1}")
            break
    
    if ais_info_line_idx < 0:
        print("❌ 未找到 ais_info 初始化行")
        return
    
    # 找到 "if w_ais > 0:" 这一行（在 ais_info 之后）
    ais_if_line_idx = -1
    for i in range(ais_info_line_idx, min(ais_info_line_idx + 5, len(lines))):
        if 'if w_ais > 0:' in lines[i]:
            ais_if_line_idx = i
            print(f"✅ 找到 if w_ais > 0 行：{i+1}")
            break
    
    if ais_if_line_idx < 0:
        print("❌ 未找到 if w_ais > 0 行")
        return
    
    # 找到这个 if 块的结束位置（下一个 if 或 # 注释，且缩进级别相同）
    ais_block_end = -1
    base_indent = len(lines[ais_if_line_idx]) - len(lines[ais_if_line_idx].lstrip())
    
    for i in range(ais_if_line_idx + 1, len(lines)):
        line = lines[i]
        if line.strip() == '':
            continue
        
        current_indent = len(line) - len(line.lstrip())
        
        # 如果缩进回到基础级别，说明 if 块结束
        if current_indent <= base_indent and line.strip():
            ais_block_end = i
            print(f"✅ 找到 if 块结束位置：{i+1}")
            break
    
    if ais_block_end < 0:
        print("❌ 未找到 if 块结束位置")
        return
    
    # 现在我们有了：
    # - ais_info_line_idx: ais_info 初始化行
    # - ais_if_line_idx: if w_ais > 0 行
    # - ais_block_end: if 块结束行
    
    print(f"\n[object Object]IS 块范围：{ais_info_line_idx+1} - {ais_block_end}")
    print(f"原始块行数：{ais_block_end - ais_info_line_idx}")
    
    # 创建新的 AIS 加载块
    new_ais_block = '''        # ====================================================================
        # 任务 A：AIS 密度加载与状态管理
        # 确保 AIS 步骤完成时不停留在 pending（成功加载或跳过都标记为 done）
        # ====================================================================
        ais_info = {"loaded": False, "error": None, "shape": None, "num_points": 0, "num_binned": 0}
        ais_da_loaded = None
        
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
                pass

'''
    
    # 替换行
    new_lines = lines[:ais_info_line_idx] + [new_ais_block] + lines[ais_block_end:]
    
    # 保存修改
    with open('arcticroute/ui/planner_minimal.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("\n✅ 任务 A 完成：修正 AIS 状态处理")
    print(f"📊 修改统计：删除 {ais_block_end - ais_info_line_idx} 行，添加 {len(new_ais_block.split(chr(10)))} 行")

if __name__ == "__main__":
    main()
