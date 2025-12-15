#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
任务 C1：改进 AIS 选择器 - 按网格过滤 + 自动清空旧选择
"""

def main():
    # 读取原文件
    with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # ========================================================================
    # 任务 C1：找到网格 signature 计算部分，增强网格变化检测
    # ========================================================================
    
    # 找到 "try:" 块中的 "if grid_mode == "demo":" 这一行
    grid_sig_start = -1
    for i, line in enumerate(lines):
        if 'if grid_mode == "demo":' in line and i > 700 and i < 800:
            grid_sig_start = i
            print(f"✅ 找到网格 signature 计算块起始行：{i+1}")
            break
    
    if grid_sig_start < 0:
        print("❌ 未找到网格 signature 计算块")
        return
    
    # 找到这个块的结束位置（找到 "st.session_state["grid_signature"]" 的最后一行）
    grid_sig_end = -1
    for i in range(grid_sig_start, min(grid_sig_start + 50, len(lines))):
        if 'st.session_state["grid_signature"]' in lines[i] and 'except' not in lines[i]:
            grid_sig_end = i + 1
    
    if grid_sig_end < 0:
        print("❌ 未找到网格 signature 块结束位置")
        return
    
    print(f"✅ 网格 signature 块范围：{grid_sig_start+1} - {grid_sig_end}")
    
    # 在这个块之前插入网格变化检测逻辑
    # 找到 "try:" 这一行
    try_line = -1
    for i in range(grid_sig_start - 5, grid_sig_start):
        if 'try:' in lines[i]:
            try_line = i
            break
    
    if try_line < 0:
        print("❌ 未找到 try 块")
        return
    
    print(f"✅ 找到 try 块起始行：{try_line+1}")
    
    # 在 try 块之后插入网格变化检测代码
    grid_change_detection = '''        # ====================================================================
        # 任务 C1：检查网格是否变化，若变化则清空 AIS 选择
        # 这样可以避免用户在切换网格后仍然使用旧网格的 AIS 密度文件
        # ====================================================================
        previous_grid_signature = st.session_state.get("previous_grid_signature", None)
        
'''
    
    # 在 try 块之后插入
    new_lines = lines[:try_line+1] + [grid_change_detection] + lines[try_line+1:]
    
    # 现在找到 "current_grid_signature = compute_grid_signature" 这一行，在其后添加检测逻辑
    for i in range(try_line, len(new_lines)):
        if 'current_grid_signature = compute_grid_signature' in new_lines[i]:
            # 在这一行之后插入检测逻辑
            detection_logic = '''
            # 检查网格是否变化
            if previous_grid_signature is not None and previous_grid_signature != current_grid_signature:
                # 网格已切换，清空 AIS 密度选择
                st.session_state["ais_density_path"] = None
                st.session_state["ais_density_path_selected"] = None
                st.session_state["ais_density_cache_key"] = None
                st.info(f"🔄 网格已切换，已清空 AIS 密度选择以避免维度错配")
                print(f"[UI] Grid changed: {previous_grid_signature[:30]}... -> {current_grid_signature[:30]}...")
            
            # 更新当前网格 signature
            st.session_state["previous_grid_signature"] = current_grid_signature
'''
            new_lines.insert(i + 1, detection_logic)
            print(f"✅ 在第 {i+2} 行插入网格变化检测逻辑")
            break
    
    # 保存修改
    with open('arcticroute/ui/planner_minimal.py', 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
    
    print("\n✅ 任务 C1 完成：添加网格变化检测和 AIS 自动清空")

if __name__ == "__main__":
    main()








