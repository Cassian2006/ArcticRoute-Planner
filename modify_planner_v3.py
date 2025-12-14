#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 planner_minimal.py 以在规划流程中集成 Pipeline 的 start/done/fail 调用 - 版本 3
"""

from pathlib import Path

def modify_planner_minimal():
    """在规划流程中添加 Pipeline 调用"""
    
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    lines = planner_path.read_text(encoding='utf-8').split('\n')
    
    # 1. 在 "with st.spinner("加载网格与规划路线.."):" 之后添加 grid_env stage 的启动
    spinner_idx = None
    for i, line in enumerate(lines):
        if 'with st.spinner("加载网格与规划路线..")' in line or 'with st.spinner("加载网格与规划路线..")' in line:
            spinner_idx = i
            break
    
    if spinner_idx is None:
        print("WARNING: Could not find spinner line, skipping grid_env stage start")
    else:
        # 在 spinner 块的第一行有意义的代码处添加 start 调用
        # 找到 "grid_source_label = "demo"" 这一行
        grid_label_idx = None
        for i in range(spinner_idx + 1, min(spinner_idx + 20, len(lines))):
            if 'grid_source_label = "demo"' in lines[i]:
                grid_label_idx = i
                break
        
        if grid_label_idx is not None:
            # 在这一行之前添加 start 调用
            indent = "        "  # 8 spaces for inside spinner
            start_code = [
                f"{indent}# 启动 grid_env stage",
                f"{indent}pipeline.start('grid_env')",
                f"{indent}",
            ]
            
            for j, code_line in enumerate(start_code):
                lines.insert(grid_label_idx + j, code_line)
            
            print(f"✅ Added grid_env stage start at line {grid_label_idx}")
    
    # 2. 在 "with st.spinner" 块结束后添加 grid_env 的 done 调用
    # 这比较复杂，因为需要找到 spinner 块的结束
    # 我们查找 "ais_info = " 这一行，它应该在 spinner 块内
    ais_info_idx = None
    for i, line in enumerate(lines):
        if 'ais_info = {"loaded": False' in line:
            ais_info_idx = i
            break
    
    if ais_info_idx is not None:
        # 在这一行之前添加 grid_env done 调用
        indent = "        "
        done_code = [
            f"{indent}# 完成 grid_env stage",
            f"{indent}grid_shape = grid.shape() if hasattr(grid, 'shape') else (0, 0)",
            f"{indent}pipeline.done('grid_env', extra_info=f'grid={{grid_shape[0]}}×{{grid_shape[1]}}')",
            f"{indent}",
        ]
        
        for j, code_line in enumerate(done_code):
            lines.insert(ais_info_idx + j, code_line)
        
        print(f"✅ Added grid_env stage done at line {ais_info_idx}")
    
    # 3. 在 AIS 加载逻辑中添加 ais stage 的 start/done
    # 查找 "if w_ais > 0:" 这一行（在 spinner 块内）
    w_ais_check_idx = None
    for i in range(ais_info_idx if ais_info_idx else spinner_idx, min(len(lines), (ais_info_idx if ais_info_idx else spinner_idx) + 50)):
        if 'if w_ais > 0:' in lines[i] and 'try:' in lines[i + 1]:
            w_ais_check_idx = i
            break
    
    if w_ais_check_idx is not None:
        # 在 try 块之前添加 start
        indent = "            "  # 12 spaces
        start_code = f"{indent}pipeline.start('ais')"
        lines.insert(w_ais_check_idx + 1, start_code)
        
        print(f"✅ Added ais stage start at line {w_ais_check_idx + 1}")
    
    # 4. 在 cost_build 阶段（plan_three_routes 调用）之前添加相关 stages
    # 查找 "routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes("
    plan_three_routes_idx = None
    for i, line in enumerate(lines):
        if 'routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(' in line:
            plan_three_routes_idx = i
            break
    
    if plan_three_routes_idx is not None:
        # 在这一行之前添加 cost_build, snap, astar 的 start 调用
        indent = "        "
        start_code = [
            f"{indent}# 启动后续 stages",
            f"{indent}pipeline.start('cost_build')",
            f"{indent}pipeline.start('snap')",
            f"{indent}pipeline.start('astar')",
            f"{indent}",
        ]
        
        for j, code_line in enumerate(start_code):
            lines.insert(plan_three_routes_idx + j, code_line)
        
        print(f"✅ Added cost_build/snap/astar stage starts at line {plan_three_routes_idx}")
    
    # 5. 在 plan_three_routes 调用之后添加这些 stages 的 done 调用
    if plan_three_routes_idx is not None:
        # 找到 plan_three_routes 调用的结束（通常是一个 ) 在某一行）
        # 我们需要找到这个调用的最后一个 )
        paren_count = 0
        call_end_idx = plan_three_routes_idx
        for i in range(plan_three_routes_idx, len(lines)):
            for char in lines[i]:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
            if paren_count == 0 and '=' in lines[plan_three_routes_idx]:
                call_end_idx = i
                break
        
        # 在调用结束后添加 done 调用
        indent = "        "
        done_code = [
            f"{indent}",
            f"{indent}# 完成 cost_build/snap/astar stages",
            f"{indent}pipeline.done('cost_build')",
            f"{indent}pipeline.done('snap')",
            f"{indent}num_reachable = sum(1 for r in routes_info.values() if r.reachable)",
            f"{indent}pipeline.done('astar', extra_info=f'routes reachable={{num_reachable}}/3')",
            f"{indent}",
        ]
        
        for j, code_line in enumerate(done_code):
            lines.insert(call_end_idx + 1 + j, code_line)
        
        print(f"✅ Added cost_build/snap/astar stage done at line {call_end_idx + 1}")
    
    # 6. 在分析阶段之前添加 analysis stage
    # 查找第一个 st.subheader("KPI 总览") 或类似的
    analysis_idx = None
    for i, line in enumerate(lines):
        if 'st.subheader("KPI 总览")' in line:
            analysis_idx = i
            break
    
    if analysis_idx is not None:
        indent = "    "
        start_code = [
            f"{indent}# 启动 analysis stage",
            f"{indent}pipeline.start('analysis')",
            f"{indent}",
        ]
        
        for j, code_line in enumerate(start_code):
            lines.insert(analysis_idx + j, code_line)
        
        print(f"✅ Added analysis stage start at line {analysis_idx}")
    
    # 7. 在 render 阶段之前添加
    # 查找 st.subheader("路线对比地图") 或类似的
    render_idx = None
    for i, line in enumerate(lines):
        if 'st.subheader("路线对比地图")' in line:
            render_idx = i
            break
    
    if render_idx is not None:
        indent = "    "
        code = [
            f"{indent}# 完成 analysis 并启动 render",
            f"{indent}pipeline.done('analysis')",
            f"{indent}pipeline.start('render')",
            f"{indent}",
        ]
        
        for j, code_line in enumerate(code):
            lines.insert(render_idx + j, code_line)
        
        print(f"✅ Added analysis done and render start at line {render_idx}")
    
    # 8. 在最后添加 render done 和自动折叠逻辑
    # 查找最后一个主要的 st.subheader 或类似的
    last_section_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if 'st.subheader("📥 导出当前规划结果")' in lines[i]:
            last_section_idx = i
            break
    
    if last_section_idx is not None:
        indent = "    "
        code = [
            f"{indent}",
            f"{indent}# 完成 render stage 并保存结果到 session_state",
            f"{indent}pipeline.done('render')",
            f"{indent}",
            f"{indent}# 将规划结果保存到 session_state，以便在 rerun 后仍可用",
            f"{indent}st.session_state['last_plan_result'] = {{",
            f"{indent}    'routes_info': routes_info,",
            f"{indent}    'cost_fields': cost_fields,",
            f"{indent}    'cost_meta': cost_meta,",
            f"{indent}    'scores_by_key': scores_by_key,",
            f"{indent}    'recommended_key': recommended_key,",
            f"{indent}}}",
            f"{indent}",
            f"{indent}# 规划完成后自动折叠 pipeline",
            f"{indent}st.session_state['pipeline_expanded'] = False",
            f"{indent}st.rerun()",
        ]
        
        for j, code_line in enumerate(code):
            lines.insert(last_section_idx + j, code_line)
        
        print(f"✅ Added render done and auto-collapse logic at line {last_section_idx}")
    
    # 保存修改后的文件
    planner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ Successfully modified planner_minimal.py with pipeline calls")
    return True

if __name__ == "__main__":
    modify_planner_minimal()


