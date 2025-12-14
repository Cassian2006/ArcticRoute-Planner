#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 planner_minimal.py 以在 pipeline 的每个 stage 完成时实时更新显示 - 版本 4
"""

from pathlib import Path

def modify_planner_minimal():
    """在 pipeline 的每个 stage 完成时添加实时更新"""
    
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    lines = planner_path.read_text(encoding='utf-8').split('\n')
    
    # 找到 "# 完成 grid_env stage" 这一行，并在其后添加 render 调用
    grid_env_done_idx = None
    for i, line in enumerate(lines):
        if "# 完成 grid_env stage" in line:
            grid_env_done_idx = i
            break
    
    if grid_env_done_idx is not None:
        # 找到 pipeline.done('grid_env'...) 这一行
        done_line_idx = None
        for i in range(grid_env_done_idx, min(grid_env_done_idx + 5, len(lines))):
            if "pipeline.done('grid_env'" in lines[i]:
                done_line_idx = i
                break
        
        if done_line_idx is not None:
            # 在 done 调用之后添加 render 调用
            indent = "        "
            render_code = [
                f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
            ]
            
            for j, code_line in enumerate(render_code):
                lines.insert(done_line_idx + 1 + j, code_line)
            
            print(f"✅ Added render call after grid_env done at line {done_line_idx + 1}")
    
    # 在 AIS stage 的 done 之后添加 render
    # 查找 "if w_ais > 0:" 块中的 done 调用
    # 这比较复杂，因为 AIS 加载在 try-except 块中
    # 我们需要在 AIS 加载完成后添加 done 和 render
    
    # 查找 "ais_info.update({" 这一行
    ais_update_idx = None
    for i, line in enumerate(lines):
        if 'ais_info.update({' in line:
            ais_update_idx = i
            break
    
    if ais_update_idx is not None:
        # 在这个 update 块之后添加 done 调用
        # 找到这个块的结束（通常是 })）
        block_end_idx = None
        for i in range(ais_update_idx, min(ais_update_idx + 10, len(lines))):
            if '})' in lines[i]:
                block_end_idx = i
                break
        
        if block_end_idx is not None:
            indent = "                    "
            code = [
                f"{indent}pipeline.done('ais', extra_info=f'candidates={{len(ais_density.flat)}}')",
                f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
            ]
            
            for j, code_line in enumerate(code):
                lines.insert(block_end_idx + 1 + j, code_line)
            
            print(f"✅ Added ais done and render at line {block_end_idx + 1}")
    
    # 在 cost_build/snap/astar 的 done 之后添加 render
    astar_done_idx = None
    for i, line in enumerate(lines):
        if "pipeline.done('astar'" in line:
            astar_done_idx = i
            break
    
    if astar_done_idx is not None:
        indent = "        "
        render_code = [
            f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
        ]
        
        for j, code_line in enumerate(render_code):
            lines.insert(astar_done_idx + 1 + j, code_line)
        
        print(f"✅ Added render call after astar done at line {astar_done_idx + 1}")
    
    # 在 analysis done 之后添加 render
    analysis_done_idx = None
    for i, line in enumerate(lines):
        if "pipeline.done('analysis')" in line:
            analysis_done_idx = i
            break
    
    if analysis_done_idx is not None:
        indent = "    "
        render_code = [
            f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
        ]
        
        for j, code_line in enumerate(render_code):
            lines.insert(analysis_done_idx + 1 + j, code_line)
        
        print(f"✅ Added render call after analysis done at line {analysis_done_idx + 1}")
    
    # 在 render done 之前添加 render
    render_done_idx = None
    for i, line in enumerate(lines):
        if "pipeline.done('render')" in line:
            render_done_idx = i
            break
    
    if render_done_idx is not None:
        indent = "    "
        render_code = [
            f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
        ]
        
        for j, code_line in enumerate(render_code):
            lines.insert(render_done_idx + j, code_line)
        
        print(f"✅ Added render call before render done at line {render_done_idx}")
    
    # 保存修改后的文件
    planner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ Successfully added render_pipeline calls")
    return True

if __name__ == "__main__":
    modify_planner_minimal()


