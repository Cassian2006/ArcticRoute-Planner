#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 routes_info 的定义顺序问题
"""

from pathlib import Path

def fix():
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    lines = planner_path.read_text(encoding='utf-8').split('\n')
    
    # 找到有问题的部分
    # 需要找到 "# 完成 cost_build/snap/astar stages" 这一行
    problem_start_idx = None
    for i, line in enumerate(lines):
        if "# 完成 cost_build/snap/astar stages" in line:
            problem_start_idx = i
            break
    
    if problem_start_idx is None:
        print("ERROR: Could not find problem section")
        return False
    
    # 找到 "routes_info, cost_fields, cost_meta" 这一行
    plan_three_routes_idx = None
    for i in range(problem_start_idx, min(problem_start_idx + 50, len(lines))):
        if "routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(" in lines[i]:
            plan_three_routes_idx = i
            break
    
    if plan_three_routes_idx is None:
        print("ERROR: Could not find plan_three_routes call")
        return False
    
    # 找到 plan_three_routes 调用的结束
    call_end_idx = None
    paren_count = 0
    for i in range(plan_three_routes_idx, len(lines)):
        for char in lines[i]:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
        if paren_count == 0 and i > plan_three_routes_idx:
            call_end_idx = i
            break
    
    if call_end_idx is None:
        print("ERROR: Could not find end of plan_three_routes call")
        return False
    
    print(f"Problem section: {problem_start_idx} to {plan_three_routes_idx}")
    print(f"plan_three_routes call: {plan_three_routes_idx} to {call_end_idx}")
    
    # 删除有问题的部分（从 "# 完成" 到 "pipeline.start('astar')" 之前）
    # 找到 "pipeline.start('astar')" 这一行
    astar_start_idx = None
    for i in range(problem_start_idx, plan_three_routes_idx):
        if "pipeline.start('astar')" in lines[i]:
            astar_start_idx = i
            break
    
    if astar_start_idx is not None:
        # 删除从 problem_start_idx 到 astar_start_idx 的所有行
        del lines[problem_start_idx:astar_start_idx + 1]
        print(f"Deleted lines {problem_start_idx} to {astar_start_idx}")
    
    # 现在在 plan_three_routes 调用之后添加正确的 done 调用
    # 找到新的 plan_three_routes 位置（因为我们删除了一些行）
    plan_three_routes_idx_new = None
    for i, line in enumerate(lines):
        if "routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(" in line:
            plan_three_routes_idx_new = i
            break
    
    if plan_three_routes_idx_new is None:
        print("ERROR: Could not find plan_three_routes after deletion")
        return False
    
    # 找到 plan_three_routes 调用的结束
    call_end_idx_new = None
    paren_count = 0
    for i in range(plan_three_routes_idx_new, len(lines)):
        for char in lines[i]:
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
        if paren_count == 0 and i > plan_three_routes_idx_new:
            call_end_idx_new = i
            break
    
    if call_end_idx_new is None:
        print("ERROR: Could not find end of plan_three_routes call after deletion")
        return False
    
    # 在 plan_three_routes 调用之后添加 done 调用
    indent = "        "
    done_code = [
        f"{indent}",
        f"{indent}# 完成 cost_build/snap/astar stages",
        f"{indent}pipeline.done('cost_build')",
        f"{indent}pipeline.done('snap')",
        f"{indent}num_reachable = sum(1 for r in routes_info.values() if r.reachable)",
        f"{indent}pipeline.done('astar', extra_info=f'routes reachable={{num_reachable}}/3')",
        f"{indent}render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)",
    ]
    
    for j, code_line in enumerate(done_code):
        lines.insert(call_end_idx_new + 1 + j, code_line)
    
    print(f"Added done calls after line {call_end_idx_new + 1}")
    
    # 保存修改
    planner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ Fixed routes_info order")
    return True

if __name__ == "__main__":
    fix()








