#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 pipeline_placeholder 的作用域问题
"""

from pathlib import Path

def fix_placeholder():
    """修复 placeholder 的作用域"""
    
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    lines = planner_path.read_text(encoding='utf-8').split('\n')
    
    # 找到 "with st.expander("⏱️ 计算流程管线"" 这一行
    expander_idx = None
    for i, line in enumerate(lines):
        if 'with st.expander("⏱️ 计算流程管线"' in line:
            expander_idx = i
            break
    
    if expander_idx is None:
        print("ERROR: Could not find expander line")
        return False
    
    # 在 expander 之前创建 placeholder
    indent = "    "
    new_code = [
        f"{indent}# 创建 Pipeline 展示容器",
        f"{indent}pipeline_placeholder = st.empty()",
        f"{indent}with st.expander(\"⏱️ 计算流程管线\", expanded=st.session_state.get(\"pipeline_expanded\", True)):",
        f"{indent}    pass  # placeholder 在 expander 外部创建",
    ]
    
    # 替换原来的 expander 块
    # 首先找到 "pipeline_placeholder = st.empty()" 这一行
    placeholder_idx = None
    for i in range(expander_idx, min(expander_idx + 5, len(lines))):
        if "pipeline_placeholder = st.empty()" in lines[i]:
            placeholder_idx = i
            break
    
    if placeholder_idx is not None:
        # 删除原来的 expander 块（包括 with 和 placeholder 行）
        # 替换为新的代码
        lines[expander_idx] = new_code[0]
        lines[expander_idx + 1] = new_code[1]
        lines[placeholder_idx] = new_code[3]
        
        print(f"✅ Fixed placeholder scope at line {expander_idx}")
    
    # 保存修改后的文件
    planner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ Successfully fixed placeholder scope")
    return True

if __name__ == "__main__":
    fix_placeholder()




