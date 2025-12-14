#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修复 placeholder 问题 - 版本 2
"""

from pathlib import Path

def fix():
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    content = planner_path.read_text(encoding='utf-8')
    
    # 替换有问题的部分
    old_text = """    # 主区域逻辑
    # 创建 Pipeline 展示容器
    # 创建 Pipeline 展示容器
        pass  # placeholder 在 expander 外部创建

    if not do_plan:"""
    
    new_text = """    # 主区域逻辑
    # 创建 Pipeline 展示容器
    pipeline_placeholder = st.empty()
    with st.expander("⏱️ 计算流程管线", expanded=st.session_state.get("pipeline_expanded", True)):
        pass  # 展示容器在 expander 内

    if not do_plan:"""
    
    if old_text in content:
        content = content.replace(old_text, new_text)
        planner_path.write_text(content, encoding='utf-8')
        print("✅ Fixed placeholder")
        return True
    else:
        print("WARNING: Could not find old text to replace")
        # 尝试另一种方式
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if '# 创建 Pipeline 展示容器' in line and i > 0:
                # 检查是否是重复的
                if '# 创建 Pipeline 展示容器' in lines[i-1]:
                    print(f"Found duplicate comment at line {i}")
                    # 删除这一行和下一行
                    if i + 1 < len(lines) and 'pass' in lines[i + 1]:
                        # 替换这两行
                        lines[i-1] = "    # 创建 Pipeline 展示容器"
                        lines[i] = "    pipeline_placeholder = st.empty()"
                        lines[i+1] = '    with st.expander("⏱️ 计算流程管线", expanded=st.session_state.get("pipeline_expanded", True)):'
                        lines.insert(i+2, "        pass  # 展示容器在 expander 内")
                        
                        planner_path.write_text('\n'.join(lines), encoding='utf-8')
                        print("✅ Fixed placeholder (alternative method)")
                        return True
        
        return False

if __name__ == "__main__":
    fix()




