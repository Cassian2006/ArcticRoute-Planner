#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
修改 planner_minimal.py 以集成 Pipeline Timeline 组件 - 版本 2
"""

from pathlib import Path

def modify_planner_minimal():
    """修改 planner_minimal.py 文件"""
    
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    lines = planner_path.read_text(encoding='utf-8').split('\n')
    
    # 1. 在导入部分添加 Pipeline 导入（在第 30 行左右）
    # 查找 "from scripts.export_defense_bundle" 这一行
    import_insert_idx = None
    for i, line in enumerate(lines):
        if "from scripts.export_defense_bundle" in line:
            import_insert_idx = i + 1
            break
    
    if import_insert_idx is None:
        print("ERROR: Could not find import section")
        return False
    
    # 添加新的导入
    new_imports = [
        "",
        "# 导入 Pipeline Timeline 组件",
        "from arcticroute.ui.components import (",
        "    Pipeline,",
        "    PipelineStage,",
        "    render_pipeline,",
        "    init_pipeline_in_session,",
        "    get_pipeline,",
        ")",
    ]
    
    for j, imp in enumerate(new_imports):
        lines.insert(import_insert_idx + j, imp)
    
    print(f"✅ Added imports at line {import_insert_idx}")
    
    # 2. 在规划按钮之后添加 Pipeline 初始化
    # 查找 "do_plan = st.button" 这一行
    plan_button_idx = None
    for i, line in enumerate(lines):
        if 'do_plan = st.button("规划三条方案"' in line:
            plan_button_idx = i
            break
    
    if plan_button_idx is None:
        print("ERROR: Could not find plan button")
        return False
    
    # 在规划按钮之后添加 Pipeline 初始化代码
    pipeline_init_code = [
        "",
        "    # 初始化 Pipeline",
        "    pipeline = init_pipeline_in_session()",
        "    ",
        "    # 定义 Pipeline stages",
        "    pipeline_stages = [",
        '        ("grid_env", "加载网格"),',
        '        ("ais", "加载 AIS"),',
        '        ("cost_build", "构建成本场"),',
        '        ("snap", "起止点吸附"),',
        '        ("astar", "A* 路由"),',
        '        ("analysis", "成本分析"),',
        '        ("render", "数据准备"),',
        "    ]",
        "    ",
        "    # 添加所有 stages 到 pipeline",
        "    for stage_key, stage_label in pipeline_stages:",
        "        pipeline.add_stage(stage_key, stage_label)",
        "    ",
        "    # 初始化 session state 中的 pipeline 控制变量",
        '    if "pipeline_expanded" not in st.session_state:',
        "        st.session_state.pipeline_expanded = True",
        "    ",
        "    # 规划按钮被点击时，强制展开 pipeline",
        "    if do_plan:",
        "        st.session_state.pipeline_expanded = True",
    ]
    
    for j, code_line in enumerate(pipeline_init_code):
        lines.insert(plan_button_idx + 1 + j, code_line)
    
    print(f"✅ Added pipeline initialization after plan button at line {plan_button_idx}")
    
    # 3. 在 "if not do_plan:" 之前添加 Pipeline 展示容器
    not_do_plan_idx = None
    for i, line in enumerate(lines):
        if line.strip() == "if not do_plan:":
            not_do_plan_idx = i
            break
    
    if not_do_plan_idx is None:
        print("ERROR: Could not find 'if not do_plan:' line")
        return False
    
    # 在这行之前添加 Pipeline 展示器
    pipeline_display_code = [
        "    # 创建 Pipeline 展示容器",
        '    with st.expander("⏱️ 计算流程管线", expanded=st.session_state.get("pipeline_expanded", True)):',
        "        pipeline_placeholder = st.empty()",
        "    ",
    ]
    
    for j, code_line in enumerate(pipeline_display_code):
        lines.insert(not_do_plan_idx + j, code_line)
    
    print(f"✅ Added pipeline display container before 'if not do_plan:' at line {not_do_plan_idx}")
    
    # 保存修改后的文件
    planner_path.write_text('\n'.join(lines), encoding='utf-8')
    print("✅ Successfully modified planner_minimal.py")
    return True

if __name__ == "__main__":
    modify_planner_minimal()








