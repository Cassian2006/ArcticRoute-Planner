#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
将 CMEMS 面板集成到 planner_minimal.py

这个脚本会在 planner_minimal.py 中添加 CMEMS 数据源选择和刷新面板。
"""

import sys
from pathlib import Path


def integrate_cmems_ui():
    """
    将 CMEMS UI 集成到 planner_minimal.py
    """
    planner_file = Path("arcticroute/ui/planner_minimal.py")
    
    if not planner_file.exists():
        print(f"[ERROR] 找不到 {planner_file}")
        return False
    
    content = planner_file.read_text(encoding="utf-8")
    
    # 检查是否已经集成
    if "from arcticroute.ui.cmems_panel import" in content:
        print("[OK] CMEMS UI 已经集成")
        return True
    
    # 添加导入语句
    import_section = """# 导入 CMEMS 面板组件
from arcticroute.ui.cmems_panel import (
    render_env_source_selector,
    render_cmems_panel,
    render_manual_nc_selector,
    get_env_source_config,
)"""
    
    # 在现有导入后添加
    insert_pos = content.find("# 导入 AIS Density 面板组件")
    if insert_pos == -1:
        print("[ERROR] 找不到导入位置")
        return False
    
    content = content[:insert_pos] + import_section + "\n\n" + content[insert_pos:]
    
    # 在场景与环境部分后添加 CMEMS 面板
    cmems_panel_code = """
        # ====================================================================
        # CMEMS 环境数据源选择与刷新
        # ====================================================================
        with st.expander("☁️ CMEMS 近实时数据 (可选)", expanded=False):
            env_source = render_env_source_selector()
            
            if env_source == "cmems_latest":
                render_cmems_panel()
            elif env_source == "manual_nc":
                render_manual_nc_selector()
            
            # 获取环境数据源配置
            env_source_config = get_env_source_config()
            st.session_state["env_source_config"] = env_source_config
"""
    
    # 在网格模式选择后添加
    insert_pos = content.find('st.session_state["grid_mode_pref"] = grid_mode')
    if insert_pos == -1:
        print("[ERROR] 找不到网格模式选择位置")
        return False
    
    # 找到该行的结尾
    line_end = content.find("\n", insert_pos) + 1
    
    content = content[:line_end] + cmems_panel_code + "\n" + content[line_end:]
    
    # 保存修改
    planner_file.write_text(content, encoding="utf-8")
    print(f"[OK] CMEMS UI 已集成到 {planner_file}")
    
    return True


if __name__ == "__main__":
    success = integrate_cmems_ui()
    sys.exit(0 if success else 1)

