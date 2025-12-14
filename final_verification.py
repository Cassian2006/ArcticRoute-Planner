#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终验证脚本 - 确保所有文件都在正确的位置并且功能正常
"""

import sys
from pathlib import Path

def verify_files():
    """验证所有必需的文件都存在"""
    print("=" * 60)
    print("文件验证")
    print("=" * 60)
    
    required_files = [
        "arcticroute/ui/components/pipeline_timeline.py",
        "arcticroute/ui/components/__init__.py",
        "arcticroute/ui/planner_minimal.py",
        "test_pipeline_integration.py",
        "PIPELINE_TIMELINE_IMPLEMENTATION.md",
        "PIPELINE_QUICK_START.md",
        "PIPELINE_COMPLETION_SUMMARY.md",
        "IMPLEMENTATION_CHECKLIST.md",
        "FINAL_DELIVERY_REPORT.md",
        "QUICK_REFERENCE.md",
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            size = path.stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - NOT FOUND")
            all_exist = False
    
    return all_exist

def verify_imports():
    """验证所有导入都正常"""
    print("\n" + "=" * 60)
    print("导入验证")
    print("=" * 60)
    
    try:
        from arcticroute.ui.components import (
            Pipeline,
            PipelineStage,
            render_pipeline,
            init_pipeline_in_session,
            get_pipeline,
        )
        print("✅ Pipeline 组件导入成功")
        
        # 测试创建对象
        pipeline = Pipeline()
        print("✅ Pipeline 对象创建成功")
        
        pipeline.add_stage("test", "Test")
        print("✅ add_stage() 方法正常")
        
        pipeline.start("test")
        print("✅ start() 方法正常")
        
        pipeline.done("test")
        print("✅ done() 方法正常")
        
        stages = pipeline.get_stages_list()
        print(f"✅ get_stages_list() 返回 {len(stages)} 个 stage")
        
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_planner_integration():
    """验证 planner_minimal.py 中的集成"""
    print("\n" + "=" * 60)
    print("集成验证")
    print("=" * 60)
    
    try:
        planner_path = Path("arcticroute/ui/planner_minimal.py")
        content = planner_path.read_text(encoding='utf-8')
        
        checks = [
            ("Pipeline 导入", "from arcticroute.ui.components import"),
            ("Pipeline 初始化", "init_pipeline_in_session()"),
            ("Pipeline stages", "pipeline.add_stage"),
            ("Pipeline start", "pipeline.start("),
            ("Pipeline done", "pipeline.done("),
            ("render_pipeline", "render_pipeline("),
            ("session_state 控制", "st.session_state['pipeline_expanded']"),
            ("st.rerun()", "st.rerun()"),
        ]
        
        all_found = True
        for check_name, check_str in checks:
            if check_str in content:
                print(f"✅ {check_name}")
            else:
                print(f"❌ {check_name} - NOT FOUND")
                all_found = False
        
        return all_found
    except Exception as e:
        print(f"❌ 集成验证失败: {e}")
        return False

def verify_syntax():
    """验证 Python 语法"""
    print("\n" + "=" * 60)
    print("语法验证")
    print("=" * 60)
    
    import py_compile
    
    files_to_check = [
        "arcticroute/ui/components/pipeline_timeline.py",
        "arcticroute/ui/planner_minimal.py",
        "test_pipeline_integration.py",
    ]
    
    all_valid = True
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✅ {file_path}")
        except Exception as e:
            print(f"❌ {file_path}: {e}")
            all_valid = False
    
    return all_valid

def verify_documentation():
    """验证文档文件"""
    print("\n" + "=" * 60)
    print("文档验证")
    print("=" * 60)
    
    doc_files = [
        "PIPELINE_TIMELINE_IMPLEMENTATION.md",
        "PIPELINE_QUICK_START.md",
        "PIPELINE_COMPLETION_SUMMARY.md",
        "IMPLEMENTATION_CHECKLIST.md",
        "FINAL_DELIVERY_REPORT.md",
        "QUICK_REFERENCE.md",
    ]
    
    all_exist = True
    for doc_file in doc_files:
        path = Path(doc_file)
        if path.exists():
            lines = path.read_text(encoding='utf-8').split('\n')
            print(f"✅ {doc_file} ({len(lines)} 行)")
        else:
            print(f"❌ {doc_file} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """运行所有验证"""
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Pipeline Timeline 最终验证".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = []
    results.append(("文件验证", verify_files()))
    results.append(("导入验证", verify_imports()))
    results.append(("集成验证", verify_planner_integration()))
    results.append(("语法验证", verify_syntax()))
    results.append(("文档验证", verify_documentation()))
    
    print("\n" + "=" * 60)
    print("验证总结")
    print("=" * 60)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 验证通过")
    
    if passed == total:
        print("\n" + "=" * 60)
        print("🎉 所有验证通过！")
        print("=" * 60)
        print("\n✅ Pipeline Timeline 已准备好投入生产使用")
        print("\n快速开始:")
        print("  1. 运行测试: python test_pipeline_integration.py")
        print("  2. 启动 UI: streamlit run run_ui.py")
        print("  3. 点击'规划三条方案'查看 Pipeline Timeline")
        print("\n文档:")
        print("  - 快速启动: PIPELINE_QUICK_START.md")
        print("  - 详细实现: PIPELINE_TIMELINE_IMPLEMENTATION.md")
        print("  - 快速参考: QUICK_REFERENCE.md")
        return 0
    else:
        print("\n" + "=" * 60)
        print(f"⚠️ {total - passed} 个验证失败")
        print("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())








