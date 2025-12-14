#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 Pipeline Timeline 集成
"""

import sys
from pathlib import Path

def test_imports():
    """测试导入"""
    print("Testing imports...")
    try:
        from arcticroute.ui.components import (
            Pipeline,
            PipelineStage,
            render_pipeline,
            init_pipeline_in_session,
            get_pipeline,
        )
        print("✅ Pipeline components imported successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to import pipeline components: {e}")
        return False

def test_pipeline_class():
    """测试 Pipeline 类"""
    print("\nTesting Pipeline class...")
    try:
        from arcticroute.ui.components import Pipeline, PipelineStage
        
        # 创建 pipeline
        pipeline = Pipeline()
        
        # 添加 stages
        pipeline.add_stage("test1", "Test Stage 1")
        pipeline.add_stage("test2", "Test Stage 2")
        
        # 测试 start/done
        pipeline.start("test1")
        assert pipeline.stages["test1"].status == "running"
        print("✅ Stage start works")
        
        pipeline.done("test1", extra_info="test_info")
        assert pipeline.stages["test1"].status == "done"
        assert pipeline.stages["test1"].extra_info == "test_info"
        assert pipeline.stages["test1"].dt_s >= 0  # dt_s 可能是 0（执行很快）
        print("✅ Stage done works with timing")
        
        # 测试 fail
        pipeline.start("test2")
        pipeline.fail("test2", fail_reason="test_failure")
        assert pipeline.stages["test2"].status == "fail"
        assert pipeline.stages["test2"].fail_reason == "test_failure"
        print("✅ Stage fail works")
        
        # 测试 get_stages_list
        stages = pipeline.get_stages_list()
        assert len(stages) == 2
        print("✅ get_stages_list works")
        
        return True
    except Exception as e:
        print(f"❌ Pipeline class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_planner_syntax():
    """测试 planner_minimal.py 的语法"""
    print("\nTesting planner_minimal.py syntax...")
    try:
        import py_compile
        py_compile.compile("arcticroute/ui/planner_minimal.py", doraise=True)
        print("✅ planner_minimal.py syntax is valid")
        return True
    except Exception as e:
        print(f"❌ planner_minimal.py syntax error: {e}")
        return False

def test_pipeline_in_planner():
    """测试 planner_minimal.py 中的 pipeline 导入"""
    print("\nTesting pipeline integration in planner_minimal.py...")
    try:
        # 检查文件内容
        planner_path = Path("arcticroute/ui/planner_minimal.py")
        content = planner_path.read_text(encoding='utf-8')
        
        # 检查导入
        if "from arcticroute.ui.components import" in content:
            print("✅ Pipeline import found in planner_minimal.py")
        else:
            print("❌ Pipeline import not found in planner_minimal.py")
            return False
        
        # 检查 pipeline 初始化
        if "init_pipeline_in_session()" in content:
            print("✅ Pipeline initialization found")
        else:
            print("❌ Pipeline initialization not found")
            return False
        
        # 检查 pipeline stages
        if "pipeline.add_stage" in content:
            print("✅ Pipeline stages found")
        else:
            print("❌ Pipeline stages not found")
            return False
        
        # 检查 pipeline start/done 调用
        if "pipeline.start(" in content and "pipeline.done(" in content:
            print("✅ Pipeline start/done calls found")
        else:
            print("❌ Pipeline start/done calls not found")
            return False
        
        # 检查 render_pipeline 调用
        if "render_pipeline(" in content:
            print("✅ render_pipeline calls found")
        else:
            print("❌ render_pipeline calls not found")
            return False
        
        # 检查 session_state 控制
        if "st.session_state['pipeline_expanded']" in content or 'st.session_state["pipeline_expanded"]' in content:
            print("✅ Pipeline session state control found")
        else:
            print("❌ Pipeline session state control not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("=" * 60)
    print("Pipeline Timeline Integration Tests")
    print("=" * 60)
    
    results = []
    results.append(("Imports", test_imports()))
    results.append(("Pipeline Class", test_pipeline_class()))
    results.append(("Planner Syntax", test_planner_syntax()))
    results.append(("Pipeline Integration", test_pipeline_in_planner()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

