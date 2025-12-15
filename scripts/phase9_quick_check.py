#!/usr/bin/env python3
"""
Phase 9 快速检查脚本 - 验证 CMEMS 集成的关键点
"""
import json
import sys
from pathlib import Path

def check_file_exists(path, description):
    """检查文件是否存在"""
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        print(f"[OK] {description}: {path} ({size} bytes)")
        return True
    else:
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False

def check_json_valid(path, description):
    """检查 JSON 文件是否有效"""
    p = Path(path)
    if not p.exists():
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False
    
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[OK] {description}: {path} (valid JSON)")
        return True
    except Exception as e:
        print(f"[FAIL] {description}: {path} (invalid JSON: {e})")
        return False

def check_code_contains(path, pattern, description):
    """检查代码文件是否包含特定模式"""
    p = Path(path)
    if not p.exists():
        print(f"[FAIL] {description}: {path} NOT FOUND")
        return False
    
    try:
        content = p.read_text(encoding="utf-8")
        if pattern in content:
            print(f"[OK] {description}: found in {path}")
            return True
        else:
            print(f"[FAIL] {description}: NOT found in {path}")
            return False
    except Exception as e:
        print(f"[FAIL] {description}: error reading {path}: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("Phase 9 Quick Check - CMEMS Integration Verification")
    print("="*70 + "\n")
    
    checks = []
    
    # 1. 检查 CMEMS 配置文件
    print("[1] Configuration Files")
    checks.append(check_json_valid("reports/cmems_resolved.json", "CMEMS resolved config"))
    
    # 2. 检查脚本
    print("\n[2] CMEMS Scripts")
    checks.append(check_file_exists("scripts/cmems_resolve.py", "cmems_resolve.py"))
    checks.append(check_file_exists("scripts/cmems_refresh_and_export.py", "cmems_refresh_and_export.py"))
    checks.append(check_file_exists("scripts/cmems_newenv_sync.py", "cmems_newenv_sync.py"))
    
    # 3. 检查 UI 集成
    print("\n[3] UI Integration")
    checks.append(check_file_exists("arcticroute/ui/cmems_panel.py", "cmems_panel.py"))
    checks.append(check_code_contains("arcticroute/ui/planner_minimal.py", "cmems_panel", "cmems_panel import in planner_minimal.py"))
    
    # 4. 检查环境加载
    print("\n[4] Environment Loading")
    checks.append(check_code_contains("arcticroute/core/planner_service.py", "use_newenv_for_cost", "use_newenv_for_cost parameter in planner_service.py"))
    
    # 5. 检查测试
    print("\n[5] Tests")
    checks.append(check_file_exists("tests/test_cmems_planner_integration.py", "test_cmems_planner_integration.py"))
    
    # 总结
    print("\n" + "="*70)
    passed = sum(checks)
    total = len(checks)
    print(f"Results: {passed}/{total} checks passed")
    
    if passed == total:
        print("[OK] All checks passed!")
        return 0
    else:
        print(f"[WARN] {total - passed} checks failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())

