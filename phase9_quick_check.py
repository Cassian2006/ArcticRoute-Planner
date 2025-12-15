#!/usr/bin/env python3
"""
Phase 9 快速检查 - 验证关键文件和配置
"""
import json
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("Phase 9 快速检查")
    print("="*70)
    
    # 1. 检查 SIC describe
    print("\n[1] SIC describe 文件")
    sic_path = Path("reports/cmems_sic_describe.json")
    if sic_path.exists():
        size = sic_path.stat().st_size
        print(f"✅ 存在，大小: {size} 字节")
        if size > 100:
            try:
                content = sic_path.read_text(encoding="utf-8")
                print(f"   开头 200 字符: {content[:200]}")
            except:
                pass
    else:
        print(f"❌ 不存在")
    
    # 2. 检查 resolved.json
    print("\n[2] cmems_resolved.json")
    resolved_path = Path("reports/cmems_resolved.json")
    if resolved_path.exists():
        try:
            content = json.loads(resolved_path.read_text(encoding="utf-8"))
            print(f"✅ 存在，内容:")
            print(json.dumps(content, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ 读取失败: {e}")
    else:
        print(f"❌ 不存在")
    
    # 3. 检查 CMEMS 面板导入
    print("\n[3] CMEMS 面板集成")
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    if planner_path.exists():
        content = planner_path.read_text(encoding="utf-8")
        has_import = "from arcticroute.ui.cmems_panel import" in content
        has_call = "render_env_source_selector()" in content or "render_cmems_panel()" in content
        
        print(f"  导入: {'✅' if has_import else '❌'}")
        print(f"  调用: {'✅' if has_call else '❌'}")
        
        if has_import and not has_call:
            print("  ⚠️  需要在 UI 中插入调用")
    
    # 4. 检查 planner_service
    print("\n[4] planner_service 集成")
    service_path = Path("arcticroute/core/planner_service.py")
    if service_path.exists():
        content = service_path.read_text(encoding="utf-8")
        has_load_env = "def load_environment" in content
        has_newenv = "use_newenv_for_cost" in content
        
        print(f"  load_environment: {'✅' if has_load_env else '❌'}")
        print(f"  use_newenv_for_cost: {'✅' if has_newenv else '❌'}")
    
    # 5. 检查 sync 脚本
    print("\n[5] cmems_newenv_sync 脚本")
    sync_path = Path("scripts/cmems_newenv_sync.py")
    if sync_path.exists():
        print(f"✅ 存在")
    else:
        print(f"❌ 不存在")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()

