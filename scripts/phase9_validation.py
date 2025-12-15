#!/usr/bin/env python3
"""
Phase 9 验证脚本 - 按顺序执行关键步骤
"""
import subprocess
import json
from pathlib import Path
import sys

def run_cmd(cmd, description):
    """运行命令并报告结果"""
    print(f"\n{'='*70}")
    print(f"[步骤] {description}")
    print(f"{'='*70}")
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"✅ 成功")
            if result.stdout:
                print("输出:")
                print(result.stdout[:500])
            return True
        else:
            print(f"❌ 失败 (rc={result.returncode})")
            if result.stderr:
                print("错误:")
                print(result.stderr[:500])
            return False
    except subprocess.TimeoutExpired:
        print(f"❌ 超时")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def check_file(path, description):
    """检查文件是否存在及大小"""
    p = Path(path)
    if p.exists():
        size = p.stat().st_size
        print(f"✅ {description}: {path}")
        print(f"   大小: {size} 字节")
        return True
    else:
        print(f"❌ {description}: {path} 不存在")
        return False

def main():
    print("\n" + "="*70)
    print("Phase 9 验证流程")
    print("="*70)
    
    # Step 1: 检查 SIC describe 文件
    print("\n[检查] SIC describe 文件状态")
    sic_describe_path = Path("reports/cmems_sic_describe.json")
    if sic_describe_path.exists():
        size = sic_describe_path.stat().st_size
        print(f"✅ 文件存在: {size} 字节")
        if size < 100:
            print(f"⚠️  文件太小，可能为空或未正确生成")
        else:
            content = sic_describe_path.read_text(encoding="utf-8")
            lines = content.split("\n")[:5]
            print("开头 5 行:")
            for i, line in enumerate(lines, 1):
                print(f"  {i}: {line[:80]}")
    else:
        print(f"❌ 文件不存在")
    
    # Step 2: 运行 cmems_resolve.py
    print("\n[步骤 1] 运行 cmems_resolve.py")
    success = run_cmd(
        ["python", "scripts/cmems_resolve.py"],
        "解析 CMEMS 数据源"
    )
    
    if success:
        # 检查 resolved 文件
        resolved_path = Path("reports/cmems_resolved.json")
        if resolved_path.exists():
            content = json.loads(resolved_path.read_text(encoding="utf-8"))
            print("\n[结果] cmems_resolved.json 内容:")
            print(json.dumps(content, indent=2, ensure_ascii=False))
    
    # Step 3: 运行 cmems_refresh_and_export
    print("\n[步骤 2] 运行 cmems_refresh_and_export（end-to-end 测试）")
    success = run_cmd(
        ["python", "-m", "scripts.cmems_refresh_and_export", "--days", "2", "--bbox", "-40", "60", "65", "85"],
        "刷新并导出 CMEMS 数据"
    )
    
    if success:
        # 检查刷新记录
        refresh_path = Path("reports/cmems_refresh_last.json")
        if refresh_path.exists():
            content = json.loads(refresh_path.read_text(encoding="utf-8"))
            print("\n[结果] cmems_refresh_last.json 摘要:")
            print(f"  时间范围: {content.get('start_date')} 至 {content.get('end_date')}")
            print(f"  下载结果:")
            for key, val in content.get("downloads", {}).items():
                status = "✅" if val.get("success") else "❌"
                print(f"    {status} {key}: {val.get('filename', val.get('error', '?'))}")
    
    # Step 4: 检查 CMEMS 面板集成
    print("\n[步骤 3] 检查 CMEMS 面板集成")
    planner_path = Path("arcticroute/ui/planner_minimal.py")
    content = planner_path.read_text(encoding="utf-8")
    
    if "render_cmems_panel" in content:
        print("✅ cmems_panel 已导入")
        if "render_env_source_selector()" in content or "render_cmems_panel()" in content:
            print("✅ cmems_panel 已在 UI 中调用")
        else:
            print("⚠️  cmems_panel 已导入但未在 UI 中调用，需要手动插入")
    else:
        print("❌ cmems_panel 未导入")
    
    # Step 5: 检查 planner_service 集成
    print("\n[步骤 4] 检查 planner_service 集成")
    if "load_environment" in content:
        print("✅ load_environment 已在代码中")
        if "use_newenv_for_cost" in content:
            print("✅ use_newenv_for_cost 参数已集成")
        else:
            print("⚠️  use_newenv_for_cost 参数未集成，需要检查")
    else:
        print("❌ load_environment 未在代码中")
    
    # Step 6: 运行测试
    print("\n[步骤 5] 运行测试套件")
    success = run_cmd(
        ["python", "-m", "pytest", "-q"],
        "运行 pytest"
    )
    
    print("\n" + "="*70)
    print("Phase 9 验证完成")
    print("="*70)

if __name__ == "__main__":
    main()

