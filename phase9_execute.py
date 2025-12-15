#!/usr/bin/env python3
"""
Phase 9 执行脚本 - 按顺序执行关键步骤
"""
import subprocess
import json
from pathlib import Path
import sys
import time

def run_python_script(script_path, args=None, description=""):
    """运行 Python 脚本"""
    print(f"\n{'='*70}")
    print(f"[执行] {description}")
    print(f"{'='*70}")
    
    cmd = ["python", script_path]
    if args:
        cmd.extend(args)
    
    print(f"命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ 超时")
        return False
    except Exception as e:
        print(f"❌ 异常: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("Phase 9 关键步骤执行")
    print("="*70)
    
    # Step 1: 强制生成 SIC describe
    print("\n[步骤 1] 强制生成 SIC describe")
    success = run_python_script(
        "scripts/force_sic_describe.py",
        description="强制 SIC describe 落盘"
    )
    
    # 检查结果
    sic_path = Path("reports/cmems_sic_describe.json")
    if sic_path.exists():
        size = sic_path.stat().st_size
        print(f"\n✅ SIC describe 文件大小: {size} 字节")
        if size > 100:
            content = sic_path.read_text(encoding="utf-8")
            lines = content.split("\n")[:10]
            print("开头 10 行:")
            for i, line in enumerate(lines, 1):
                print(f"  {i}: {line}")
    else:
        print(f"❌ SIC describe 文件未生成")
    
    # Step 2: 运行 cmems_resolve.py
    print("\n[步骤 2] 运行 cmems_resolve.py")
    success = run_python_script(
        "scripts/cmems_resolve.py",
        description="解析 CMEMS 数据源"
    )
    
    # 检查 resolved.json
    resolved_path = Path("reports/cmems_resolved.json")
    if resolved_path.exists():
        try:
            content = json.loads(resolved_path.read_text(encoding="utf-8"))
            print("\n✅ cmems_resolved.json 内容:")
            print(json.dumps(content, indent=2, ensure_ascii=False))
        except Exception as e:
            print(f"❌ 读取 resolved.json 失败: {e}")
    
    # Step 3: 运行 cmems_refresh_and_export
    print("\n[步骤 3] 运行 cmems_refresh_and_export（end-to-end 测试）")
    success = run_python_script(
        "-m",
        ["scripts.cmems_refresh_and_export", "--days", "2", "--bbox", "-40", "60", "65", "85"],
        description="刷新并导出 CMEMS 数据"
    )
    
    # 检查刷新记录
    refresh_path = Path("reports/cmems_refresh_last.json")
    if refresh_path.exists():
        try:
            content = json.loads(refresh_path.read_text(encoding="utf-8"))
            print("\n✅ cmems_refresh_last.json 摘要:")
            print(f"  时间范围: {content.get('start_date')} 至 {content.get('end_date')}")
            print(f"  下载结果:")
            for key, val in content.get("downloads", {}).items():
                status = "✅" if val.get("success") else "❌"
                print(f"    {status} {key}: {val.get('filename', val.get('error', '?'))}")
        except Exception as e:
            print(f"⚠️  读取刷新记录失败: {e}")
    
    # Step 4: 运行测试
    print("\n[步骤 4] 运行测试套件")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "-q"],
            capture_output=True,
            text=True,
            timeout=300
        )
        print(result.stdout)
        if result.returncode == 0:
            print("✅ 测试通过")
        else:
            print(f"⚠️  测试有失败 (rc={result.returncode})")
    except Exception as e:
        print(f"❌ 测试执行失败: {e}")
    
    print("\n" + "="*70)
    print("Phase 9 执行完成")
    print("="*70)

if __name__ == "__main__":
    main()

