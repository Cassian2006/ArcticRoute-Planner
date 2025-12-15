#!/usr/bin/env python3
"""快速检查 resolved 状态"""
import json
from pathlib import Path

# 直接读取已有的 resolved.json
resolved_path = Path("reports/cmems_resolved.json")
sic_describe_path = Path("reports/cmems_sic_describe.json")

print("="*70)
print("Phase 9 快速状态检查")
print("="*70)

# 检查 SIC describe
print("\n[1] SIC describe 文件")
if sic_describe_path.exists():
    size = sic_describe_path.stat().st_size
    print(f"✅ 存在，大小: {size} 字节")
    if size < 100:
        print("⚠️  文件太小，可能为空")
    else:
        content = sic_describe_path.read_text(encoding="utf-8")
        lines = content.split("\n")[:20]
        print("\n开头 20 行:")
        for i, line in enumerate(lines, 1):
            print(f"{i:2d}: {line[:80]}")
else:
    print("❌ 不存在")

# 检查 resolved.json
print("\n[2] cmems_resolved.json")
if resolved_path.exists():
    try:
        content = json.loads(resolved_path.read_text(encoding="utf-8"))
        print("✅ 存在，内容:")
        print(json.dumps(content, indent=2, ensure_ascii=False))
        
        # 分析 SIC 配置
        if "sic" in content:
            sic_cfg = content["sic"]
            print(f"\n[SIC 配置分析]")
            print(f"  Dataset ID: {sic_cfg.get('dataset_id', '?')}")
            print(f"  变量数: {len(sic_cfg.get('variables', []))}")
            print(f"  变量列表: {sic_cfg.get('variables', [])}")
    except Exception as e:
        print(f"❌ 读取失败: {e}")
else:
    print("❌ 不存在")

print("\n" + "="*70)

