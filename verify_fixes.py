#!/usr/bin/env python3
"""
验证两项修复是否生效：
1. 乱码修复
2. 地图固定在北极区域 + 限制缩放/禁止拖动
"""

from pathlib import Path
from arcticroute.core.scenarios import load_all_scenarios

print("=" * 70)
print("验证修复 U1：乱码修复")
print("=" * 70)

# 检查 scenarios 中的标题
scenarios = load_all_scenarios()
print(f"\n✅ 成功加载 {len(scenarios)} 个 scenario")

# 显示前 5 个 scenario 的标题
print("\n前 5 个 scenario 的标题：")
for i, (scenario_id, scenario) in enumerate(list(scenarios.items())[:5]):
    print(f"  {i+1}. {scenario_id}: {scenario.title}")

# 检查是否有乱码
mojibake_chars = {'æ', 'ä', 'ç', 'ö', 'ü'}
has_mojibake = False
for scenario in scenarios.values():
    for char in mojibake_chars:
        if char in scenario.title or char in scenario.description:
            has_mojibake = True
            print(f"❌ 发现乱码: {scenario.title}")

if not has_mojibake:
    print("\n✅ 所有 scenario 标题都没有乱码")

# 检查 planner_minimal.py 中的标签
print("\n" + "=" * 70)
print("验证修复 U2：地图固定在北极区域 + 限制缩放/禁止拖动")
print("=" * 70)

planner_path = Path("arcticroute/ui/planner_minimal.py")
planner_content = planner_path.read_text(encoding='utf-8')

# 检查配置是否存在
checks = [
    ("ARCTIC_VIEW 配置", "ARCTIC_VIEW = {"),
    ("MAP_CONTROLLER 配置", "MAP_CONTROLLER = {"),
    ("dragPan: False", '"dragPan": False'),
    ("min_zoom 限制", '"min_zoom": 2.2'),
    ("max_zoom 限制", '"max_zoom": 6.0'),
    ("北极纬度设置", '"latitude": 75.0'),
    ("北极经度设置", '"longitude": 30.0'),
]

print("\n地图配置检查：")
all_passed = True
for check_name, check_pattern in checks:
    if check_pattern in planner_content:
        print(f"  ✅ {check_name}")
    else:
        print(f"  ❌ {check_name}")
        all_passed = False

# 检查 ViewState 是否使用了配置
print("\n地图使用配置检查：")
arctic_view_usage = planner_content.count('ARCTIC_VIEW["')
map_controller_usage = planner_content.count('MAP_CONTROLLER')

print(f"  ARCTIC_VIEW 被使用了 {arctic_view_usage} 次")
print(f"  MAP_CONTROLLER 被使用了 {map_controller_usage} 次")

if arctic_view_usage >= 4 and map_controller_usage >= 2:
    print("  ✅ 配置被正确使用")
else:
    print("  ⚠️  配置使用次数可能不足")

print("\n" + "=" * 70)
print("总结")
print("=" * 70)

if not has_mojibake and all_passed and arctic_view_usage >= 4:
    print("\n✅ 所有修复都已成功应用！")
    print("\n接下来可以运行：")
    print("  streamlit run run_ui.py")
    print("\n然后进入'航线规划驾驶舱'检查：")
    print("  1. 左侧预设/模式文字不乱码")
    print("  2. 地图无法拖到赤道/南半球")
    print("  3. 地图无法缩放到无限小/无限大")
else:
    print("\n❌ 某些修复可能未完成，请检查上面的输出")


