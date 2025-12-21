import re
from collections import Counter

with open('reports/phase13_warnings_full_v2.txt', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

lines = content.splitlines()

# 查找包含 Warning 的行
warning_lines = [line for line in lines if 'Warning' in line and ':' in line]

print('=== 警告分析报告 ===')
print(f'总警告行数: {len(warning_lines)}')

# 简单统计
warn_types = Counter()
modules = Counter()

for line in warning_lines:
    # 提取警告类型
    if 'UndefinedMetricWarning' in line:
        warn_types['UndefinedMetricWarning'] += 1
    elif 'RuntimeWarning' in line:
        warn_types['RuntimeWarning'] += 1
    else:
        warn_types['Other'] += 1
    
    # 提取模块
    if 'site-packages' in line:
        modules['site-packages'] += 1
    elif 'arcticroute' in line:
        modules['arcticroute'] += 1
    else:
        modules['other'] += 1

print('\n=== 按警告类型分类 ===')
for wtype, count in warn_types.items():
    print(f'{wtype}: {count}')

print('\n=== 按模块分类 ===')
for mod, count in modules.items():
    print(f'{mod}: {count}')

print('\n=== 前10个警告样例 ===')
for i, line in enumerate(warning_lines[:10]):
    print(f'{i+1}: {line.strip()}')

print('\n=== 结论 ===')
arcticroute_count = modules.get('arcticroute', 0)
if arcticroute_count == 0:
    print('✅ 没有来自 arcticroute 代码的警告')
else:
    print(f'❌ 发现 {arcticroute_count} 个来自 arcticroute 代码的警告')

deprecation_count = warn_types.get('DeprecationWarning', 0) + warn_types.get('PendingDeprecationWarning', 0)
if deprecation_count == 0:
    print('✅ 没有 DeprecationWarning')
else:
    print(f'❌ 发现 {deprecation_count} 个 DeprecationWarning')
