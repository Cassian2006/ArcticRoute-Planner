#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""找到 discover_ais_density_candidates 的实际使用位置"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到 discover_ais_density_candidates 的实际使用位置（不是导入）
found_import = False
for i, line in enumerate(lines, start=1):
    if 'discover_ais_density_candidates' in line and 'import' not in line:
        if not found_import:
            found_import = True
            print(f'Line {i}: discover_ais_density_candidates usage found')
            # 打印前后 30 行
            for j in range(max(0, i-5), min(len(lines), i+50)):
                try:
                    print(f'{j+1:4d}: {lines[j][:100]}')
                except:
                    print(f'{j+1:4d}: [encoding error]')
            break

