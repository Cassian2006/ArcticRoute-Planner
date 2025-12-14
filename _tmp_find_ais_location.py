#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""找到 discover_ais_density_candidates 的使用位置"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到 discover_ais_density_candidates 的使用位置
for i, line in enumerate(lines, start=1):
    if 'discover_ais_density_candidates' in line:
        print(f'Line {i}: discover_ais_density_candidates found')
        # 打印前后 20 行
        for j in range(max(0, i-5), min(len(lines), i+30)):
            try:
                print(f'{j+1:4d}: {lines[j][:80]}')
            except:
                print(f'{j+1:4d}: [encoding error]')
        break

