#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""找到 ais_weights_enabled 的位置"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 找到第一个 ais_weights_enabled 的位置
for i, line in enumerate(lines, start=1):
    if 'ais_weights_enabled = any' in line and i < 1000:
        print(f'Found at line {i}')
        # 打印前后 10 行
        for j in range(max(0, i-5), min(len(lines), i+15)):
            try:
                print(f'{j+1:4d}: {lines[j][:100]}')
            except:
                print(f'{j+1:4d}: [encoding error]')
        break

