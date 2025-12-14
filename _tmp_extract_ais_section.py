#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取 AIS 候选项部分的代码"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 从第 856 行开始，找到这个部分的结束
start_line = 855  # 0-indexed
end_line = start_line + 150

print(f"=== Lines {start_line+1} to {end_line+1} ===")
for i in range(start_line, min(end_line, len(lines))):
    try:
        print(f'{i+1:4d}: {lines[i]}', end='')
    except:
        print(f'{i+1:4d}: [encoding error]')

