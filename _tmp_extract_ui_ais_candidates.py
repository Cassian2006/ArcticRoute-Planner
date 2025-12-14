#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取 planner_minimal.py 中的 AIS 候选项 UI 代码"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找 AIS 候选项部分（大约在 844-960 行）
print("=== AIS 候选项 UI 部分 (844-960) ===")
for i in range(843, min(960, len(lines))):
    print(f'{i+1:4d}: {lines[i]}', end='')

