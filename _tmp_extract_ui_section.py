#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""提取 planner_minimal.py 中的 AIS 相关 UI 代码"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 查找 AIS 权重部分（大约在 791-843 行）
print("=== AIS 权重 UI 部分 (791-843) ===")
for i in range(790, min(843, len(lines))):
    print(f'{i+1:4d}: {lines[i]}', end='')


