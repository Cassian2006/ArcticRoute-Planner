#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""临时脚本：读取 planner_minimal.py 中的 AIS 相关代码"""

with open('arcticroute/ui/planner_minimal.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    
# 查找 AIS density 相关的行
print("=== AIS density related lines ===")
for i, line in enumerate(lines, start=1):
    if 'ais_density' in line.lower():
        # 只打印行号和简化的内容
        print(f'{i:4d}: {line[:80]}')
