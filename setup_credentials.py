#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
创建 Copernicus Marine 凭据文件
"""
from pathlib import Path
import os
import sys

# 设置输出编码
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

username = "caiyuanqi2006@outlook.com"
password = "Asswecan661@"

# 创建凭据目录
cred_dir = Path.home() / '.copernicusmarine'
cred_dir.mkdir(parents=True, exist_ok=True)

# 创建凭据文件
cred_file = cred_dir / 'credentials.txt'

# 写入凭据（格式：username:password）
with open(cred_file, 'w') as f:
    f.write(f"{username}:{password}\n")

print(f"[OK] Credentials file created: {cred_file}")
print(f"[OK] Content: {username}:****")

# 验证文件
if cred_file.exists():
    print(f"[OK] File verification passed")
    with open(cred_file, 'r') as f:
        content = f.read()
        if username in content:
            print(f"[OK] Username saved successfully")
