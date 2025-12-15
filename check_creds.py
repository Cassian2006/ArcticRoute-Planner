#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
检查 Copernicus Marine 凭据
"""
from pathlib import Path
import os

# 检查凭据文件
cred_dir = Path.home() / '.copernicusmarine'
cred_file = cred_dir / 'credentials.txt'

print(f"[*] 凭据目录: {cred_dir}")
print(f"[*] 凭据文件: {cred_file}")
print(f"[*] 目录存在: {cred_dir.exists()}")
print(f"[*] 文件存在: {cred_file.exists()}")

if cred_file.exists():
    with open(cred_file, 'r') as f:
        content = f.read()
    print(f"[*] 文件内容长度: {len(content)} 字符")
    print(f"[*] 文件内容: {content[:100]}...")
    
    # 尝试导入 copernicusmarine 并检查配置
    try:
        from copernicusmarine import CopernicusMarineUserCredentials
        print("[OK] 成功导入 CopernicusMarineUserCredentials")
        
        # 尝试读取凭据
        try:
            creds = CopernicusMarineUserCredentials.load()
            print(f"[OK] 凭据已加载")
            print(f"[*] 用户名: {creds.username}")
        except Exception as e:
            print(f"[ERROR] 加载凭据失败: {e}")
    except ImportError as e:
        print(f"[ERROR] 导入失败: {e}")

