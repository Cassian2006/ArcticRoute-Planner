#!/usr/bin/env python
"""
自动化登录脚本，用于生成 Copernicus Marine 凭据文件
"""
import subprocess
import sys
from pathlib import Path

# 凭据信息
username = "caiyuanqi2006@outlook.com"
password = "Asswecan661@"

# 执行登录命令
try:
    print("Starting Copernicus Marine login...")
    process = subprocess.Popen(
        ["copernicusmarine", "login"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # 发送用户名和密码
    stdout, stderr = process.communicate(input=f"{username}\n{password}\n", timeout=30)
    
    print("STDOUT:")
    print(stdout)
    if stderr:
        print("STDERR:")
        print(stderr)
    
    print(f"\nReturn code: {process.returncode}")
    
    # 检查凭据文件
    creds_file = Path.home() / ".copernicusmarine" / "credentials.txt"
    if creds_file.exists():
        print(f"\nSuccess! Credentials file created: {creds_file}")
        print(f"File size: {creds_file.stat().st_size} bytes")
    else:
        print(f"\nWarning: Credentials file not found: {creds_file}")
        
except subprocess.TimeoutExpired:
    print("Login timeout")
    process.kill()
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
