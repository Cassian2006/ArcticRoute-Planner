#!/usr/bin/env python
"""
直接创建 Copernicus Marine 凭据文件
"""
from pathlib import Path
import base64

# 凭据信息
username = "caiyuanqi2006@outlook.com"
password = "Asswecan661@"

# 创建凭据目录
creds_dir = Path.home() / ".copernicusmarine"
creds_dir.mkdir(parents=True, exist_ok=True)

# 创建凭据文件
creds_file = creds_dir / "credentials.txt"

# Copernicus Marine 使用 base64 编码的格式: username:password
credentials = f"{username}:{password}"
encoded = base64.b64encode(credentials.encode()).decode()

# 写入凭据文件
with open(creds_file, 'w') as f:
    f.write(encoded)

print(f"Credentials file created: {creds_file}")
print(f"File size: {creds_file.stat().st_size} bytes")
print(f"Credentials: {encoded[:50]}...")


