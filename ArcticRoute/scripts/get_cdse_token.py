"""Obtain/refresh auth token for Copernicus/CDSE access.

@role: pipeline
"""

import requests

# === 替换为你的 CDSE 登录信息 ===
u = "caiyuanqi2006@outlook.com"
p = "Asswecan661@"

url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

resp = requests.post(url, data={
    "client_id": "cdse-public",
    "grant_type": "password",
    "username": u,
    "password": p
})

if resp.ok:
    token = resp.json().get("access_token")
    print("\nAccess Token:\n", token)
else:
    print("\n❌ 获取失败:", resp.status_code, resp.text)
