#!/usr/bin/env python3
"""
强制 SIC describe 落盘写实 - 使用临时文件+原子替换+校验大小
"""
import subprocess
import json
from pathlib import Path
import time

def run_describe(fields="datasets,variables"):
    """运行 copernicusmarine describe 并返回输出"""
    try:
        result = subprocess.run(
            [
                "copernicusmarine", "describe",
                "--contains", "cmems_mod_arc_phy_anfc_nextsim_hm",
                "--return-fields", fields
            ],
            capture_output=True,
            text=True,
            timeout=120
        )
        return result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return "", "TIMEOUT", -1
    except Exception as e:
        return "", str(e), -1

def main():
    tmp_path = Path("reports/cmems_sic_describe.tmp.json")
    out_path = Path("reports/cmems_sic_describe.json")
    
    print("[Step 1] 尝试仅获取 datasets,variables（降低体积）...")
    stdout, stderr, rc = run_describe("datasets,variables")
    
    if rc != 0:
        print(f"[WARN] 第一次尝试失败: {stderr[:200]}")
        print("[Step 2] 降级到 all fields...")
        stdout, stderr, rc = run_describe("all")
        if rc != 0:
            print(f"[ERROR] 两次尝试都失败: {stderr}")
            return False
    
    # 写入临时文件
    tmp_path.write_text(stdout, encoding="utf-8")
    tmp_size = tmp_path.stat().st_size
    print(f"[OK] 临时文件大小: {tmp_size} 字节")
    
    # 校验大小
    if tmp_size < 1000:
        print(f"[WARN] 文件太小 ({tmp_size} < 1000)，可能没有匹配到结果")
        print("[Step 3] 尝试用 --contains nextsim 探测...")
        stdout2, stderr2, rc2 = run_describe("datasets")
        if rc2 == 0 and len(stdout2) > tmp_size:
            print(f"[INFO] nextsim 探测返回更多数据，使用该结果")
            tmp_path.write_text(stdout2, encoding="utf-8")
            tmp_size = tmp_path.stat().st_size
            print(f"[OK] 更新临时文件大小: {tmp_size} 字节")
    
    # 原子替换
    print(f"[Step 4] 原子替换到 {out_path}...")
    tmp_path.replace(out_path)
    final_size = out_path.stat().st_size
    print(f"[OK] 最终文件大小: {final_size} 字节")
    
    # 打印开头 20 行
    content = out_path.read_text(encoding="utf-8")
    lines = content.split("\n")[:20]
    print("\n[开头 20 行]")
    for i, line in enumerate(lines, 1):
        print(f"{i:2d}: {line}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

