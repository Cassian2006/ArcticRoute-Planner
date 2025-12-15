#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CMEMS 数据下载管道测试脚本

验证整个闭环的各个步骤是否正常工作。
"""

import json
import sys
from pathlib import Path

def test_describe_files_exist():
    """测试 describe JSON 文件是否存在"""
    print("\n[TEST 1] 检查 describe JSON 文件...")
    
    sic_path = Path("reports/cmems_sic_describe.json")
    wav_path = Path("reports/cmems_wav_describe.json")
    
    if not sic_path.exists():
        print(f"  [FAIL] {sic_path} 不存在")
        return False
    
    if not wav_path.exists():
        print(f"  [FAIL] {wav_path} 不存在")
        return False
    
    print(f"  [OK] {sic_path} 存在 ({sic_path.stat().st_size} bytes)")
    print(f"  [OK] {wav_path} 存在 ({wav_path.stat().st_size} bytes)")
    return True

def test_describe_json_valid():
    """测试 describe JSON 是否有效"""
    print("\n[TEST 2] 验证 describe JSON 格式...")
    
    try:
        sic_path = Path("reports/cmems_sic_describe.json")
        sic = json.loads(sic_path.read_text(encoding="utf-8-sig"))
        print(f"  [OK] cmems_sic_describe.json 有效 (keys: {list(sic.keys())})")
    except Exception as e:
        print(f"  [FAIL] cmems_sic_describe.json 无效: {e}")
        return False
    
    try:
        wav_path = Path("reports/cmems_wav_describe.json")
        wav = json.loads(wav_path.read_text(encoding="utf-8-sig"))
        print(f"  [OK] cmems_wav_describe.json 有效 (keys: {list(wav.keys())})")
    except Exception as e:
        print(f"  [FAIL] cmems_wav_describe.json 无效: {e}")
        return False
    
    return True

def test_resolved_config_exists():
    """测试解析后的配置文件是否存在"""
    print("\n[TEST 3] 检查解析后的配置文件...")
    
    resolved_path = Path("reports/cmems_resolved.json")
    
    if not resolved_path.exists():
        print(f"  [FAIL] {resolved_path} 不存在")
        print("     请先运行: python scripts/cmems_resolve.py")
        return False
    
    print(f"  [OK] {resolved_path} 存在 ({resolved_path.stat().st_size} bytes)")
    return True

def test_resolved_config_valid():
    """测试解析后的配置是否有效"""
    print("\n[TEST 4] 验证解析后的配置...")
    
    try:
        resolved_path = Path("reports/cmems_resolved.json")
        config = json.loads(resolved_path.read_text(encoding="utf-8"))
        
        # 检查 SIC 配置
        if "sic" not in config:
            print("  [FAIL] 配置中缺少 'sic' 键")
            return False
        
        sic = config["sic"]
        if not sic.get("dataset_id"):
            print("  [FAIL] SIC 配置缺少 dataset_id")
            return False
        
        if not sic.get("variables"):
            print("  [FAIL] SIC 配置缺少 variables")
            return False
        
        print(f"  [OK] SIC 配置有效")
        print(f"     - dataset_id: {sic['dataset_id']}")
        print(f"     - variables: {sic['variables']}")
        
        # 检查 WAV 配置
        if "wav" not in config:
            print("  [FAIL] 配置中缺少 'wav' 键")
            return False
        
        wav = config["wav"]
        if not wav.get("dataset_id"):
            print("  [FAIL] WAV 配置缺少 dataset_id")
            return False
        
        if not wav.get("variables"):
            print("  [FAIL] WAV 配置缺少 variables")
            return False
        
        print(f"  [OK] WAV 配置有效")
        print(f"     - dataset_id: {wav['dataset_id']}")
        print(f"     - variables ({len(wav['variables'])} 个): {wav['variables'][:3]}...")
        
        return True
    
    except Exception as e:
        print(f"  [FAIL] 配置无效: {e}")
        return False

def test_output_directory_exists():
    """测试输出目录是否存在"""
    print("\n[TEST 5] 检查输出目录...")
    
    output_dir = Path("data/cmems_cache")
    
    if not output_dir.exists():
        print(f"  [INFO] {output_dir} 不存在，将在下载时创建")
        return True
    
    print(f"  [OK] {output_dir} 存在")
    
    # 检查是否有下载的数据
    sic_file = output_dir / "sic_latest.nc"
    swh_file = output_dir / "swh_latest.nc"
    
    if sic_file.exists():
        print(f"  [OK] {sic_file} 存在 ({sic_file.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"  [INFO] {sic_file} 不存在 (尚未下载)")
    
    if swh_file.exists():
        print(f"  [OK] {swh_file} 存在 ({swh_file.stat().st_size / 1024 / 1024:.2f} MB)")
    else:
        print(f"  [INFO] {swh_file} 不存在 (尚未下载)")
    
    return True

def test_scripts_exist():
    """测试脚本文件是否存在"""
    print("\n[TEST 6] 检查脚本文件...")
    
    scripts = [
        "scripts/cmems_resolve.py",
        "scripts/cmems_download.py",
        "scripts/cmems_download.ps1",
    ]
    
    all_exist = True
    for script in scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"  [OK] {script} 存在")
        else:
            print(f"  [FAIL] {script} 不存在")
            all_exist = False
    
    return all_exist

def test_docs_exist():
    """测试文档文件是否存在"""
    print("\n[TEST 7] 检查文档文件...")
    
    docs = [
        "docs/CMEMS_DOWNLOAD_GUIDE.md",
        "docs/CMEMS_WORKFLOW.md",
        "CMEMS_QUICK_START.md",
    ]
    
    all_exist = True
    for doc in docs:
        doc_path = Path(doc)
        if doc_path.exists():
            print(f"  [OK] {doc} 存在")
        else:
            print(f"  [FAIL] {doc} 不存在")
            all_exist = False
    
    return all_exist

def main():
    """运行所有测试"""
    print("=" * 60)
    print("CMEMS 数据下载管道测试")
    print("=" * 60)
    
    tests = [
        ("describe 文件存在", test_describe_files_exist),
        ("describe JSON 有效", test_describe_json_valid),
        ("解析配置文件存在", test_resolved_config_exists),
        ("解析配置有效", test_resolved_config_valid),
        ("输出目录", test_output_directory_exists),
        ("脚本文件", test_scripts_exist),
        ("文档文件", test_docs_exist),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  [ERROR] 测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n[SUCCESS] 所有测试通过！管道已准备好。")
        print("\n下一步:")
        print("  1. 运行: python scripts/cmems_download.py")
        print("  2. 检查: data/cmems_cache/ 中的数据文件")
        return 0
    else:
        print("\n[WARNING] 部分测试失败。请检查上述错误。")
        print("\n常见问题:")
        print("  - describe 文件不存在: 运行 describe 命令获取元数据")
        print("  - 解析配置不存在: 运行 python scripts/cmems_resolve.py")
        print("  - 脚本文件不存在: 检查 scripts/ 目录")
        return 1

if __name__ == "__main__":
    sys.exit(main())
