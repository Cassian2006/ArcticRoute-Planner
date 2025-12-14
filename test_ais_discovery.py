#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 AIS 密度发现和加载功能
"""

from pathlib import Path
from arcticroute.core.cost import discover_ais_density_candidates, load_ais_density_for_grid

def test_discover():
    """测试 discover_ais_density_candidates() 函数"""
    print("=" * 60)
    print("测试 discover_ais_density_candidates()")
    print("=" * 60)
    
    candidates = discover_ais_density_candidates()
    print(f"\n发现的 AIS 密度文件数量: {len(candidates)}")
    
    for i, cand in enumerate(candidates, 1):
        print(f"\n候选文件 {i}:")
        print(f"  Label: {cand['label']}")
        print(f"  Path:  {cand['path']}")
        print(f"  Path type: {type(cand['path'])}")
        
        # 验证路径是否有效
        p = Path(cand["path"])
        if not p.is_absolute():
            p = Path.cwd() / p
        print(f"  Absolute path: {p}")
        print(f"  Exists: {p.exists()}")
    
    return candidates

def test_load_with_explicit_path(candidates):
    """测试 load_ais_density_for_grid() 使用显式路径"""
    print("\n" + "=" * 60)
    print("测试 load_ais_density_for_grid() 使用显式路径")
    print("=" * 60)
    
    if not candidates:
        print("\n没有发现任何 AIS 密度文件，跳过测试")
        return
    
    # 测试第一个候选文件
    first_cand = candidates[0]
    path_str = first_cand["path"]
    
    print(f"\n尝试加载: {path_str}")
    print(f"路径类型: {type(path_str)}")
    
    try:
        result = load_ais_density_for_grid(explicit_path=path_str)
        if result is not None:
            print(f"✅ 成功加载! 数据形状: {result.shape}")
        else:
            print("⚠️  加载返回 None")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        import traceback
        traceback.print_exc()

def test_load_auto():
    """测试 load_ais_density_for_grid() 自动发现"""
    print("\n" + "=" * 60)
    print("测试 load_ais_density_for_grid() 自动发现")
    print("=" * 60)
    
    print("\n尝试自动发现并加载...")
    try:
        result = load_ais_density_for_grid()
        if result is not None:
            print(f"✅ 成功自动发现并加载! 数据形状: {result.shape}")
        else:
            print("⚠️  自动发现返回 None")
    except Exception as e:
        print(f"❌ 自动发现失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("\n🧪 开始测试 AIS 密度发现和加载功能\n")
    
    candidates = test_discover()
    test_load_with_explicit_path(candidates)
    test_load_auto()
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)









