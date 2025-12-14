#!/usr/bin/env python
"""
测试脚本，验证真实数据加载失败时的容错能力。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arcticroute.core.grid import make_demo_grid, load_real_grid_from_nc
from arcticroute.core.landmask import load_real_landmask_from_nc
from arcticroute.core.env_real import load_real_env_for_grid


def test_real_grid_loading():
    """测试真实网格加载的容错能力。"""
    print("\n" + "="*80)
    print("TEST: Real Grid Loading Fallback")
    print("="*80)
    
    # 尝试加载真实网格
    print("\n1. Attempting to load real grid...")
    real_grid = load_real_grid_from_nc()
    
    if real_grid is None:
        print("   ✓ Real grid not available (expected in test environment)")
        print("   ✓ No exception raised - graceful fallback works")
    else:
        print(f"   ✓ Real grid loaded: shape={real_grid.shape()}")
        
        # 尝试加载真实 landmask
        print("\n2. Attempting to load real landmask...")
        land_mask = load_real_landmask_from_nc(real_grid)
        
        if land_mask is None:
            print("   ✓ Real landmask not available")
            print("   ✓ No exception raised - graceful fallback works")
        else:
            print(f"   ✓ Real landmask loaded: shape={land_mask.shape}")
    
    print("\n" + "="*80)
    print("✓ TEST PASSED: Real data loading is gracefully handled")
    print("="*80 + "\n")
    return True


def test_real_env_loading():
    """测试真实环境数据加载的容错能力。"""
    print("\n" + "="*80)
    print("TEST: Real Environment Data Loading Fallback")
    print("="*80)
    
    # 创建 demo 网格用于测试
    grid, _ = make_demo_grid()
    print(f"\n✓ Created demo grid: shape={grid.shape()}")
    
    # 尝试加载真实环境数据
    print("\n1. Attempting to load real environment data...")
    try:
        real_env = load_real_env_for_grid(grid)
        
        if real_env is None:
            print("   ✓ Real environment data not available (expected in test environment)")
            print("   ✓ No exception raised - graceful fallback works")
        else:
            print(f"   ✓ Real environment data loaded:")
            print(f"     - SIC: {real_env.sic is not None}")
            print(f"     - Wave: {real_env.wave_swh is not None}")
            print(f"     - Ice thickness: {real_env.ice_thickness_m is not None}")
    except Exception as e:
        print(f"   ✗ FAILED: Exception raised: {e}")
        return False
    
    print("\n" + "="*80)
    print("✓ TEST PASSED: Real environment loading is gracefully handled")
    print("="*80 + "\n")
    return True


if __name__ == "__main__":
    success = test_real_grid_loading() and test_real_env_loading()
    sys.exit(0 if success else 1)

