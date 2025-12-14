#!/usr/bin/env python
"""
AIS Phase 1 验证脚本

验证所有 AIS Phase 1 的组件是否正确安装和工作。
"""

import sys
from pathlib import Path


def check_files():
    """检查所有必需的文件是否存在。"""
    print("\n" + "=" * 70)
    print("检查文件...")
    print("=" * 70)
    
    files_to_check = [
        # 核心代码
        "arcticroute/core/ais_ingest.py",
        "arcticroute/core/cost.py",
        "arcticroute/ui/planner_minimal.py",
        
        # 测试文件
        "tests/test_ais_ingest_schema.py",
        "tests/test_ais_density_rasterize.py",
        "tests/test_cost_with_ais_density.py",
        "tests/test_ais_phase1_integration.py",
        
        # 数据文件
        "tests/data/ais_sample.csv",
        "data_real/ais/raw/ais_2024_sample.csv",
        
        # 文档文件
        "AIS_PHASE1_IMPLEMENTATION_SUMMARY.md",
        "AIS_PHASE1_QUICK_START.md",
        "AIS_PHASE1_VERIFICATION_REPORT.md",
        "AIS_PHASE1_中文总结.md",
        "AIS_PHASE1_INDEX.md",
    ]
    
    all_exist = True
    for file_path in files_to_check:
        exists = Path(file_path).exists()
        status = "✅" if exists else "❌"
        print(f"{status} {file_path}")
        if not exists:
            all_exist = False
    
    return all_exist


def check_imports():
    """检查所有必需的模块是否可以导入。"""
    print("\n" + "=" * 70)
    print("检查导入...")
    print("=" * 70)
    
    imports_to_check = [
        ("arcticroute.core.ais_ingest", "AISSchemaSummary"),
        ("arcticroute.core.ais_ingest", "inspect_ais_csv"),
        ("arcticroute.core.ais_ingest", "rasterize_ais_density_to_grid"),
        ("arcticroute.core.ais_ingest", "AISDensityResult"),
        ("arcticroute.core.ais_ingest", "build_ais_density_for_grid"),
        ("arcticroute.core.cost", "build_cost_from_real_env"),
    ]
    
    all_imported = True
    for module_name, class_or_func in imports_to_check:
        try:
            module = __import__(module_name, fromlist=[class_or_func])
            getattr(module, class_or_func)
            print(f"✅ {module_name}.{class_or_func}")
        except Exception as e:
            print(f"❌ {module_name}.{class_or_func}: {e}")
            all_imported = False
    
    return all_imported


def check_ais_data():
    """检查 AIS 数据文件的内容。"""
    print("\n" + "=" * 70)
    print("检查 AIS 数据...")
    print("=" * 70)
    
    try:
        import pandas as pd
        
        # 检查测试数据
        test_csv = "tests/data/ais_sample.csv"
        if Path(test_csv).exists():
            df = pd.read_csv(test_csv)
            print(f"✅ 测试数据: {test_csv}")
            print(f"   - 行数: {len(df)}")
            print(f"   - 列: {', '.join(df.columns)}")
        else:
            print(f"❌ 测试数据不存在: {test_csv}")
            return False
        
        # 检查真实数据
        real_csv = "data_real/ais/raw/ais_2024_sample.csv"
        if Path(real_csv).exists():
            df = pd.read_csv(real_csv)
            print(f"✅ 真实数据: {real_csv}")
            print(f"   - 行数: {len(df)}")
            print(f"   - 列: {', '.join(df.columns)}")
        else:
            print(f"❌ 真实数据不存在: {real_csv}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ 检查 AIS 数据失败: {e}")
        return False


def run_tests():
    """运行所有 AIS Phase 1 测试。"""
    print("\n" + "=" * 70)
    print("运行测试...")
    print("=" * 70)
    
    try:
        import subprocess
        
        test_files = [
            "tests/test_ais_ingest_schema.py",
            "tests/test_ais_density_rasterize.py",
            "tests/test_cost_with_ais_density.py",
            "tests/test_ais_phase1_integration.py",
        ]
        
        cmd = ["python", "-m", "pytest"] + test_files + ["-v", "--tb=short"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 所有测试通过")
            # 提取测试统计
            for line in result.stdout.split("\n"):
                if "passed" in line:
                    print(f"   {line.strip()}")
            return True
        else:
            print("❌ 测试失败")
            print(result.stdout)
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ 运行测试失败: {e}")
        return False


def check_api():
    """检查 API 是否正常工作。"""
    print("\n" + "=" * 70)
    print("检查 API...")
    print("=" * 70)
    
    try:
        from arcticroute.core.ais_ingest import inspect_ais_csv, build_ais_density_for_grid
        from arcticroute.core.grid import make_demo_grid
        
        # 测试 schema 探测
        summary = inspect_ais_csv("data_real/ais/raw/ais_2024_sample.csv")
        if summary.num_rows > 0:
            print(f"✅ inspect_ais_csv() 工作正常")
            print(f"   - 数据行数: {summary.num_rows}")
            print(f"   - 纬度范围: {summary.lat_min:.1f} ~ {summary.lat_max:.1f}")
            print(f"   - 经度范围: {summary.lon_min:.1f} ~ {summary.lon_max:.1f}")
        else:
            print(f"❌ inspect_ais_csv() 返回空数据")
            return False
        
        # 测试栅格化
        grid, _ = make_demo_grid(ny=20, nx=20)
        ais_result = build_ais_density_for_grid(
            "data_real/ais/raw/ais_2024_sample.csv",
            grid.lat2d, grid.lon2d,
            max_rows=1000
        )
        if ais_result.num_binned > 0:
            print(f"✅ build_ais_density_for_grid() 工作正常")
            print(f"   - 有效点数: {ais_result.num_binned}/{ais_result.num_points}")
            print(f"   - 密度场形状: {ais_result.da.shape}")
            print(f"   - 最大密度: {ais_result.da.max().values:.3f}")
        else:
            print(f"❌ build_ais_density_for_grid() 返回空结果")
            return False
        
        return True
    except Exception as e:
        print(f"❌ API 检查失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主验证函数。"""
    print("\n" + "=" * 70)
    print("AIS Phase 1 验证脚本")
    print("=" * 70)
    
    results = {
        "文件检查": check_files(),
        "导入检查": check_imports(),
        "数据检查": check_ais_data(),
        "API 检查": check_api(),
        "测试运行": run_tests(),
    }
    
    # 打印总结
    print("\n" + "=" * 70)
    print("验证总结")
    print("=" * 70)
    
    all_passed = True
    for check_name, result in results.items():
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ AIS Phase 1 验证完全通过！")
        print("系统已准备好进入生产环境。")
        return 0
    else:
        print("❌ AIS Phase 1 验证失败！")
        print("请检查上面的错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())






