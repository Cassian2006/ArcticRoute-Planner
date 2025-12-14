"""
EDL Miles Smoke Test - 验证 mlguess 是否正确安装并可用。

功能：
  1. 尝试导入 mlguess 和 regression_uq
  2. 打印 mlguess 版本号
  3. 使用 regression_uq 计算不确定性指标
  4. 捕获所有异常并打印，但脚本本身不会因单个函数失败而崩溃

运行方式：
  conda activate ar_edl
  python -m scripts.edl_miles_smoke_test
"""

import sys
import traceback
import numpy as np


def main():
    """主函数：运行 smoke test。"""
    print("[EDL_SMOKE] Starting mlguess smoke test...")
    print()

    # ========== 第一步：导入 mlguess ==========
    try:
        import mlguess
        print(f"[EDL_SMOKE] mlguess imported successfully")
        
        # 尝试获取版本号
        try:
            version = mlguess.__version__
            print(f"[EDL_SMOKE] mlguess version = {version}")
        except AttributeError:
            print("[EDL_SMOKE] mlguess.__version__ not available, trying alternative methods...")
            try:
                import importlib.metadata
                version = importlib.metadata.version("mlguess")
                print(f"[EDL_SMOKE] mlguess version = {version}")
            except Exception as e:
                print(f"[EDL_SMOKE] Could not determine mlguess version: {e}")
    except ImportError as e:
        print(f"[EDL_SMOKE] Failed to import mlguess: {e}")
        traceback.print_exc()
        return False

    print()

    # ========== 第二步：导入 regression_uq ==========
    try:
        from mlguess import regression_uq
        print(f"[EDL_SMOKE] regression_uq imported successfully")
    except ImportError as e:
        print(f"[EDL_SMOKE] Failed to import regression_uq: {e}")
        traceback.print_exc()
        return False

    print()

    # ========== 第三步：构造测试数据 ==========
    print("[EDL_SMOKE] Constructing test data...")
    try:
        np.random.seed(42)  # 固定随机种子以保证可重现性
        
        y_true = np.linspace(0, 1, 100)
        mu = y_true + 0.1 * np.random.randn(100)
        sigma = 0.1 + 0.05 * np.random.rand(100)
        
        print(f"[EDL_SMOKE] y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        print(f"[EDL_SMOKE] mu shape: {mu.shape}, dtype: {mu.dtype}")
        print(f"[EDL_SMOKE] sigma shape: {sigma.shape}, dtype: {sigma.dtype}")
        print(f"[EDL_SMOKE] mu[:5] = {mu[:5]}")
        print(f"[EDL_SMOKE] sigma[:5] = {sigma[:5]}")
    except Exception as e:
        print(f"[EDL_SMOKE] Failed to construct test data: {e}")
        traceback.print_exc()
        return False

    print()

    # ========== 第四步：调用 regression_uq 函数 ==========
    
    # 尝试 compute_coverage
    print("[EDL_SMOKE] Testing regression_uq.compute_coverage()...")
    try:
        if hasattr(regression_uq, 'compute_coverage'):
            result = regression_uq.compute_coverage(y_true, mu, sigma)
            print(f"[EDL_SMOKE] compute_coverage result type: {type(result)}")
            if isinstance(result, np.ndarray):
                print(f"[EDL_SMOKE] compute_coverage result shape: {result.shape}")
                print(f"[EDL_SMOKE] compute_coverage result[:5]: {result[:5]}")
            else:
                print(f"[EDL_SMOKE] compute_coverage result: {result}")
        else:
            print("[EDL_SMOKE] compute_coverage not found in regression_uq")
    except Exception as e:
        print(f"[EDL_SMOKE] compute_coverage failed: {e}")
        traceback.print_exc()

    print()

    # 尝试 calibration
    print("[EDL_SMOKE] Testing regression_uq.calibration()...")
    try:
        if hasattr(regression_uq, 'calibration'):
            result = regression_uq.calibration(y_true, mu, sigma)
            print(f"[EDL_SMOKE] calibration result type: {type(result)}")
            if isinstance(result, np.ndarray):
                print(f"[EDL_SMOKE] calibration result shape: {result.shape}")
                print(f"[EDL_SMOKE] calibration result[:5]: {result[:5]}")
            elif isinstance(result, dict):
                print(f"[EDL_SMOKE] calibration result keys: {result.keys()}")
                for key, val in result.items():
                    if isinstance(val, np.ndarray):
                        print(f"[EDL_SMOKE]   {key}: shape={val.shape}, dtype={val.dtype}")
                    else:
                        print(f"[EDL_SMOKE]   {key}: {val}")
            else:
                print(f"[EDL_SMOKE] calibration result: {result}")
        else:
            print("[EDL_SMOKE] calibration not found in regression_uq")
    except Exception as e:
        print(f"[EDL_SMOKE] calibration failed: {e}")
        traceback.print_exc()

    print()

    # 尝试 prediction_interval
    print("[EDL_SMOKE] Testing regression_uq.prediction_interval()...")
    try:
        if hasattr(regression_uq, 'prediction_interval'):
            result = regression_uq.prediction_interval(mu, sigma, confidence=0.95)
            print(f"[EDL_SMOKE] prediction_interval result type: {type(result)}")
            if isinstance(result, tuple):
                print(f"[EDL_SMOKE] prediction_interval result is tuple with {len(result)} elements")
                for i, elem in enumerate(result):
                    if isinstance(elem, np.ndarray):
                        print(f"[EDL_SMOKE]   element {i}: shape={elem.shape}, dtype={elem.dtype}, first 5: {elem[:5]}")
                    else:
                        print(f"[EDL_SMOKE]   element {i}: {elem}")
            elif isinstance(result, np.ndarray):
                print(f"[EDL_SMOKE] prediction_interval result shape: {result.shape}")
                print(f"[EDL_SMOKE] prediction_interval result[:5]: {result[:5]}")
            else:
                print(f"[EDL_SMOKE] prediction_interval result: {result}")
        else:
            print("[EDL_SMOKE] prediction_interval not found in regression_uq")
    except Exception as e:
        print(f"[EDL_SMOKE] prediction_interval failed: {e}")
        traceback.print_exc()

    print()

    # 列出 regression_uq 中所有可用的函数
    print("[EDL_SMOKE] Available functions in regression_uq:")
    try:
        available_funcs = [name for name in dir(regression_uq) if not name.startswith('_')]
        for func_name in available_funcs:
            print(f"[EDL_SMOKE]   - {func_name}")
    except Exception as e:
        print(f"[EDL_SMOKE] Failed to list functions: {e}")

    print()
    print("[EDL_SMOKE] Smoke test completed!")
    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[EDL_SMOKE] Unexpected error in main: {e}")
        traceback.print_exc()
        sys.exit(1)










