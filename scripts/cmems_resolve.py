"""
CMEMS 策略解析脚本：nextsim 可用则优先，否则回退到 L4 观测。

输出：
- 可用产品列表
- 选择的产品（nextsim 或 L4）
- 回退原因（如果适用）
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def check_nextsim_available() -> tuple[bool, str]:
    """
    检查 nextsim 产品是否可用。
    
    Returns:
        (is_available, reason)
    """
    try:
        result = subprocess.run(
            ["copernicusmarine", "describe", "--contains", "nextsim"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return False, f"Command failed with exit code {result.returncode}"
        
        output = result.stdout + result.stderr
        
        if "cmems_mod_arc_phy_anfc_nextsim_hm" in output:
            return True, "nextsim product found"
        else:
            return False, "nextsim product not found in catalog"
            
    except subprocess.TimeoutExpired:
        return False, "Command timeout"
    except FileNotFoundError:
        return False, "copernicusmarine CLI not found"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def check_l4_available() -> tuple[bool, str]:
    """
    检查 L4 观测产品是否可用。
    
    Returns:
        (is_available, reason)
    """
    try:
        result = subprocess.run(
            ["copernicusmarine", "describe", "--contains", "siconc", "--contains", "ARCTIC"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        if result.returncode != 0:
            return False, f"Command failed with exit code {result.returncode}"
        
        output = result.stdout + result.stderr
        
        if "ARCTIC_ANALYSISFORECAST_PHY_ICE" in output or "siconc" in output:
            return True, "L4 product found"
        else:
            return False, "L4 product not found in catalog"
            
    except subprocess.TimeoutExpired:
        return False, "Command timeout"
    except FileNotFoundError:
        return False, "copernicusmarine CLI not found"
    except Exception as e:
        return False, f"Unexpected error: {e}"


def main() -> None:
    """主函数：解析 CMEMS 策略。"""
    print("=" * 60)
    print("CMEMS Strategy Resolution")
    print("=" * 60)
    
    # 检查 nextsim
    print("\n[1/2] Checking nextsim availability...")
    nextsim_available, nextsim_reason = check_nextsim_available()
    print(f"  Status: {'[OK] AVAILABLE' if nextsim_available else '[X] NOT AVAILABLE'}")
    print(f"  Reason: {nextsim_reason}")
    
    # 检查 L4
    print("\n[2/2] Checking L4 product availability...")
    l4_available, l4_reason = check_l4_available()
    print(f"  Status: {'[OK] AVAILABLE' if l4_available else '[X] NOT AVAILABLE'}")
    print(f"  Reason: {l4_reason}")
    
    # 决策
    print("\n" + "=" * 60)
    print("RESOLUTION:")
    print("=" * 60)
    
    if nextsim_available:
        print("[OK] Using nextsim HM (high-resolution model)")
        selected = "nextsim"
        fallback = None
    elif l4_available:
        print("[WARN] nextsim not available, falling back to L4 observations")
        print(f"  Fallback reason: {nextsim_reason}")
        selected = "L4"
        fallback = nextsim_reason
    else:
        print("[ERROR] Neither nextsim nor L4 products are available")
        print(f"  nextsim: {nextsim_reason}")
        print(f"  L4: {l4_reason}")
        selected = None
        fallback = f"nextsim: {nextsim_reason}, L4: {l4_reason}"
    
    # 保存结果
    result = {
        "nextsim_available": nextsim_available,
        "nextsim_reason": nextsim_reason,
        "l4_available": l4_available,
        "l4_reason": l4_reason,
        "selected": selected,
        "fallback_reason": fallback,
    }
    
    output_path = Path("reports/cmems_strategy.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    
    print(f"\n[OK] Strategy saved to: {output_path}")
    
    if selected is None:
        sys.exit(1)


if __name__ == "__main__":
    main()

