#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute 可用性探测脚本 (Phase 5A)

功能：
1. 检测 import polar_route
2. 检测 CLI：optimise_routes --help
3. 打印版本/路径（能定位装到哪个 venv）

使用：
    python -m scripts.polarroute_doctor
"""

import sys
import subprocess
import shutil
from pathlib import Path
import io

# 设置 stdout 编码为 UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def check_import() -> bool:
    """检测 polar_route 包是否可导入。"""
    print("=" * 70)
    print("1. 检测 polar_route 包导入...")
    print("=" * 70)
    try:
        import polar_route
        print(f"✓ polar_route 导入成功")
        print(f"  位置: {polar_route.__file__}")
        if hasattr(polar_route, "__version__"):
            print(f"  版本: {polar_route.__version__}")
        return True
    except ImportError as e:
        print(f"✗ polar_route 导入失败: {e}")
        return False


def check_cli() -> bool:
    """检测 optimise_routes CLI 是否可用。"""
    print("\n" + "=" * 70)
    print("2. 检测 optimise_routes CLI...")
    print("=" * 70)
    
    # 查找 optimise_routes 命令（包括 .exe 后缀）
    optimise_routes_path = shutil.which("optimise_routes")
    if not optimise_routes_path:
        # 在 Windows 上尝试查找 .exe 版本
        optimise_routes_path = shutil.which("optimise_routes.exe")
    
    # 如果仍未找到，尝试通过 Python 模块调用
    if not optimise_routes_path:
        try:
            from polar_route.cli import optimise_routes_cli
            print(f"✓ optimise_routes 可通过 Python 模块调用")
            print(f"  模块: polar_route.cli.optimise_routes_cli")
            return True
        except ImportError:
            print(f"✗ optimise_routes 命令未找到")
            print(f"  提示：在 Windows 上，请确保 .venv\\Scripts 在 PATH 中")
            return False
    
    print(f"✓ optimise_routes 命令找到")
    print(f"  位置: {optimise_routes_path}")
    
    # 尝试运行 --help
    try:
        result = subprocess.run(
            ["optimise_routes", "--help"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ optimise_routes --help 执行成功")
            print(f"\n{result.stdout}")
            return True
        else:
            print(f"✗ optimise_routes --help 返回码: {result.returncode}")
            if result.stderr:
                print(f"  错误: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"✗ optimise_routes --help 超时")
        return False
    except Exception as e:
        print(f"✗ optimise_routes --help 执行失败: {e}")
        return False


def check_other_commands() -> None:
    """检测其他 PolarRoute CLI 命令。"""
    print("\n" + "=" * 70)
    print("3. 检测其他 PolarRoute 命令...")
    print("=" * 70)
    
    commands = ["add_vehicle", "calculate_route"]
    for cmd in commands:
        cmd_path = shutil.which(cmd)
        if cmd_path:
            print(f"✓ {cmd} 命令找到: {cmd_path}")
        else:
            print(f"✗ {cmd} 命令未找到")


def check_python_version() -> None:
    """检测 Python 版本。"""
    print("\n" + "=" * 70)
    print("4. Python 环境信息")
    print("=" * 70)
    print(f"Python 版本: {sys.version}")
    print(f"Python 可执行文件: {sys.executable}")
    print(f"sys.prefix (venv): {sys.prefix}")


def main():
    """主函数。"""
    print("\n" + "=" * 70)
    print("PolarRoute 可用性探测 (Doctor Script)")
    print("=" * 70)
    
    # 检查 import
    import_ok = check_import()
    
    # 检查 CLI
    cli_ok = check_cli()
    
    # 检查其他命令
    check_other_commands()
    
    # 检查 Python 环境
    check_python_version()
    
    # 总结
    print("\n" + "=" * 70)
    print("诊断总结")
    print("=" * 70)
    if import_ok and cli_ok:
        print("✓ PolarRoute 已正确安装并可用")
        print("  可以继续进行 Phase 5A 集成")
        return 0
    else:
        print("✗ PolarRoute 未完全可用")
        if not import_ok:
            print("  - polar_route 包未安装")
        if not cli_ok:
            print("  - optimise_routes CLI 不可用")
        print("\n建议：")
        print("  1. 安装 PolarRoute: pip install polar-route")
        print("  2. 验证 venv 激活: source .venv/bin/activate (Linux/Mac)")
        print("                    或 .venv\\Scripts\\Activate.ps1 (Windows)")
        return 1


if __name__ == "__main__":
    sys.exit(main())

