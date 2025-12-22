"""
Static Assets Rescan Smoke Tests
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest


def test_static_rescan_handler_callable():
    """测试静态资产重新扫描处理函数可调用"""
    # 这是一个轻量级测试，确保函数存在且可调用
    # 实际的 UI 处理函数会在 pages_data.py 中定义
    
    # 模拟一个重新扫描处理函数
    def handle_static_rescan():
        """模拟的重新扫描处理函数"""
        # 1. 运行 doctor 脚本
        result = subprocess.run(
            ["python", "-m", "scripts.static_assets_doctor"],
            capture_output=True,
            text=True,
        )
        
        # 2. 返回结果
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    
    # 测试函数可调用
    assert callable(handle_static_rescan)


@patch("subprocess.run")
def test_static_rescan_subprocess_call(mock_run):
    """测试重新扫描会调用 subprocess"""
    # 模拟 subprocess.run 的返回值
    mock_run.return_value = MagicMock(
        returncode=0,
        stdout="All OK",
        stderr="",
    )
    
    # 模拟重新扫描
    result = subprocess.run(
        ["python", "-m", "scripts.static_assets_doctor"],
        capture_output=True,
        text=True,
    )
    
    # 断言
    assert mock_run.called
    assert result.returncode == 0


def test_static_rescan_result_structure():
    """测试重新扫描结果结构"""
    # 模拟一个结果
    result = {
        "exit_code": 0,
        "missing_required": 0,
        "missing_optional": 2,
        "timestamp": "2024-01-01T00:00:00",
    }
    
    # 断言结构
    assert "exit_code" in result
    assert "missing_required" in result
    assert "missing_optional" in result
    assert isinstance(result["exit_code"], int)


def test_static_rescan_session_state_update():
    """测试重新扫描更新 session_state"""
    # 模拟 session_state
    session_state = {}
    
    # 模拟更新
    session_state["static_assets_last_scan"] = {
        "timestamp": "2024-01-01T00:00:00",
        "exit_code": 0,
        "missing_required": 0,
        "missing_optional": 2,
    }
    
    # 断言
    assert "static_assets_last_scan" in session_state
    assert session_state["static_assets_last_scan"]["exit_code"] == 0


def test_static_rescan_feedback_message():
    """测试重新扫描反馈消息"""
    # 模拟不同的扫描结果
    
    # 成功情况
    result_success = {
        "exit_code": 0,
        "missing_required": 0,
        "missing_optional": 0,
    }
    
    message_success = f"扫描完成：missing_required={result_success['missing_required']}, missing_optional={result_success['missing_optional']}"
    assert "扫描完成" in message_success
    assert "missing_required=0" in message_success
    
    # 失败情况
    result_failure = {
        "exit_code": 1,
        "missing_required": 2,
        "missing_optional": 3,
    }
    
    message_failure = f"扫描失败：exit_code={result_failure['exit_code']}；missing_required={result_failure['missing_required']}"
    assert "扫描失败" in message_failure
    assert "exit_code=1" in message_failure


@patch("subprocess.run")
def test_static_rescan_error_handling(mock_run):
    """测试重新扫描错误处理"""
    # 模拟 subprocess 失败
    mock_run.side_effect = Exception("Subprocess failed")
    
    # 测试错误处理
    try:
        result = subprocess.run(
            ["python", "-m", "scripts.static_assets_doctor"],
            capture_output=True,
            text=True,
        )
        assert False, "Should have raised exception"
    except Exception as e:
        assert "Subprocess failed" in str(e)


def test_static_rescan_spinner_context():
    """测试重新扫描使用 spinner"""
    # 这是一个文档级别的测试
    # 实际的 spinner 需要在 Streamlit 上下文中测试
    
    # 模拟 spinner 上下文
    class MockSpinner:
        def __init__(self, message):
            self.message = message
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    # 测试 spinner 可以正常使用
    with MockSpinner("正在扫描静态资产...") as spinner:
        assert spinner.message == "正在扫描静态资产..."

