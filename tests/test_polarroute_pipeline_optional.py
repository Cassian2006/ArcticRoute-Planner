#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PolarRoute Pipeline 可选测试 (Phase 5B)

这些测试仅在以下条件满足时运行：
1. pipeline CLI 可用
2. 设置了 AR_POLAR_PIPELINE_DIR 环境变量

否则自动 skip。

运行方式：
    python -m pytest tests/test_polarroute_pipeline_optional.py -v
"""

import os
import subprocess
import shutil
import pytest
from pathlib import Path

# 检查 pipeline CLI 是否可用
PIPELINE_CLI_AVAILABLE = shutil.which("pipeline") is not None

# 检查环境变量
PIPELINE_DIR = os.getenv("AR_POLAR_PIPELINE_DIR")


@pytest.mark.skipif(
    not PIPELINE_CLI_AVAILABLE,
    reason="pipeline CLI 不可用"
)
class TestPipelineCLI:
    """测试 pipeline CLI 基本功能"""
    
    def test_pipeline_help(self):
        """测试 pipeline --help"""
        result = subprocess.run(
            ["pipeline", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"pipeline --help 失败: {result.stderr}"
    
    def test_pipeline_status_help(self):
        """测试 pipeline status --help"""
        result = subprocess.run(
            ["pipeline", "status", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode == 0, f"pipeline status --help 失败: {result.stderr}"


@pytest.mark.skipif(
    not PIPELINE_CLI_AVAILABLE or not PIPELINE_DIR,
    reason="pipeline CLI 不可用或未设置 AR_POLAR_PIPELINE_DIR"
)
class TestPipelineIntegration:
    """测试 pipeline 集成"""
    
    def test_pipeline_dir_exists(self):
        """测试 pipeline 目录存在"""
        assert Path(PIPELINE_DIR).exists(), f"Pipeline 目录不存在: {PIPELINE_DIR}"
    
    def test_pipeline_status_short(self):
        """测试 pipeline status --short"""
        result = subprocess.run(
            ["pipeline", "status", PIPELINE_DIR, "--short"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"pipeline status --short 失败 (返回码 {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )
    
    def test_find_latest_vessel_mesh(self):
        """测试查找最新的 vessel_mesh.json"""
        from arcticroute.integrations.polarroute_artifacts import find_latest_vessel_mesh
        
        mesh_path = find_latest_vessel_mesh(PIPELINE_DIR)
        
        # 如果找不到，提示用户先执行 pipeline execute
        if not mesh_path:
            pytest.skip(
                f"未找到 vessel_mesh.json 在 {PIPELINE_DIR}。"
                "请先执行 pipeline execute"
            )
        
        assert Path(mesh_path).exists(), f"vessel_mesh.json 不存在: {mesh_path}"
        assert mesh_path.endswith(".json"), f"文件不是 JSON: {mesh_path}"


@pytest.mark.skipif(
    not PIPELINE_CLI_AVAILABLE,
    reason="pipeline CLI 不可用"
)
class TestPipelineDoctor:
    """测试 pipeline 医生脚本"""
    
    def test_doctor_script_exists(self):
        """测试医生脚本存在"""
        doctor_script = Path("scripts/polarroute_pipeline_doctor.py")
        assert doctor_script.exists(), f"医生脚本不存在: {doctor_script}"
    
    def test_doctor_script_basic(self):
        """测试医生脚本基本运行"""
        result = subprocess.run(
            ["python", "-m", "scripts.polarroute_pipeline_doctor"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"医生脚本失败 (返回码 {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )


@pytest.mark.skipif(
    not PIPELINE_CLI_AVAILABLE,
    reason="pipeline CLI 不可用"
)
class TestPipelineIntegrationModule:
    """测试 polarroute_pipeline 集成模块"""
    
    def test_import_pipeline_module(self):
        """测试导入 polarroute_pipeline 模块"""
        from arcticroute.integrations import polarroute_pipeline
        assert polarroute_pipeline is not None
    
    def test_import_artifacts_module(self):
        """测试导入 polarroute_artifacts 模块"""
        from arcticroute.integrations import polarroute_artifacts
        assert polarroute_artifacts is not None
    
    def test_pipeline_functions_exist(self):
        """测试 pipeline 函数存在"""
        from arcticroute.integrations.polarroute_pipeline import (
            pipeline_build,
            pipeline_status,
            pipeline_execute,
            pipeline_reset,
            pipeline_halt,
        )
        
        assert callable(pipeline_build)
        assert callable(pipeline_status)
        assert callable(pipeline_execute)
        assert callable(pipeline_reset)
        assert callable(pipeline_halt)
    
    def test_artifacts_functions_exist(self):
        """测试 artifacts 函数存在"""
        from arcticroute.integrations.polarroute_artifacts import (
            find_latest_vessel_mesh,
            find_latest_route_json,
            find_latest_route_config,
        )
        
        assert callable(find_latest_vessel_mesh)
        assert callable(find_latest_route_json)
        assert callable(find_latest_route_config)


@pytest.mark.skipif(
    not PIPELINE_CLI_AVAILABLE,
    reason="pipeline CLI 不可用"
)
class TestPolarRouteBackendPipelineMode:
    """测试 PolarRouteBackend 的 pipeline_dir 模式"""
    
    def test_polarroute_backend_import(self):
        """测试导入 PolarRouteBackend"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        assert PolarRouteBackend is not None
    
    def test_polarroute_backend_external_mode(self):
        """测试 PolarRouteBackend 外部文件模式（Phase 5A）"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        from arcticroute.core.planners.base import PlannerBackendError
        
        # 使用不存在的文件测试错误处理
        with pytest.raises(PlannerBackendError):
            PolarRouteBackend(
                vessel_mesh_path="/nonexistent/vessel_mesh.json",
                route_config_path="/nonexistent/route_config.json",
            )
    
    @pytest.mark.skipif(
        not PIPELINE_DIR,
        reason="未设置 AR_POLAR_PIPELINE_DIR"
    )
    def test_polarroute_backend_pipeline_mode(self):
        """测试 PolarRouteBackend pipeline_dir 模式（Phase 5B）"""
        from arcticroute.core.planners.polarroute_backend import PolarRouteBackend
        from arcticroute.core.planners.base import PlannerBackendError
        
        # 如果 pipeline 目录不存在，应该抛出错误
        if not Path(PIPELINE_DIR).exists():
            with pytest.raises(PlannerBackendError):
                PolarRouteBackend(pipeline_dir=PIPELINE_DIR)
        else:
            # 如果目录存在但没有 vessel_mesh，也应该抛出错误
            try:
                backend = PolarRouteBackend(pipeline_dir=PIPELINE_DIR)
                # 如果初始化成功，说明找到了 vessel_mesh
                assert backend.vessel_mesh_path is not None
                assert backend.route_config_path is not None
            except PlannerBackendError as e:
                # 预期的错误：未找到 vessel_mesh 或 route_config
                assert "未找到" in str(e) or "不存在" in str(e)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

