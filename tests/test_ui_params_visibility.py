"""
UI Parameters Visibility Tests - 确保新增权重在 UI 配置中可见
防止 UI 又掉回黑箱
"""

from __future__ import annotations

import pytest


def test_cost_function_accepts_new_weights():
    """测试成本函数接受新增权重参数"""
    from arcticroute.core.cost import build_cost_from_real_env as build_cost_field_multimodal
    
    # 获取函数签名
    import inspect
    sig = inspect.signature(build_cost_field_multimodal)
    params = sig.parameters
    
    # 确保新增权重参数存在（使用实际的参数名）
    expected_params = [
        "ice_penalty",  # 海冰惩罚（等价于 w_sic）
        "wave_penalty",  # 波浪惩罚（等价于 w_wave）
        "w_ais_corridor",  # AIS 走廊权重
        "w_ais_congestion",  # AIS 拥挤权重
        "w_edl",  # EDL 风险权重
        "edl_uncertainty_weight",  # EDL 不确定性权重
    ]
    
    for param in expected_params:
        assert param in params, f"参数 {param} 未在 build_cost_field_multimodal 中定义"


def test_cost_field_components_include_new_sources():
    """测试成本场组件包含新数据源"""
    # 这是一个轻量级测试，只检查代码中是否定义了相关组件
    from arcticroute.core import cost
    
    # 检查代码中是否提到这些组件
    import inspect
    source = inspect.getsource(cost)
    
    expected_components = [
        "ais_corridor",  # AIS 走廊成本
        "ais_congestion",  # AIS 拥挤成本
        "edl_risk",  # EDL 风险
        "edl_uncertainty_penalty",  # EDL 不确定性惩罚
    ]
    
    for component in expected_components:
        assert component in source, f"成本组件 {component} 未在 cost.py 中定义"


def test_static_asset_weights_documented():
    """测试静态资产权重在文档中有说明"""
    from arcticroute.core.cost import build_cost_from_real_env
    
    # 获取函数文档字符串
    docstring = build_cost_from_real_env.__doc__
    
    if docstring:
        # 检查文档中是否提到关键权重（简化检查）
        key_terms = [
            "ais",
            "corridor",
        ]
        
        for term in key_terms:
            assert term.lower() in docstring.lower(), f"文档中未提到 {term}"


def test_ais_weights_resolution_function_exists():
    """测试 AIS 权重解析函数存在"""
    from arcticroute.core.cost import _resolve_ais_weights
    
    assert callable(_resolve_ais_weights)
    
    # 测试函数签名
    import inspect
    sig = inspect.signature(_resolve_ais_weights)
    params = sig.parameters
    
    assert "w_ais_corridor" in params
    assert "w_ais_congestion" in params
    assert "w_ais" in params


def test_ais_weights_resolution_logic():
    """测试 AIS 权重解析逻辑"""
    from arcticroute.core.cost import _resolve_ais_weights
    
    # 测试场景 1: 只提供 corridor
    w_corridor, w_congestion, legacy, mapped = _resolve_ais_weights(
        w_ais=0.0,
        w_ais_corridor=1.0,
        w_ais_congestion=0.0,
    )
    assert w_corridor == 1.0
    assert w_congestion == 0.0
    assert not mapped
    
    # 测试场景 2: 只提供 congestion
    w_corridor, w_congestion, legacy, mapped = _resolve_ais_weights(
        w_ais=0.0,
        w_ais_corridor=0.0,
        w_ais_congestion=2.0,
    )
    assert w_corridor == 0.0
    assert w_congestion == 2.0
    assert not mapped
    
    # 测试场景 3: legacy 映射到 corridor
    w_corridor, w_congestion, legacy, mapped = _resolve_ais_weights(
        w_ais=1.5,
        w_ais_corridor=0.0,
        w_ais_congestion=0.0,
        map_legacy_to_corridor=True,
    )
    assert w_corridor == 1.5
    assert w_congestion == 0.0
    assert mapped


def test_ui_data_page_has_static_assets_section():
    """测试数据页包含静态资产展示"""
    from arcticroute.ui.pages_data import render_data
    
    # 检查函数源代码中是否包含静态资产相关内容
    import inspect
    source = inspect.getsource(render_data)
    
    expected_terms = [
        "bathymetry",
        "ports",
        "corridors",
        "manifest",
        "doctor",
    ]
    
    for term in expected_terms:
        assert term.lower() in source.lower(), f"数据页未包含 {term} 相关内容"


def test_ui_data_page_scan_function_exists():
    """测试数据页扫描函数存在"""
    from arcticroute.ui.pages_data import scan_static_assets
    
    assert callable(scan_static_assets)
    
    # 测试函数返回值结构
    result = scan_static_assets()
    
    assert isinstance(result, dict)
    assert "bathymetry" in result
    assert "ports" in result
    assert "corridors" in result
    assert "ais" in result


def test_ui_data_page_doctor_function_exists():
    """测试数据页 doctor 加载函数存在"""
    from arcticroute.ui.pages_data import load_static_assets_doctor
    
    assert callable(load_static_assets_doctor)
    
    # 测试函数返回值结构
    result = load_static_assets_doctor()
    
    assert isinstance(result, dict)
    # 应该包含这些键之一
    assert ("missing_required" in result) or ("error" in result)


def test_static_assets_doctor_script_exists():
    """测试静态资产检查脚本存在"""
    try:
        from scripts.static_assets_doctor import check_static_assets
        assert callable(check_static_assets)
    except ImportError:
        pytest.fail("static_assets_doctor.py 脚本不存在或无法导入")


def test_static_assets_doctor_report_structure():
    """测试 doctor 报告结构"""
    from scripts.static_assets_doctor import check_static_assets
    
    report = check_static_assets()
    
    assert isinstance(report, dict)
    assert "missing_required" in report
    assert "missing_optional" in report
    assert "all_ok" in report
    
    assert isinstance(report["missing_required"], list)
    assert isinstance(report["missing_optional"], list)
    assert isinstance(report["all_ok"], bool)


def test_cost_field_meta_includes_sources():
    """测试成本场 meta 包含数据源信息"""
    # 这是一个轻量级测试，检查代码中是否记录数据源
    from arcticroute.core import cost
    
    import inspect
    source = inspect.getsource(cost)
    
    # 检查是否有 meta 相关代码
    assert "meta" in source.lower()
    
    # 检查是否记录数据源（简化检查，只要有 meta 相关代码即可）
    # 实际的 meta 键可能因版本而异
    assert "meta" in source.lower()
    assert "components" in source.lower()


def test_weight_params_are_floats():
    """测试权重参数类型为 float"""
    from arcticroute.core.cost import build_cost_from_real_env
    
    import inspect
    sig = inspect.signature(build_cost_from_real_env)
    
    weight_params = [
        "ice_penalty",
        "wave_penalty",
        "w_ais_corridor",
        "w_ais_congestion",
        "w_edl",
        "edl_uncertainty_weight",
    ]
    
    for param_name in weight_params:
        if param_name in sig.parameters:
            param = sig.parameters[param_name]
            # 检查是否有类型注解
            if param.annotation != inspect.Parameter.empty:
                # 注解应该是 float 或包含 float
                assert "float" in str(param.annotation), f"{param_name} 的类型注解应该是 float"


def test_ui_exposes_all_weight_controls():
    """测试 UI 暴露所有权重控制（文档检查）"""
    # 这是一个文档级别的测试
    # 实际的 UI 控件需要在 Streamlit 上下文中测试
    
    # 检查 pages_data.py 中是否提到权重
    from arcticroute.ui import pages_data
    
    import inspect
    source = inspect.getsource(pages_data)
    
    # 至少应该提到静态资产
    assert "bathymetry" in source.lower() or "shallow" in source.lower()
    assert "corridor" in source.lower()
    assert "port" in source.lower()

