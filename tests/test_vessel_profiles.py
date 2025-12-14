#!/usr/bin/env python3
"""
船舶参数配置系统的单元测试。

测试覆盖：
  - VesselProfile 数据类
  - 冰级参数映射
  - 工厂函数
  - 工具函数
"""

from __future__ import annotations

import pytest

from arcticroute.core.eco.vessel_profiles import (
    VesselProfile,
    VesselType,
    IceClass,
    ICE_CLASS_PARAMETERS,
    VESSEL_TYPE_PARAMETERS,
    create_vessel_profile,
    get_default_profiles,
    get_profile_by_key,
    list_available_profiles,
    get_ice_class_options,
    get_vessel_type_options,
)


# ============================================================================
# 测试数据类
# ============================================================================

def test_vessel_profile_creation():
    """测试 VesselProfile 数据类的创建。"""
    profile = VesselProfile(
        key="test",
        name="Test Vessel",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.POLAR_PC7,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=1.20,
        ice_margin_factor=0.95,
    )
    
    assert profile.key == "test"
    assert profile.name == "Test Vessel"
    assert profile.dwt == 75000.0
    assert profile.max_ice_thickness_m == 1.20


def test_vessel_profile_ice_class_label():
    """测试 ice_class_label 的自动设置。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.POLAR_PC7,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=1.20,
    )
    
    assert profile.ice_class_label == "Polar Class PC7"


def test_get_effective_max_ice_thickness():
    """测试有效最大冰厚的计算。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.POLAR_PC7,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=1.20,
        ice_margin_factor=0.95,
    )
    
    effective = profile.get_effective_max_ice_thickness()
    assert effective == pytest.approx(1.14, abs=0.01)


def test_get_soft_constraint_threshold():
    """测试软约束阈值的计算。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.POLAR_PC7,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=1.20,
    )
    
    soft_threshold = profile.get_soft_constraint_threshold()
    assert soft_threshold == pytest.approx(0.84, abs=0.01)


def test_get_ice_class_info():
    """测试冰级信息的获取。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.POLAR_PC7,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=1.20,
    )
    
    info = profile.get_ice_class_info()
    assert "label" in info
    assert "description" in info
    assert "standard" in info
    assert info["label"] == "Polar Class PC7"
    assert "IMO Polar Code" in info["standard"]


# ============================================================================
# 测试冰级参数映射
# ============================================================================

def test_ice_class_parameters():
    """测试冰级参数映射的完整性。"""
    assert len(ICE_CLASS_PARAMETERS) == len(IceClass)
    
    for ice_class in IceClass:
        assert ice_class in ICE_CLASS_PARAMETERS
        params = ICE_CLASS_PARAMETERS[ice_class]
        assert "label" in params
        assert "max_ice_thickness_m" in params
        assert "description" in params
        assert "standard" in params


def test_ice_class_thickness_values():
    """测试冰级厚度值的合理性。"""
    # 验证厚度值递增
    thickness_values = [
        ICE_CLASS_PARAMETERS[IceClass.NO_ICE_CLASS]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.FSICR_1C]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.FSICR_1B]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.FSICR_1A]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.FSICR_1A_SUPER]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.POLAR_PC7]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.POLAR_PC6]["max_ice_thickness_m"],
        ICE_CLASS_PARAMETERS[IceClass.POLAR_PC5]["max_ice_thickness_m"],
    ]
    
    for i in range(len(thickness_values) - 1):
        assert thickness_values[i] < thickness_values[i + 1]


# ============================================================================
# 测试业务船型参数映射
# ============================================================================

def test_vessel_type_parameters():
    """测试业务船型参数映射的完整性。"""
    assert len(VESSEL_TYPE_PARAMETERS) == len(VesselType)
    
    for vessel_type in VesselType:
        assert vessel_type in VESSEL_TYPE_PARAMETERS
        params = VESSEL_TYPE_PARAMETERS[vessel_type]
        assert "label" in params
        assert "dwt_range" in params
        assert "design_speed_kn" in params
        assert "base_fuel_per_km" in params
        assert "description" in params


def test_vessel_type_dwt_ranges():
    """测试业务船型 DWT 范围的合理性。"""
    for vessel_type, params in VESSEL_TYPE_PARAMETERS.items():
        dwt_min, dwt_max = params["dwt_range"]
        assert dwt_min > 0
        assert dwt_max > dwt_min


# ============================================================================
# 测试工厂函数
# ============================================================================

def test_create_vessel_profile_with_defaults():
    """测试使用默认参数创建船舶配置。"""
    profile = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.POLAR_PC7,
    )
    
    assert profile.vessel_type == VesselType.PANAMAX
    assert profile.ice_class == IceClass.POLAR_PC7
    assert profile.max_ice_thickness_m == 1.20
    assert profile.dwt > 0
    assert profile.design_speed_kn > 0
    assert profile.base_fuel_per_km > 0


def test_create_vessel_profile_with_custom_params():
    """测试使用自定义参数创建船舶配置。"""
    profile = create_vessel_profile(
        VesselType.HANDYSIZE,
        IceClass.FSICR_1A,
        dwt=35000.0,
        design_speed_kn=12.5,
        base_fuel_per_km=0.040,
        ice_margin_factor=0.85,
    )
    
    assert profile.dwt == 35000.0
    assert profile.design_speed_kn == 12.5
    assert profile.base_fuel_per_km == 0.040
    assert profile.ice_margin_factor == 0.85


def test_create_vessel_profile_key_and_name():
    """测试自动生成的 key 和 name。"""
    profile = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.POLAR_PC7,
    )
    
    assert "panamax" in profile.key.lower()
    assert "polar_pc7" in profile.key.lower()
    assert "Panamax" in profile.name
    assert "PC7" in profile.name


# ============================================================================
# 测试工具函数
# ============================================================================

def test_get_default_profiles():
    """测试获取默认配置。"""
    profiles = get_default_profiles()
    
    assert isinstance(profiles, dict)
    assert len(profiles) > 0
    assert "handy" in profiles
    assert "panamax" in profiles
    assert "ice_class" in profiles
    
    for key, profile in profiles.items():
        assert isinstance(profile, VesselProfile)
        # 注意：profile.key 是由 create_vessel_profile 生成的，
        # 格式为 "vessel_type_ice_class"，可能与字典 key 不同
        assert isinstance(profile.key, str)


def test_get_profile_by_key():
    """测试按 key 获取配置。"""
    profile = get_profile_by_key("panamax")
    
    assert profile is not None
    assert "Panamax" in profile.name
    
    # 不存在的 key
    assert get_profile_by_key("nonexistent") is None


def test_list_available_profiles():
    """测试列出可用配置。"""
    profiles = list_available_profiles()
    
    assert isinstance(profiles, dict)
    assert len(profiles) > 0
    assert "handy" in profiles
    assert "panamax" in profiles
    
    # 验证值是字符串
    for key, name in profiles.items():
        assert isinstance(key, str)
        assert isinstance(name, str)


def test_get_ice_class_options():
    """测试获取冰级选项。"""
    options = get_ice_class_options()
    
    assert isinstance(options, dict)
    assert len(options) == len(IceClass)
    assert "no_ice_class" in options
    assert "fsicr_1a" in options
    assert "polar_pc7" in options


def test_get_vessel_type_options():
    """测试获取业务船型选项。"""
    options = get_vessel_type_options()
    
    assert isinstance(options, dict)
    assert len(options) == len(VesselType)
    assert "handysize" in options
    assert "panamax" in options
    assert "capesize" in options


# ============================================================================
# 集成测试
# ============================================================================

def test_default_profiles_consistency():
    """测试默认配置的一致性。"""
    profiles = get_default_profiles()
    
    for key, profile in profiles.items():
        # 验证基本字段
        assert profile.vessel_type in VesselType
        assert profile.ice_class in IceClass
        assert profile.dwt > 0
        assert profile.design_speed_kn > 0
        assert profile.base_fuel_per_km > 0
        assert 0 < profile.ice_margin_factor <= 1.0
        
        # 验证冰厚参数
        assert profile.max_ice_thickness_m > 0
        assert profile.get_effective_max_ice_thickness() > 0
        assert profile.get_soft_constraint_threshold() > 0
        
        # 验证冰级信息
        info = profile.get_ice_class_info()
        assert info["label"] == profile.ice_class_label


def test_all_combinations_creatable():
    """测试所有业务船型和冰级的组合都可创建。"""
    # 测试几个关键组合
    combinations = [
        (VesselType.HANDYSIZE, IceClass.NO_ICE_CLASS),
        (VesselType.PANAMAX, IceClass.NO_ICE_CLASS),
        (VesselType.HANDYSIZE, IceClass.FSICR_1A),
        (VesselType.PANAMAX, IceClass.POLAR_PC7),
        (VesselType.CAPESIZE, IceClass.POLAR_PC5),
    ]
    
    for vessel_type, ice_class in combinations:
        profile = create_vessel_profile(vessel_type, ice_class)
        assert profile is not None
        assert profile.vessel_type == vessel_type
        assert profile.ice_class == ice_class


def test_ice_margin_factor_effect():
    """测试安全裕度系数的影响。"""
    profile1 = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.POLAR_PC7,
        ice_margin_factor=0.90,
    )
    
    profile2 = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.POLAR_PC7,
        ice_margin_factor=0.80,
    )
    
    # 较小的安全裕度系数应该导致较小的有效冰厚
    assert profile1.get_effective_max_ice_thickness() > profile2.get_effective_max_ice_thickness()


# ============================================================================
# 边界情况测试
# ============================================================================

def test_minimum_effective_ice_thickness():
    """测试最小有效冰厚（至少 0.01m）。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.HANDYSIZE,
        ice_class=IceClass.NO_ICE_CLASS,
        dwt=30000.0,
        design_speed_kn=13.0,
        base_fuel_per_km=0.035,
        max_ice_thickness_m=0.01,
        ice_margin_factor=0.01,  # 非常小的裕度
    )
    
    # 应该至少返回 0.01m
    effective = profile.get_effective_max_ice_thickness()
    assert effective >= 0.01


def test_soft_constraint_threshold_minimum():
    """测试软约束阈值的最小值。"""
    profile = VesselProfile(
        key="test",
        name="Test",
        vessel_type=VesselType.HANDYSIZE,
        ice_class=IceClass.NO_ICE_CLASS,
        dwt=30000.0,
        design_speed_kn=13.0,
        base_fuel_per_km=0.035,
        max_ice_thickness_m=0.25,
    )
    
    soft_threshold = profile.get_soft_constraint_threshold()
    assert soft_threshold > 0
    assert soft_threshold < profile.max_ice_thickness_m


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

