"""
船舶冰级能力的测试。

测试 VesselProfile 中的冰厚参数和相关方法。
"""

from __future__ import annotations

import pytest

from arcticroute.core.eco.vessel_profiles import VesselProfile, get_default_profiles


class TestVesselProfileIceCapability:
    """测试 VesselProfile 的冰级能力。"""

    def test_vessel_profile_has_ice_thickness_fields(self):
        """测试 VesselProfile 包含冰厚相关字段。"""
        profile = VesselProfile(
            key="test",
            name="Test Vessel",
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.8,
            ice_margin_factor=0.9,
        )

        assert hasattr(profile, "max_ice_thickness_m")
        assert hasattr(profile, "ice_margin_factor")
        assert profile.max_ice_thickness_m == 0.8
        assert profile.ice_margin_factor == 0.9

    def test_vessel_profile_default_ice_thickness_values(self):
        """测试 VesselProfile 的冰厚默认值。"""
        profile = VesselProfile(
            key="test",
            name="Test Vessel",
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
        )

        # 应该有默认值
        assert profile.max_ice_thickness_m == 0.7
        assert profile.ice_margin_factor == 0.9

    def test_get_effective_max_ice_thickness(self):
        """测试 get_effective_max_ice_thickness 方法。"""
        profile = VesselProfile(
            key="test",
            name="Test Vessel",
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=1.0,
            ice_margin_factor=0.8,
        )

        effective = profile.get_effective_max_ice_thickness()
        # 应该是 1.0 * 0.8 = 0.8
        assert abs(effective - 0.8) < 1e-6

    def test_get_effective_max_ice_thickness_minimum_bound(self):
        """测试 get_effective_max_ice_thickness 的最小值约束。"""
        profile = VesselProfile(
            key="test",
            name="Test Vessel",
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.001,
            ice_margin_factor=0.5,
        )

        effective = profile.get_effective_max_ice_thickness()
        # 应该至少为 0.01
        assert effective >= 0.01

    def test_default_profiles_have_ice_parameters(self):
        """测试默认船型都有冰厚参数。"""
        profiles = get_default_profiles()

        for key, profile in profiles.items():
            assert profile.max_ice_thickness_m > 0, f"{key} should have positive max_ice_thickness_m"
            assert 0.0 < profile.ice_margin_factor <= 1.0, f"{key} should have valid ice_margin_factor"

    def test_default_profiles_ice_capability_ordering(self):
        """测试默认船型的冰厚能力排序。

        预期：Handy < Panamax < Ice-Class
        """
        profiles = get_default_profiles()

        handy = profiles["handy"]
        panamax = profiles["panamax"]
        ice_class = profiles["ice_class"]

        # 冰厚能力应该按此顺序递增
        assert handy.max_ice_thickness_m < panamax.max_ice_thickness_m
        assert panamax.max_ice_thickness_m < ice_class.max_ice_thickness_m

    def test_default_profiles_effective_ice_thickness(self):
        """测试默认船型的有效冰厚。"""
        profiles = get_default_profiles()

        for key, profile in profiles.items():
            effective = profile.get_effective_max_ice_thickness()
            # 有效冰厚应该小于等于名义冰厚
            assert effective <= profile.max_ice_thickness_m
            # 有效冰厚应该是正数
            assert effective > 0

    def test_handy_ice_capability(self):
        """测试 Handy 船型的冰厚能力。"""
        profiles = get_default_profiles()
        handy = profiles["handy"]

        # Handy 应该是无冰级船，冰厚能力较弱
        assert handy.max_ice_thickness_m == 0.3
        assert handy.ice_margin_factor == 0.85
        effective = handy.get_effective_max_ice_thickness()
        assert abs(effective - 0.3 * 0.85) < 1e-6

    def test_panamax_ice_capability(self):
        """测试 Panamax 船型的冰厚能力。"""
        profiles = get_default_profiles()
        panamax = profiles["panamax"]

        # Panamax 应该是无冰级船，冰厚能力中等
        assert panamax.max_ice_thickness_m == 0.5
        assert panamax.ice_margin_factor == 0.90
        effective = panamax.get_effective_max_ice_thickness()
        assert abs(effective - 0.5 * 0.90) < 1e-6

    def test_ice_class_ice_capability(self):
        """测试 Ice-Class 船型的冰厚能力。"""
        profiles = get_default_profiles()
        ice_class = profiles["ice_class"]

        # Ice-Class 应该是冰级船，冰厚能力最强
        assert ice_class.max_ice_thickness_m == 1.2
        assert ice_class.ice_margin_factor == 0.95
        effective = ice_class.get_effective_max_ice_thickness()
        assert abs(effective - 1.2 * 0.95) < 1e-6

    def test_vessel_profile_ice_parameters_are_positive(self):
        """测试所有船型的冰厚参数都是正数。"""
        profiles = get_default_profiles()

        for key, profile in profiles.items():
            assert profile.max_ice_thickness_m > 0, f"{key}: max_ice_thickness_m should be positive"
            assert profile.ice_margin_factor > 0, f"{key}: ice_margin_factor should be positive"
            assert profile.ice_margin_factor <= 1.0, f"{key}: ice_margin_factor should be <= 1.0"











