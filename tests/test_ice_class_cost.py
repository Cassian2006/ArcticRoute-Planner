"""
冰级成本约束的测试。

测试冰厚与船舶冰级的软+硬约束逻辑。
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import Grid2D
from arcticroute.core.eco.vessel_profiles import VesselProfile, VesselType, IceClass


class TestIceClassCostConstraints:
    """测试冰级成本约束。"""

    def _make_test_grid(self, ny: int = 5, nx: int = 10) -> tuple[Grid2D, np.ndarray]:
        """创建测试网格和陆地掩码。"""
        lat_1d = np.linspace(65.0, 70.0, ny)
        lon_1d = np.linspace(0.0, 10.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
        land_mask = np.zeros((ny, nx), dtype=bool)
        return grid, land_mask

    def test_ice_class_no_thickness_keeps_cost_unchanged(self):
        """
        测试当 ice_thickness_m 为 None 时，成本不变。

        构造两个成本场：一个不传 vessel_profile，一个传入但 thickness 为 None。
        两者成本应该相同。
        """
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造环境数据（仅 sic，无 thickness）
        sic = np.full((ny, nx), 0.5)
        env_no_thickness = RealEnvLayers(sic=sic, ice_thickness_m=None)

        # 构造船舶配置
        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        # 构建两个成本场
        cost_field_no_vessel = build_cost_from_real_env(
            grid, land_mask, env_no_thickness, ice_penalty=4.0, vessel_profile=None
        )
        cost_field_with_vessel = build_cost_from_real_env(
            grid, land_mask, env_no_thickness, ice_penalty=4.0, vessel_profile=vessel
        )

        # 两者成本应该相同（因为 thickness 为 None）
        assert np.allclose(cost_field_no_vessel.cost, cost_field_with_vessel.cost)

        # 新增的冰级组件不应该出现
        assert "ice_class_soft" not in cost_field_with_vessel.components
        assert "ice_class_hard" not in cost_field_with_vessel.components

    def test_ice_class_hard_forbidden_sets_inf_cost(self):
        """
        测试硬禁区（T > T_max）设置 inf 成本。

        构造一个小网格，其中某些区域的冰厚超过船舶能力。
        """
        ny, nx = 5, 5
        lat_1d = np.linspace(65.0, 69.0, ny)
        lon_1d = np.linspace(0.0, 5.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
        land_mask = np.zeros((ny, nx), dtype=bool)

        # 构造冰厚：上半部分厚冰，下半部分薄冰
        thickness = np.zeros((ny, nx))
        thickness[:2, :] = 1.5  # 上半部分：1.5m（超过限制）
        thickness[2:, :] = 0.3  # 下半部分：0.3m（在限制内）

        env = RealEnvLayers(sic=np.full((ny, nx), 0.5), ice_thickness_m=thickness)

        # 构造船舶配置：T_max = 0.7m，有效 T_max = 0.7 * 0.9 = 0.63m
        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel
        )

        # 上半部分（厚冰）应该是 inf
        assert np.all(np.isinf(cost_field.cost[:2, :]))

        # 下半部分（薄冰）应该是有限值
        assert np.all(np.isfinite(cost_field.cost[2:, :]))

        # ice_class_hard 组件应该存在且标记硬禁区
        assert "ice_class_hard" in cost_field.components
        hard_component = cost_field.components["ice_class_hard"]
        assert np.all(hard_component[:2, :] == 1.0)  # 硬禁区标记为 1
        assert np.all(hard_component[2:, :] == 0.0)  # 其他区域标记为 0

    def test_ice_class_soft_penalty_increases_cost_gradually(self):
        """
        测试软风险区的成本随冰厚单调增加。

        构造一个网格，冰厚从安全区逐渐增加到硬禁区。
        """
        ny, nx = 1, 20
        lat_1d = np.array([65.0])
        lon_1d = np.linspace(0.0, 20.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)
        land_mask = np.zeros((ny, nx), dtype=bool)

        # 冰厚从 0 到 1.0m 线性增加
        thickness = np.linspace(0.0, 1.0, nx).reshape(ny, nx)

        env = RealEnvLayers(sic=np.full((ny, nx), 0.5), ice_thickness_m=thickness)

        # 构造船舶配置：T_max = 0.7m，有效 T_max = 0.7 * 0.9 = 0.63m
        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel, ice_class_soft_weight=3.0
        )

        # 获取有效阈值
        T_max_effective = vessel.get_effective_max_ice_thickness()
        safe_threshold = 0.7 * T_max_effective

        # 找到安全区、软风险区和硬禁区的索引
        safe_idx = thickness[0] <= safe_threshold
        soft_idx = (thickness[0] > safe_threshold) & (thickness[0] <= T_max_effective)
        hard_idx = thickness[0] > T_max_effective

        # 安全区的成本应该不包含 ice_class_soft
        if np.any(safe_idx):
            safe_costs = cost_field.components["ice_class_soft"][0, safe_idx]
            assert np.allclose(safe_costs, 0.0)

        # 软风险区的成本应该单调递增
        if np.any(soft_idx):
            soft_costs = cost_field.components["ice_class_soft"][0, soft_idx]
            # 检查单调性
            for i in range(len(soft_costs) - 1):
                assert soft_costs[i] <= soft_costs[i + 1], "soft costs should be monotonically increasing"

        # 硬禁区的成本应该是 inf
        if np.any(hard_idx):
            hard_costs = cost_field.cost[0, hard_idx]
            assert np.all(np.isinf(hard_costs))

    def test_ice_class_soft_component_exists(self):
        """测试冰级软约束组件存在。"""
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造冰厚数据
        thickness = np.full((ny, nx), 0.5)  # 在软风险区

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel
        )

        # ice_class_soft 应该在 components 中
        assert "ice_class_soft" in cost_field.components
        soft_component = cost_field.components["ice_class_soft"]

        # 形状应该正确
        assert soft_component.shape == (ny, nx)

        # 应该有非零值（因为冰厚在软风险区）
        assert np.any(soft_component > 0)

    def test_ice_class_hard_component_exists(self):
        """测试冰级硬约束组件存在。"""
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造冰厚数据（全部超过限制）
        thickness = np.full((ny, nx), 2.0)

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel
        )

        # ice_class_hard 应该在 components 中
        assert "ice_class_hard" in cost_field.components
        hard_component = cost_field.components["ice_class_hard"]

        # 形状应该正确
        assert hard_component.shape == (ny, nx)

        # 应该全部为 1（因为冰厚全部超过限制）
        assert np.all(hard_component == 1.0)

    def test_ice_class_soft_weight_scaling(self):
        """测试 ice_class_soft_weight 对软约束的影响。"""
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造冰厚数据
        thickness = np.full((ny, nx), 0.5)

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        # 构建两个成本场，不同的 ice_class_soft_weight
        cost_field_low = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel, ice_class_soft_weight=1.0
        )
        cost_field_high = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel, ice_class_soft_weight=5.0
        )

        # 高权重的软约束应该产生更大的成本
        assert np.all(
            cost_field_high.components["ice_class_soft"]
            >= cost_field_low.components["ice_class_soft"]
        )

        # 总成本也应该相应增加
        assert np.all(cost_field_high.cost >= cost_field_low.cost)

    def test_ice_class_respects_land_mask(self):
        """测试冰级约束尊重陆地掩码。"""
        ny, nx = 5, 5
        lat_1d = np.linspace(65.0, 69.0, ny)
        lon_1d = np.linspace(0.0, 5.0, nx)
        lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)
        grid = Grid2D(lat2d=lat2d, lon2d=lon2d)

        # 右侧为陆地
        land_mask = np.zeros((ny, nx), dtype=bool)
        land_mask[:, -2:] = True

        # 冰厚全部超过限制
        thickness = np.full((ny, nx), 2.0)

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        vessel = VesselProfile(
            key="test",
            name="Test Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.FSICR_1A,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.05,
            max_ice_thickness_m=0.7,
            ice_margin_factor=0.9,
        )

        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=vessel
        )

        # 陆地格点应该是 inf
        assert np.all(np.isinf(cost_field.cost[:, -2:]))

        # 海洋格点应该受冰级约束（硬禁区 = inf）
        assert np.all(np.isinf(cost_field.cost[:, :-2]))

    def test_ice_class_with_different_vessels(self):
        """测试不同船型的冰级约束差异。"""
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造冰厚数据
        thickness = np.full((ny, nx), 0.6)

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        # 弱冰级船
        weak_vessel = VesselProfile(
            key="weak",
            name="Weak Vessel",
            vessel_type=VesselType.HANDYSIZE,
            ice_class=IceClass.FSICR_1C,
            dwt=30000.0,
            design_speed_kn=13.0,
            base_fuel_per_km=0.035,
            max_ice_thickness_m=0.3,
            ice_margin_factor=0.85,
        )

        # 强冰级船
        strong_vessel = VesselProfile(
            key="strong",
            name="Strong Vessel",
            vessel_type=VesselType.PANAMAX,
            ice_class=IceClass.POLAR_PC7,
            dwt=50000.0,
            design_speed_kn=12.0,
            base_fuel_per_km=0.060,
            max_ice_thickness_m=1.2,
            ice_margin_factor=0.95,
        )

        cost_field_weak = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=weak_vessel
        )
        cost_field_strong = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=strong_vessel
        )

        # 弱船的成本应该更高（因为冰厚对它来说更危险）
        weak_cost = np.nanmean(cost_field_weak.cost[~land_mask])
        strong_cost = np.nanmean(cost_field_strong.cost[~land_mask])

        assert weak_cost > strong_cost, "Weak vessel should have higher cost in thick ice"

    def test_ice_class_no_vessel_profile_no_constraint(self):
        """测试当 vessel_profile 为 None 时，不应用冰级约束。"""
        grid, land_mask = self._make_test_grid()
        ny, nx = grid.shape()

        # 构造冰厚数据
        thickness = np.full((ny, nx), 2.0)  # 超过任何合理的限制

        env = RealEnvLayers(sic=np.full((ny, nx), 0.3), ice_thickness_m=thickness)

        # 不传 vessel_profile
        cost_field = build_cost_from_real_env(
            grid, land_mask, env, ice_penalty=4.0, vessel_profile=None
        )

        # 冰级约束组件不应该存在
        assert "ice_class_soft" not in cost_field.components
        assert "ice_class_hard" not in cost_field.components

        # 成本应该只受 sic 影响，不受冰厚影响
        # 所有海洋格点应该是有限值
        assert np.all(np.isfinite(cost_field.cost[~land_mask]))













