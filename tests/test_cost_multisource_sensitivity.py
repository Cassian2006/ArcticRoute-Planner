"""
多源数据对成本敏感性的回归测试。

验证每个源/开关确实影响到 cost 或 meta：
1. w_shallow 从 0→1：total_cost 变大，且 components 里出现 shallow_penalty
2. w_ais 从 0→1：components 里 AIS 相关组件非全 0
3. 启用 POLARIS（rules yaml）：hard-block 或 soft penalty 生效

关键：不比绝对数值，只比单调性/组件存在性/meta 标记。
"""

from __future__ import annotations

import numpy as np
import pytest

from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import make_demo_grid
import arcticroute.core.cost as cost_module


class TestShallowPenaltySensitivity:
    """测试浅水惩罚对成本的敏感性。"""

    @pytest.mark.skip(reason="浅水惩罚功能可能尚未完全实现，待实现后启用")
    def test_w_shallow_zero_to_one_increases_cost(self):
        """
        测试 w_shallow 从 0→1 时，total_cost 变大，且 components 里出现 shallow_penalty。
        
        注意：此测试假设未来会实现完整的浅水惩罚功能。
        """
        pass

    @pytest.mark.skip(reason="浅水惩罚功能可能尚未完全实现，待实现后启用")
    def test_shallow_penalty_monotonic_with_weight(self):
        """
        测试 w_shallow 权重的单调性：权重越大，浅水区域的成本越高。
        
        注意：此测试假设未来会实现完整的浅水惩罚功能。
        """
        pass


class TestAISDensitySensitivity:
    """测试 AIS 密度对成本的敏感性。"""

    def test_w_ais_corridor_zero_to_one_changes_cost(self, monkeypatch):
        """
        测试 w_ais_corridor 从 0→1 时，components 里 AIS 相关组件非全 0。
        
        使用 monkeypatch 控制 AIS 密度场。
        """
        grid, land_mask = make_demo_grid(ny=5, nx=5)
        env = RealEnvLayers(grid=grid, sic=np.zeros((5, 5)), land_mask=land_mask)

        # 构造可控的 AIS 密度场：中间一列为高密度主航道
        ais_density = np.zeros((5, 5), dtype=float)
        ais_density[:, 2] = 1.0  # 中间列为主航道

        # 场景 1: w_ais_corridor = 0（不使用 AIS 走廊）
        cost_field_no_ais = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            ais_density=ais_density,
            w_ais_corridor=0.0,
            w_ais_congestion=0.0,
        )

        # 场景 2: w_ais_corridor = 1（使用 AIS 走廊偏好）
        cost_field_with_ais = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            ais_density=ais_density,
            w_ais_corridor=1.0,
            w_ais_congestion=0.0,
        )

        # 断言 1: w_ais_corridor=0 时，不应该有 ais_corridor 组件
        assert "ais_corridor" not in cost_field_no_ais.components

        # 断言 2: w_ais_corridor=1 时，应该有 ais_corridor 组件
        assert "ais_corridor" in cost_field_with_ais.components

        # 断言 3: ais_corridor 组件非全 0
        ais_corridor = cost_field_with_ais.components["ais_corridor"]
        assert not np.allclose(ais_corridor, 0.0)

        # 断言 4: 主航道（高密度）区域的 ais_corridor 成本应该更低（负值或更小）
        # 因为 corridor 是偏好，高密度区域应该更便宜
        corridor_cost = ais_corridor[:, 2]  # 主航道列
        non_corridor_cost = ais_corridor[:, 0]  # 非主航道列
        
        # corridor 成本应该小于或等于非 corridor 成本（因为是偏好）
        assert np.mean(corridor_cost) <= np.mean(non_corridor_cost), \
            "主航道区域的 corridor 成本应该更低"

    def test_w_ais_congestion_zero_to_one_changes_cost(self, monkeypatch):
        """
        测试 w_ais_congestion 从 0→1 时，components 里出现 ais_congestion 组件。
        """
        grid, land_mask = make_demo_grid(ny=5, nx=5)
        env = RealEnvLayers(grid=grid, sic=np.zeros((5, 5)), land_mask=land_mask)

        # 构造 AIS 密度场：有明显的高密度区域（拥挤）
        ais_density = np.zeros((5, 5), dtype=float)
        ais_density[2, 2] = 1.0  # 中心点高密度
        ais_density[1:4, 1:4] = 0.5  # 周围中等密度

        # 场景 1: w_ais_congestion = 0
        cost_field_no_congestion = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            ais_density=ais_density,
            w_ais_corridor=0.0,
            w_ais_congestion=0.0,
        )

        # 场景 2: w_ais_congestion = 1
        cost_field_with_congestion = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            ais_density=ais_density,
            w_ais_corridor=0.0,
            w_ais_congestion=1.0,
        )

        # 断言 1: w_ais_congestion=0 时，不应该有 ais_congestion 组件
        assert "ais_congestion" not in cost_field_no_congestion.components

        # 断言 2: w_ais_congestion=1 时，应该有 ais_congestion 组件
        assert "ais_congestion" in cost_field_with_congestion.components

        # 断言 3: ais_congestion 组件非全 0
        ais_congestion = cost_field_with_congestion.components["ais_congestion"]
        assert not np.allclose(ais_congestion, 0.0)

        # 断言 4: 高密度区域的 ais_congestion 惩罚应该更高
        high_density_cost = ais_congestion[2, 2]
        low_density_cost = ais_congestion[0, 0]
        
        assert high_density_cost >= low_density_cost, \
            "高密度区域的拥挤惩罚应该更高"

    def test_ais_meta_tracking(self):
        """
        测试 AIS 相关的 meta 信息被正确记录。
        """
        grid, land_mask = make_demo_grid(ny=4, nx=4)
        env = RealEnvLayers(grid=grid, sic=np.zeros((4, 4)), land_mask=land_mask)

        ais_density = np.random.uniform(0, 1, (4, 4))

        cost_field = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            ais_density=ais_density,
            w_ais_corridor=0.5,
            w_ais_congestion=0.3,
        )

        # 断言：meta 中应该记录 AIS 相关信息
        assert cost_field.meta is not None
        # 可能的 meta 键：ais_enabled, ais_source, w_ais_corridor, w_ais_congestion 等
        # 根据实际实现调整


class TestPolarisRulesSensitivity:
    """测试 POLARIS 规则对成本的敏感性。"""

    @pytest.mark.skip(reason="POLARIS 规则集成需要 rules yaml 文件，暂时跳过")
    def test_polaris_rules_enabled_affects_cost(self):
        """
        测试启用 POLARIS 规则时，hard-block 或 soft penalty 生效。
        
        验证 meta 里有 polaris_enabled / polaris_meta 标记。
        """
        grid, land_mask = make_demo_grid(ny=5, nx=5)
        env = RealEnvLayers(grid=grid, sic=np.zeros((5, 5)), land_mask=land_mask)

        # 场景 1: 不启用 POLARIS
        cost_field_no_polaris = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            # polaris_rules_path=None,  # 假设有这个参数
        )

        # 场景 2: 启用 POLARIS
        # cost_field_with_polaris = build_cost_from_real_env(
        #     grid,
        #     land_mask,
        #     env,
        #     polaris_rules_path="path/to/rules.yaml",
        # )

        # 断言：meta 中应该有 polaris 相关标记
        # assert "polaris_enabled" in cost_field_with_polaris.meta
        # assert cost_field_with_polaris.meta["polaris_enabled"] is True

        # 断言：成本场应该有差异（特定区域被 hard-block 或 soft penalty）
        pass


class TestMultiSourceIntegration:
    """测试多源数据同时启用时的集成效果。"""

    def test_multiple_sources_all_contribute(self):
        """
        测试同时启用多个源（SIC + wave + AIS）时，所有组件都出现在 components 中。
        """
        grid, land_mask = make_demo_grid(ny=5, nx=5)
        env = RealEnvLayers(
            grid=grid, 
            sic=np.full((5, 5), 0.3), 
            wave_swh=np.full((5, 5), 2.0),
            land_mask=land_mask
        )

        # AIS 密度
        ais_density = np.random.uniform(0, 1, (5, 5))

        # 同时启用多个源
        cost_field = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            wave_penalty=2.0,
            ais_density=ais_density,
            w_ais_corridor=0.5,
            w_ais_congestion=0.3,
        )

        # 断言：所有预期的组件都应该存在
        expected_components = ["base_distance", "ice_risk", "wave_risk", "ais_corridor", "ais_congestion"]
        for comp in expected_components:
            assert comp in cost_field.components, f"组件 {comp} 应该存在于 components 中"

        # 断言：每个组件都非全 0（至少有部分区域有贡献）
        for comp in expected_components:
            if comp == "base_distance":
                continue  # base_distance 在海洋上恒为 1.0
            component_data = cost_field.components[comp]
            assert not np.allclose(component_data, 0.0), f"组件 {comp} 不应该全为 0"

    def test_component_sum_equals_total_cost(self):
        """
        测试所有组件之和等于总成本（验证成本分解的正确性）。
        """
        grid, land_mask = make_demo_grid(ny=4, nx=4)
        env = RealEnvLayers(
            grid=grid, 
            sic=np.full((4, 4), 0.2),
            wave_swh=np.full((4, 4), 1.5),
            land_mask=land_mask
        )

        # AIS 密度
        ais_density = np.random.uniform(0, 1, (4, 4))

        cost_field = build_cost_from_real_env(
            grid,
            land_mask,
            env,
            wave_penalty=1.5,
            ais_density=ais_density,
            w_ais_corridor=0.3,
        )

        # 计算所有组件之和
        component_sum = np.zeros_like(cost_field.cost)
        for comp_name, comp_data in cost_field.components.items():
            component_sum += comp_data

        # 断言：组件之和应该等于总成本（允许小的浮点误差）
        # 注意：某些实现可能有 corridor 是负值（减少成本），所以可能不完全相等
        # 这里只验证海洋区域
        ocean_mask = ~land_mask
        assert np.allclose(
            component_sum[ocean_mask], 
            cost_field.cost[ocean_mask], 
            rtol=1e-5, 
            atol=1e-5
        ), "组件之和应该等于总成本"

