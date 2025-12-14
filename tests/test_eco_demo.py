"""
ECO 模块的 demo 测试。

包含：
- 船型配置烟雾测试
- ECO 估算功能测试
"""

from arcticroute.core.eco.vessel_profiles import get_default_profiles
from arcticroute.core.eco.eco_model import estimate_route_eco


def test_default_vessels_exist():
    """验证默认船型配置存在。"""
    profiles = get_default_profiles()
    assert "panamax" in profiles
    assert "handy" in profiles
    assert "ice_class" in profiles


def test_default_vessels_have_required_fields():
    """验证每个船型都有必要的字段。"""
    profiles = get_default_profiles()
    for key, vessel in profiles.items():
        assert hasattr(vessel, "key")
        assert hasattr(vessel, "name")
        assert hasattr(vessel, "dwt")
        assert hasattr(vessel, "design_speed_kn")
        assert hasattr(vessel, "base_fuel_per_km")
        assert vessel.key == key
        assert vessel.dwt > 0
        assert vessel.design_speed_kn > 0
        assert vessel.base_fuel_per_km > 0


def test_eco_scales_with_distance():
    """验证 ECO 指标随距离增加而增加。"""
    profiles = get_default_profiles()
    vessel = profiles["panamax"]

    # 一条短路线 & 一条长路线
    short_route = [(70.0, 10.0), (70.1, 10.0)]
    long_route = [(70.0, 10.0), (71.0, 10.0)]

    eco_short = estimate_route_eco(short_route, vessel)
    eco_long = estimate_route_eco(long_route, vessel)

    assert eco_long.distance_km > eco_short.distance_km
    assert eco_long.fuel_total_t > eco_short.fuel_total_t
    assert eco_long.co2_total_t > eco_short.co2_total_t


def test_empty_route_eco_zero():
    """验证空路线返回全 0 的 ECO。"""
    vessel = get_default_profiles()["panamax"]
    eco = estimate_route_eco([], vessel)
    assert eco.distance_km == 0
    assert eco.travel_time_h == 0
    assert eco.fuel_total_t == 0
    assert eco.co2_total_t == 0


def test_single_point_route_eco_zero():
    """验证单点路线返回全 0 的 ECO。"""
    vessel = get_default_profiles()["panamax"]
    eco = estimate_route_eco([(70.0, 10.0)], vessel)
    assert eco.distance_km == 0
    assert eco.travel_time_h == 0
    assert eco.fuel_total_t == 0
    assert eco.co2_total_t == 0


def test_eco_fuel_calculation():
    """验证燃油计算的正确性。"""
    profiles = get_default_profiles()
    vessel = profiles["panamax"]

    # 一条简单路线：两个点
    route = [(70.0, 10.0), (70.0, 11.0)]
    eco = estimate_route_eco(route, vessel)

    # 验证燃油 = 距离 * base_fuel_per_km
    expected_fuel = eco.distance_km * vessel.base_fuel_per_km
    assert abs(eco.fuel_total_t - expected_fuel) < 0.001


def test_eco_co2_calculation():
    """验证 CO2 计算的正确性。"""
    profiles = get_default_profiles()
    vessel = profiles["panamax"]

    route = [(70.0, 10.0), (70.0, 11.0)]
    eco = estimate_route_eco(route, vessel)

    # 验证 CO2 = 燃油 * co2_per_ton_fuel
    expected_co2 = eco.fuel_total_t * 3.114
    assert abs(eco.co2_total_t - expected_co2) < 0.001


def test_eco_travel_time_calculation():
    """验证航行时间计算的正确性。"""
    profiles = get_default_profiles()
    vessel = profiles["panamax"]

    route = [(70.0, 10.0), (70.0, 11.0)]
    eco = estimate_route_eco(route, vessel)

    # 航速：节 -> km/h，1 节 = 1.852 km/h
    speed_kmh = vessel.design_speed_kn * 1.852
    expected_time = eco.distance_km / speed_kmh
    assert abs(eco.travel_time_h - expected_time) < 0.001


def test_eco_different_vessels():
    """验证不同船型的 ECO 差异。"""
    profiles = get_default_profiles()
    route = [(70.0, 10.0), (70.0, 11.0)]

    eco_handy = estimate_route_eco(route, profiles["handy"])
    eco_panamax = estimate_route_eco(route, profiles["panamax"])
    eco_ice_class = estimate_route_eco(route, profiles["ice_class"])

    # 距离相同，但燃油消耗不同（base_fuel_per_km 不同）
    assert eco_handy.distance_km == eco_panamax.distance_km
    assert eco_handy.distance_km == eco_ice_class.distance_km

    # Ice-Class 油耗最高
    assert eco_ice_class.fuel_total_t > eco_panamax.fuel_total_t
    assert eco_panamax.fuel_total_t > eco_handy.fuel_total_t

    # 航速不同，时间也不同
    assert eco_handy.travel_time_h != eco_panamax.travel_time_h


def test_eco_custom_co2_coefficient():
    """验证自定义 CO2 系数的效果。"""
    profiles = get_default_profiles()
    vessel = profiles["panamax"]
    route = [(70.0, 10.0), (70.0, 11.0)]

    eco_default = estimate_route_eco(route, vessel, co2_per_ton_fuel=3.114)
    eco_custom = estimate_route_eco(route, vessel, co2_per_ton_fuel=3.5)

    # CO2 应该按比例增加
    ratio = 3.5 / 3.114
    assert abs(eco_custom.co2_total_t / eco_default.co2_total_t - ratio) < 0.001











