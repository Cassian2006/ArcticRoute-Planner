"""
场景预设配置模块。

定义标准场景库，用于 EDL 灵敏度分析和 UI 演示。

四个标准场景：
  - barents_to_chukchi: 巴伦支海到楚科奇海（高冰区，长距离）
  - kara_short: 卡拉海短途（中等冰区，冰级船）
  - southern_route: 南向北冰洋边缘（低冰区，短距离）
  - west_to_east_demo: 西向东跨越北冰洋（全程高纬，多冰区）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Scenario:
    """单个场景定义。
    
    Attributes:
        name: 场景名称（英文标识，用于代码）
        description: 场景描述（中文，用于 UI 显示）
        ym: 年月，格式 "YYYYMM"（例如 "202412"）
        start_lat: 起点纬度（度）
        start_lon: 起点经度（度）
        end_lat: 终点纬度（度）
        end_lon: 终点经度（度）
        vessel_profile: 船舶配置名称（例如 "panamax", "ice_class", "handy"）
        sic_exp: 海冰浓度指数（默认 1.5）
        wave_exp: 波浪高度指数（默认 1.5）
    """
    
    name: str
    description: str
    ym: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    vessel_profile: str
    sic_exp: float = 1.5
    wave_exp: float = 1.5
    
    def __str__(self) -> str:
        """返回场景的显示字符串。"""
        return f"{self.description} ({self.name})"


# ============================================================================
# 标准场景库
# ============================================================================

SCENARIOS: List[Scenario] = [
    Scenario(
        name="barents_to_chukchi",
        description="巴伦支海到楚科奇海（高冰区，长距离）",
        ym="202412",
        start_lat=69.0,
        start_lon=33.0,
        end_lat=70.5,
        end_lon=170.0,
        vessel_profile="panamax",
    ),
    Scenario(
        name="kara_short",
        description="卡拉海短途（中等冰区，冰级船）",
        ym="202412",
        start_lat=73.0,
        start_lon=60.0,
        end_lat=76.0,
        end_lon=120.0,
        vessel_profile="ice_class",
    ),
    Scenario(
        name="southern_route",
        description="南向北冰洋边缘（低冰区，短距离）",
        ym="202412",
        start_lat=60.0,
        start_lon=30.0,
        end_lat=68.0,
        end_lon=90.0,
        vessel_profile="panamax",
    ),
    Scenario(
        name="west_to_east_demo",
        description="西向东跨越北冰洋（全程高纬，多冰区）",
        ym="202412",
        start_lat=66.0,
        start_lon=5.0,
        end_lat=78.0,
        end_lon=150.0,
        vessel_profile="handy",
    ),
]


# ============================================================================
# 工具函数
# ============================================================================

def get_scenario_by_name(name: str) -> Optional[Scenario]:
    """
    按名称获取场景。
    
    Args:
        name: 场景名称
    
    Returns:
        Scenario 对象，若不存在则返回 None
    """
    for scenario in SCENARIOS:
        if scenario.name == name:
            return scenario
    return None


def list_scenarios() -> List[str]:
    """
    列出所有场景名称。
    
    Returns:
        场景名称列表
    """
    return [s.name for s in SCENARIOS]


def list_scenario_descriptions() -> dict[str, str]:
    """
    列出所有场景的名称和描述。
    
    Returns:
        {场景名称: 场景描述} 字典
    """
    return {s.name: s.description for s in SCENARIOS}


def get_scenario_display_name(name: str) -> str:
    """
    获取场景的显示名称。
    
    Args:
        name: 场景名称
    
    Returns:
        显示名称（格式：描述 (名称)）
    
    Raises:
        ValueError: 如果场景不存在
    """
    scenario = get_scenario_by_name(name)
    if scenario is None:
        raise ValueError(f"Unknown scenario: {name}")
    return str(scenario)








