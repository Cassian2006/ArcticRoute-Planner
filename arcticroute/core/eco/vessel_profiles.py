"""
船舶参数配置模块 - 两层结构（业务船型 × 冰级标准）。

本模块提供：
  1. 业务船型（Vessel Type）：Handysize、Panamax、Capesize 等
  2. 冰级标准（Ice Class）：No ice class、FSICR 1C/1B/1A/1A Super、Polar Class PC7~PC3

关键说明：
  - 冰厚阈值（max_ice_thickness_m）是工程代理参数，基于 Polar Class 和冰情分级体系
  - 这些阈值是初始工程估计，后续将通过 AIS 轨迹和 EDL 模型进行校准
  - ice_margin_factor 用于计算安全工作冰厚（考虑设计裕度）

参考标准：
  - Polar Class (PC): IMO Polar Code 定义的冰级标准
  - FSICR: Finnish-Swedish Ice Class Rules（芬兰-瑞典冰级规则）
  - 冰厚定义：指一年冰（first-year ice）的厚度
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple


# ============================================================================
# 枚举定义
# ============================================================================

class VesselType(Enum):
    """业务船型枚举。"""
    FEEDER = "feeder"
    HANDYSIZE = "handysize"
    PANAMAX = "panamax"
    AFRAMAX = "aframax"
    SUEZMAX = "suezmax"
    CAPESIZE = "capesize"
    CONTAINER = "container"
    LNG = "lng"
    TANKER = "tanker"
    BULK_CARRIER = "bulk_carrier"


class IceClass(Enum):
    """冰级标准枚举。"""
    NO_ICE_CLASS = "no_ice_class"
    FSICR_1C = "fsicr_1c"
    FSICR_1B = "fsicr_1b"
    FSICR_1A = "fsicr_1a"
    FSICR_1A_SUPER = "fsicr_1a_super"
    POLAR_PC7 = "polar_pc7"
    POLAR_PC6 = "polar_pc6"
    POLAR_PC5 = "polar_pc5"
    POLAR_PC4 = "polar_pc4"
    POLAR_PC3 = "polar_pc3"


# ============================================================================
# 冰级参数映射表
# ============================================================================

ICE_CLASS_PARAMETERS = {
    # 无冰级
    IceClass.NO_ICE_CLASS: {
        "label": "No Ice Class",
        "max_ice_thickness_m": 0.25,  # 仅可通行薄冰
        "description": "非冰级船，仅可通行薄冰（<0.25m）",
        "standard": "N/A",
    },
    
    # FSICR（芬兰-瑞典冰级规则）
    IceClass.FSICR_1C: {
        "label": "FSICR 1C",
        "max_ice_thickness_m": 0.30,
        "description": "芬兰-瑞典冰级 1C，可通行厚度 ~0.3m 的一年冰",
        "standard": "Finnish-Swedish Ice Class Rules",
    },
    IceClass.FSICR_1B: {
        "label": "FSICR 1B",
        "max_ice_thickness_m": 0.50,
        "description": "芬兰-瑞典冰级 1B，可通行厚度 ~0.5m 的一年冰",
        "standard": "Finnish-Swedish Ice Class Rules",
    },
    IceClass.FSICR_1A: {
        "label": "FSICR 1A",
        "max_ice_thickness_m": 0.80,
        "description": "芬兰-瑞典冰级 1A，可通行厚度 ~0.8m 的一年冰",
        "standard": "Finnish-Swedish Ice Class Rules",
    },
    IceClass.FSICR_1A_SUPER: {
        "label": "FSICR 1A Super",
        "max_ice_thickness_m": 1.00,
        "description": "芬兰-瑞典冰级 1A Super，可通行厚度 ~1.0m 的一年冰",
        "standard": "Finnish-Swedish Ice Class Rules",
    },
    
    # Polar Class（IMO Polar Code）
    IceClass.POLAR_PC7: {
        "label": "Polar Class PC7",
        "max_ice_thickness_m": 1.20,
        "description": "IMO Polar Class PC7，可通行厚度 ~1.2m 的一年冰",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC6: {
        "label": "Polar Class PC6",
        "max_ice_thickness_m": 1.50,
        "description": "IMO Polar Class PC6，可通行厚度 ~1.5m 的一年冰",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC5: {
        "label": "Polar Class PC5",
        "max_ice_thickness_m": 2.00,
        "description": "IMO Polar Class PC5，可通行厚度 ~2.0m 的一年冰",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC4: {
        "label": "Polar Class PC4",
        "max_ice_thickness_m": 2.50,
        "description": "IMO Polar Class PC4，可通行厚度 ~2.5m 的一年冰（多年冰）",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC3: {
        "label": "Polar Class PC3",
        "max_ice_thickness_m": 3.00,
        "description": "IMO Polar Class PC3，可通行厚度 ~3.0m 的多年冰",
        "standard": "IMO Polar Code",
    },
}


# ============================================================================
# 业务船型参数映射表
# ============================================================================

VESSEL_TYPE_PARAMETERS = {
    VesselType.FEEDER: {
        "label": "Feeder",
        "dwt_range": (5000, 15000),
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.020,
        "description": "支线船，小型通用船",
    },
    VesselType.HANDYSIZE: {
        "label": "Handysize",
        "dwt_range": (20000, 40000),
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.035,
        "description": "灵便型散货船",
    },
    VesselType.PANAMAX: {
        "label": "Panamax",
        "dwt_range": (65000, 85000),
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.050,
        "description": "巴拿马型船，可通过巴拿马运河",
    },
    VesselType.AFRAMAX: {
        "label": "Aframax",
        "dwt_range": (80000, 120000),
        "design_speed_kn": 13.5,
        "base_fuel_per_km": 0.055,
        "description": "阿芙拉型油轮",
    },
    VesselType.SUEZMAX: {
        "label": "Suezmax",
        "dwt_range": (120000, 200000),
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.070,
        "description": "苏伊士型船，可通过苏伊士运河",
    },
    VesselType.CAPESIZE: {
        "label": "Capesize",
        "dwt_range": (150000, 220000),
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.080,
        "description": "好望角型散货船，最大型通用船",
    },
    VesselType.CONTAINER: {
        "label": "Container",
        "dwt_range": (40000, 200000),
        "design_speed_kn": 18.0,
        "base_fuel_per_km": 0.065,
        "description": "集装箱船",
    },
    VesselType.LNG: {
        "label": "LNG Carrier",
        "dwt_range": (130000, 180000),
        "design_speed_kn": 19.0,
        "base_fuel_per_km": 0.045,
        "description": "液化天然气运输船",
    },
    VesselType.TANKER: {
        "label": "Tanker",
        "dwt_range": (30000, 150000),
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.055,
        "description": "油轮",
    },
    VesselType.BULK_CARRIER: {
        "label": "Bulk Carrier",
        "dwt_range": (30000, 200000),
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.050,
        "description": "散货船",
    },
}


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class VesselProfile:
    """船舶参数配置。
    
    两层结构：
      - vessel_type: 业务船型（Handysize、Panamax 等）
      - ice_class: 冰级标准（No ice class、FSICR 1C、Polar Class PC7 等）
    
    Attributes:
        key: 唯一标识符（例如 "panamax_pc7"）
        name: 显示名称（例如 "Panamax + Polar Class PC7"）
        vessel_type: 业务船型
        ice_class: 冰级标准
        dwt: 载重吨（Deadweight tonnage）
        design_speed_kn: 设计航速（节）
        base_fuel_per_km: 基础单位油耗（t/km，无冰/平静海况）
        max_ice_thickness_m: 设计可通行最大冰厚（米）
        ice_margin_factor: 冰厚安全裕度系数（0..1），用于计算有效最大冰厚
        ice_class_label: 冰级标签（用于 UI 显示）
    """
    
    key: str
    name: str
    vessel_type: VesselType
    ice_class: IceClass
    dwt: float
    design_speed_kn: float
    base_fuel_per_km: float
    max_ice_thickness_m: float
    ice_margin_factor: float = 0.90
    ice_class_label: str = ""
    
    def __post_init__(self):
        """初始化后处理：设置 ice_class_label。"""
        if not self.ice_class_label:
            self.ice_class_label = ICE_CLASS_PARAMETERS[self.ice_class]["label"]
    
    def get_effective_max_ice_thickness(self) -> float:
        """
        获取考虑安全裕度后的有效最大冰厚。
        
        有效冰厚 = max_ice_thickness_m × ice_margin_factor
        
        这个值用于成本构建中的 hard constraint（超过此值则成本为 inf）。
        
        Returns:
            有效最大冰厚（米）
        """
        return max(self.max_ice_thickness_m * self.ice_margin_factor, 0.01)
    
    def get_soft_constraint_threshold(self) -> float:
        """
        获取软约束阈值（冰级约束的软风险区起点）。
        
        软约束区间：[soft_threshold, max_ice_thickness_m]
        在此区间内，冰厚会施加二次惩罚，但不会导致不可通行。
        
        soft_threshold = 0.7 × max_ice_thickness_m
        
        Returns:
            软约束阈值（米）
        """
        return 0.7 * self.max_ice_thickness_m
    
    def get_ice_class_info(self) -> Dict[str, str]:
        """
        获取冰级的详细信息。
        
        Returns:
            包含 label、description、standard 的字典
        """
        return {
            "label": self.ice_class_label,
            "description": ICE_CLASS_PARAMETERS[self.ice_class]["description"],
            "standard": ICE_CLASS_PARAMETERS[self.ice_class]["standard"],
        }


# ============================================================================
# 预定义配置
# ============================================================================

def create_vessel_profile(
    vessel_type: VesselType,
    ice_class: IceClass,
    dwt: Optional[float] = None,
    design_speed_kn: Optional[float] = None,
    base_fuel_per_km: Optional[float] = None,
    ice_margin_factor: float = 0.90,
) -> VesselProfile:
    """
    工厂函数：根据船型和冰级创建船舶配置。
    
    如果未指定参数，则使用默认值。
    
    Args:
        vessel_type: 业务船型
        ice_class: 冰级标准
        dwt: 载重吨（可选，使用默认值的中点）
        design_speed_kn: 设计航速（可选，使用默认值）
        base_fuel_per_km: 基础油耗（可选，使用默认值）
        ice_margin_factor: 冰厚安全裕度系数（默认 0.90）
    
    Returns:
        VesselProfile 对象
    """
    # 获取船型参数
    vessel_params = VESSEL_TYPE_PARAMETERS[vessel_type]
    if dwt is None:
        dwt = sum(vessel_params["dwt_range"]) / 2.0
    if design_speed_kn is None:
        design_speed_kn = vessel_params["design_speed_kn"]
    if base_fuel_per_km is None:
        base_fuel_per_km = vessel_params["base_fuel_per_km"]
    
    # 获取冰级参数
    ice_params = ICE_CLASS_PARAMETERS[ice_class]
    max_ice_thickness_m = ice_params["max_ice_thickness_m"]
    
    # 构造 key 和 name
    key = f"{vessel_type.value}_{ice_class.value}"
    name = f"{vessel_params['label']} + {ice_params['label']}"
    
    return VesselProfile(
        key=key,
        name=name,
        vessel_type=vessel_type,
        ice_class=ice_class,
        dwt=dwt,
        design_speed_kn=design_speed_kn,
        base_fuel_per_km=base_fuel_per_km,
        max_ice_thickness_m=max_ice_thickness_m,
        ice_margin_factor=ice_margin_factor,
        ice_class_label=ice_params["label"],
    )


def get_default_profiles() -> Dict[str, VesselProfile]:
    """
    返回内置的默认船型配置字典。
    
    包含常见的业务场景组合：
      - 无冰级船（Handysize、Panamax）
      - 冰级船（Handysize + FSICR 1A、Panamax + PC7 等）
    
    Returns:
        {key: VesselProfile} 字典
    """
    profiles: Dict[str, VesselProfile] = {}
    
    # 无冰级船
    profiles["handy"] = create_vessel_profile(
        VesselType.HANDYSIZE,
        IceClass.NO_ICE_CLASS,
        ice_margin_factor=0.85,
    )
    
    profiles["panamax"] = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.NO_ICE_CLASS,
        ice_margin_factor=0.90,
    )
    
    profiles["capesize"] = create_vessel_profile(
        VesselType.CAPESIZE,
        IceClass.NO_ICE_CLASS,
        ice_margin_factor=0.90,
    )
    
    # 冰级船
    profiles["handy_1a"] = create_vessel_profile(
        VesselType.HANDYSIZE,
        IceClass.FSICR_1A,
        ice_margin_factor=0.90,
    )
    
    profiles["panamax_pc7"] = create_vessel_profile(
        VesselType.PANAMAX,
        IceClass.POLAR_PC7,
        ice_margin_factor=0.95,
    )
    
    profiles["ice_class"] = create_vessel_profile(
        VesselType.HANDYSIZE,
        IceClass.POLAR_PC7,
        ice_margin_factor=0.95,
    )
    
    # LNG 运输船（特殊用途）
    profiles["lng"] = create_vessel_profile(
        VesselType.LNG,
        IceClass.NO_ICE_CLASS,
        ice_margin_factor=0.85,
    )
    
    return profiles


def get_profile_by_key(key: str) -> Optional[VesselProfile]:
    """
    按 key 获取预定义的船舶配置。
    
    Args:
        key: 配置 key（例如 "panamax"、"panamax_pc7"）
    
    Returns:
        VesselProfile 对象，或 None 如果不存在
    """
    profiles = get_default_profiles()
    return profiles.get(key)


def list_available_profiles() -> Dict[str, str]:
    """
    列出所有可用的预定义配置。
    
    Returns:
        {key: name} 字典
    """
    profiles = get_default_profiles()
    return {key: profile.name for key, profile in profiles.items()}


def get_ice_class_options() -> Dict[str, str]:
    """
    获取所有冰级选项。
    
    Returns:
        {ice_class.value: label} 字典
    """
    return {
        ice_class.value: params["label"]
        for ice_class, params in ICE_CLASS_PARAMETERS.items()
    }


def get_vessel_type_options() -> Dict[str, str]:
    """
    获取所有业务船型选项。
    
    Returns:
        {vessel_type.value: label} 字典
    """
    return {
        vessel_type.value: params["label"]
        for vessel_type, params in VESSEL_TYPE_PARAMETERS.items()
    }


# ============================================================================
# 文档和参考
# ============================================================================

"""
冰厚阈值参考标准
==================

本模块中的冰厚阈值（max_ice_thickness_m）基于以下标准：

1. FSICR（芬兰-瑞典冰级规则）
   - 1C: 0.3m（薄冰）
   - 1B: 0.5m（中等冰）
   - 1A: 0.8m（厚冰）
   - 1A Super: 1.0m（很厚冰）

2. IMO Polar Class（国际海事组织极地规则）
   - PC7: 1.2m（一年冰）
   - PC6: 1.5m（一年冰）
   - PC5: 2.0m（一年冰）
   - PC4: 2.5m（多年冰）
   - PC3: 3.0m（多年冰）

3. 冰情分级体系
   - 薄冰（thin ice）: <0.3m
   - 一年冰（first-year ice）: 0.3-2.0m
   - 多年冰（multi-year ice）: >2.0m

重要说明
========

这些阈值是初始工程估计，基于冰级标准的定义。后续将通过以下方式进行校准：

1. AIS 轨迹分析
   - 收集实际船舶在不同冰厚条件下的通行数据
   - 分析船舶的实际冰厚容忍度

2. EDL（Evidential Deep Learning）模型
   - 使用 AIS 轨迹作为正样本
   - 训练模型预测最优冰厚阈值
   - 估计模型的不确定性

3. 参数优化
   - 使用 scripts/calibrate_env_exponents.py 进行网格搜索
   - 优化 max_ice_thickness_m 和 ice_margin_factor
   - 生成校准报告和置信区间

使用建议
========

1. 短期（立即使用）
   - 使用本模块提供的默认参数
   - 在 UI 中提供冰级选择
   - 根据用户反馈调整参数

2. 中期（1-2 周）
   - 收集 AIS 轨迹数据
   - 进行初步的参数校准
   - 更新默认参数

3. 长期（1-3 月）
   - 使用 EDL 模型进行深度学习校准
   - 建立自动参数更新机制
   - 支持多月份、多船型的参数定制

参考文献
========

1. IMO Polar Code
   https://www.imo.org/en/OurWork/Environment/PolarCode/Pages/default.aspx

2. Finnish-Swedish Ice Class Rules
   https://www.abb.com/marine

3. Arctic Shipping and Ice Class Ships
   https://www.arctictoday.com/

4. Polar Class Definitions
   https://www.dnvgl.com/
"""
