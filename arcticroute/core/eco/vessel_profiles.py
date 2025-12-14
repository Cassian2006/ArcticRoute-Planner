# -*- coding: utf-8 -*-
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "ArcticRoute" / "config" / "vessel_profiles.yaml"


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


# ============================================================================
# 冰级参数映射
# ============================================================================

ICE_CLASS_PARAMETERS: Dict[IceClass, Dict[str, Any]] = {
    IceClass.NO_ICE_CLASS: {
        "label": "No Ice Class",
        "max_ice_thickness_m": 0.0,
        "description": "No ice capability",
        "standard": "N/A",
    },
    IceClass.FSICR_1C: {
        "label": "FSICR 1C",
        "max_ice_thickness_m": 0.30,
        "description": "Finnish-Swedish Ice Class 1C",
        "standard": "Finnish-Swedish Ice Class",
    },
    IceClass.FSICR_1B: {
        "label": "FSICR 1B",
        "max_ice_thickness_m": 0.60,
        "description": "Finnish-Swedish Ice Class 1B",
        "standard": "Finnish-Swedish Ice Class",
    },
    IceClass.FSICR_1A: {
        "label": "FSICR 1A",
        "max_ice_thickness_m": 0.85,
        "description": "Finnish-Swedish Ice Class 1A",
        "standard": "Finnish-Swedish Ice Class",
    },
    IceClass.FSICR_1A_SUPER: {
        "label": "FSICR 1A Super",
        "max_ice_thickness_m": 1.00,
        "description": "Finnish-Swedish Ice Class 1A Super",
        "standard": "Finnish-Swedish Ice Class",
    },
    IceClass.POLAR_PC7: {
        "label": "Polar Class PC7",
        "max_ice_thickness_m": 1.20,
        "description": "IMO Polar Code PC7",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC6: {
        "label": "Polar Class PC6",
        "max_ice_thickness_m": 1.50,
        "description": "IMO Polar Code PC6",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC5: {
        "label": "Polar Class PC5",
        "max_ice_thickness_m": 2.00,
        "description": "IMO Polar Code PC5",
        "standard": "IMO Polar Code",
    },
}


# ============================================================================
# 业务船型参数映射
# ============================================================================

VESSEL_TYPE_PARAMETERS: Dict[VesselType, Dict[str, Any]] = {
    VesselType.FEEDER: {
        "label": "Feeder",
        "dwt_range": [5000, 15000],
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.020,
        "description": "Feeder vessel, small general cargo ship",
    },
    VesselType.HANDYSIZE: {
        "label": "Handysize",
        "dwt_range": [20000, 40000],
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.035,
        "description": "Handy-size bulk carrier",
    },
    VesselType.PANAMAX: {
        "label": "Panamax",
        "dwt_range": [65000, 85000],
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.050,
        "description": "Panamax vessel, can transit Panama Canal",
    },
    VesselType.AFRAMAX: {
        "label": "Aframax",
        "dwt_range": [80000, 120000],
        "design_speed_kn": 13.5,
        "base_fuel_per_km": 0.055,
        "description": "Aframax tanker",
    },
    VesselType.SUEZMAX: {
        "label": "Suezmax",
        "dwt_range": [120000, 200000],
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.070,
        "description": "Suezmax vessel, can transit Suez Canal",
    },
    VesselType.CAPESIZE: {
        "label": "Capesize",
        "dwt_range": [150000, 220000],
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.080,
        "description": "Capesize bulk carrier, largest general cargo ship",
    },
    VesselType.CONTAINER: {
        "label": "Container",
        "dwt_range": [40000, 200000],
        "design_speed_kn": 18.0,
        "base_fuel_per_km": 0.065,
        "description": "Container ship",
    },
    VesselType.LNG: {
        "label": "LNG Carrier",
        "dwt_range": [130000, 180000],
        "design_speed_kn": 19.0,
        "base_fuel_per_km": 0.045,
        "description": "Liquefied Natural Gas carrier",
    },
    VesselType.TANKER: {
        "label": "Tanker",
        "dwt_range": [30000, 150000],
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.055,
        "description": "Oil tanker",
    },
    VesselType.BULK_CARRIER: {
        "label": "Bulk Carrier",
        "dwt_range": [30000, 200000],
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.060,
        "description": "Bulk carrier",
    },
}


# ============================================================================
# VesselProfile 数据类
# ============================================================================

@dataclass
class VesselProfile:
    """船舶参数数据类。"""
    key: str
    name: str
    vessel_type: VesselType
    ice_class: IceClass
    dwt: float
    design_speed_kn: float
    base_fuel_per_km: float
    max_ice_thickness_m: float
    ice_margin_factor: float = 0.95
    
    @property
    def ice_class_label(self) -> str:
        """获取冰级标签。"""
        return ICE_CLASS_PARAMETERS[self.ice_class]["label"]
    
    def get_effective_max_ice_thickness(self) -> float:
        """计算有效最大冰厚（应用安全裕度）。"""
        effective = self.max_ice_thickness_m * self.ice_margin_factor
        return max(effective, 0.01)  # 至少 0.01m
    
    def get_soft_constraint_threshold(self) -> float:
        """计算软约束阈值（有效冰厚的 70%）。"""
        return self.get_effective_max_ice_thickness() * 0.7
    
    def get_ice_class_info(self) -> Dict[str, Any]:
        """获取冰级信息。"""
        return ICE_CLASS_PARAMETERS[self.ice_class].copy()
    
    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> VesselProfile:
        """从字典创建 VesselProfile 实例。"""
        return cls(
            key=data.get("key", name),
            name=data.get("name", name),
            vessel_type=data.get("vessel_type", VesselType.HANDYSIZE),
            ice_class=data.get("ice_class", IceClass.NO_ICE_CLASS),
            dwt=data.get("dwt", 30000.0),
            design_speed_kn=data.get("design_speed_kn", 13.0),
            base_fuel_per_km=data.get("base_fuel_per_km", 0.05),
            max_ice_thickness_m=data.get("max_ice_thickness_m", 0.0),
            ice_margin_factor=data.get("ice_margin_factor", 0.95),
        )


# ============================================================================
# 工厂函数
# ============================================================================

def create_vessel_profile(
    vessel_type: VesselType,
    ice_class: IceClass,
    dwt: Optional[float] = None,
    design_speed_kn: Optional[float] = None,
    base_fuel_per_km: Optional[float] = None,
    ice_margin_factor: float = 0.95,
) -> VesselProfile:
    """创建船舶配置。
    
    Args:
        vessel_type: 业务船型
        ice_class: 冰级标准
        dwt: 载重吨（可选，默认使用船型参数）
        design_speed_kn: 设计航速（可选，默认使用船型参数）
        base_fuel_per_km: 基础燃油消耗（可选，默认使用船型参数）
        ice_margin_factor: 安全裕度系数
    
    Returns:
        VesselProfile 实例
    """
    vessel_params = VESSEL_TYPE_PARAMETERS[vessel_type]
    ice_params = ICE_CLASS_PARAMETERS[ice_class]
    
    # 使用提供的参数或默认参数
    dwt_val = dwt if dwt is not None else (vessel_params["dwt_range"][0] + vessel_params["dwt_range"][1]) / 2
    speed_val = design_speed_kn if design_speed_kn is not None else vessel_params["design_speed_kn"]
    fuel_val = base_fuel_per_km if base_fuel_per_km is not None else vessel_params["base_fuel_per_km"]
    
    # 生成 key 和 name
    key = f"{vessel_type.value}_{ice_class.value}"
    name = f"{vessel_params['label']} ({ice_params['label']})"
    
    return VesselProfile(
        key=key,
        name=name,
        vessel_type=vessel_type,
        ice_class=ice_class,
        dwt=dwt_val,
        design_speed_kn=speed_val,
        base_fuel_per_km=fuel_val,
        max_ice_thickness_m=ice_params["max_ice_thickness_m"],
        ice_margin_factor=ice_margin_factor,
    )


# ============================================================================
# 工具函数
# ============================================================================

def load_all_profiles() -> Dict[str, Dict[str, Any]]:
    """从 YAML 文件加载所有配置。"""
    if not PROFILES_PATH.exists():
        return {}
    try:
        obj = yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8")) or {}
        profs = obj.get("profiles") or {}
        if isinstance(profs, dict):
            return profs
        return {}
    except Exception:
        return {}


def load_vessel_profile(name: str) -> Dict[str, Any] | None:
    """按名称加载船舶配置。"""
    profs = load_all_profiles()
    if not profs:
        return None
    p = profs.get(name)
    if not p and profs:
        # fallback first
        k0 = next(iter(profs.keys()))
        p = profs.get(k0)
    return p


def get_default_profiles() -> Dict[str, VesselProfile]:
    """获取所有默认船舶参数。"""
    # 创建一些常见的组合
    profiles = {}
    
    # Handy（无冰级）- 冰厚能力较弱
    profiles["handy"] = VesselProfile(
        key="handy",
        name="Handysize (No Ice Class)",
        vessel_type=VesselType.HANDYSIZE,
        ice_class=IceClass.NO_ICE_CLASS,
        dwt=30000.0,
        design_speed_kn=13.0,
        base_fuel_per_km=0.035,
        max_ice_thickness_m=0.3,
        ice_margin_factor=0.85,
    )
    
    # Handy with Ice Class
    profiles["handy_ice"] = create_vessel_profile(VesselType.HANDYSIZE, IceClass.FSICR_1A)
    
    # Panamax（无冰级）- 冰厚能力中等
    profiles["panamax"] = VesselProfile(
        key="panamax",
        name="Panamax (No Ice Class)",
        vessel_type=VesselType.PANAMAX,
        ice_class=IceClass.NO_ICE_CLASS,
        dwt=75000.0,
        design_speed_kn=14.0,
        base_fuel_per_km=0.050,
        max_ice_thickness_m=0.5,
        ice_margin_factor=0.90,
    )
    
    # Panamax with Ice Class
    profiles["panamax_ice"] = create_vessel_profile(VesselType.PANAMAX, IceClass.POLAR_PC7)
    
    # Capesize（无冰级）
    profiles["capesize"] = VesselProfile(
        key="capesize",
        name="Capesize (No Ice Class)",
        vessel_type=VesselType.CAPESIZE,
        ice_class=IceClass.NO_ICE_CLASS,
        dwt=185000.0,
        design_speed_kn=13.0,
        base_fuel_per_km=0.080,
        max_ice_thickness_m=0.2,  # 大型船舶冰厚能力较弱
        ice_margin_factor=0.90,
    )
    
    # Ice-Class（强冰级）- 冰厚能力强
    profiles["ice_class"] = create_vessel_profile(VesselType.HANDYSIZE, IceClass.POLAR_PC7)
    
    return profiles


def get_profile_by_key(key: str) -> Optional[VesselProfile]:
    """按 key 获取配置。"""
    profiles = get_default_profiles()
    return profiles.get(key)


def list_available_profiles() -> Dict[str, str]:
    """列出可用配置。"""
    profiles = get_default_profiles()
    return {key: profile.name for key, profile in profiles.items()}


def get_ice_class_options() -> Dict[str, str]:
    """获取冰级选项。"""
    return {ic.value: ICE_CLASS_PARAMETERS[ic]["label"] for ic in IceClass}


def get_vessel_type_options() -> Dict[str, str]:
    """获取业务船型选项。"""
    return {vt.value: VESSEL_TYPE_PARAMETERS[vt]["label"] for vt in VesselType}
