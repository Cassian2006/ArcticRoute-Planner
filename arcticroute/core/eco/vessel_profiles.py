# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml

# Optional external YAML support (not required for tests)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES_PATH = PROJECT_ROOT / "ArcticRoute" / "config" / "vessel_profiles.yaml"


class VesselType(Enum):
    """业务船型枚举（匹配测试所需名称）。"""
    HANDYSIZE = "handysize"
    PANAMAX = "panamax"
    CAPESIZE = "capesize"


class IceClass(Enum):
    """冰级枚举（匹配测试所需名称及顺序）。"""
    NO_ICE_CLASS = "no_ice_class"
    FSICR_1C = "fsicr_1c"
    FSICR_1B = "fsicr_1b"
    FSICR_1A = "fsicr_1a"
    FSICR_1A_SUPER = "fsicr_1a_super"
    POLAR_PC7 = "polar_pc7"
    POLAR_PC6 = "polar_pc6"
    POLAR_PC5 = "polar_pc5"


# 冰级参数映射（以枚举为键，满足测试字段需求）
ICE_CLASS_PARAMETERS: Dict[IceClass, Dict[str, Any]] = {
    IceClass.NO_ICE_CLASS: {
        "label": "No Ice Class",
        "max_ice_thickness_m": 0.00,
        "description": "No specific ice strengthening.",
        "standard": "General navigation",
    },
    IceClass.FSICR_1C: {
        "label": "FSICR 1C",
        "max_ice_thickness_m": 0.25,
        "description": "Finnish-Swedish Ice Class 1C.",
        "standard": "FSICR",
    },
    IceClass.FSICR_1B: {
        "label": "FSICR 1B",
        "max_ice_thickness_m": 0.50,
        "description": "Finnish-Swedish Ice Class 1B.",
        "standard": "FSICR",
    },
    IceClass.FSICR_1A: {
        "label": "FSICR 1A",
        "max_ice_thickness_m": 0.80,
        "description": "Finnish-Swedish Ice Class 1A.",
        "standard": "FSICR",
    },
    IceClass.FSICR_1A_SUPER: {
        "label": "FSICR 1A Super",
        "max_ice_thickness_m": 1.00,
        "description": "Finnish-Swedish Ice Class 1A Super.",
        "standard": "FSICR",
    },
    IceClass.POLAR_PC7: {
        "label": "Polar Class PC7",
        "max_ice_thickness_m": 1.20,
        "description": "Year-round operation in thin first-year ice.",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC6: {
        "label": "Polar Class PC6",
        "max_ice_thickness_m": 1.50,
        "description": "Summer/autumn operation in medium first-year ice.",
        "standard": "IMO Polar Code",
    },
    IceClass.POLAR_PC5: {
        "label": "Polar Class PC5",
        "max_ice_thickness_m": 2.00,
        "description": "Year-round operation in medium first-year ice.",
        "standard": "IMO Polar Code",
    },
}


# 船舶类型参数映射（以枚举为键，满足测试字段需求）
VESSEL_TYPE_PARAMETERS: Dict[VesselType, Dict[str, Any]] = {
    VesselType.HANDYSIZE: {
        "label": "Handysize",
        "dwt_range": (10000, 40000),
        "design_speed_kn": 13.0,
        "base_fuel_per_km": 0.035,
        "description": "General-purpose handy bulk carrier.",
    },
    VesselType.PANAMAX: {
        "label": "Panamax",
        "dwt_range": (60000, 80000),
        "design_speed_kn": 14.0,
        "base_fuel_per_km": 0.050,
        "description": "Panamax-size bulk carrier.",
    },
    VesselType.CAPESIZE: {
        "label": "Capesize",
        "dwt_range": (150000, 200000),
        "design_speed_kn": 15.0,
        "base_fuel_per_km": 0.080,
        "description": "Large capesize bulk carrier.",
    },
}


@dataclass
class VesselProfile:
    """船舶配置数据类（最小实现，覆盖测试所需字段与方法）。"""

    key: str
    name: str
    vessel_type: VesselType
    ice_class: IceClass
    dwt: float
    design_speed_kn: float
    base_fuel_per_km: float
    max_ice_thickness_m: float
    ice_margin_factor: float = 1.0

    @property
    def ice_class_label(self) -> str:
        return ICE_CLASS_PARAMETERS[self.ice_class]["label"]

    def get_effective_max_ice_thickness(self) -> float:
        # 应用安全裕度系数，并设置下限 0.01m
        effective = float(self.max_ice_thickness_m) * float(self.ice_margin_factor or 1.0)
        return max(0.01, effective)

    def get_soft_constraint_threshold(self) -> float:
        # 经验性阈值：70% 的最大冰厚
        return 0.7 * float(self.max_ice_thickness_m)

    def get_ice_class_info(self) -> Dict[str, Any]:
        return dict(ICE_CLASS_PARAMETERS[self.ice_class])

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"VesselProfile(key={self.key!r}, name={self.name!r}, "
            f"vessel_type={self.vessel_type}, ice_class={self.ice_class})"
        )


# ----------------------------------------------------------------------------
# 工厂与工具函数
# ----------------------------------------------------------------------------

def _default_for(vessel_type: VesselType) -> Tuple[float, float, float]:
    p = VESSEL_TYPE_PARAMETERS[vessel_type]
    # 选取 dwt_range 中值作为默认 DWT
    dwt_min, dwt_max = p["dwt_range"]
    dwt_default = 0.5 * (dwt_min + dwt_max)
    return (
        dwt_default,
        float(p["design_speed_kn"]),
        float(p["base_fuel_per_km"]),
    )


def create_vessel_profile(
    vessel_type: VesselType,
    ice_class: IceClass,
    *,
    dwt: float | None = None,
    design_speed_kn: float | None = None,
    base_fuel_per_km: float | None = None,
    ice_margin_factor: float = 1.0,
) -> VesselProfile:
    """根据业务船型与冰级创建一个默认的 VesselProfile，可被参数覆盖。

    - key: 形如 "panamax_polar_pc7"
    - name: 形如 "Panamax (PC7)"
    """
    d_dwt, d_speed, d_fuel = _default_for(vessel_type)
    dwt = float(dwt if dwt is not None else d_dwt)
    design_speed_kn = float(design_speed_kn if design_speed_kn is not None else d_speed)
    base_fuel_per_km = float(base_fuel_per_km if base_fuel_per_km is not None else d_fuel)

    # 由冰级参数决定默认最大冰厚
    max_ice_thickness_m = ICE_CLASS_PARAMETERS[ice_class]["max_ice_thickness_m"]

    key = f"{vessel_type.value}_{ice_class.value}"
    name = f"{VESSEL_TYPE_PARAMETERS[vessel_type]['label']} ({ICE_CLASS_PARAMETERS[ice_class]['label'].split()[-1]})"

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
    )


def get_default_profiles() -> Dict[str, VesselProfile]:
    """获取默认船舶配置字典。

    返回键："handy"、"panamax"、"ice_class"（满足测试断言）。
    """
    profiles: Dict[str, VesselProfile] = {
        "handy": create_vessel_profile(VesselType.HANDYSIZE, IceClass.FSICR_1A),
        "panamax": create_vessel_profile(VesselType.PANAMAX, IceClass.POLAR_PC7),
        "ice_class": create_vessel_profile(VesselType.HANDYSIZE, IceClass.FSICR_1A_SUPER),
    }

    # 如果外部 YAML 存在，可进一步覆盖或扩展（非测试必需，容错）
    if PROFILES_PATH.exists():
        try:
            obj = yaml.safe_load(PROFILES_PATH.read_text(encoding="utf-8")) or {}
            extra = obj.get("profiles") or {}
            if isinstance(extra, dict):
                for k, v in extra.items():
                    if isinstance(v, dict):
                        vt = v.get("vessel_type", "panamax").lower()
                        ic = v.get("ice_class", "no_ice_class").lower()
                        vt_enum = VesselType(vt) if vt in [e.value for e in VesselType] else VesselType.PANAMAX
                        ic_enum = IceClass(ic) if ic in [e.value for e in IceClass] else IceClass.NO_ICE_CLASS
                        profiles[k] = create_vessel_profile(
                            vt_enum,
                            ic_enum,
                            dwt=v.get("dwt"),
                            design_speed_kn=v.get("design_speed_kn"),
                            base_fuel_per_km=v.get("base_fuel_per_km"),
                            ice_margin_factor=v.get("ice_margin_factor", 1.0),
                        )
        except Exception:
            pass

    return profiles


def get_profile_by_key(key: str) -> VesselProfile | None:
    return get_default_profiles().get(key)


def list_available_profiles() -> Dict[str, str]:
    """列出可用配置（key -> 友好名称）。"""
    profs = get_default_profiles()
    return {k: v.name for k, v in profs.items()}


def get_ice_class_options() -> Dict[str, str]:
    """返回可选冰级映射（值为显示名称）。"""
    return {ic.value: ICE_CLASS_PARAMETERS[ic]["label"] for ic in IceClass}


def get_vessel_type_options() -> Dict[str, str]:
    """返回可选业务船型映射（值为显示名称）。"""
    return {vt.value: VESSEL_TYPE_PARAMETERS[vt]["label"] for vt in VesselType}
