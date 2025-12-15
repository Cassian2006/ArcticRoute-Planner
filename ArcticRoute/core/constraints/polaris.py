from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Tuple
import numpy as np

# WMO 厚度分段（用于把 thickness 映射到 ice type）：
# grey-white 15-30 cm；thin FY 30-70；medium FY 70-120；thick FY >120；
# old ice typical up to 3m+；second-year typical up to 2.5m
IceType = Literal[
    "ice_free",
    "new_ice",
    "grey_ice",
    "grey_white_ice",
    "thin_fy_1st",
    "thin_fy_2nd",
    "medium_fy",
    "thick_fy",
    "second_year",
    "multi_year",
]

OperationLevel = Literal["normal", "elevated", "special"]

@dataclass(frozen=True)
class PolarisMeta:
    ice_type: IceType
    rio: float
    level: OperationLevel
    speed_limit_knots: Optional[float]
    riv_used: str  # "table_1_3" or "table_1_4"

# RIV tables (values from MSC.1/Circ.1519 Annex Table 1.3/1.4; encoded numerically)
# Keep tables minimal in code review: only numeric mapping, no verbatim doc text.
# Keys: (ice_class, ice_type) -> RIV
# ice_class naming: PC1..PC7, "IC", "NOICE"
RIV_TABLE_1_3: Dict[Tuple[str, IceType], int] = {}
RIV_TABLE_1_4: Dict[Tuple[str, IceType], int] = {}

def _load_riv_tables():
    # Table 1.3 / 1.4 columns: Ice-free, New, Grey, Grey-white,
    # Thin FY 1st, Thin FY 2nd, Medium FY, Thick FY, Second-year, Multi-year
    ice_cols: list[IceType] = [
        "ice_free","new_ice","grey_ice","grey_white_ice",
        "thin_fy_1st","thin_fy_2nd","medium_fy","thick_fy","second_year","multi_year"
    ]

    # Table 1.3 values (PC1..PC7) + IC + Not Ice Strengthened
    # Source: MSC.1/Circ.1519 Annex Table 1.3 (standard conditions).
    table_1_3_rows = {
        "PC1":  [ 3, 3, 3, 3, 3, 3, 2, 1, 1, 1],
        "PC2":  [ 3, 3, 3, 3, 3, 2, 1, 1, 0, 0],
        "PC3":  [ 3, 3, 3, 3, 2, 1, 1, 0,-1,-1],
        "PC4":  [ 3, 3, 3, 2, 1, 1, 0,-1,-2,-3],
        "PC5":  [ 3, 3, 2, 1, 1, 0,-1,-2,-3,-4],
        "PC6":  [ 3, 2, 1, 1, 0,-1,-2,-3,-4,-5],
        "PC7":  [ 3, 1, 0,-1,-1,-2,-3,-4,-5,-6],
        "IC":   [ 3, 2, 1, 0,-1,-2,-3,-4,-5,-6],
        "NOICE":[ 3, 0,-1,-2,-3,-4,-5,-6,-7,-8],
    }

    # Table 1.4 values (decayed conditions)
    table_1_4_rows = {
        "PC1":  [ 3, 3, 3, 3, 3, 3, 3, 2, 2, 2],
        "PC2":  [ 3, 3, 3, 3, 3, 3, 2, 1, 1, 1],
        "PC3":  [ 3, 3, 3, 3, 3, 2, 1, 1, 0, 0],
        "PC4":  [ 3, 3, 3, 3, 2, 1, 1, 0,-1,-1],
        "PC5":  [ 3, 3, 3, 2, 1, 1, 0,-1,-2,-3],
        "PC6":  [ 3, 3, 2, 1, 1, 0,-1,-2,-3,-4],
        "PC7":  [ 3, 2, 1, 0,-1,-2,-3,-4,-5,-6],
        "IC":   [ 3, 2, 1, 0,-1,-2,-3,-4,-5,-6],
        "NOICE":[ 3, 1, 0,-1,-2,-3,-4,-5,-6,-7],
    }

    for cls, vals in table_1_3_rows.items():
        for ice_t, riv in zip(ice_cols, vals):
            RIV_TABLE_1_3[(cls, ice_t)] = int(riv)
    for cls, vals in table_1_4_rows.items():
        for ice_t, riv in zip(ice_cols, vals):
            RIV_TABLE_1_4[(cls, ice_t)] = int(riv)

_load_riv_tables()

def thickness_to_ice_type(thickness_m: float, sic: float, ice_free_sic_threshold: float = 0.05) -> IceType:
    if np.isnan(thickness_m) or thickness_m <= 0.0 or sic < ice_free_sic_threshold:
        return "ice_free"
    # Below 10 cm: treat as "new_ice" for our grid approximation
    if thickness_m < 0.10:
        return "new_ice"
    if thickness_m < 0.15:
        return "grey_ice"          # 10–15 cm
    if thickness_m < 0.30:
        return "grey_white_ice"    # 15–30 cm
    if thickness_m < 0.50:
        return "thin_fy_1st"       # 30–50 cm
    if thickness_m < 0.70:
        return "thin_fy_2nd"       # 50–70 cm
    if thickness_m < 1.20:
        return "medium_fy"         # 70–120 cm
    if thickness_m < 2.00:
        return "thick_fy"          # >120 cm, FY up to ~2 m
    if thickness_m < 2.50:
        return "second_year"       # typical up to 2.5 m
    return "multi_year"            # old ice typical up to 3 m or more

def classify_operation_level(rio: float, ice_class: str) -> OperationLevel:
    # Table 1.1 logic
    pc_set = {"PC1","PC2","PC3","PC4","PC5","PC6","PC7"}
    if rio >= 0:
        return "normal"
    if ice_class in pc_set:
        return "elevated" if rio >= -10 else "special"
    return "special"

def recommended_speed_limit_knots(level: OperationLevel, ice_class: str) -> Optional[float]:
    if level != "elevated":
        return None
    # Table 1.2
    if ice_class == "PC1":
        return 11.0
    if ice_class == "PC2":
        return 8.0
    if ice_class in {"PC3","PC4","PC5"}:
        return 5.0
    return 3.0  # below PC5

def compute_rio_for_cell(sic: float, thickness_m: float, ice_class: str, use_decayed_table: bool = False) -> PolarisMeta:
    ice_class = ice_class.upper()
    ice_t = thickness_to_ice_type(thickness_m, sic)
    c_ice = int(np.clip(np.rint(10.0 * float(sic)), 0, 10))
    c_open = 10 - c_ice

    table = RIV_TABLE_1_4 if use_decayed_table else RIV_TABLE_1_3
    riv_used = "table_1_4" if use_decayed_table else "table_1_3"

    riv_ice = table.get((ice_class, ice_t), None)
    riv_open = table.get((ice_class, "ice_free"), None)
    if riv_ice is None or riv_open is None:
        raise KeyError(f"Missing RIV mapping for ice_class={ice_class}, ice_type={ice_t}")

    rio = (c_open * riv_open) + (c_ice * riv_ice)
    level = classify_operation_level(rio, ice_class)
    spd = recommended_speed_limit_knots(level, ice_class)
    return PolarisMeta(ice_type=ice_t, rio=float(rio), level=level, speed_limit_knots=spd, riv_used=riv_used)


