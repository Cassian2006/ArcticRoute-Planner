from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Tuple

import streamlit as st

from arcticroute.core.eco.vessel_profiles import get_default_profiles


def render_vessel_selector(key_prefix: str = "vessel") -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (profile_key, ice_class_label, meta_dict)
    profile_key: the chosen key in get_default_profiles()
    ice_class_label: best-effort label parsed from key
    meta_dict: serializable info for cost_breakdown/meta
    """
    profiles = get_default_profiles()
    keys = sorted(list(profiles.keys()))
    if not keys:
        st.warning("No vessel profiles available.")
        return ("", "", {"vessel_profile_key": "", "vessel_profile": None})

    # 从 session_state 获取默认值
    default_key = st.session_state.get("vessel_profile_key", keys[0])
    if default_key not in keys:
        default_key = keys[0]
    
    profile_key = st.selectbox(
        "船舶/冰级（Vessel Profile）",
        options=keys,
        index=keys.index(default_key),
        key=f"{key_prefix}_profile_key",
        help="来自 arcticroute.core.eco.vessel_profiles.get_default_profiles()",
    )

    prof = profiles.get(profile_key)
    # best-effort parse: keys often contain PC? or ice class tokens; keep raw key anyway
    ice_label = ""
    for token in ["PC1","PC2","PC3","PC4","PC5","PC6","PC7"]:
        if token in profile_key.upper():
            ice_label = token
            break

    meta = {
        "vessel_profile_key": profile_key,
        "ice_class_label": ice_label,
        "vessel_profile": asdict(prof) if hasattr(prof, "__dataclass_fields__") else (prof.__dict__ if prof is not None else None),
    }
    return profile_key, ice_label, meta

