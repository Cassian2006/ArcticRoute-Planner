"""
Polar Rules Engine - Traceable, Configurable, Testable

Implements hard constraints (blocked cells) and soft penalties based on:
- Wave (SWH) thresholds
- Sea ice concentration (SIC) thresholds
- Ice thickness thresholds
- Land mask (always blocked)

All thresholds are loaded from configuration file (arcticroute/config/polar_rules.yaml)
and are fully traceable to authoritative sources.

Missing values do not crash the system; they trigger warnings and optionally disable rules.
"""

import logging
import warnings
from typing import Dict, Any, Tuple, Optional
import numpy as np
import yaml
from pathlib import Path

# Optional POLARIS integration
from arcticroute.core.constraints.polaris import compute_rio_for_cell

logger = logging.getLogger(__name__)


class PolarRulesConfig:
    """Load and validate polar rules configuration."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to polar_rules.yaml. If None, uses default location.
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "polar_rules.yaml"
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.warnings = []

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config file not found: {self.config_path}")
            return self._default_config()
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config or self._default_config()
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return self._default_config()

    @staticmethod
    def _default_config() -> Dict[str, Any]:
        """Return default (conservative) configuration."""
        return {
            "version": "0.1",
            "global": {
                "enabled": True,
                "missing_value_policy": "warn_and_disable_rule",
                "land_is_blocked": True,
            },
            "rules": {
                "wave": {"enabled": True, "swh_max_m": {"default": None}},
                "sic": {"enabled": True, "sic_max": {"default": None}},
                "ice_thickness": {"enabled": True, "thickness_max_m": {"default": None}},
                "speed_penalty": {"enabled": False},
            },
        }

    def is_global_enabled(self) -> bool:
        """Check if global rules are enabled."""
        return self.config.get("global", {}).get("enabled", True)

    def get_rule_enabled(self, rule_name: str) -> bool:
        """Check if a specific rule is enabled."""
        if not self.is_global_enabled():
            return False
        return self.config.get("rules", {}).get(rule_name, {}).get("enabled", False)

    def get_missing_value_policy(self) -> str:
        """Get policy for missing threshold values."""
        return self.config.get("global", {}).get("missing_value_policy", "warn_and_disable_rule")

    def is_land_blocked(self) -> bool:
        """Check if land is always blocked."""
        return self.config.get("global", {}).get("land_is_blocked", True)


def load_polar_rules_config(path: Optional[str] = None) -> PolarRulesConfig:
    """
    Load polar rules configuration.

    Args:
        path: Path to polar_rules.yaml

    Returns:
        PolarRulesConfig instance
    """
    return PolarRulesConfig(path)


def resolve_threshold(
    rule_key: str,
    param_key: str,
    vessel_profile: Optional[Dict[str, Any]],
    rules_cfg: PolarRulesConfig,
) -> Optional[float]:
    """
    Resolve a threshold value based on vessel profile and configuration.

    Args:
        rule_key: e.g., "wave", "sic", "ice_thickness"
        param_key: e.g., "swh_max_m", "sic_max", "thickness_max_m"
        vessel_profile: Dict with vessel_type, ice_class, etc.
        rules_cfg: PolarRulesConfig instance

    Returns:
        Threshold value (float) or None if not found/disabled
    """
    if not rules_cfg.get_rule_enabled(rule_key):
        return None

    rule_cfg = rules_cfg.config.get("rules", {}).get(rule_key, {})
    param_cfg = rule_cfg.get(param_key, {})

    if not isinstance(param_cfg, dict):
        return None

    # Try vessel_type first
    if vessel_profile:
        vessel_type = vessel_profile.get("vessel_type")
        if vessel_type and vessel_type in param_cfg.get("by_vessel_type", {}):
            return param_cfg["by_vessel_type"][vessel_type]

        # Try ice_class
        ice_class = vessel_profile.get("ice_class")
        if ice_class and ice_class in param_cfg.get("by_ice_class", {}):
            return param_cfg["by_ice_class"][ice_class]

    # Fall back to default
    default_val = param_cfg.get("default")
    return default_val


def apply_hard_constraints(
    env: Dict[str, Any],
    vessel_profile: Optional[Dict[str, Any]],
    rules_cfg: PolarRulesConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply hard constraints (blocked cells) based on rules.

    Args:
        env: Environment dict with keys like 'landmask', 'sic', 'wave', 'ice_thickness'
             Each is expected to be a 2D numpy array
        vessel_profile: Dict with vessel_type, ice_class, etc.
        rules_cfg: PolarRulesConfig instance

    Returns:
        (blocked_mask, meta)
        - blocked_mask: bool 2D array, True = blocked
        - meta: Dict with rule application details
    """
    meta = {
        "rules_enabled": rules_cfg.is_global_enabled(),
        "rules_applied": [],
        "warnings": [],
        "blocked_count": 0,
        "polaris_meta": {},
    }

    if not rules_cfg.is_global_enabled():
        # Return all False (nothing blocked by rules)
        shape = env.get("landmask", np.zeros((1, 1))).shape
        return np.zeros(shape, dtype=bool), meta

    # Initialize with land mask
    shape = env.get("landmask", np.zeros((1, 1))).shape
    blocked = np.zeros(shape, dtype=bool)

    if rules_cfg.is_land_blocked():
        landmask = env.get("landmask")
        if landmask is not None:
            blocked = blocked | (landmask > 0)
            meta["rules_applied"].append("land_blocked")

    # Wave constraint
    if rules_cfg.get_rule_enabled("wave"):
        swh_threshold = resolve_threshold("wave", "swh_max_m", vessel_profile, rules_cfg)
        if swh_threshold is not None:
            wave_data = env.get("wave")
            if wave_data is not None:
                wave_blocked = wave_data > swh_threshold
                blocked = blocked | wave_blocked
                meta["rules_applied"].append(f"wave (swh_max={swh_threshold}m)")
                meta["wave_blocked_count"] = int(np.sum(wave_blocked))
        else:
            msg = "wave rule enabled but swh_max_m threshold not found"
            if rules_cfg.get_missing_value_policy() == "error":
                raise ValueError(msg)
            else:
                meta["warnings"].append(msg)

    # SIC constraint
    if rules_cfg.get_rule_enabled("sic"):
        sic_threshold = resolve_threshold("sic", "sic_max", vessel_profile, rules_cfg)
        if sic_threshold is not None:
            sic_data = env.get("sic")
            if sic_data is not None:
                sic_blocked = sic_data > sic_threshold
                blocked = blocked | sic_blocked
                meta["rules_applied"].append(f"sic (sic_max={sic_threshold})")
                meta["sic_blocked_count"] = int(np.sum(sic_blocked))
        else:
            msg = "sic rule enabled but sic_max threshold not found"
            if rules_cfg.get_missing_value_policy() == "error":
                raise ValueError(msg)
            else:
                meta["warnings"].append(msg)

    # Ice thickness constraint
    if rules_cfg.get_rule_enabled("ice_thickness"):
        thickness_threshold = resolve_threshold(
            "ice_thickness", "thickness_max_m", vessel_profile, rules_cfg
        )
        if thickness_threshold is not None:
            thickness_data = env.get("ice_thickness")
            if thickness_data is not None:
                thickness_blocked = thickness_data > thickness_threshold
                blocked = blocked | thickness_blocked
                meta["rules_applied"].append(f"ice_thickness (max={thickness_threshold}m)")
                meta["thickness_blocked_count"] = int(np.sum(thickness_blocked))
        else:
            msg = "ice_thickness rule enabled but thickness_max_m threshold not found"
            if rules_cfg.get_missing_value_policy() == "error":
                raise ValueError(msg)
            else:
                meta["warnings"].append(msg)

    # POLARIS constraint (hard block for "special" level)
    if rules_cfg.get_rule_enabled("polaris"):
        polaris_blocked, polaris_meta = _apply_polaris_hard_constraints(
            env, vessel_profile, rules_cfg
        )
        blocked = blocked | polaris_blocked
        meta["rules_applied"].append("polaris_hard_block")
        meta["polaris_meta"] = polaris_meta

    meta["blocked_count"] = int(np.sum(blocked))
    meta["total_cells"] = int(np.prod(shape))
    meta["blocked_fraction"] = (
        meta["blocked_count"] / meta["total_cells"] if meta["total_cells"] > 0 else 0.0
    )

    return blocked, meta


def _apply_polaris_hard_constraints(
    env: Dict[str, Any],
    vessel_profile: Optional[Dict[str, Any]],
    rules_cfg: PolarRulesConfig,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply POLARIS hard constraints (block cells with level == "special").

    Args:
        env: Environment dict with 'sic' and 'ice_thickness'
        vessel_profile: Dict with 'ice_class'
        rules_cfg: PolarRulesConfig instance

    Returns:
        (blocked_mask, meta)
    """
    meta = {
        "polaris_enabled": True,
        "rio_min": None,
        "rio_mean": None,
        "special_fraction": 0.0,
        "elevated_fraction": 0.0,
        "riv_table_used": "table_1_3",
        "special_count": 0,
        "elevated_count": 0,
        "warnings": [],
    }

    sic_data = env.get("sic")
    thickness_data = env.get("ice_thickness")
    
    if sic_data is None or thickness_data is None:
        meta["warnings"].append("Missing sic or ice_thickness data for POLARIS")
        shape = sic_data.shape if sic_data is not None else thickness_data.shape
        return np.zeros(shape, dtype=bool), meta

    if vessel_profile is None or "ice_class" not in vessel_profile:
        meta["warnings"].append("Missing ice_class in vessel_profile for POLARIS")
        return np.zeros(sic_data.shape, dtype=bool), meta

    ice_class = vessel_profile.get("ice_class", "PC6").upper()
    polaris_cfg = rules_cfg.config.get("rules", {}).get("polaris", {})
    use_decayed = polaris_cfg.get("use_decayed_table", False)
    hard_block_level = polaris_cfg.get("hard_block_level", "special")

    # Compute RIO for each cell
    shape = sic_data.shape
    rio_field = np.full(shape, np.nan, dtype=float)
    level_field = np.full(shape, "", dtype=object)
    speed_field = np.full(shape, np.nan, dtype=float)

    blocked = np.zeros(shape, dtype=bool)
    special_count = 0
    elevated_count = 0
    rio_values = []

    for i in range(shape[0]):
        for j in range(shape[1]):
            sic = float(sic_data[i, j])
            thickness = float(thickness_data[i, j])

            # Skip NaN values
            if np.isnan(sic) or np.isnan(thickness):
                continue

            try:
                polaris_meta = compute_rio_for_cell(
                    sic=sic,
                    thickness_m=thickness,
                    ice_class=ice_class,
                    use_decayed_table=use_decayed,
                )
                rio_field[i, j] = polaris_meta.rio
                level_field[i, j] = polaris_meta.level
                speed_field[i, j] = polaris_meta.speed_limit_knots if polaris_meta.speed_limit_knots else np.nan
                rio_values.append(polaris_meta.rio)

                # Hard block if level matches hard_block_level
                if polaris_meta.level == hard_block_level:
                    blocked[i, j] = True
                    special_count += 1
                elif polaris_meta.level == "elevated":
                    elevated_count += 1

                meta["riv_table_used"] = polaris_meta.riv_used

            except Exception as e:
                meta["warnings"].append(f"POLARIS computation failed at ({i},{j}): {e}")

    # Compute statistics
    if rio_values:
        meta["rio_min"] = float(np.min(rio_values))
        meta["rio_mean"] = float(np.mean(rio_values))
    
    total_valid = len(rio_values)
    if total_valid > 0:
        meta["special_fraction"] = special_count / total_valid
        meta["elevated_fraction"] = elevated_count / total_valid
    
    meta["special_count"] = special_count
    meta["elevated_count"] = elevated_count
    meta["total_valid_cells"] = total_valid

    # Store fields for later use (e.g., in soft penalties)
    meta["rio_field"] = rio_field
    meta["level_field"] = level_field
    meta["speed_field"] = speed_field

    return blocked, meta


def apply_soft_penalties(
    cost_field: np.ndarray,
    env: Dict[str, Any],
    vessel_profile: Optional[Dict[str, Any]],
    rules_cfg: PolarRulesConfig,
    polaris_meta: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Apply soft penalties (cost modifications) based on rules.

    Includes POLARIS elevated-level penalty and optional speed/fuel penalties.

    Args:
        cost_field: 2D cost array
        env: Environment dict
        vessel_profile: Vessel profile dict
        rules_cfg: PolarRulesConfig instance
        polaris_meta: Optional metadata from hard constraints (contains rio_field, level_field)

    Returns:
        (modified_cost_field, meta)
    """
    meta = {
        "soft_penalties_applied": [],
        "warnings": [],
        "polaris_penalty_added": False,
        "elevated_penalty_count": 0,
    }

    cost_modified = cost_field.copy()

    # POLARIS elevated penalty
    if rules_cfg.get_rule_enabled("polaris") and polaris_meta is not None:
        polaris_cfg = rules_cfg.config.get("rules", {}).get("polaris", {})
        elevated_cfg = polaris_cfg.get("elevated_penalty", {})
        
        if elevated_cfg.get("enabled", False):
            rio_field = polaris_meta.get("rio_field")
            level_field = polaris_meta.get("level_field")
            
            if rio_field is not None and level_field is not None:
                scale = float(elevated_cfg.get("scale", 1.0))
                shape = cost_modified.shape
                
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        if level_field[i, j] == "elevated":
                            rio = rio_field[i, j]
                            # penalty = scale * max(0, -rio) / 10
                            penalty = scale * max(0.0, -rio) / 10.0
                            cost_modified[i, j] += penalty
                            meta["elevated_penalty_count"] += 1
                
                meta["soft_penalties_applied"].append(f"polaris_elevated (scale={scale})")
                meta["polaris_penalty_added"] = True

    # Placeholder: speed_penalty rule is not yet implemented
    if rules_cfg.get_rule_enabled("speed_penalty"):
        meta["warnings"].append("speed_penalty rule enabled but not yet implemented")

    return cost_modified, meta


def integrate_hard_constraints_into_cost(
    cost_field: np.ndarray,
    blocked_mask: np.ndarray,
    blocked_value: float = 1e10,
) -> np.ndarray:
    """
    Integrate hard constraint mask into cost field.

    Args:
        cost_field: 2D cost array
        blocked_mask: bool 2D array, True = blocked
        blocked_value: Cost value to assign to blocked cells (default: 1e10)

    Returns:
        Modified cost field with blocked cells set to blocked_value
    """
    cost_modified = cost_field.copy()
    cost_modified[blocked_mask] = blocked_value
    return cost_modified

