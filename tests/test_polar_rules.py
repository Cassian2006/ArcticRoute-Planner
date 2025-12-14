"""
Unit tests for Polar Rules Engine

Tests cover:
- Configuration loading and validation
- Threshold resolution (by vessel type, ice class, default)
- Hard constraints (blocked cells)
- Missing value handling (no crash, warnings)
- Land mask always blocked
- Boundary conditions (equal to threshold)
- Soft penalties (placeholder)
- Cost field integration
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import yaml

from arcticroute.core.constraints.polar_rules import (
    PolarRulesConfig,
    load_polar_rules_config,
    resolve_threshold,
    apply_hard_constraints,
    apply_soft_penalties,
    integrate_hard_constraints_into_cost,
)


class TestPolarRulesConfig:
    """Test configuration loading and validation."""

    def test_load_default_config(self):
        """Test loading default config when file doesn't exist."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        assert cfg.is_global_enabled()
        assert cfg.is_land_blocked()
        assert cfg.get_missing_value_policy() == "warn_and_disable_rule"

    def test_load_from_file(self):
        """Test loading config from actual file."""
        # Use the default config file location
        cfg = load_polar_rules_config()
        assert cfg.is_global_enabled()
        assert cfg.is_land_blocked()

    def test_rule_enabled_checks(self):
        """Test rule enabled/disabled checks."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        assert cfg.get_rule_enabled("wave")
        assert cfg.get_rule_enabled("sic")
        assert cfg.get_rule_enabled("ice_thickness")
        assert not cfg.get_rule_enabled("speed_penalty")

    def test_global_disabled_disables_all_rules(self):
        """Test that disabling global rules disables all individual rules."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": False},
                "rules": {"wave": {"enabled": True}}
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            assert not cfg.get_rule_enabled("wave")


class TestThresholdResolution:
    """Test threshold resolution logic."""

    def test_resolve_default_threshold(self):
        """Test resolving default threshold."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        # Default config has None values, so should return None
        threshold = resolve_threshold("wave", "swh_max_m", None, cfg)
        assert threshold is None

    def test_resolve_by_vessel_type(self):
        """Test resolving threshold by vessel type."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True},
                "rules": {
                    "wave": {
                        "enabled": True,
                        "swh_max_m": {
                            "default": 5.0,
                            "by_vessel_type": {"PC6": 4.0, "PC7": 3.5},
                            "by_ice_class": {}
                        }
                    }
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            # Should return vessel-type-specific value
            vessel = {"vessel_type": "PC6", "ice_class": "1A"}
            threshold = resolve_threshold("wave", "swh_max_m", vessel, cfg)
            assert threshold == 4.0

    def test_resolve_by_ice_class(self):
        """Test resolving threshold by ice class."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True},
                "rules": {
                    "sic": {
                        "enabled": True,
                        "sic_max": {
                            "default": 0.95,
                            "by_vessel_type": {},
                            "by_ice_class": {"1A": 0.80, "1B": 0.85}
                        }
                    }
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            vessel = {"vessel_type": "Generic", "ice_class": "1A"}
            threshold = resolve_threshold("sic", "sic_max", vessel, cfg)
            assert threshold == 0.80

    def test_resolve_fallback_to_default(self):
        """Test fallback to default when vessel type/ice class not found."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True},
                "rules": {
                    "wave": {
                        "enabled": True,
                        "swh_max_m": {
                            "default": 5.0,
                            "by_vessel_type": {"PC6": 4.0},
                            "by_ice_class": {}
                        }
                    }
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            # Unknown vessel type should fall back to default
            vessel = {"vessel_type": "Unknown", "ice_class": "1C"}
            threshold = resolve_threshold("wave", "swh_max_m", vessel, cfg)
            assert threshold == 5.0


class TestHardConstraints:
    """Test hard constraint application."""

    def test_land_always_blocked(self):
        """Test that land is always blocked."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        
        env = {
            "landmask": np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]),
            "sic": np.array([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]),
            "wave": np.array([[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]),
        }
        
        blocked, meta = apply_hard_constraints(env, None, cfg)
        
        # Land cells should be blocked
        assert blocked[0, 1] == True
        assert blocked[1, 0] == True
        assert blocked[1, 2] == True
        assert blocked[2, 1] == True
        
        # Ocean cells should not be blocked (no thresholds set)
        assert blocked[0, 0] == False
        assert blocked[0, 2] == False

    def test_wave_constraint_blocks_high_waves(self):
        """Test that high wave cells are blocked."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True, "land_is_blocked": False},
                "rules": {
                    "wave": {
                        "enabled": True,
                        "swh_max_m": {
                            "default": 4.0,
                            "by_vessel_type": {},
                            "by_ice_class": {}
                        }
                    },
                    "sic": {"enabled": False},
                    "ice_thickness": {"enabled": False},
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            env = {
                "landmask": np.zeros((3, 3)),
                "wave": np.array([[2.0, 4.0, 5.0], [3.0, 4.0, 6.0], [2.0, 3.0, 4.0]]),
            }
            
            blocked, meta = apply_hard_constraints(env, None, cfg)
            
            # Cells with wave > 4.0 should be blocked
            assert blocked[0, 2] == True  # 5.0
            assert blocked[1, 2] == True  # 6.0
            
            # Cells with wave <= 4.0 should not be blocked
            assert blocked[0, 0] == False  # 2.0
            assert blocked[0, 1] == False  # 4.0 (boundary, not blocked)
            assert blocked[1, 1] == False  # 4.0 (boundary, not blocked)

    def test_sic_constraint_blocks_high_concentration(self):
        """Test that high SIC cells are blocked."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True, "land_is_blocked": False},
                "rules": {
                    "wave": {"enabled": False},
                    "sic": {
                        "enabled": True,
                        "sic_max": {
                            "default": 0.80,
                            "by_vessel_type": {},
                            "by_ice_class": {}
                        }
                    },
                    "ice_thickness": {"enabled": False},
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            env = {
                "landmask": np.zeros((3, 3)),
                "sic": np.array([[0.5, 0.8, 0.9], [0.7, 0.8, 0.95], [0.6, 0.75, 0.8]]),
            }
            
            blocked, meta = apply_hard_constraints(env, None, cfg)
            
            # Cells with sic > 0.80 should be blocked
            assert blocked[0, 2] == True  # 0.9
            assert blocked[1, 2] == True  # 0.95
            
            # Cells with sic <= 0.80 should not be blocked
            assert blocked[0, 0] == False  # 0.5
            assert blocked[0, 1] == False  # 0.8 (boundary, not blocked)

    def test_ice_thickness_constraint(self):
        """Test ice thickness constraint."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True, "land_is_blocked": False},
                "rules": {
                    "wave": {"enabled": False},
                    "sic": {"enabled": False},
                    "ice_thickness": {
                        "enabled": True,
                        "thickness_max_m": {
                            "default": 2.0,
                            "by_vessel_type": {},
                            "by_ice_class": {}
                        }
                    },
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            env = {
                "landmask": np.zeros((3, 3)),
                "ice_thickness": np.array([[1.0, 2.0, 2.5], [1.5, 2.0, 3.0], [0.5, 1.5, 2.0]]),
            }
            
            blocked, meta = apply_hard_constraints(env, None, cfg)
            
            # Cells with thickness > 2.0 should be blocked
            assert blocked[0, 2] == True  # 2.5
            assert blocked[1, 2] == True  # 3.0
            
            # Cells with thickness <= 2.0 should not be blocked
            assert blocked[0, 0] == False  # 1.0
            assert blocked[0, 1] == False  # 2.0 (boundary, not blocked)

    def test_missing_threshold_warning(self):
        """Test that missing thresholds generate warnings, not crashes."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        
        env = {
            "landmask": np.zeros((3, 3)),
            "wave": np.array([[2.0, 3.0, 4.0], [2.0, 3.0, 4.0], [2.0, 3.0, 4.0]]),
        }
        
        # Should not crash, should generate warning
        blocked, meta = apply_hard_constraints(env, None, cfg)
        
        assert len(meta["warnings"]) > 0
        assert "swh_max_m threshold not found" in meta["warnings"][0]

    def test_blocked_fraction_calculation(self):
        """Test blocked fraction calculation."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True, "land_is_blocked": True},
                "rules": {
                    "wave": {"enabled": False},
                    "sic": {"enabled": False},
                    "ice_thickness": {"enabled": False},
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            # 2x2 grid with 1 land cell
            env = {
                "landmask": np.array([[1, 0], [0, 0]]),
            }
            
            blocked, meta = apply_hard_constraints(env, None, cfg)
            
            assert meta["total_cells"] == 4
            assert meta["blocked_count"] == 1
            assert meta["blocked_fraction"] == 0.25


class TestSoftPenalties:
    """Test soft penalty application."""

    def test_soft_penalties_placeholder(self):
        """Test that soft penalties is a placeholder (no-op)."""
        cfg = PolarRulesConfig(config_path="/nonexistent/path.yaml")
        
        cost = np.array([[1.0, 2.0], [3.0, 4.0]])
        env = {}
        
        cost_modified, meta = apply_soft_penalties(cost, env, None, cfg)
        
        # Should return copy of original cost (no modifications)
        assert np.allclose(cost_modified, cost)


class TestCostIntegration:
    """Test integration of constraints into cost field."""

    def test_integrate_blocked_into_cost(self):
        """Test integrating blocked mask into cost field."""
        cost = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        blocked = np.array([[False, True, False], [True, False, False], [False, False, True]])
        
        cost_integrated = integrate_hard_constraints_into_cost(cost, blocked, blocked_value=1e10)
        
        # Blocked cells should have high cost
        assert cost_integrated[0, 1] == 1e10
        assert cost_integrated[1, 0] == 1e10
        assert cost_integrated[2, 2] == 1e10
        
        # Non-blocked cells should keep original cost
        assert cost_integrated[0, 0] == 1.0
        assert cost_integrated[0, 2] == 3.0
        assert cost_integrated[1, 1] == 5.0


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_constraint_pipeline(self):
        """Test full pipeline: config -> constraints -> cost integration."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({
                "global": {"enabled": True, "land_is_blocked": True},
                "rules": {
                    "wave": {
                        "enabled": True,
                        "swh_max_m": {
                            "default": 4.0,
                            "by_vessel_type": {},
                            "by_ice_class": {}
                        }
                    },
                    "sic": {"enabled": False},
                    "ice_thickness": {"enabled": False},
                }
            }, f)
            f.flush()
            cfg = PolarRulesConfig(config_path=f.name)
            
            # Create environment
            env = {
                "landmask": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]]),
                "wave": np.array([[2.0, 3.0, 5.0], [2.0, 4.0, 6.0], [3.0, 3.0, 2.0]]),
            }
            
            # Apply constraints
            blocked, meta = apply_hard_constraints(env, None, cfg)
            
            # Integrate into cost
            cost = np.ones((3, 3))
            cost_integrated = integrate_hard_constraints_into_cost(cost, blocked, blocked_value=1e10)
            
            # Check results
            assert cost_integrated[0, 0] == 1e10  # Land
            assert cost_integrated[0, 2] == 1e10  # High wave (5.0 > 4.0)
            assert cost_integrated[1, 2] == 1e10  # High wave (6.0 > 4.0)
            assert cost_integrated[2, 2] == 1e10  # Land
            
            assert cost_integrated[0, 1] == 1.0  # Ocean, low wave
            assert cost_integrated[1, 1] == 1.0  # Ocean, wave at threshold


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

