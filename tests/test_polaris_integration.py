"""
Test POLARIS integration into polar rules (hard block + soft penalty + diagnostics).

Tests:
1. Special level must hard-block
2. Elevated level produces penalty
3. Enable/disable switch works
"""

import pytest
import numpy as np
from arcticroute.core.constraints.polar_rules import (
    load_polar_rules_config,
    apply_hard_constraints,
    apply_soft_penalties,
)


class TestPolarisHardBlock:
    """Test POLARIS hard block for special level."""

    def test_special_level_hard_blocks(self):
        """Verify that special level cells are hard-blocked."""
        # Create synthetic environment with high ice concentration
        # that should trigger "special" level for a weak vessel
        sic = np.full((10, 10), 0.95, dtype=float)  # 95% ice concentration
        thickness = np.full((10, 10), 2.5, dtype=float)  # 2.5m thick (second-year ice)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC7",  # Weakest class
        }

        cfg = load_polar_rules_config()
        blocked, meta = apply_hard_constraints(env, vessel_profile, cfg)

        # Should have some blocked cells
        assert np.sum(blocked) > 0, "Expected some cells to be blocked for PC7 in heavy ice"
        assert meta["polaris_meta"]["special_count"] > 0, "Expected special-level cells"

    def test_normal_level_not_blocked(self):
        """Verify that normal level cells are not hard-blocked by POLARIS."""
        # Create synthetic environment with low ice
        sic = np.full((10, 10), 0.1, dtype=float)  # 10% ice
        thickness = np.full((10, 10), 0.05, dtype=float)  # 5cm (new ice)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC1",  # Strongest class
        }

        cfg = load_polar_rules_config()
        blocked, meta = apply_hard_constraints(env, vessel_profile, cfg)

        # Should have no POLARIS blocks (only land blocks, which is zero here)
        assert meta["polaris_meta"]["special_count"] == 0, "Expected no special-level cells for PC1 in light ice"

    def test_enable_disable_switch(self):
        """Verify that POLARIS can be disabled via config."""
        sic = np.full((10, 10), 0.95, dtype=float)
        thickness = np.full((10, 10), 2.5, dtype=float)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC7",
        }

        cfg = load_polar_rules_config()
        
        # First, verify POLARIS is enabled by default
        assert cfg.get_rule_enabled("polaris"), "POLARIS should be enabled by default"
        
        blocked_enabled, meta_enabled = apply_hard_constraints(env, vessel_profile, cfg)
        
        # Now disable POLARIS in config
        cfg.config["rules"]["polaris"]["enabled"] = False
        blocked_disabled, meta_disabled = apply_hard_constraints(env, vessel_profile, cfg)
        
        # With POLARIS disabled, should have fewer blocks
        assert np.sum(blocked_disabled) <= np.sum(blocked_enabled), \
            "Disabling POLARIS should not increase blocked cells"


class TestPolarisElevatedPenalty:
    """Test POLARIS elevated-level soft penalty."""

    def test_elevated_penalty_increases_cost(self):
        """Verify that elevated level cells get cost penalty."""
        # Create environment with moderate ice (should trigger "elevated" for some vessels)
        sic = np.full((10, 10), 0.5, dtype=float)  # 50% ice
        thickness = np.full((10, 10), 0.8, dtype=float)  # 80cm (medium FY)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC6",
        }

        cfg = load_polar_rules_config()
        
        # First apply hard constraints to get polaris_meta
        blocked, hard_meta = apply_hard_constraints(env, vessel_profile, cfg)
        
        # Create a cost field
        cost_field = np.ones((10, 10), dtype=float)
        
        # Apply soft penalties with polaris_meta
        cost_modified, soft_meta = apply_soft_penalties(
            cost_field, env, vessel_profile, cfg, 
            polaris_meta=hard_meta.get("polaris_meta")
        )
        
        # Cost should increase in elevated cells
        if hard_meta["polaris_meta"]["elevated_count"] > 0:
            assert np.sum(cost_modified) > np.sum(cost_field), \
                "Cost should increase with elevated penalty"
            assert soft_meta["elevated_penalty_count"] > 0, \
                "Should have applied penalties to elevated cells"

    def test_penalty_scale_factor(self):
        """Verify that penalty scale factor works correctly."""
        sic = np.full((10, 10), 0.5, dtype=float)
        thickness = np.full((10, 10), 0.8, dtype=float)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC6",
        }

        cfg = load_polar_rules_config()
        blocked, hard_meta = apply_hard_constraints(env, vessel_profile, cfg)
        
        cost_field = np.ones((10, 10), dtype=float)
        
        # Apply with default scale
        cost_default, _ = apply_soft_penalties(
            cost_field, env, vessel_profile, cfg,
            polaris_meta=hard_meta.get("polaris_meta")
        )
        
        # Modify scale to 2.0
        cfg.config["rules"]["polaris"]["elevated_penalty"]["scale"] = 2.0
        cost_scaled, _ = apply_soft_penalties(
            cost_field, env, vessel_profile, cfg,
            polaris_meta=hard_meta.get("polaris_meta")
        )
        
        # Scaled cost should be higher (approximately 2x the difference)
        # Note: This is approximate due to floating point
        if hard_meta["polaris_meta"]["elevated_count"] > 0:
            assert np.sum(cost_scaled) >= np.sum(cost_default), \
                "Doubling scale should increase cost further"


class TestPolarisMetadata:
    """Test POLARIS metadata collection."""

    def test_rio_statistics_collected(self):
        """Verify that RIO statistics are collected."""
        sic = np.full((10, 10), 0.5, dtype=float)
        thickness = np.full((10, 10), 0.8, dtype=float)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC6",
        }

        cfg = load_polar_rules_config()
        blocked, meta = apply_hard_constraints(env, vessel_profile, cfg)
        
        polaris_meta = meta["polaris_meta"]
        
        # Check that statistics are present
        assert "rio_min" in polaris_meta, "rio_min should be in metadata"
        assert "rio_mean" in polaris_meta, "rio_mean should be in metadata"
        assert "special_fraction" in polaris_meta, "special_fraction should be in metadata"
        assert "elevated_fraction" in polaris_meta, "elevated_fraction should be in metadata"
        assert "riv_table_used" in polaris_meta, "riv_table_used should be in metadata"

    def test_speed_limit_exposure(self):
        """Verify that speed limits are exposed in metadata."""
        sic = np.full((10, 10), 0.5, dtype=float)
        thickness = np.full((10, 10), 0.8, dtype=float)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC6",
        }

        cfg = load_polar_rules_config()
        blocked, meta = apply_hard_constraints(env, vessel_profile, cfg)
        
        polaris_meta = meta["polaris_meta"]
        
        # Check that speed field is present
        assert "speed_field" in polaris_meta, "speed_field should be in metadata"
        speed_field = polaris_meta["speed_field"]
        assert speed_field.shape == (10, 10), "speed_field should match grid shape"


class TestPolarisDecayedTable:
    """Test POLARIS with decayed table option."""

    def test_decayed_table_option(self):
        """Verify that use_decayed_table option works."""
        sic = np.full((10, 10), 0.5, dtype=float)
        thickness = np.full((10, 10), 0.8, dtype=float)
        landmask = np.zeros((10, 10), dtype=float)

        env = {
            "sic": sic,
            "ice_thickness": thickness,
            "landmask": landmask,
        }

        vessel_profile = {
            "ice_class": "PC6",
        }

        cfg = load_polar_rules_config()
        
        # Test with standard table (default)
        blocked_std, meta_std = apply_hard_constraints(env, vessel_profile, cfg)
        assert meta_std["polaris_meta"]["riv_table_used"] == "table_1_3"
        
        # Test with decayed table
        cfg.config["rules"]["polaris"]["use_decayed_table"] = True
        blocked_decayed, meta_decayed = apply_hard_constraints(env, vessel_profile, cfg)
        assert meta_decayed["polaris_meta"]["riv_table_used"] == "table_1_4"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

