"""
Polar rules engine with authoritative rules support.

This module provides functionality to load and apply polar rules,
including support for authoritative rules compiled from templates and overrides.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

from .polar_rules_compile import compile_authoritative_rules


class PolarRulesEngine:
    """Engine for applying polar rules with authoritative rules support."""
    
    def __init__(self, rules_config_path: Optional[str] = None):
        """
        Initialize the polar rules engine.
        
        Args:
            rules_config_path: Path to rules configuration file.
                              If contains 'authoritative_template' or 'authoritative',
                              will use compile_authoritative_rules.
        """
        self.rules_config_path = rules_config_path
        self.rules_config = None
        self.meta = {}
        self.missing_thresholds = []
        
        if rules_config_path:
            self._load_rules(rules_config_path)
    
    def _load_rules(self, config_path: str) -> None:
        """Load rules configuration from file."""
        path = Path(config_path)
        
        # Check if this is an authoritative rules file
        if any(keyword in path.name.lower() for keyword in ['authoritative_template', 'authoritative']):
            # Use compile_authoritative_rules
            override_path = os.getenv("ARCTICROUTE_RULES_OVERRIDE")
            compiled, meta = compile_authoritative_rules(
                template_path=config_path,
                override_path=override_path
            )
            self.rules_config = compiled
            self.meta = meta
            
            # Check missing value policy
            missing_policy = (self.rules_config.get("rules") or {}).get("missing_value_policy", "warn")
            
            # Identify missing thresholds
            thresholds = self.rules_config.get("thresholds") or {}
            self.missing_thresholds = [
                k for k, v in thresholds.items() 
                if isinstance(v, dict) and v.get("value") is None
            ]
            
            # Handle missing values according to policy
            if missing_policy == "block" and self.missing_thresholds:
                raise ValueError(
                    f"Missing required threshold values: {self.missing_thresholds}. "
                    f"Set missing_value_policy to 'warn' or 'ignore' to proceed."
                )
            
        else:
            # Load regular polar rules file
            if not path.exists():
                raise FileNotFoundError(f"Rules configuration file not found: {config_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                self.rules_config = yaml.safe_load(f) or {}
            
            self.meta = {
                "ruleset_id": (self.rules_config.get("meta") or {}).get("ruleset_id", "legacy"),
                "template_path": None,
                "override_path": None,
                "missing_count": 0,
                "filled_count": 0,
            }
    
    def get_threshold(self, threshold_key: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific threshold configuration.
        
        Args:
            threshold_key: The key of the threshold to retrieve.
            
        Returns:
            Threshold configuration dict or None if not found.
        """
        if not self.rules_config:
            return None
            
        thresholds = self.rules_config.get("thresholds") or {}
        return thresholds.get(threshold_key)
    
    def check_threshold(self, threshold_key: str, value: Any) -> bool:
        """
        Check if a value violates a threshold.
        
        Args:
            threshold_key: The key of the threshold to check.
            value: The value to check.
            
        Returns:
            True if the value violates the threshold (i.e., should be blocked).
        """
        threshold = self.get_threshold(threshold_key)
        if not threshold or threshold.get("value") is None:
            return False  # No threshold configured, don't block
            
        threshold_value = threshold.get("value")
        op = threshold.get("op", ">=")
        
        try:
            if op == ">=":
                return float(value) >= float(threshold_value)
            elif op == ">":
                return float(value) > float(threshold_value)
            elif op == "<=":
                return float(value) <= float(threshold_value)
            elif op == "<":
                return float(value) < float(threshold_value)
            elif op == "==":
                return float(value) == float(threshold_value)
            elif op == "!=":
                return float(value) != float(threshold_value)
            else:
                return False  # Unknown operator
        except (ValueError, TypeError):
            return False  # Can't compare, don't block
    
    def get_vessel_override(self, vessel_name: str, threshold_key: str) -> Optional[Dict[str, Any]]:
        """
        Get vessel-specific threshold override.
        
        Args:
            vessel_name: Name of the vessel.
            threshold_key: The key of the threshold.
            
        Returns:
            Override configuration or None.
        """
        if not self.rules_config:
            return None
            
        overrides = (self.rules_config.get("overrides_by_vessel") or {})
        vessel_overrides = overrides.get(vessel_name, {})
        return vessel_overrides.get(threshold_key)
    
    def get_sources(self) -> List[Dict[str, Any]]:
        """Get the list of sources."""
        if not self.rules_config:
            return []
        return (self.rules_config.get("sources") or {}).get("items", [])
    
    def is_enabled(self) -> bool:
        """Check if rules are enabled."""
        if not self.rules_config:
            return False
        return (self.rules_config.get("rules") or {}).get("enabled", True)
    
    def get_meta(self) -> Dict[str, Any]:
        """Get metadata about the loaded rules."""
        return self.meta.copy()
    
    def get_missing_thresholds(self) -> List[str]:
        """Get list of thresholds with missing values."""
        return self.missing_thresholds.copy()


# Convenience function for backward compatibility
def load_polar_rules(rules_config_path: str) -> PolarRulesEngine:
    """
    Load polar rules from configuration file.
    
    Args:
        rules_config_path: Path to rules configuration file.
        
    Returns:
        PolarRulesEngine instance.
    """
    return PolarRulesEngine(rules_config_path)
