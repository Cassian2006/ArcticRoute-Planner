"""
ArcticRoute 配置模块。

包含 EDL 模式、场景预设等共享配置。
"""

from .edl_modes import EDL_MODES, get_edl_mode_config, list_edl_modes, get_edl_mode_display_name
from .scenarios import SCENARIOS, get_scenario_by_name, list_scenarios, list_scenario_descriptions

__all__ = [
    "EDL_MODES",
    "get_edl_mode_config",
    "list_edl_modes",
    "get_edl_mode_display_name",
    "SCENARIOS",
    "get_scenario_by_name",
    "list_scenarios",
    "list_scenario_descriptions",
]

