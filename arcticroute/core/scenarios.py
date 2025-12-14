"""Scenario configuration loading helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

DEFAULT_SCENARIOS_PATH = Path(__file__).resolve().parents[2] / "configs" / "scenarios.yaml"
ALLOWED_GRID_MODES = {"demo", "real"}
ALLOWED_BASE_PROFILES = {"efficient", "edl_safe", "edl_robust"}


@dataclass
class ScenarioConfig:
    id: str
    title: str
    description: str
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    ym: str
    grid_mode: str
    base_profile: str
    vessel: str
    w_ice: float
    w_wave: float
    w_ais: float
    w_ais_corridor: float
    w_ais_congestion: float
    use_edl: bool
    use_edl_uncertainty: bool
    reserved: Dict[str, Any] | None = None


def _ensure_grid_mode(mode: str, scenario_id: str) -> str:
    normalized = str(mode).lower()
    if normalized not in ALLOWED_GRID_MODES:
        raise ValueError(f"Scenario '{scenario_id}' has invalid grid_mode '{mode}', expected one of {sorted(ALLOWED_GRID_MODES)}")
    return normalized


def _ensure_base_profile(profile: str, scenario_id: str) -> str:
    normalized = str(profile).lower()
    if normalized not in ALLOWED_BASE_PROFILES:
        raise ValueError(
            f"Scenario '{scenario_id}' has invalid base_profile '{profile}', expected one of {sorted(ALLOWED_BASE_PROFILES)}"
        )
    return normalized


def load_all_scenarios(config_path: str | Path | None = None) -> dict[str, ScenarioConfig]:
    """Load all scenarios from configs/scenarios.yaml and return a mapping of id -> ScenarioConfig."""
    path = Path(config_path) if config_path is not None else DEFAULT_SCENARIOS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Scenario config not found: {path}")

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Scenario config must be a mapping at top level: {path}")

    raw_scenarios = payload.get("scenarios") or {}
    if not isinstance(raw_scenarios, dict):
        raise ValueError(f"'scenarios' must be a mapping of id -> config in {path}")

    required_fields = [
        "title",
        "description",
        "start_lat",
        "start_lon",
        "end_lat",
        "end_lon",
        "ym",
        "grid_mode",
        "base_profile",
        "vessel",
        "w_ice",
        "w_wave",
        "w_ais",
        "use_edl",
        "use_edl_uncertainty",
    ]
    optional_fields = ["w_ais_corridor", "w_ais_congestion"]

    scenarios: dict[str, ScenarioConfig] = {}
    for scen_id, raw in raw_scenarios.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Scenario '{scen_id}' must be a mapping of fields")

        missing = [key for key in required_fields if key not in raw]
        if missing:
            raise ValueError(f"Scenario '{scen_id}' is missing required fields: {missing}")

        grid_mode = _ensure_grid_mode(raw["grid_mode"], scen_id)
        base_profile = _ensure_base_profile(raw["base_profile"], scen_id)

        known_keys = set(required_fields + optional_fields)
        reserved = {k: v for k, v in raw.items() if k not in known_keys}

        w_ais_corridor = float(raw.get("w_ais_corridor", raw["w_ais"]))
        w_ais_congestion = float(raw.get("w_ais_congestion", 0.0))

        scenarios[scen_id] = ScenarioConfig(
            id=str(scen_id),
            title=str(raw["title"]),
            description=str(raw["description"]),
            start_lat=float(raw["start_lat"]),
            start_lon=float(raw["start_lon"]),
            end_lat=float(raw["end_lat"]),
            end_lon=float(raw["end_lon"]),
            ym=str(raw["ym"]),
            grid_mode=grid_mode,
            base_profile=base_profile,
            vessel=str(raw["vessel"]),
            w_ice=float(raw["w_ice"]),
            w_wave=float(raw["w_wave"]),
            w_ais=float(raw["w_ais"]),
            w_ais_corridor=w_ais_corridor,
            w_ais_congestion=w_ais_congestion,
            use_edl=bool(raw["use_edl"]),
            use_edl_uncertainty=bool(raw["use_edl_uncertainty"]),
            reserved=reserved or None,
        )

    return scenarios


def get_scenario_ids() -> List[str]:
    """Return all scenario IDs from the default config file."""
    return list(load_all_scenarios().keys())


# Backward-compatible alias used in a few legacy call sites
def load_scenarios(config_path: str | Path | None = None) -> list[ScenarioConfig]:
    return list(load_all_scenarios(config_path).values())
