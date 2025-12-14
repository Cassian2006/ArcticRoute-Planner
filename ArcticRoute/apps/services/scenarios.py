"""
Scenario presets loader for Planner "Scenario Lab".
@role: core
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml


# 复用项目根定位方式（参考已有 get_project_root 实现）
def get_project_root() -> Path:
    # 该文件位于 minimum/ArcticRoute/apps/services/scenarios.py
    # __file__.parents: [0]=.../services, [1]=.../apps, [2]=.../ArcticRoute, [3]=.../minimum
    return Path(__file__).resolve().parents[3]  # .../minimum


@dataclass
class ScenarioPreset:
    key: str
    name: str
    description: str
    params: Dict[str, Any]


def load_planner_presets() -> List["ScenarioPreset"]:
    """
    从 configs/scenarios.yaml 加载 planner_presets 列表。
    若文件或字段不存在，则返回一个空列表。
    """
    root = get_project_root()
    cfg_path = root / "configs" / "scenarios.yaml"
    if not cfg_path.exists():
        return []

    with cfg_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    items = data.get("planner_presets") or []
    presets: List[ScenarioPreset] = []
    for item in items:
        key = str(item.get("key") or "").strip()
        name = str(item.get("name") or key)
        desc = str(item.get("description") or "")
        params = dict(item.get("params") or {})
        if not key:
            continue
        presets.append(ScenarioPreset(key=key, name=name, description=desc, params=params))
    return presets

