from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception:
        return {}
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@dataclass
class UIFeatureFlags:
    ui_sync_strict: bool = True
    pages: Dict[str, bool] = None  # type: ignore[assignment]
    components: Dict[str, bool] = None  # type: ignore[assignment]
    defaults: Dict[str, Any] = None  # type: ignore[assignment]
    eco: Dict[str, Any] = None  # type: ignore[assignment]
    advanced: Dict[str, Any] = None  # type: ignore[assignment]


class UIRegistry:
    """
    读取 config/runtime.yaml 的 ui.* 配置，提供统一的页面与组件开关查询。
    - 页面可插拔：通过 enabled_pages() 返回当前启用的页面名集合。
    - 组件开关：is_component_enabled(name) 查询是否渲染某组件。
    - 默认值：get_default(key, fallback) 读取 ui.defaults.*（如 risk_source/risk_agg/alpha 等）。
    - 高级：get_advanced_default(key, fallback) 读取 ui.advanced.*（如 fusion_mode_default 等）。
    """

    def __init__(self, runtime_path: Optional[Path] = None) -> None:
        self.repo = _repo_root()
        self.runtime_path = runtime_path or (self.repo / "ArcticRoute" / "config" / "runtime.yaml")
        self.flags = self._load_flags()

    def _load_flags(self) -> UIFeatureFlags:
        data = _load_yaml(self.runtime_path)
        ui = data.get("ui", {}) if isinstance(data, dict) else {}
        ff = UIFeatureFlags(
            ui_sync_strict=bool(ui.get("ui_sync_strict", True)),
            pages=dict(ui.get("pages", {})) if isinstance(ui.get("pages"), dict) else {},
            components=dict(ui.get("components", {})) if isinstance(ui.get("components"), dict) else {},
            defaults=dict(ui.get("defaults", {})) if isinstance(ui.get("defaults"), dict) else {},
            eco=dict(ui.get("eco", {})) if isinstance(ui.get("eco"), dict) else {},
            advanced=dict(ui.get("advanced", {})) if isinstance(ui.get("advanced"), dict) else {},
        )
        return ff

    def refresh(self) -> None:
        self.flags = self._load_flags()

    # ---- Pages ----
    def enabled_pages(self) -> Dict[str, bool]:
        return dict(self.flags.pages or {})

    def is_page_enabled(self, name: str, default: bool = False) -> bool:
        pages = self.flags.pages or {}
        if name in pages:
            return bool(pages[name])
        # 未配置时，严格模式默认关闭；宽松模式使用 default
        return False if self.flags.ui_sync_strict else bool(default)

    # ---- Components ----
    def is_component_enabled(self, name: str, default: bool = True) -> bool:
        comps = self.flags.components or {}
        if name in comps:
            return bool(comps[name])
        return bool(default)

    # ---- Eco Module ----
    def is_eco_enabled(self) -> bool:
        eco_cfg = self.flags.eco or {}
        return bool(eco_cfg.get("enabled", False))

    # ---- Defaults ----
    def get_default(self, key: str, fallback: Any = None) -> Any:
        return (self.flags.defaults or {}).get(key, fallback)

    # ---- Advanced Defaults & toggles ----
    def get_advanced_default(self, key: str, fallback: Any = None) -> Any:
        return (self.flags.advanced or {}).get(key, fallback)

    def is_advanced_enabled(self, key: str, default: bool = True) -> bool:
        adv = self.flags.advanced or {}
        if key in adv:
            v = adv[key]
            if isinstance(v, bool):
                return v
        return bool(default)

    # ---- Export for debugging ----
    def to_json(self) -> str:
        payload = {
            "runtime": str(self.runtime_path),
            "ui_sync_strict": self.flags.ui_sync_strict,
            "pages": self.flags.pages,
            "components": self.flags.components,
            "defaults": self.flags.defaults,
            "eco": self.flags.eco,
            "advanced": self.flags.advanced,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
