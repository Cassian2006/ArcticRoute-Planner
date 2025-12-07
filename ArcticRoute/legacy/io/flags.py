"""Feature Flags 读取（Phase A，默认全关闭）

- 配置文件：configs/feature_flags.yaml
- 使用方式：from ArcticRoute.io.flags import is_enabled
- 语义：缺失/解析失败/未声明 -> False
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CFG = _REPO_ROOT / "configs" / "feature_flags.yaml"
_CACHE: Dict[str, bool] | None = None


def _load() -> Dict[str, bool]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    flags: Dict[str, bool] = {}
    if yaml is None or not _CFG.exists():
        _CACHE = flags
        return flags
    try:
        data = yaml.safe_load(_CFG.read_text(encoding="utf-8")) or {}
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(k, str):
                    flags[k] = bool(v)
    except Exception:
        pass
    _CACHE = flags
    return flags


def is_enabled(name: str) -> bool:
    """返回功能开关是否启用（未声明/配置缺失 -> False）。"""
    if not isinstance(name, str) or not name:
        return False
    flags = _load()
    return bool(flags.get(name, False))

__all__ = ["is_enabled"]

