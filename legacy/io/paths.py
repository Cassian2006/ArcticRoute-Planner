"""通用路径访问器（Phase A）

- 读取项目根下 configs/paths.yaml（只读）
- 提供只读访问器 P：from io.paths import P
- 不会创建/修改目录；仅返回声明的路径字符串（保持 YAML 中的原样或转为绝对路径可选）
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # 兜底：避免因缺少依赖而崩溃
    yaml = None  # type: ignore


_ROOT = Path(__file__).resolve().parents[1]  # 项目根（包含 configs/）
_CFG_PATH = _ROOT / "configs" / "paths.yaml"

_DEFAULTS: Dict[str, str] = {
    "data_raw": "ArcticRoute/data_raw",
    "data_processed": "ArcticRoute/data_processed",
    "cache": ".cache",
    "reports": "reports",
    "outputs": "outputs",
    "logs": "ArcticRoute/logs",
}


def _load_config() -> Dict[str, str]:
    cfg: Dict[str, str] = dict(_DEFAULTS)
    if _CFG_PATH.exists() and yaml is not None:
        try:
            text = _CFG_PATH.read_text(encoding="utf-8")
            data = yaml.safe_load(text) or {}
            if isinstance(data, dict):
                for k, v in data.items():
                    if isinstance(k, str) and isinstance(v, (str, os.PathLike)):
                        cfg[k] = str(v)
        except Exception:
            # 读取失败时使用默认
            pass
    return cfg


class _ReadOnlyPaths:
    """只读路径映射：属性与字典方式访问。

    行为：
    - 不创建目录，不规范化；按 YAML 原样返回（相对路径基于项目根可由调用方自行拼接）。
    - 支持 .as_dict() 导出副本。
    """

    def __init__(self, mapping: Dict[str, str]):
        self._m = dict(mapping)

    def __getattr__(self, name: str) -> str:
        if name in self._m:
            return self._m[name]
        raise AttributeError(name)

    def __getitem__(self, key: str) -> str:
        return self._m[key]

    def keys(self):
        return self._m.keys()

    def as_dict(self) -> Dict[str, str]:
        return dict(self._m)

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self._m.keys()))
        return f"<Paths keys=[{keys}]>"


# 模块级单例
P = _ReadOnlyPaths(_load_config())

__all__ = ["P", "_ReadOnlyPaths"]

