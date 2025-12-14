"""Predictor sub-modules.

将 cv_sat 设为可选导入，避免包初始化时的级联副作用（例如依赖缺失/本地环境无GDAL等）。
若导入失败，SatCVPredictor 名字仍然导出但值为 None，
以便上层通过 `if SatCVPredictor is None:` 优雅降级。
"""

from .env_nc import EnvNCPredictor
from .dl_ice import DLIcePredictor

# 可选导入：cv_sat
try:  # pragma: no cover - 依赖可能在CI/最小环境中缺失
    from .cv_sat import SatCVPredictor  # type: ignore
except Exception:  # noqa: BLE001 - 任何导入期副作用都拦截
    SatCVPredictor = None  # type: ignore

__all__ = [
    "EnvNCPredictor",
    "DLIcePredictor",
    "SatCVPredictor",
]
