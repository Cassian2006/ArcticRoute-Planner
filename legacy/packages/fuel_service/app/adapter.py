from __future__ import annotations

from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import os
import math

# 经验规则 + 可选真实模型 适配器：
# - 若设置 FUEL_MODEL_PATH/MODE 且可加载，则使用真实模型预测每海里燃油（per-nm）
# - 否则采用规则版（vessel_class 基准 × 季节/图层修正）


def _find_repo_root() -> Path:
    p = Path(__file__).resolve()
    for q in p.parents:
        if (q / "pyproject.toml").exists() and (q / "README.md").exists():
            return q
    return p.parents[3]


class FuelModelAdapter:
    def __init__(self, model_path: Optional[str] = None, mode: str = "sklearn") -> None:
        self.model_path = model_path
        self.mode = (mode or "sklearn").lower()
        self._repo = _find_repo_root()
        self._eco = self._load_eco_cfg()
        # 真实模型句柄
        self._mdl = None
        self._feat_names: Optional[List[str]] = None
        # 采样器（懒加载）
        self._sampler = None
        # 尝试加载真实模型（若配置存在）
        self._try_load_model()

    # ---------------- eco 配置与通用工具 ----------------
    def _load_eco_cfg(self) -> Dict[str, Any]:
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None  # type: ignore
        eco_path = self._repo / "ArcticRoute" / "config" / "eco.yaml"
        if yaml is None or not eco_path.exists():
            return {}
        try:
            data = yaml.safe_load(eco_path.read_text(encoding="utf-8")) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _exists(self, rel: str) -> bool:
        return (self._repo / rel).exists()

    # ---------------- 真实模型加载 ----------------
    def _try_load_model(self) -> None:
        path = (self.model_path or "").strip()
        if not path:
            return
        p = Path(path)
        if not p.exists():
            return
        try:
            if self.mode in ("sk", "sklearn", "joblib"):
                import joblib  # type: ignore
                payload = joblib.load(p)
                # 兼容两种保存方式：
                if hasattr(payload, "predict"):
                    self._mdl = payload
                    self._feat_names = None
                elif isinstance(payload, dict) and payload.get("model") is not None:
                    self._mdl = payload.get("model")
                    fns = payload.get("feature_names")
                    if isinstance(fns, list) and all(isinstance(x, str) for x in fns):
                        self._feat_names = list(fns)
                else:
                    self._mdl = None
            elif self.mode in ("xg", "xgb", "xgboost"):
                import xgboost as xgb  # type: ignore
                booster = xgb.Booster()
                booster.load_model(str(p))
                self._mdl = booster
                self._feat_names = None
            else:
                self._mdl = None
        except Exception:
            self._mdl = None

    # ---------------- 规则版：季节/图层/基准 ----------------
    def _season_factor(self, ym: Optional[str]) -> float:
        if not ym or len(str(ym)) < 6:
            return 1.05
        try:
            m = int(str(ym)[4:6])
        except Exception:
            return 1.05
        if m in (12, 1, 2):
            return 1.15
        if m in (3, 4, 5, 9, 10, 11):
            return 1.05
        return 1.0

    def _layer_factor(self, ym: Optional[str]) -> float:
        if not ym:
            return 1.0
        try:
            alpha_ice = float((((self._eco.get("eco") or {}).get("alpha_ice", 0.8))))
        except Exception:
            alpha_ice = 0.8
        f = 1.0
        p_ice = self._repo / "ArcticRoute" / "data_processed" / "risk" / f"R_ice_eff_{ym}.nc"
        if p_ice.exists():
            f *= (1.0 + 0.20 * max(0.0, min(alpha_ice, 1.5)))
        p_risk = self._repo / "ArcticRoute" / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
        if p_risk.exists():
            f *= 1.05
        return f

    def _base_per_nm(self, vessel_class: str) -> float:
        defaults = {
            "icebreaker": 0.020,
            "cargo_iceclass": 0.015,
            "cargo_standard": 0.012,
            "fishing_small": 0.006,
        }
        v = (vessel_class or "cargo_iceclass").strip()
        base = defaults.get(v, 0.015)
        try:
            eco = (self._eco.get("eco") or {})
            vmap = (eco.get("vessel_classes") or {})
            if v in vmap:
                b = vmap[v].get("fuel_per_nm_base")
                if isinstance(b, (int, float)) and b > 0:
                    return float(b)
        except Exception:
            pass
        return float(base)

    # ---------------- 采样器（路线→环境特征） ----------------
    def _get_sampler(self):
        if self._sampler is not None:
            return self._sampler
        try:
            from .sampler import RouteSampler  # type: ignore
            self._sampler = RouteSampler(self._repo)
        except Exception:
            self._sampler = None
        return self._sampler

    def _vclass_index(self, vessel_class: str) -> int:
        v = (vessel_class or "cargo_iceclass").lower()
        if "ice" in v:
            return 1
        return 0

    def _sample_env_features(self, coords: List[Tuple[float, float]], ym: Optional[str]) -> Dict[str, float]:
        # 先取规则默认，再尝试用采样器覆盖
        env = self._default_env_features(ym)
        if not ym:
            return env
        smp = self._get_sampler()
        if smp is None:
            return env
        try:
            ice = smp.sample(coords, ym, kind="ice")
            risk = smp.sample(coords, ym, kind="risk")
            wave = smp.sample(coords, ym, kind="wave")
            # 将采样值映射到合理范围（占位）：
            # risk/ice 为 0..1 概率，wave 近似映射到 0..6m，取缺省回退
            if ice is not None and math.isfinite(ice):
                env["ice_risk"] = float(max(0.0, min(1.0, ice)))
            if risk is not None and math.isfinite(risk):
                # risk 可弱映射到风速：0..1 → 0..20 kn（仅占位）
                env["wind_speed"] = float(max(0.0, min(20.0, risk * 20.0)))
            if wave is not None and math.isfinite(wave):
                # wave 栅格若存在，则 0..1 映射到 0..6m
                env["wave_height"] = float(max(0.0, min(6.0, wave * 6.0)))
        except Exception:
            pass
        return env

    def _default_env_features(self, ym: Optional[str]) -> Dict[str, float]:
        # 简单占位：可在未来接入 ArcticRoute 的栅格采样
        # 根据季节略微调整默认风/浪/冰
        sf = self._season_factor(ym)
        wind = 8.0 * sf  # 冬季稍高
        wave = 1.8 * sf
        ice = 0.2 if sf >= 1.15 else (0.1 if sf > 1.0 else 0.05)
        return {"wind_speed": float(wind), "wave_height": float(wave), "ice_risk": float(ice)}

    def _build_feature_row(self, coords: List[Tuple[float, float]], ym: Optional[str], vessel_class: str, feature_names: Optional[List[str]]) -> List[float]:
        # 估算平均分段长度（nm）作为代表
        def _hav(a, b):
            R_m = 6_371_000.0
            import math as _m
            phi1 = _m.radians(a[1]); phi2 = _m.radians(b[1])
            dphi = _m.radians(b[1] - a[1]); dl = _m.radians(b[0] - a[0])
            h = _m.sin(dphi/2)**2 + _m.cos(phi1)*_m.cos(phi2)*_m.sin(dl/2)**2
            d_m = 2*R_m*_m.atan2(_m.sqrt(h), _m.sqrt(1-h))
            return d_m/1852.0
        if len(coords) >= 2:
            segs = [_hav(coords[i], coords[i+1]) for i in range(len(coords)-1)]
            seg_len_nm = (sum(segs)/len(segs)) if segs else 10.0
        else:
            seg_len_nm = 10.0
        # 优先使用采样特征
        env = self._sample_env_features(coords, ym)
        vix = self._vclass_index(vessel_class)
        # 期望特征名：若导出时包含 feature_names 列表则按顺序构造；否则按默认顺序
        f_order = feature_names or [
            "segment_length_nm", "wind_speed", "wave_height", "ice_risk", "vclass_ix"
        ]
        fmap = {
            "segment_length_nm": seg_len_nm,
            "wind_speed": env["wind_speed"],
            "wave_height": env["wave_height"],
            "ice_risk": env["ice_risk"],
            "vclass_ix": float(vix),
        }
        row = [float(fmap.get(k, 0.0)) for k in f_order]
        return row

    # ---------------- 主推理接口 ----------------
    def predict_per_nm(self, coords: List[Tuple[float, float]], ym: Optional[str], vessel_class: str) -> float:
        # 真实模型优先
        if self._mdl is not None:
            try:
                if self.mode in ("sk", "sklearn", "joblib"):
                    import numpy as np  # type: ignore
                    row = self._build_feature_row(coords, ym, vessel_class, self._feat_names)
                    X = np.asarray(row, dtype=float).reshape(1, -1)
                    y = float(self._mdl.predict(X)[0])  # type: ignore[attr-defined]
                    # 防御：per-nm 不应为负或过大
                    if math.isfinite(y) and y > 0 and y < 1.0:
                        return y
                elif self.mode in ("xg", "xgb", "xgboost"):
                    import numpy as np  # type: ignore
                    import xgboost as xgb  # type: ignore
                    row = self._build_feature_row(coords, ym, vessel_class, self._feat_names)
                    X = np.asarray(row, dtype=float).reshape(1, -1)
                    dmat = xgb.DMatrix(X)
                    y = float(self._mdl.predict(dmat)[0])  # type: ignore[attr-defined]
                    if math.isfinite(y) and y > 0 and y < 1.0:
                        return y
            except Exception:
                pass
        # 规则版回退：base * season * layer
        base = self._base_per_nm(vessel_class)
        f_season = self._season_factor(ym)
        f_layer = self._layer_factor(ym)
        per_nm = base * f_season * f_layer
        return float(per_nm)


def build_adapter_from_env() -> FuelModelAdapter:
    path = os.environ.get("FUEL_MODEL_PATH")
    mode = os.environ.get("FUEL_MODEL_MODE", "sklearn")
    return FuelModelAdapter(model_path=path, mode=mode)
