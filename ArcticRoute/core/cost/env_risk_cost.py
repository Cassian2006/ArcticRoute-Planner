from __future__ import annotations

from typing import Optional

import numpy as np
import xarray as xr

from ..interfaces import CostProvider


class EnvRiskCostProvider(CostProvider):
    """根据 risk_env、corridor_prob 与事故密度计算代价系数。
    
    扩展（Phase F）：支持交互风险与先验惩罚（非破坏性，REUSE）。
    """

    def __init__(
        self,
        beta: float,
        p_exp: float,
        gamma: float,
        *,
        beta_accident: float = 0.0,
        alpha_ice: float = 0.0,
        interact_weight: float = 0.0,
        prior_penalty_weight: float = 0.0,
    ):
        self.beta = beta
        self.p_exp = p_exp
        self.gamma = gamma
        self.beta_accident = beta_accident
        self.alpha_ice = float(np.clip(alpha_ice, 0.0, 1.0))
        # Phase F: 交互风险与先验惩罚开关（默认关闭）
        self.interact_weight = float(np.clip(interact_weight, 0.0, 1.0))
        self.prior_penalty_weight = float(np.clip(prior_penalty_weight, 0.0, 1.0))
        # Phase G: 距离权重（w_d），作为代价基线乘数
        self.distance_weight = 1.0
        # Phase M: ECO 权重（与归一化 eco_cost_norm 相乘并按距离积分）
        self.eco_weight = 0.0


    @staticmethod
    def blend_with_ice(
        risk_da: xr.DataArray,
        ice_prob: Optional[xr.DataArray],
        alpha_ice: float,
    ) -> xr.DataArray:
        """Fuse risk field with ice probability according to alpha_ice."""
        alpha = float(np.clip(alpha_ice, 0.0, 1.0))
        if ice_prob is None or alpha <= 0.0:
            return risk_da

        ice_aligned = ice_prob
        for dim in risk_da.dims:
            if dim not in ice_aligned.dims:
                ice_aligned = ice_aligned.expand_dims({dim: risk_da.coords[dim]})
        ice_aligned = ice_aligned.transpose(*risk_da.dims, missing_dims="ignore")
        ice_aligned = ice_aligned.broadcast_like(risk_da)

        risk_base = risk_da.astype("float32")
        ice_cast = ice_aligned.astype("float32")
        ice_cast = ice_cast.where(np.isfinite(ice_cast), risk_base)

        # Blend environmental risk with ice probability and clip to [0, 1].
        final_risk = alpha * risk_base + (1.0 - alpha) * ice_cast
        final_risk = final_risk.clip(0.0, 1.0)
        final_risk.name = "final_risk"
        final_risk.attrs.update(risk_base.attrs)
        final_risk.attrs["alpha_ice"] = alpha
        return final_risk

    def compute(self, risk: float, corridor: Optional[float] = None, accident: Optional[float] = None,
                interact: Optional[float] = None, prior_penalty: Optional[float] = None) -> float:
        risk_clamped = np.clip(risk, 0.0, 1.0)
        coef_env = 1.0 + self.beta * (risk_clamped ** self.p_exp)

        corridor_factor = 1.0
        if corridor is not None:
            corridor_factor = max(1.0 - self.gamma * float(np.clip(corridor, 0.0, 1.0)), 0.0)

        accident_factor = 1.0
        if accident is not None and self.beta_accident > 0.0:
            accident_factor += self.beta_accident * float(np.clip(accident, 0.0, 1.0))

        # Phase F/G: 交互风险与先验惩罚（乘性系数）。
        # 兼容 planner 仅传递 corridor/accident 的情况：
        # - 若 interact 缺失且 accident 有值且 interact_weight>0，则把 accident 视为 interact。
        # - 若 prior_penalty 缺失且 corridor 有值且 prior_penalty_weight>0，则把 corridor 视为 prior_penalty。
        if interact is None and accident is not None and self.interact_weight > 0.0:
            interact = accident
        if prior_penalty is None and corridor is not None and self.prior_penalty_weight > 0.0:
            prior_penalty = corridor

        interact_factor = 1.0
        if interact is not None and self.interact_weight > 0.0:
            interact_factor += self.interact_weight * float(np.clip(interact, 0.0, 1.0))

        prior_factor = 1.0
        if prior_penalty is not None and self.prior_penalty_weight > 0.0:
            prior_factor += self.prior_penalty_weight * float(np.clip(prior_penalty, 0.0, 1.0))

        return coef_env * corridor_factor * accident_factor * interact_factor * prior_factor
