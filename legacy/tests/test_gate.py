from __future__ import annotations
import numpy as np
from ArcticRoute.core.fusion_adv.gate import GateMLP, gate_weights


def test_gate_softmax_sums_to_one():
    k = 3
    w = gate_weights(prior_embed=[0.1, 0.2, 0.3], region_1h=[1,0], season_1h=[0,1,0,0], vessel_1h=[0,1,0,0,0], k=k, temperature=1.0)
    assert np.isclose(w.sum(), 1.0, atol=1e-6), w
    assert (w >= 0).all() and (w <= 1).all()


def test_gate_temperature_uniformity():
    # 高温应更接近均匀分布
    k = 4
    xargs = dict(prior_embed=[1,2,3,4], region_1h=[1,0,0], season_1h=[0,1,0,0], vessel_1h=[0,0,1,0,0], k=k)
    w_cold = gate_weights(**xargs, temperature=0.5)
    w_hot = gate_weights(**xargs, temperature=5.0)
    uni = np.ones(k) / k
    def _l2(a,b):
        return float(np.sqrt(((a-b)**2).sum()))
    assert _l2(w_hot, uni) < _l2(w_cold, uni)

