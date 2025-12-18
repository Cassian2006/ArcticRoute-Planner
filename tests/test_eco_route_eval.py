import numpy as np
from arcticroute.core.eco.route_eval import eval_route_eco

def test_eval_route_eco_returns_zeros_with_shape():
    land = np.zeros((4, 6), dtype=bool)
    out = eval_route_eco(None, land)
    assert out.shape == land.shape
    assert np.all(np.isfinite(out))
    assert np.all(out >= 0.0)

def test_eval_route_eco_fallback_no_land_mask():
    out = eval_route_eco()
    assert out.ndim == 2 and out.size > 0
    assert np.all(out == 0.0)






