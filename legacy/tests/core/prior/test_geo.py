import math
import pytest

try:
    from ArcticRoute.core.prior.geo import to_xy, gc_distance_nm
    _has_pyproj = True
except Exception:
    _has_pyproj = False

pytestmark = pytest.mark.skipif(not _has_pyproj, reason="pyproj not installed")


def test_lon_wrap_equivalence():
    # 350° 等价于 -10°
    x1, y1 = to_xy(75.0, 350.0)
    x2, y2 = to_xy(75.0, -10.0)
    assert abs(x1 - x2) < 1e-6
    assert abs(y1 - y2) < 1e-6


def test_gc_distance_basic():
    # 赤道 1 度经差约 60 nm（赤道附近）
    d = gc_distance_nm(0.0, 0.0, 0.0, 1.0)
    assert 59.0 < d < 61.5
    # 高纬 1 度纬向约 60 nm
    d2 = gc_distance_nm(80.0, 0.0, 81.0, 0.0)
    assert 59.0 < d2 < 61.5


def test_polar_projection_local_metric_stability():
    # 在 80N 附近，投影到 EPSG:3413 后，局部欧氏距离与大圆距离近似相等（小位移）
    lat, lon = 80.0, 10.0
    dlat = 0.05  # ~3nm
    dlon = 0.1   # ~?nm，足够小

    x1, y1 = to_xy(lat, lon)
    x2, y2 = to_xy(lat + dlat, lon + dlon)
    planar_m = math.hypot(x2 - x1, y2 - y1)
    planar_nm = planar_m / 1852.0

    gc_nm = gc_distance_nm(lat, lon, lat + dlat, lon + dlon)

    # 允许 5% 相对误差
    if gc_nm > 0:
        rel = abs(planar_nm - gc_nm) / gc_nm
        assert rel < 0.05


def test_projection_monotonicity():
    # 东移，x 增大；北移，y 增大（在 EPSG:3413 范围内）
    lat, lon = 75.0, -30.0
    x0, y0 = to_xy(lat, lon)
    x1, y1 = to_xy(lat, lon + 1.0)
    x2, y2 = to_xy(lat + 1.0, lon)
    assert x1 > x0
    assert y2 > y0














