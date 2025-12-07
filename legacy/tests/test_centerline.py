from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from ArcticRoute.core.prior.centerline import _cluster_centerline_latlon, _bandwidth_profile


@pytest.fixture
def sample_trajs() -> list[tuple[np.ndarray, np.ndarray]]:
    # 两条相似轨迹，一条有噪声
    t1_lat = np.linspace(70, 71, 10)
    t1_lon = np.linspace(-150, -148, 10)
    t2_lat = t1_lat + 0.05 * np.random.randn(10)
    t2_lon = t1_lon + 0.05 * np.random.randn(10)
    return [(t1_lat, t1_lon), (t2_lat, t2_lon)]


def test_centerline_dba_approx(sample_trajs):
    """DBA 近似法能输出有序 polyline。"""
    cl_lat, cl_lon = _cluster_centerline_latlon(sample_trajs, m=50)
    assert cl_lat.shape == (50,)
    assert cl_lon.shape == (50,)
    # 检查单调性（近似）：经度应大致单调增
    assert np.all(np.diff(cl_lon) >= -1e-6)  # 允许数值误差


def test_bandwidth_positive(sample_trajs):
    """带宽 > 0。"""
    cl_lat, cl_lon = _cluster_centerline_latlon(sample_trajs, m=50)
    bw, stats = _bandwidth_profile(sample_trajs, cl_lat, cl_lon, q=0.75)
    assert bw.shape == (50,)
    assert np.all(bw > 0)
    assert stats["p75"] > 0












