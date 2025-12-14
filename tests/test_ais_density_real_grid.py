import numpy as np

from arcticroute.core.ais_ingest import rasterize_ais_density_to_grid


def test_rasterize_on_real_like_grid():
    """真实网格模式下栅格化 AIS 密度应产生非零结果，形状匹配。"""
    ny, nx = 10, 20
    lat_1d = np.linspace(60.0, 80.0, ny)
    lon_1d = np.linspace(-180.0, 180.0, nx)
    lon2d, lat2d = np.meshgrid(lon_1d, lat_1d)

    # 构造一些假 AIS 点，覆盖中间和边缘
    lat_points = np.concatenate([
        lat_1d[::3],
        np.array([lat_1d[-1]]),
    ])
    lon_points = np.concatenate([
        lon_1d[::4],
        np.array([lon_1d[0]]),
    ])

    density = rasterize_ais_density_to_grid(
        lat_points=lat_points,
        lon_points=lon_points,
        grid_lat2d=lat2d,
        grid_lon2d=lon2d,
    )

    assert density.shape == (ny, nx)
    assert np.count_nonzero(density.values) > 0
