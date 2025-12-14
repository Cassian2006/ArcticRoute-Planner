from __future__ import annotations

from typing import Tuple
import math

try:
    from pyproj import CRS, Transformer  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    CRS = None  # type: ignore
    Transformer = None  # type: ignore


def _wrap_lon(lon: float) -> float:
    """将经度规范化到 [-180, 180)。同时接受 0..360 的输入并自动折返。"""
    try:
        x = float(lon)
    except Exception:
        raise ValueError(f"invalid lon: {lon!r}")
    # 先将任何角度折返到 0..360
    x = x % 360.0
    # 再映射到 [-180, 180)
    if x >= 180.0:
        x -= 360.0
    return x


def to_xy(lat: float, lon: float, crs: str = "EPSG:3413") -> Tuple[float, float]:
    """将经纬坐标投影到平面坐标（米）。

    默认使用 EPSG:3413（NSIDC Sea Ice Polar Stereographic North），适用于 60°N 以北。
    依赖 pyproj；若未安装则抛出 ImportError 提示安装。
    """
    if Transformer is None or CRS is None:
        raise ImportError("pyproj 未安装。请先安装: pip install pyproj")

    try:
        lat_f = float(lat)
        lon_f = _wrap_lon(float(lon))
    except Exception:
        raise ValueError(f"invalid lat/lon: lat={lat!r}, lon={lon!r}")

    src = CRS.from_epsg(4326)
    dst = CRS.from_user_input(crs)
    # always_xy=True 确保经度先于纬度
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    x, y = transformer.transform(lon_f, lat_f)
    return float(x), float(y)


def gc_distance_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点间大圆距离（海里）。WGS84 球面近似。

    返回单位：海里（nautical miles）。
    """
    # 半径（海里）：地球半径 3440.065 nm
    R_nm = 3440.065

    def rad(deg: float) -> float:
        return math.radians(float(deg))

    lat1r = rad(lat1)
    lat2r = rad(lat2)
    dphi = rad(lat2 - lat1)
    # 经度先折返再差值
    lon1w = _wrap_lon(float(lon1))
    lon2w = _wrap_lon(float(lon2))
    dlam = rad(lon2w - lon1w)

    a = math.sin(dphi / 2) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(max(1e-16, 1 - a)))
    return R_nm * c


__all__ = ["to_xy", "gc_distance_nm"]














