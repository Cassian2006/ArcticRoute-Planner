"""
快速检查历史主航线 GeoJSON (原始 vs WGS84) 的 CRS 与经纬范围；
当原始数据本就处于经纬度坐标时，生成一个“干净”的 WGS84 文件（仅 set_crs，不做投影转换）。

用法：
    python ArcticRoute/scripts/debug_prior_centerlines.py
输出：
    - ORIG/WGS 的 CRS 与 bounds
    - 如发现原始即为 WGS84，经判断后会写入 data_processed/prior/prior_centerlines_all_wgs84.geojson
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import geopandas as gpd
except Exception as e:
    raise SystemExit("[ERR] 需要安装 geopandas 才能运行本脚本: pip install geopandas")

REPO_ROOT = Path(__file__).resolve().parents[1]
ORIG_PATH = REPO_ROOT / "ArcticRoute" / "reports" / "phaseE" / "center" / "prior_centerlines_all.geojson"
WGS_PATH = REPO_ROOT / "ArcticRoute" / "data_processed" / "prior" / "prior_centerlines_all_wgs84.geojson"
WGS_PATH.parent.mkdir(parents=True, exist_ok=True)

def _print_gdf_info(tag: str, gdf: "gpd.GeoDataFrame") -> None:
    try:
        crs = gdf.crs
    except Exception:
        crs = None
    try:
        tb = gdf.total_bounds  # [minx, miny, maxx, maxy]
        tb = [float(x) for x in tb]
    except Exception:
        tb = [float("nan")] * 4
    print(f"{tag} CRS:", crs)
    print(f"{tag} bounds:", tb)


def _is_bounds_plausible_wgs84(bounds: list[float]) -> bool:
    # 粗略判定：经度应在 [-180, 180] 左右；纬度应在 [45, 90] 区间（北极高纬，放宽下限避免过严）
    if not bounds or len(bounds) != 4:
        return False
    minx, miny, maxx, maxy = bounds
    lon_ok = (minx >= -200.0) and (maxx <= 200.0)
    lat_ok = (miny >= 45.0) and (maxy <= 90.5)
    return lon_ok and lat_ok


def main() -> None:
    if not ORIG_PATH.exists():
        print("[WARN] 原始主航线文件不存在:", ORIG_PATH)
    else:
        g_orig = gpd.read_file(ORIG_PATH)
        _print_gdf_info("ORIG", g_orig)
        orig_bounds = [float(x) for x in g_orig.total_bounds]

    g_wgs = None
    if WGS_PATH.exists():
        try:
            g_wgs = gpd.read_file(WGS_PATH)
            _print_gdf_info("WGS", g_wgs)
        except Exception as e:
            print("[WARN] 读取 WGS 文件失败:", e)

    # 决策：若原始数据看起来就是 WGS84（坐标本来正常），则生成一个“干净”的 WGS 文件（仅 set_crs，不做 to_crs）
    try:
        if ORIG_PATH.exists():
            g_orig = gpd.read_file(ORIG_PATH)
            ob = [float(x) for x in g_orig.total_bounds]
            if _is_bounds_plausible_wgs84(ob):
                g_clean = g_orig.copy()
                # 不要投影转换：仅当缺 CRS 时标注 EPSG:4326
                if g_clean.crs is None:
                    g_clean.set_crs(epsg=4326, inplace=True)
                # 若已有 CRS 但值异常、仍属于经纬度，应保持坐标不变，仅修正声明；这里防御起见只在 None 时 set_crs
                g_clean.to_file(WGS_PATH, driver="GeoJSON")
                print("[FIX] 已生成干净的 WGS84 文件:", WGS_PATH)
            else:
                print("[INFO] 原始 bounds 不像 WGS84，保留现有 wgs 文件（若存在）。")
        else:
            print("[INFO] 无原始文件，跳过重建。")
    except Exception as e:
        print("[WARN] 生成 WGS84 文件时出错:", e)

    print("[DONE] prior centerlines debug finished.")


if __name__ == "__main__":
    main()

