"""Inspects prior centerline GeoJSONs and checks CRS/bounds/high-lat coverage.

@role: analysis
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, List, Tuple

try:
    import geopandas as gpd  # type: ignore
except Exception as e:
    raise SystemExit("[ERR] 需要安装 geopandas: pip install geopandas")

# 统一项目根路径：包含 pyproject.toml、Dockerfile 的那层（本仓库为 minimum/）

def get_project_root() -> Path:
    # 该文件位于 minimum/ArcticRoute/scripts/... 下
    # parents[0] -> .../ArcticRoute/scripts/inspect_prior_centerlines.py
    # parents[1] -> .../ArcticRoute
    # parents[2] -> .../minimum
    return Path(__file__).resolve().parents[2]

root = get_project_root()             # .../minimum
arctic_dir = root / "ArcticRoute"     # .../minimum/ArcticRoute

# 路径统一
prior_orig_path = arctic_dir / "reports" / "phaseE" / "center" / "prior_centerlines_all.geojson"
prior_wgs_path  = arctic_dir / "data_processed" / "prior" / "prior_centerlines_all_wgs84.geojson"


def _print_basic(tag: str, gdf: "gpd.GeoDataFrame") -> None:
    crs = getattr(gdf, "crs", None)
    try:
        bounds = [float(x) for x in gdf.total_bounds]
    except Exception:
        bounds = [float("nan")] * 4
    print(f"{tag} CRS:", crs)
    print(f"{tag} bounds [minx, miny, maxx, maxy]:", bounds)


def _geom_head_coords(gdf: "gpd.GeoDataFrame", max_geoms: int = 3, max_pts: int = 6) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    try:
        from shapely.geometry import LineString, MultiLineString  # type: ignore
    except Exception:
        return out
    for geom in list(gdf.geometry)[:max_geoms]:
        try:
            if geom is None:
                continue
            if isinstance(geom, LineString):
                pts = list(geom.coords)[:max_pts]
                out.append([(float(x), float(y)) for (x, y) in pts])
            elif isinstance(geom, MultiLineString):
                pts_all: List[Tuple[float, float]] = []
                for line in geom.geoms:
                    pts_all.extend(list(line.coords))
                    if len(pts_all) >= max_pts:
                        break
                out.append([(float(x), float(y)) for (x, y) in pts_all[:max_pts]])
        except Exception:
            continue
    return out


def _fraction_high_lat(gdf: "gpd.GeoDataFrame", thresh_lat: float = 55.0) -> float:
    try:
        g2 = gdf
        if getattr(g2, "crs", None) is None or str(g2.crs).lower() not in ("epsg:4326", "wgs84"):
            # 不改变坐标，只设置声明，避免错误投影
            g2 = g2.set_crs(epsg=4326, allow_override=True)
        cent = g2.geometry.centroid
        if len(cent) == 0:
            return 0.0
        return float((cent.y > float(thresh_lat)).mean())
    except Exception:
        return 0.0


def _inspect_one(tag: str, path: Path) -> None:
    if not path.exists():
        print(f"[{tag}] 文件不存在:", path)
        return
    try:
        g = gpd.read_file(path)
    except Exception as e:
        print(f"[{tag}] 读取失败:", e)
        return
    _print_basic(tag, g)
    heads = _geom_head_coords(g)
    print(f"{tag} head coords (lon,lat):", heads)
    frac = _fraction_high_lat(g, 55.0)
    print(f"{tag} fraction of segments with lat > 55N:", round(frac, 3))
    if frac < 0.6:
        print(f">>> {tag} PRIOR CENTERLINES LOOK SUSPICIOUS (most not in Arctic)")
    else:
        print(f">>> {tag} PRIOR CENTERLINES LOOK ROUGHLY OK")


def main() -> None:
    print("--- Inspect ORIG ---")
    _inspect_one("ORIG", prior_orig_path)
    print("\n--- Inspect WGS ---")
    _inspect_one("WGS", prior_wgs_path)
    print("\n[DONE] inspect finished.")


if __name__ == "__main__":
    main()

