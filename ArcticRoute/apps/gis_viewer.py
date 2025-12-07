# -*- coding: utf-8 -*-
"""
GIS Viewer (embedded)
- 复用并改造自 streamlit-gis-tool-main（MIT License, (c) 2025 Xu Qiongjie）
- 嵌入到 ArcticRoute/minimum 项目，提供：
  1) 上传/浏览 GeoJSON 或 Shapefile(zip)
  2) 2D 地图可视化（folium + streamlit-folium）
  3) 属性表查看
  4) 缓冲分析（米）
  5) 3D 点云柱状展示（pydeck，仅 Point 支持）

注意：Windows 安装 GeoPandas/Shapely 建议使用 conda-forge。
"""
from __future__ import annotations

import io
import os
import zipfile
import tempfile
from pathlib import Path
from typing import Optional

import streamlit as st

# 依赖导入（带提示）
try:
    import geopandas as gpd  # type: ignore
    import pandas as pd  # type: ignore
    from shapely.geometry import base as _shp_base  # type: ignore
    from shapely.geometry import Point  # type: ignore
    import shapely  # type: ignore
except Exception as _e:
    st.error("缺少 GeoPandas/Shapely 相关依赖，请先安装：conda install -c conda-forge geopandas shapely fiona pyproj")
    raise

try:
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:
    st.error("缺少 folium/streamlit-folium，请安装：pip install folium streamlit-folium")
    raise

try:
    import pydeck as pdk  # type: ignore
except Exception:
    pdk = None  # 允许缺失，仅禁用 3D


# ---------- 路径与辅助 ----------

def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def outputs_dir() -> Path:
    return repo_root() / "outputs"


def _ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    try:
        if gdf.crs is None:
            # 无 CRS 时默认假定为 WGS84（经纬度），保守处理
            gdf = gdf.set_crs(4326, allow_override=True)
        else:
            gdf = gdf.to_crs(epsg=4326)
    except Exception:
        pass
    return gdf


def _safe_centroid_latlon(gdf: gpd.GeoDataFrame) -> list[float]:
    """返回 [lat, lon] 作为地图中心。若失败则回退 [0,0]。"""
    try:
        gg = _ensure_wgs84(gdf)
        # 对 Multi/Line/Polygon 使用 bounds 中心更稳妥
        b = gg.total_bounds  # [minx, miny, maxx, maxy]
        if not b or len(b) != 4:
            raise ValueError("bounds invalid")
        lon = float((b[0] + b[2]) / 2)
        lat = float((b[1] + b[3]) / 2)
        return [lat, lon]
    except Exception:
        return [0.0, 0.0]


def _read_zip_shapefile(uploaded) -> Optional[gpd.GeoDataFrame]:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(uploaded, "r") as z:
                z.extractall(tmp)
            # 兼容子目录
            shp_candidates = []
            for root, _dirs, files in os.walk(tmp):
                for f in files:
                    if f.lower().endswith(".shp"):
                        shp_candidates.append(Path(root) / f)
            if not shp_candidates:
                return None
            return gpd.read_file(str(shp_candidates[0]))
    except Exception:
        return None


def load_vector_data(file) -> Optional[gpd.GeoDataFrame]:
    try:
        name = getattr(file, "name", "") or ""
        if name.lower().endswith(".geojson") or name.lower().endswith(".json"):
            return _ensure_wgs84(gpd.read_file(file))
        if name.lower().endswith(".zip"):
            return _ensure_wgs84(_read_zip_shapefile(file))
        return None
    except Exception as e:
        st.error(f"读取矢量数据失败：{e}")
        return None


def apply_buffer_meters(gdf: gpd.GeoDataFrame, distance_m: float) -> gpd.GeoDataFrame:
    """在 Web Mercator(3857) 下按米缓冲，结果再转回 WGS84。"""
    try:
        gg = gdf.copy()
        gg = _ensure_wgs84(gg)
        gg = gg.to_crs(epsg=3857)
        gg["geometry"] = gg.buffer(float(distance_m))
        gg = gg.to_crs(epsg=4326)
        return gg
    except Exception as e:
        st.warning(f"缓冲失败：{e}")
        return gdf


def render_map_2d(gdf: gpd.GeoDataFrame, height: int = 520, width: int | None = None) -> None:
    gg = _ensure_wgs84(gdf)
    lat, lon = _safe_centroid_latlon(gg)
    m = folium.Map(location=[lat, lon], tiles="cartodbpositron", zoom_start=8)
    try:
        folium.GeoJson(gg).add_to(m)
    except Exception:
        # 若因坐标或 MultiGeom 导致失败，尝试投影到 EPSG:4326 后简化
        try:
            tmp = gg.to_crs(4326)
            folium.GeoJson(tmp).add_to(m)
        except Exception as e:
            st.warning(f"GeoJson 渲染失败：{e}")
    st_folium(m, height=height, width=width)


def render_map_3d_points(gdf: gpd.GeoDataFrame) -> None:
    if pdk is None:
        st.info("未安装 pydeck，无法渲染 3D 视图。pip install pydeck")
        return
    if gdf.empty:
        st.info("无数据")
        return
    # 仅 Point 支持
    geom0 = gdf.geometry.iloc[0]
    if getattr(geom0, "geom_type", "") != "Point":
        st.info("3D 视图目前仅支持 Point 几何。")
        return
    gg = _ensure_wgs84(gdf)
    df = gg.copy()
    df["lon"] = df.geometry.x
    df["lat"] = df.geometry.y
    df["elevation"] = 1000.0
    layer = pdk.Layer(
        "ColumnLayer",
        data=df,
        get_position='[lon, lat]',
        get_elevation='elevation',
        elevation_scale=1,
        radius=200,
        get_fill_color='[180, 0, 200, 140]',
        pickable=True,
        auto_highlight=True,
    )
    vs = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=8, pitch=45)
    r = pdk.Deck(layers=[layer], initial_view_state=vs)
    st.pydeck_chart(r)


# ---------- UI ----------

st.set_page_config(page_title="ArcticRoute · GIS Viewer", layout="wide")
st.title("🗺️ GIS Viewer")

with st.sidebar:
    st.markdown("### 数据源")
    st.caption("可从 outputs 选择现有 GeoJSON，或在下方上传文件。")
    # 列出 outputs 下常见 geojson
    outs = []
    try:
        od = outputs_dir()
        if od.exists():
            outs = sorted([p for p in od.glob("*.geojson")], key=lambda p: p.stat().st_mtime, reverse=True)
    except Exception:
        outs = []
    opts = ["<不使用>"] + [p.name for p in outs]
    pick = st.selectbox("浏览 outputs/", options=opts, index=0)
    st.markdown("---")
    uploaded = st.file_uploader("上传 GeoJSON 或 Shapefile(.zip)", type=["geojson","json","zip"], accept_multiple_files=False)

# 加载数据
_gdf: Optional[gpd.GeoDataFrame] = None
source_desc = None
if pick and pick != "<不使用>":
    try:
        path = outputs_dir() / pick
        _gdf = gpd.read_file(path)
        _gdf = _ensure_wgs84(_gdf)
        source_desc = f"outputs/{pick}"
    except Exception as e:
        st.error(f"读取 {pick} 失败：{e}")

if _gdf is None and uploaded is not None:
    _gdf = load_vector_data(uploaded)
    source_desc = f"uploaded:{getattr(uploaded, 'name', '')}"

if _gdf is None:
    st.info("请选择 outputs 下的数据或上传文件进行查看。")
    st.stop()

if source_desc:
    st.caption(f"数据源：{source_desc} · 记录数={len(_gdf)}")

# 属性表
try:
    st.subheader("📍 属性表")
    df_disp = _gdf.drop(columns=["geometry"]) if "geometry" in _gdf.columns else _gdf.copy()
    st.dataframe(df_disp.head(1000))
except Exception:
    st.caption("属性表展示失败或列过多。")

# 2D 地图
st.subheader("🗺️ 2D 地图")
render_map_2d(_gdf)

# 缓冲分析
st.subheader("📏 缓冲分析")
col_b1, col_b2 = st.columns(2)
with col_b1:
    buf_dist = st.slider("缓冲距离(米)", 50, 20000, 1000, 50)
with col_b2:
    do_union = st.checkbox("缓冲后 union dissolve", value=False)

_gdf_buf = apply_buffer_meters(_gdf, float(buf_dist))
if do_union:
    try:
        _gdf_buf = _gdf_buf.dissolve()
    except Exception:
        pass
st.caption("缓冲结果预览：")
render_map_2d(_gdf_buf)

# 3D 地图（仅点）
st.subheader("🌍 3D 视图（点）")
render_map_3d_points(_gdf)






