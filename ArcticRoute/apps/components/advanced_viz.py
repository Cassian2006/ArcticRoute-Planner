from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import json
import math
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry
from ArcticRoute.core import planner_service as ps

# 可选：keplergl 支持（若安装）
_HAS_KEPLER = False
try:
    from keplergl import KeplerGl  # type: ignore
    from streamlit_keplergl import keplergl_static  # type: ignore
    _HAS_KEPLER = True
except Exception:
    _HAS_KEPLER = False

# 回退：pydeck（随 Streamlit 自带）
try:
    import pydeck as pdk  # type: ignore
    _HAS_PYDECK = True
except Exception:
    _HAS_PYDECK = False


def _norm_lon(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def _extract_lonlat_from_route(route_result: Any) -> Optional[List[Tuple[float, float]]]:
    if route_result is None:
        return None
    cand = [
        getattr(route_result, "path_lonlat", None),
        getattr(route_result, "lonlat_path", None),
        getattr(route_result, "lonlat_list", None),
        getattr(route_result, "coords_lonlat", None),
    ]
    coords = None
    for c in cand:
        if c is not None:
            coords = c
            break
    if coords is None:
        return None
    out = []
    try:
        for p in coords:
            lat = float(p[0]); lon = float(p[1])
            out.append((_norm_lon(lon), lat))  # 统一为 (lon, lat)
        return out
    except Exception:
        return None


def _bounds_from_routes(all_paths: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    lons, lats = [], []
    for path in all_paths:
        for lon, lat in path:
            if math.isfinite(lon) and math.isfinite(lat):
                lons.append(lon); lats.append(lat)
    if not lons or not lats:
        return -180.0, 180.0, 50.0, 90.0
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)
    # 最小可视跨度
    if (lat_max - lat_min) < 10.0:
        mid = 0.5 * (lat_min + lat_max)
        lat_min, lat_max = mid - 5.0, mid + 5.0
    if (lon_max - lon_min) < 20.0:
        mid = 0.5 * (lon_min + lon_max)
        lon_min, lon_max = mid - 10.0, mid + 10.0
    # 裁剪合法范围
    lat_min, lat_max = max(-90.0, lat_min), min(90.0, lat_max)
    lon_min, lon_max = max(-180.0, lon_min), min(180.0, lon_max)
    return lon_min, lon_max, lat_min, lat_max


def _read_geojson_lines(content: bytes) -> Optional[List[Tuple[float, float]]]:
    try:
        obj = json.loads(content.decode("utf-8"))
        feats = obj.get("features") if isinstance(obj, dict) else None
        if not feats:
            return None
        for f in feats:
            geom = (f or {}).get("geometry") or {}
            if geom.get("type") in ("LineString", "MultiLineString"):
                coords = geom.get("coordinates")
                if geom.get("type") == "LineString":
                    path = [(_norm_lon(float(lon)), float(lat)) for lon, lat in coords]
                    return path
                else:
                    # 取第一条子线
                    if coords and isinstance(coords[0], list):
                        path = [(_norm_lon(float(lon)), float(lat)) for lon, lat in coords[0]]
                        return path
        return None
    except Exception:
        return None


def _color_palette() -> List[Tuple[int, int, int]]:
    return [
        (255, 47, 146),   # 主路线 pink
        (0, 91, 187),     # review 蓝
        (255, 140, 0),    # 橙
        (46, 204, 113),   # 绿
        (155, 89, 182),   # 紫
        (241, 196, 15),   # 黄
    ]


def get_default_kepler_config(view_bounds: Tuple[float, float, float, float], categories: List[str]) -> Dict[str, Any]:
    """
    Presentation-oriented Kepler config for ArcticRoute 2.0 advanced viz.
    - Single ROUTES layer bound to dataset "routes"
    - Read-only UI, hide developer controls and side panels
    - Ordinal color by scenario category (A/B/C) with fixed palette
    - 2D view with gentle zoom centered on provided bounds
    """
    lon_min, lon_max, lat_min, lat_max = view_bounds
    center_lon = float((lon_min + lon_max) / 2.0)
    center_lat = float((lat_min + lat_max) / 2.0)
    # Default color palette mapping A/B/C (magenta, blue, yellow)
    palette = [
        [255, 47, 146],   # A
        [0, 91, 187],     # B
        [241, 196, 15],   # C
    ]
    # Rough zoom estimation from bounds span
    span_lon = max(1.0, float(lon_max - lon_min))
    span_lat = max(1.0, float(lat_max - lat_min))
    span = max(span_lon, span_lat)
    if span <= 20:
        zoom = 4.5
    elif span <= 40:
        zoom = 4.0
    elif span <= 80:
        zoom = 3.2
    else:
        zoom = 2.4
    # Build layer config
    layer = {
        "id": "routes_layer",
        "type": "geojson",
        "config": {
            "dataId": "routes",
            "label": "ROUTES",
            "color": [255, 47, 146],  # fallback color (magenta)
            # columns left to Kepler's auto-detection for GeoJSON datasets
            "isVisible": True,
            "visConfig": {
                "opacity": 0.9,
                "thickness": 2.5,
                "stroked": True,
                "filled": False,
                "wireframe": False,
                "colorRange": {
                    "name": "Custom Ordinal",
                    "type": "ordinal",
                    "category": "Custom",
                    "colors": [
                        f"rgb({palette[0][0]},{palette[0][1]},{palette[0][2]})",
                        f"rgb({palette[1][0]},{palette[1][1]},{palette[1][2]})",
                        f"rgb({palette[2][0]},{palette[2][1]},{palette[2][2]})",
                    ],
                },
                "sizeRange": [2, 4],
            },
            "visualChannels": {
                "colorField": {"name": "scenario", "type": "string"},
                "colorScale": "ordinal",
            },
        },
        "visualChannels": {
            "colorField": {"name": "scenario", "type": "string"},
            "colorScale": "ordinal",
        },
    }
    # UI and interaction config
    ui_state = {
        "readOnly": True,
        "activeSidePanel": None,
        "currentModal": None,
        "mapControls": {
            "visibleLayers": {"show": True},
            "mapLegend": {"show": True},
            "toggle3d": {"show": False},
            "splitMap": {"show": False},
            "drawPolygon": {"show": False},
        },
    }
    interaction = {
        "tooltip": {
            "enabled": True,
            "compareMode": False,
            "fieldsToShow": {
                "routes": [
                    {"name": "方案标签", "format": None},
                    {"name": "累计距离_km", "format": None}
                ]
            },
        },
        "brush": {"enabled": False},
        "geocoder": {"enabled": False},
        "coordinate": {"enabled": True},
    }
    map_state = {
        "bearing": 0,
        "dragRotate": False,
        "latitude": center_lat,
        "longitude": center_lon,
        "pitch": 0,
        "zoom": 3,
        "minZoom": 2.0,
        "maxZoom": 6.0,
        "isSplit": False,
    }
    cfg = {
        "version": "v1",
        "config": {
            "visState": {"layers": [layer], "interactionConfig": interaction},
            "mapState": map_state,
            "mapStyle": {"styleType": "dark"},
            "uiState": ui_state,
        },
    }
    return cfg


def _haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    try:
        import math as _m
        R = 6371.0088
        phi1, phi2 = _m.radians(lat1), _m.radians(lat2)
        dphi = _m.radians(lat2 - lat1)
        dl = _m.radians(lon2 - lon1)
        a = _m.sin(dphi/2)**2 + _m.cos(phi1) * _m.cos(phi2) * _m.sin(dl/2)**2
        return 2 * R * _m.asin(_m.sqrt(a))
    except Exception:
        return 0.0

def _render_kepler(paths: List[Dict[str, Any]], view_bounds: Tuple[float, float, float, float]) -> None:
    # Build GeoJSON FeatureCollection, attach per-feature scenario category
    features = []
    for item in paths:
        name = item.get("name", "路线")
        # categorize by name prefix
        if name.startswith("方案 A"):
            scenario = "A"
            label_cn = "方案 A（当前）"
        elif name.startswith("方案 B"):
            scenario = "B"
            label_cn = "方案 B（应用反馈后）"
        else:
            scenario = "C"
            label_cn = "方案 C（外部导入）"
        coords = [[float(lon), float(lat)] for lon, lat in item["path"]]
        # cumulative distance (km) along the line
        total_km = 0.0
        for i in range(1, len(coords)):
            lon1, lat1 = coords[i-1]
            lon2, lat2 = coords[i]
            total_km += _haversine_km(lon1, lat1, lon2, lat2)
        features.append({
            "type": "Feature",
            "properties": {
                "name": name,
                "scenario": scenario,
                "方案标签": label_cn,
                "累计距离_km": round(total_km, 1),
            },
            "geometry": {
                "type": "LineString",
                "coordinates": coords,
            },
        })
    fc = {"type": "FeatureCollection", "features": features}

    try:
        config = get_default_kepler_config(view_bounds, categories=["A", "B", "C"])
        kmap = KeplerGl(height=520, config=config)  # presentation-oriented kepler view
        # Kepler expects tabular data; embed geojson as a single column if necessary
        # Some versions accept raw GeoJSON via add_data
        kmap.add_data(data=fc, name="routes")
        keplergl_static(kmap, center_map=False)
    except Exception as e:
        st.warning(f"kepler 展示模式加载失败，已回退到 pydeck：{e}")
        _render_pydeck(paths, view_bounds)


def _render_pydeck(paths: List[Dict[str, Any]], view_bounds: Tuple[float, float, float, float]) -> None:
    if not _HAS_PYDECK:
        st.info("未检测到 pydeck，无法回退渲染。请安装 pydeck 或启用 keplergl。")
        return
    layers = []
    # 单层多路径 PathLayer，使用每条记录的颜色
    data = []
    for item, color in zip(paths, _color_palette() * 5):
        data.append({
            "name": item["name"],
            "path": [[float(lon), float(lat)] for lon, lat in item["path"]],
            "color": list(color),
        })
    layer = pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
        pickable=True,
    )
    layers.append(layer)

    lon_min, lon_max, lat_min, lat_max = view_bounds
    view_state = pdk.ViewState(
        longitude=(lon_min + lon_max) / 2.0,
        latitude=(lat_min + lat_max) / 2.0,
        zoom=3,
        pitch=0,
        bearing=0,
    )
    r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(r, width='stretch')

def _render_pydeck_with_ship(paths: List[Dict[str, Any]], view_bounds: Tuple[float, float, float, float], ship: Dict[str, Any] | None = None, hotspots: List[Dict[str, Any]] | None = None) -> None:
    """pydeck 渲染 + 可选小船 marker（Scatterplot）。"""
    if not _HAS_PYDECK:
        st.info("未检测到 pydeck，无法回退渲染。请安装 pydeck 或启用 keplergl。")
        return
    layers = []
    # 路径层
    data = []
    for item, color in zip(paths, _color_palette() * 5):
        data.append({
            "name": item["name"],
            "path": [[float(lon), float(lat)] for lon, lat in item["path"]],
            "color": list(color),
        })
    path_layer = pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
        pickable=True,
    )
    layers.append(path_layer)

    # 小船层
    if ship is not None and all(k in ship for k in ("lon", "lat")):
        ship_data = [{
            "name": "小船",
            "lon": float(ship["lon"]),
            "lat": float(ship["lat"]),
            "s": float(ship.get("s", 0.0) or 0.0),
            "risk_total": float(ship.get("risk_total", 0.0) or 0.0),
            "sic": None if ship.get("sic") is None else float(ship.get("sic")),
            "swh": None if ship.get("swh") is None else float(ship.get("swh")),
            "co2": None if ship.get("co2") is None else float(ship.get("co2")),
            "reason": "当前回放位置",
        }]
        ship_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ship_data,
            get_position='[lon, lat]',
            get_fill_color='[255, 255, 255, 255]',
            get_radius=8000,
            radius_min_pixels=6,
            radius_max_pixels=22,
            pickable=True,
        )
        layers.append(ship_layer)
    # 热点层
    if hotspots:
        hs_data = []
        for i, h in enumerate(hotspots):
            try:
                hs_data.append({
                    "name": f"热点#{i+1}",
                    "lon": float(h.get("lon")),
                    "lat": float(h.get("lat")),
                    "s": float(h.get("s", 0.0) or 0.0),
                    "risk_total": h.get("risk_total"),
                    "risk_ice": h.get("risk_ice"),
                    "swh": h.get("wave_swh"),
                    "reason": str(h.get("reason", "热点")),
                })
            except Exception:
                continue
        if hs_data:
            hs_layer = pdk.Layer(
                "ScatterplotLayer",
                data=hs_data,
                get_position='[lon, lat]',
                get_fill_color='[255, 100, 0, 220]',
                get_radius=12000,
                radius_min_pixels=5,
                radius_max_pixels=30,
                pickable=True,
            )
            layers.append(hs_layer)
    # 统一 tooltip（兼容路径/小船/热点）
    tooltip = {
        "html": "<b>{name}</b><br/>{reason}<br/>里程: {s} km<br/>总风险: {risk_total}<br/>冰风险: {risk_ice}<br/>浪高: {swh} m<br/>CO₂ 累积: {co2}",
        "style": {"backgroundColor": "#0b1220", "color": "#E6EDF7"}
    }

    lon_min, lon_max, lat_min, lat_max = view_bounds
    view_state = pdk.ViewState(
        longitude=(lon_min + lon_max) / 2.0,
        latitude=(lat_min + lat_max) / 2.0,
        zoom=3,
        pitch=0,
        bearing=0,
    )
    r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip=tooltip)
    st.pydeck_chart(r, width='stretch')
    if not _HAS_PYDECK:
        st.info("未检测到 pydeck，无法回退渲染。请安装 pydeck 或启用 keplergl。")
        return
    layers = []
    # 单层多路径 PathLayer，使用每条记录的颜色
    data = []
    for item, color in zip(paths, _color_palette() * 5):
        data.append({
            "name": item["name"],
            "path": [[float(lon), float(lat)] for lon, lat in item["path"]],
            "color": list(color),
        })
    layer = pdk.Layer(
        "PathLayer",
        data=data,
        get_path="path",
        get_color="color",
        width_scale=2,
        width_min_pixels=3,
        pickable=True,
    )
    layers.append(layer)

    lon_min, lon_max, lat_min, lat_max = view_bounds
    view_state = pdk.ViewState(
        longitude=(lon_min + lon_max) / 2.0,
        latitude=(lat_min + lat_max) / 2.0,
        zoom=3,
        pitch=0,
        bearing=0,
    )
    r = pdk.Deck(layers=layers, initial_view_state=view_state, tooltip={"text": "{name}"})
    st.pydeck_chart(r, width='stretch')


def _get_point_along_profile(profile: Dict[str, Any], idx: int) -> Dict[str, Any]:
    if not profile:
        return {}
    lat = profile.get("lat") or []
    lon = profile.get("lon") or []
    n = min(len(lat), len(lon))
    if n == 0:
        return {}
    k = max(0, min(int(idx), n - 1))
    def _get(arr, k):
        try:
            v = arr[k]
            return None if v is None else float(v)
        except Exception:
            return None
    return {
        "lat": _get(lat, k),
        "lon": _get(lon, k),
        "s": _get(profile.get("s") or [], k),
        "risk_total": _get(profile.get("risk_total") or [], k),
        "sic": _get(profile.get("risk_ice") or [], k),  # 占位：以 risk_ice 近似展示
        "swh": _get(profile.get("wave_swh") or [], k),
        "co2": _get(profile.get("co2_cum") or [], k),
    }

def render(env_ctx: Any, route_result: Any, extra_routes: Any = None) -> None:
    """高级可视化 Tab 渲染入口。
    - env_ctx: 规划环境上下文（用于 bounds 估计）
    - route_result: 当前主路线
    - extra_routes: 可为 None/单个/列表（如 review 路线等）
    """
    st.markdown("#### 多方案路线对比")

    ureg = UIRegistry()
    replay_enabled = bool(ureg.is_advanced_enabled("enable_route_replay", False))
    hotspot_enabled = bool(ureg.is_advanced_enabled("enable_hotspot_markers", False))

    all_items: List[Dict[str, Any]] = []

    main_path = _extract_lonlat_from_route(route_result)
    if main_path:
        all_items.append({"name": "方案 A（当前）", "path": main_path})

    # 收集额外路线（列表或单个）
    if extra_routes is not None:
        if isinstance(extra_routes, list):
            for idx, r in enumerate(extra_routes):
                p = _extract_lonlat_from_route(r)
                if p:
                    all_items.append({"name": "方案 B（应用反馈后）", "path": p})
        else:
            p = _extract_lonlat_from_route(extra_routes)
            if p:
                all_items.append({"name": "方案 B（应用反馈后）", "path": p})

    # 允许用户上传更多方案（GeoJSON LineString）
    up = st.file_uploader("可选：上传其他方案 GeoJSON（LineString/MultiLineString）进行对比", type=["json", "geojson"], accept_multiple_files=True)
    if up:
        for f in up:
            content = f.read()
            p = _read_geojson_lines(content)
            if p:
                all_items.append({"name": "方案 C（外部导入）", "path": p})

    if not all_items:
        st.info("暂无可对比的路线。请先在“规划结果”页生成当前路线，或上传外部 GeoJSON。")
        return

    # 视域计算：基于所有路径
    view_bounds = _bounds_from_routes([it["path"] for it in all_items])

    # 主路线剖面（仅供回放使用；失败则不影响地图）
    profile_main: Dict[str, Any] | None = None
    if replay_enabled and route_result is not None and getattr(route_result, "reachable", False):
        try:
            profile_main = ps.sample_route_profile(route_result, env_ctx)
        except Exception:
            profile_main = None

    # 回放控制条（仅在开启开关且 profile 可用时显示）
    if replay_enabled and profile_main and (profile_main.get("lat") is not None):
        lat_arr = profile_main.get("lat") or []
        max_steps = max(0, (len(lat_arr) - 1))
        if "route_replay_idx" not in st.session_state:
            st.session_state["route_replay_idx"] = 0
        if "route_replay_playing" not in st.session_state:
            st.session_state["route_replay_playing"] = False
        with st.container():
            replay_col1, replay_col2, replay_col3 = st.columns([1, 4, 1])
            with replay_col1:
                play = st.button("▶ 播放", key="route_replay_play")
                pause = st.button("⏸ 暂停", key="route_replay_pause")
            with replay_col2:
                idx = st.slider("航程进度", 0, max_steps, int(st.session_state.get("route_replay_idx", 0)), key="route_replay_slider")
            with replay_col3:
                reset = st.button("⟲ 重置", key="route_replay_reset")
        # 状态更新
        if 'idx' in locals():
            st.session_state["route_replay_idx"] = int(idx)
        if 'play' in locals() and play:
            st.session_state["route_replay_playing"] = True
        if 'pause' in locals() and pause:
            st.session_state["route_replay_playing"] = False
        if 'reset' in locals() and reset:
            st.session_state["route_replay_idx"] = 0
            st.session_state["route_replay_playing"] = False
        # 播放推进（轻量增量 + rerun）
        try:
            import time as _t
            if st.session_state.get("route_replay_playing", False):
                k = int(st.session_state.get("route_replay_idx", 0))
                if k < max_steps:
                    st.session_state["route_replay_idx"] = k + 1
                    _t.sleep(0.12)
                    st.experimental_rerun()
                else:
                    st.session_state["route_replay_playing"] = False
        except Exception:
            pass
    else:
        max_steps = 0

    # 选择渲染后端
    backend = st.radio("渲染后端", ["自动", "keplergl（若可用）", "pydeck 回退"], index=0, horizontal=True)

    # 计算小船点（仅 pydeck 生效）
    ship = None
    if replay_enabled and profile_main and (profile_main.get("lat") is not None):
        ship = _get_point_along_profile(profile_main, int(st.session_state.get("route_replay_idx", 0)))

    # 计算热点（仅一次，不影响性能）
    hotspots = None
    if hotspot_enabled and profile_main and (profile_main.get("lat") is not None):
        try:
            hotspots = ps.extract_route_hotspots(profile_main, top_k=3)
            # 为 tooltip 提供里程 s
            try:
                s_arr = profile_main.get("s") or []
                for h in hotspots or []:
                    idx = int(h.get("idx", -1))
                    if 0 <= idx < len(s_arr):
                        h["s"] = float(s_arr[idx])
            except Exception:
                pass
            st.session_state["route_hotspots_main"] = hotspots
        except Exception:
            hotspots = None

    # 渲染
    if backend == "自动":
        if _HAS_KEPLER:
            _render_kepler(all_items, view_bounds)
            if replay_enabled:
                st.caption("提示：keplergl 当前不支持内置小船动画，已仅渲染路线图层。切换到“pydeck 回退”可查看回放。")
            if hotspot_enabled and hotspots:
                st.caption("提示：keplergl 当前未自动叠加热点点层。如需热点标记，请切换到“pydeck 回退”。")
        else:
            _render_pydeck_with_ship(all_items, view_bounds, ship=ship, hotspots=hotspots)
            st.caption("当前为简化视图（pydeck 回退）")
    elif backend.startswith("kepler"):
        if _HAS_KEPLER:
            _render_kepler(all_items, view_bounds)
            if replay_enabled:
                st.caption("提示：keplergl 当前不支持内置小船动画，已仅渲染路线图层。切换到“pydeck 回退”可查看回放。")
            if hotspot_enabled and hotspots:
                st.caption("提示：keplergl 当前未自动叠加热点点层。如需热点标记，请切换到“pydeck 回退”。")
        else:
            st.warning("未安装 keplergl，已回退至 pydeck。")
            _render_pydeck_with_ship(all_items, view_bounds, ship=ship, hotspots=hotspots)
            st.caption("当前为简化视图（pydeck 回退）")
    else:
        _render_pydeck_with_ship(all_items, view_bounds, ship=ship, hotspots=hotspots)

    # 当前位置信息（可选小卡片）
    if replay_enabled and ship and (ship.get("lat") is not None):
        st.markdown("""
        <div style="background: rgba(12,22,42,0.75); border:1px solid rgba(255,255,255,0.08); border-radius:14px; padding:10px 12px; margin-top:8px;">
          <div style="font-weight:600;color:#E6EDF7;">当前位置信息</div>
          <div style="color:#B9C6D8;font-size:13px;line-height:1.4;">
            纬度: {lat:.4f}，经度: {lon:.4f}<br/>
            里程: {s:.1f} km；总风险: {risk_total:.2f}<br/>
            冰浓度(代理): {sic}；浪高: {swh} m；CO₂ 累积: {co2} t
          </div>
        </div>
        """.format(
            lat=float(ship.get("lat", 0.0) or 0.0),
            lon=float(ship.get("lon", 0.0) or 0.0),
            s=float(ship.get("s", 0.0) or 0.0),
            risk_total=float(ship.get("risk_total", 0.0) or 0.0),
            sic=("—" if ship.get("sic") is None else f"{float(ship.get('sic')):.2f}"),
            swh=("—" if ship.get("swh") is None else f"{float(ship.get('swh')):.1f}"),
            co2=("—" if ship.get("co2") is None else f"{float(ship.get('co2')):.2f}"),
        ), unsafe_allow_html=True)

