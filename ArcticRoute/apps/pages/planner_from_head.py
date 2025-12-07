from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import pandas as pd
import copy

# Correctly import the service layer
from ArcticRoute.core import planner_service
from ArcticRoute.apps.theme import inject_theme, read_theme_flag
from ArcticRoute.apps.registry import UIRegistry
from ArcticRoute.apps.services import ai_explainer

# Optional imports for map display
try:
    import folium
    from streamlit_folium import st_folium
    _HAS_FOLIUM = True
except ImportError:
    _HAS_FOLIUM = False

@st.cache_resource
def get_base_map(center_lat, center_lon, zoom_start):
    """Creates and caches the base Folium map object."""
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=zoom_start, 
        tiles="cartodbpositron",
        max_bounds=True,  # Restrict map to a single world view
        world_copy_jump=False
    )
    return m

# ====================
# 鏁版嵁鏈夋晥鑼冨洿锛堣摑鑹茬煩褰級缁熶竴璁＄畻涓庣粯鍒?# ====================
import numpy as np


def _extract_route_lonlat(route_result):
    """
    浠?RouteResult 涓彁鍙?(lon, lat) 搴忓垪銆?    鍋氬埌瀛楁鍚嶅吋瀹癸細浼樺厛浣跨敤 path_lonlat / lonlat_path / lonlat_list / coords_lonlat 绛夈€?    鑻ユ壘涓嶅埌锛岃繑鍥?None銆?    鍏煎 (lat, lon) 涓?(lon, lat) 涓ょ椤哄簭锛屽苟鑷姩澶勭悊寮у害->搴︿笌缁忓害褰掍竴鍖栥€?    """
    if route_result is None:
        return None

    candidate_attrs = [
        "path_lonlat",
        "lonlat_path",
        "lonlat_list",
        "coords_lonlat",
    ]
    coords = None
    for name in candidate_attrs:
        val = getattr(route_result, name, None)
        if val is not None:
            coords = val
            break

    if coords is None:
        return None

    arr = np.array(coords)
    if arr.ndim != 2 or arr.shape[1] != 2:
        return None

    # 鑷姩璇嗗埆 (lat, lon) / (lon, lat) 涓庡姬搴?    a0 = arr[:, 0].astype(float)
    a1 = arr[:, 1].astype(float)
    # 寮у害鍒板害锛堣嫢涓ゅ垪閮藉儚寮у害锛?    if np.nanmax(np.abs(a0)) <= 3.2 and np.nanmax(np.abs(a1)) <= 3.2:
        a0 = np.degrees(a0)
        a1 = np.degrees(a1)

    # 鍒ゅ畾鍒楀惈涔夛細浼樺厛璁や负绾害缁濆鍊?=90
    if np.nanmax(np.abs(a0)) <= 90.0 and np.nanmax(np.abs(a1)) <= 180.0:
        lats = a0
        lons = a1
    elif np.nanmax(np.abs(a1)) <= 90.0 and np.nanmax(np.abs(a0)) <= 180.0:
        lons = a0
        lats = a1
    else:
        # 閫€鍖栵細鎸?(lon, lat)
        lons = a0
        lats = a1

    # 缁忓害缁熶竴鍒?[-180, 180]
    lons = ((lons + 180.0) % 360.0) - 180.0

    return lons, lats


essential_lat_margin = 5.0
essential_lon_margin = 15.0

def _compute_bounds_from_route(route_result, lat_margin: float = essential_lat_margin, lon_margin: float = essential_lon_margin):
    """
    浼樺厛浣跨敤璺嚎鍧愭爣璁＄畻 bounds锛屽苟鍦ㄥ洓鍛ㄥ姞 margin銆?    鍙璺嚎瀛樺湪涓旇法搴︿笉鏄繃灏忥紝灏辫繑鍥?(lat_min, lat_max, lon_min, lon_max)銆?    鍚﹀垯杩斿洖 None銆?    """
    lonlat = _extract_route_lonlat(route_result)
    if lonlat is None:
        print("[DATA_EXTENT] route_result has no usable lonlat; skip route-based bounds")
        return None

    lons, lats = lonlat
    if lons.size == 0 or lats.size == 0:
        print("[DATA_EXTENT] route lonlat empty; skip route-based bounds")
        return None

    lat_min = float(np.nanmin(lats))
    lat_max = float(np.nanmax(lats))
    lon_min = float(np.nanmin(lons))
    lon_max = float(np.nanmax(lons))

    # 璺嚎鑷韩澶煭鐨勮瘽锛屽氨涓嶈鐢ㄥ畠鏉ョ畻 bounds
    if (lat_max - lat_min) < 0.5 and (lon_max - lon_min) < 1.0:
        print("[DATA_EXTENT] route span too small; skip route-based bounds:", lat_min, lat_max, lon_min, lon_max)
        return None

    # 澧炲姞 margin锛屽苟瑁佸壀鍒板湴鐞冨悎娉曡寖鍥?    lat_min -= lat_margin
    lat_max += lat_margin
    lon_min -= lon_margin
    lon_max += lon_margin

    lat_min = max(lat_min, -90.0)
    lat_max = min(lat_max, 90.0)
    lon_min = max(lon_min, -180.0)
    lon_max = min(lon_max, 180.0)

    print("[DATA_EXTENT] ROUTE-BASED lat_min/max:", lat_min, lat_max, "lon_min/max:", lon_min, lon_max)

    return lat_min, lat_max, lon_min, lon_max


def _compute_bounds_from_grid(env_ctx):
    """
    閫€鑰屾眰鍏舵锛氱敤鐜缃戞牸 lat2d/lon2d 鐨勬暣浣撹寖鍥淬€?    鑻?lat2d/lon2d 涓嶅瓨鍦紝鍒欏皾璇?lat_arr/lon_arr銆?    鑻ユ渶缁堣寖鍥磋繃灏忥紝鍒欒繑鍥?None銆?    """
    lat2d = getattr(env_ctx, "lat2d", None)
    lon2d = getattr(env_ctx, "lon2d", None)

    if lat2d is not None and lon2d is not None:
        print("[DATA_EXTENT] using lat2d/lon2d for grid bounds")
        lat_vals = lat2d.values if hasattr(lat2d, "values") else np.array(lat2d)
        lon_vals = lon2d.values if hasattr(lon2d, "values") else np.array(lon2d)
    else:
        lat_arr = getattr(env_ctx, "lat_arr", None)
        lon_arr = getattr(env_ctx, "lon_arr", None)
        if lat_arr is None or lon_arr is None:
            print("[DATA_EXTENT] no lat2d/lon2d or lat_arr/lon_arr; skip grid bounds")
            return None

        print("[DATA_EXTENT] lat2d/lon2d missing, falling back to lat_arr/lon_arr")
        lat_1d = lat_arr.values if hasattr(lat_arr, "values") else np.array(lat_arr)
        lon_1d = lon_arr.values if hasattr(lon_arr, "values") else np.array(lon_arr)
        lon_vals, lat_vals = np.meshgrid(lon_1d, lat_1d)

    lat_flat = np.asarray(lat_vals).ravel()
    lon_flat = np.asarray(lon_vals).ravel()

    lon_flat = ((lon_flat + 180.0) % 360.0) - 180.0

    lat_min = float(np.nanmin(lat_flat))
    lat_max = float(np.nanmax(lat_flat))
    lon_min = float(np.nanmin(lon_flat))
    lon_max = float(np.nanmax(lon_flat))

    # 缃戞牸鑼冨洿澶皬灏辫涓轰笉闈犺氨锛屼緥濡?< 1掳x5掳
    if (lat_max - lat_min) < 1.0 or (lon_max - lon_min) < 5.0:
        print("[DATA_EXTENT] GRID suspiciously small:", lat_min, lat_max, lon_min, lon_max)
        return None

    print("[DATA_EXTENT] GRID lat_min/max:", lat_min, lat_max, "lon_min/max:", lon_min, lon_max)

    return lat_min, lat_max, lon_min, lon_max


def _compute_extent_bounds(env_ctx, route_result):
    """
    缁熶竴鐨?bounds 璁＄畻閫昏緫锛?    1) 浼樺厛鐢?route-based bounds (+ margin)
    2) 鍚﹀垯鐢?grid-based bounds
    3) 鑻ヤ粛鐒跺け璐ユ垨杩囧皬锛屽己纭?fallback 鍒颁竴涓浐瀹氱殑鍖楁瀬鍖呯粶妗?    """
    # 1) 鍏堝皾璇曞熀浜庤矾绾?    bounds = _compute_bounds_from_route(route_result)
    if bounds is not None:
        return bounds

    # 2) 鍐嶅皾璇曞熀浜庣綉鏍?    bounds = _compute_bounds_from_grid(env_ctx)
    if bounds is not None:
        return bounds

    # 3) 鏈€鍚庡己纭?fallback锛氬浐瀹氬寳鏋佽寖鍥?    lat_min, lat_max = 50.0, 90.0
    lon_min, lon_max = -180.0, 180.0
    print("[DATA_EXTENT] FALLBACK to static Arctic bounds:", lat_min, lat_max, lon_min, lon_max)
    return lat_min, lat_max, lon_min, lon_max


# 瀹為檯鍦?folium 涓婄粯鍒惰摑鑹茬煩褰?try:
    import folium  # 纭繚 folium 鍙敤
    def _add_data_extent_rectangle(route_map, env_ctx, route_result):
        """
        鍦?folium 鍦板浘涓婃坊鍔犺摑鑹茬煩褰紝琛ㄧず鈥滄暟鎹?璺嚎鏈夋晥鍖哄煙鈥濄€?        - 浼樺厛鍥寸粫褰撳墠瑙勫垝鍑虹殑璺嚎锛堝姞 margin锛?        - 鑻ユ棤璺嚎锛屽垯鍥寸粫鏁翠釜缃戞牸
        - 鑻ヤ粛涓嶉潬璋憋紝鍒欎娇鐢ㄥ浐瀹氱殑鍖楁瀬鑼冨洿
        """
        if env_ctx is None:
            print("[DATA_EXTENT] env_ctx is None; skip rectangle")
            return

        lat_min, lat_max, lon_min, lon_max = _compute_extent_bounds(env_ctx, route_result)
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        folium.Rectangle(
            bounds=bounds,
            color="#0077b6",
            weight=2,
            fill=False,
            dash_array="4",
            tooltip="鏁版嵁鏈夋晥鑼冨洿"
        ).add_to(route_map)
except Exception:
    # 鏃?folium 鐜涓嬪拷鐣?    pass

# UI-only display smoothing for route polyline (Chaikin corner-cutting)
# 娉ㄦ剰锛氫粎鐢ㄤ簬 Folium 鏄剧ず锛屼笉褰卞搷浠讳綍璁＄畻/瀵煎嚭/鏈嶅姟灞傘€?def smooth_path_for_display(coords: list[list[float]], iterations: int = 2) -> list[list[float]]:
    """
    浣跨敤 Chaikin 绠楁硶瀵?(lat, lon) 璺緞鍋氬钩婊戯紝浠呯敤浜?UI 鏄剧ず銆?    - coords: [[lat, lon], ...] 鍘熷璺緞锛堣嚦灏?2 鐐癸級
    - iterations: 杩唬娆℃暟锛堝缓璁?2锛?    杩斿洖鏂扮殑 [[lat, lon], ...]锛岄灏剧偣淇濇寔涓嶅彉銆?    鑻ョ偣鏁?< 3 鎴?iterations <= 0 鍒欏師鏍疯繑鍥炪€?    """
    try:
        if coords is None or len(coords) < 3 or int(iterations) <= 0:
            return coords
        arr = np.asarray(coords, dtype=float)
        for _ in range(int(iterations)):
            new_pts = [arr[0]]  # 淇濈暀棣栫偣
            for i in range(len(arr) - 1):
                p0 = arr[i]
                p1 = arr[i + 1]
                q = 0.75 * p0 + 0.25 * p1
                r = 0.25 * p0 + 0.75 * p1
                new_pts.append(q)
                new_pts.append(r)
            new_pts.append(arr[-1])  # 淇濈暀灏剧偣
            arr = np.vstack(new_pts)
        # 淇濇寔 (lat, lon) 椤哄簭杩斿洖
        out = [[float(p[0]), float(p[1])] for p in arr]
        return out
    except Exception:
        # 浠绘剰寮傚父鏃跺洖閫€涓哄師濮嬭矾寰勶紝淇濊瘉 UI 涓嶅穿婧?        return coords

def render(ctx: dict | None = None) -> None:
    """Renders the main planner page."""
    ureg = UIRegistry()
    # Eco 榛樿寮€鍏筹細鏉ヨ嚜 runtime.yaml 鐨?ui.eco.enabled锛涜嫢鏈厤缃紝鍒欓粯璁?True
    eco_cfg = ureg.flags.eco or {}
    eco_default = eco_cfg["enabled"] if isinstance(eco_cfg, dict) and ("enabled" in eco_cfg) else True

    inject_theme(read_theme_flag())

    st.title("ArcticRoute Planner 鈥?鍖楁瀬鑸嚎鏅鸿兘瑙勫垝")
    st.markdown("""
    閫夋嫨鏃堕棿鑼冨洿鍜岃捣姝㈢偣锛岀郴缁熷皢鍩轰簬娴峰啺銆侀闄╂垚鏈瓑澶氭ā鎬佹暟鎹紝
    鑷姩瑙勫垝涓€鏉″吋椤惧畨鍏ㄤ笌鏁堢巼鐨勫寳鏋佽埅绾匡紝骞剁粰鍑哄叧閿寚鏍囥€?    """)

    # --- Sidebar for Inputs ---
    with st.sidebar:
        if st.button("鍔犺浇绀轰緥鍦烘櫙", use_container_width=True):
            st.session_state.planner_ym = "202412"
            st.session_state.planner_input_method = "鏍呮牸绱㈠紩"
            # Corrected coordinates to be within the valid grid range [i: 0-120, j: 0-1160]
            st.session_state.planner_si = 60
            st.session_state.planner_sj = 150
            st.session_state.planner_gi = 60
            st.session_state.planner_gj = 1000
            st.session_state.pop("route_result", None)
            st.rerun()

        st.header("瑙勫垝鍙傛暟")
        ym = st.text_input("鏈堜唤 YYYYMM", st.session_state.get("planner_ym", "202412"))

        st.subheader("椋庨櫓鏉冮噸")
        
        # --- Planning Preference Selector ---
        preference = st.selectbox(
            "瑙勫垝鍋忓ソ",
            options=["鍧囪　妯″紡", "瀹夊叏浼樺厛", "鏁堢巼浼樺厛"],
            index=0,
            key="planning_preference"
        )

        # Eco 寮€鍏筹紙榛樿鏉ヨ嚜 runtime.yaml锛涙壘涓嶅埌鍒?True锛?        eco_enabled = st.checkbox(
            "鍚敤 Eco 缁胯壊璇勪及锛堢噧娌?+ CO鈧傦級",
            value=eco_default,
            help="鍏抽棴鏃跺皢浣跨敤鍩轰簬璺濈鐨勭畝鍗曚及绠椼€?,
            key="eco_enabled_checkbox"
        )

        # Define preset weights for each preference
        preference_weights = {
            "鍧囪　妯″紡": {"w_ice": 0.7, "w_accident": 0.2},
            "瀹夊叏浼樺厛": {"w_ice": 0.9, "w_accident": 0.5},
            "鏁堢巼浼樺厛": {"w_ice": 0.4, "w_accident": 0.1}
        }
        
        # Get the current weights based on preference
        current_weights = preference_weights[preference]
        
        # Sliders are now controlled by the preference selection
        w_ice = st.slider("娴峰啺椋庨櫓 (Ice)", 0.0, 1.0, current_weights["w_ice"], 0.05, key="w_ice_slider")
        w_accident = st.slider("浜嬫晠椋庨櫓 (Accident)", 0.0, 1.0, current_weights["w_accident"], 0.05, key="w_accident_slider")
        prior_weight = st.slider("閬靛惊鍘嗗彶鑸嚎鍋忓ソ", 0.0, 1.0, 0.0, 0.1, key="prior_weight_slider")
        # 濡傛灉褰撳墠鏃犲彲闈犱富鑸嚎鏁版嵁锛岀粰鍑烘彁绀猴紙鍗犱綅锛屼笉闅愯棌锛?        try:
            if not planner_service.prior_data_available():
                st.caption("鎻愮ず锛氬綋鍓嶆棤鏈夋晥鍘嗗彶涓昏埅绾挎暟鎹紝婊戝潡浠呭崰浣嶃€?)
        except Exception:
            pass

    # --- 楂樼骇璁剧疆锛堜晶杈规爮鎶樺彔锛?---
    with st.sidebar:
        adv_enabled = ureg.flags.advanced or {}
        with st.expander("楂樼骇椋庨櫓璁剧疆", expanded=False):
            fusion_mode = st.selectbox(
                "椋庨櫓铻嶅悎妯″紡",
                options=["baseline", "linear", "unetformer", "poe", "evidential"],
                index=["baseline", "linear", "unetformer", "poe", "evidential"].index(
                    str(ureg.get_advanced_default("fusion_mode_default", "baseline"))
                ),
                help="baseline 涓虹幇鏈夊熀纭€鎴愭湰锛涘叾浣欓€夐」灏濊瘯鍔犺浇/鏋勫缓 fused 椋庨櫓骞跺彔鍔犮€?,
                key="adv_fusion_mode",
            )
            default_w_interact = float(ureg.get_advanced_default("w_interact_default", 0.0) or 0.0)
            w_interact = st.slider("纰版挒/鎷ユ尋椋庨櫓鏉冮噸", 0.0, 1.0, default_w_interact, 0.1, key="adv_w_interact")
            default_use_escort = bool(ureg.get_advanced_default("use_escort_default", False))
            use_escort = st.checkbox("鍚敤鎶よ埅璧板粖鎶樺噺锛圧_ice_eff锛?, value=default_use_escort, key="adv_use_escort")
            risk_agg_mode = st.selectbox(
                "涓嶇‘瀹氭€ц仛鍚堟柟寮?,
                options=["mean", "quantile", "cvar"],
                index=["mean", "quantile", "cvar"].index(str(ureg.get_advanced_default("risk_agg_mode_default", "mean"))),
                key="adv_risk_agg_mode",
            )
            risk_agg_alpha = st.slider(
                "绋冲仴鎬у弬鏁?伪",
                0.5, 0.99,
                float(ureg.get_advanced_default("risk_agg_alpha_default", 0.9) or 0.9),
                0.01,
                key="adv_risk_agg_alpha",
            )
            land_mask_pad_px = st.slider(
                "闄嗗湴缂撳啿鍍忕礌",
                min_value=0,
                max_value=3,
                value=1,
                step=1,
                help="瀵归檰鍦版帺鑶滃仛褰㈡€佸鑶ㄨ儉鐨勫儚绱犳暟锛岄槻姝㈣创宀歌矾寰勭┛瓒婇檰鍦帮紙0 琛ㄧず涓嶅姞缂撳啿锛夈€?,
                key="adv_land_mask_pad_px",
            )

    # --- Load Environment Data (runs on every interaction) ---
    # 鍦ㄤ富鍖哄煙棰勭暀涓€涓€滃湴鍥句笂鏂光€濈殑鍔犺浇鏉″崰浣嶏紝纭繚浣嶇疆绋冲畾涓斿彲瑙?    env_loader_msg_ph = st.empty()
    env_loader_bar_ph = st.empty()
    try:
        import time
        with env_loader_msg_ph:
            st.info("姝ｅ湪鍔犺浇鐜鏁版嵁锛岃绋嶅€欌€?)
        pbar = env_loader_bar_ph.progress(0)
        for v in (5, 20, 35, 50, 65, 80, 90):
            pbar.progress(v)
            time.sleep(0.08)

        env_ctx = planner_service.load_environment(
            ym,
            w_ice=w_ice,
            w_accident=w_accident,
            prior_weight=prior_weight,
            fusion_mode=fusion_mode,
            w_interact=w_interact,
            use_escort=use_escort,
            risk_agg_mode=risk_agg_mode,
            risk_agg_alpha=risk_agg_alpha,
            land_mask_pad_px=land_mask_pad_px,
        )
        # 鏀跺熬锛氱‘淇濊嚦灏戞樉绀哄埌 100%锛屽苟鐭殏鍋滅暀锛岄伩鍏嶁€滈棯鐑佹劅鈥?        pbar.progress(100)
        time.sleep(0.12)
        env_loader_msg_ph.empty()
        env_loader_bar_ph.empty()

        st.session_state.env_ctx = env_ctx
        if env_ctx.cost_da is None:
            st.error(f"鏃犳硶鍔犺浇 {ym} 鏈堜唤鐨勬垚鏈暟鎹€?)
            st.stop()
    except Exception as e:
        env_loader_msg_ph.empty()
        env_loader_bar_ph.empty()
        st.error(f"鍔犺浇鐜鏁版嵁澶辫触: {e}")
        st.stop()

    with st.sidebar:
        st.subheader("璧锋鐐?)
        hint_texts = []
        if env_ctx.cost_da is not None:
            h, w = env_ctx.cost_da.shape[-2:]
            hint_texts.append(f"鏍呮牸: i鈭圼0,{h-1}], j鈭圼0,{w-1}]")
        if env_ctx.has_latlon and env_ctx.cost_da is not None:
            # 浣跨敤閲囨牱 + path_ij_to_lonlat 绋冲仴浼拌缁忕含搴﹁寖鍥达紙闃叉 0..360 涓庤法 180掳 闂锛?            try:
                h, w = env_ctx.cost_da.shape[-2:]
                sy = int(max(8, min(h, 64)))
                sx = int(max(16, min(w, 128)))
                ys = np.linspace(0, h-1, num=sy, dtype=int)
                xs = np.linspace(0, w-1, num=sx, dtype=int)
                ij_list = [(int(i), int(j)) for i in ys for j in xs]
                ll = planner_service.path_ij_to_lonlat(env_ctx, ij_list)
                lats = np.array([float(p[0]) for p in ll], dtype=float)
                lons = np.array([float(p[1]) for p in ll], dtype=float)
                mask = np.isfinite(lats) & np.isfinite(lons)
                lats = lats[mask]
                lons = lons[mask]
                # 鑻ョ枒浼煎姬搴︼紝杞负搴?                if lats.size > 0 and lons.size > 0 and (np.nanmax(np.abs(lats)) <= 3.2 and np.nanmax(np.abs(lons)) <= 3.2):
                    lats = np.degrees(lats)
                    lons = np.degrees(lons)
                # 鏈€灏忓渾寮у寘缁?                l = np.sort(((lons + 180.0) % 360.0) - 180.0)
                n = l.size
                if n > 0:
                    l_ext = np.concatenate([l, l + 360.0])
                    best_span = 1e9
                    lon_min, lon_max = -180.0, 180.0
                    for i in range(n):
                        j = i + n - 1
                        span = l_ext[j] - l_ext[i]
                        if span < best_span:
                            best_span = span
                            a = ((l_ext[i] + 180.0) % 360.0) - 180.0
                            b = ((l_ext[j] + 180.0) % 360.0) - 180.0
                            lon_min, lon_max = float(a), float(b)
                else:
                    lon_min, lon_max = -180.0, 180.0
                lat_min, lat_max = float(np.nanmin(lats)) if lats.size else -90.0, float(np.nanmax(lats)) if lats.size else 90.0
                # 鏈€灏忓彲瑙嗚法搴?                if (lat_max - lat_min) < 20.0:
                    mid = 0.5 * (lat_min + lat_max)
                    lat_min, lat_max = mid - 10.0, mid + 10.0
                if (lon_max - lon_min) < 80.0:
                    mid = 0.5 * (lon_min + lon_max)
                    lon_min, lon_max = mid - 40.0, mid + 40.0
                # 鍚堟硶瑁佸壀
                lat_min, lat_max = max(-90.0, lat_min), min(90.0, lat_max)
                lon_min, lon_max = max(-180.0, lon_min), min(180.0, lon_max)
                hint_texts.append(f"缁忕含搴? Lat鈭圼{lat_min:.1f},{lat_max:.1f}], Lon鈭圼{lon_min:.1f},{lon_max:.1f}]")
            except Exception:
                pass
        if hint_texts:
            st.caption(" 鈥?".join(hint_texts))

        input_method = st.radio("杈撳叆鏂瑰紡", ["缁忕含搴?, "鏍呮牸绱㈠紩"], index=["缁忕含搴?, "鏍呮牸绱㈠紩"].index(st.session_state.get("planner_input_method", "鏍呮牸绱㈠紩")))
        
        c1, c2, c3 = st.columns(3)
        if c1.button("閫夎捣鐐?, use_container_width=True):
            st.session_state.point_selection_mode = 'start'
        if c2.button("閫夌粓鐐?, use_container_width=True):
            st.session_state.point_selection_mode = 'goal'
        if c3.button("婕旂ず鐐?, use_container_width=True):
            st.session_state.planner_input_method = "鏍呮牸绱㈠紩"
            st.session_state.planner_si = 60
            st.session_state.planner_sj = 150
            st.session_state.planner_gi = 60
            st.session_state.planner_gj = 1000
            st.rerun()
        
        if input_method == "缁忕含搴?:
            start_lat = st.number_input("璧风偣绾害", value=st.session_state.get("planner_start_lat", 75.0))
            start_lon = st.number_input("璧风偣缁忓害", value=st.session_state.get("planner_start_lon", 15.0))
            goal_lat = st.number_input("缁堢偣绾害", value=st.session_state.get("planner_goal_lat", 72.0))
            goal_lon = st.number_input("缁堢偣缁忓害", value=st.session_state.get("planner_goal_lon", -175.0))
        else:
            si = st.number_input("璧风偣 i (y)", value=st.session_state.get("planner_si", 200))
            sj = st.number_input("璧风偣 j (x)", value=st.session_state.get("planner_sj", 150))
            gi = st.number_input("缁堢偣 i (y)", value=st.session_state.get("planner_gi", 320))
            gj = st.number_input("缁堢偣 j (x)", value=st.session_state.get("planner_gj", 480))

        st.subheader("绠楁硶鍙傛暟")
        allow_diagonal = st.checkbox("鍏佽鏂滃悜绉诲姩", value=True)
        heuristic = st.selectbox("鍚彂鍑芥暟", ["manhattan", "euclidean", "octile"], index=0)

        st.subheader("鏄剧ず鍥惧眰")
        show_sic = st.checkbox("娴峰啺娴撳害 (SIC)", value=False)
        show_cost = st.checkbox("椋庨櫓鎴愭湰 (Cost)", value=True)
        show_route = st.checkbox("瑙勫垝璺嚎", value=True)
        # 鏍规嵁鏁版嵁鍙敤鎬у喅瀹氭槸鍚﹀厑璁告樉绀哄巻鍙蹭富鑸嚎
        prior_avail = True
        try:
            prior_avail = planner_service.prior_data_available()
        except Exception:
            prior_avail = False
        show_main_routes = st.checkbox(
            "鏄剧ず鍘嗗彶涓昏埅绾?,
            value=bool(prior_avail),
            disabled=not prior_avail,
            help=None if prior_avail else "褰撳墠鏃犳湁鏁堝巻鍙蹭富鑸嚎鏁版嵁锛屽浘灞傚凡绂佺敤銆?,
        )

        # 瑙嗚骞虫粦锛堜粎褰卞搷鏄剧ず锛?        smooth_display = st.checkbox(
            "瑙嗚骞虫粦璺嚎锛堜粎褰卞搷鏄剧ず锛?,
            value=True,
            help="寮€鍚悗浠呭鍦板浘涓婄殑鑸嚎杩涜鏇茬嚎骞虫粦锛屼笉褰卞搷浠讳綍璺濈銆佹垚鏈€佺噧娌圭瓑璁＄畻缁撴灉銆?,
        )
        smooth_iterations = 2
        if smooth_display:
            smooth_iterations = st.slider(
                "骞虫粦娆℃暟",
                min_value=0,
                max_value=4,
                value=2,
                step=1,
                help="杩唬娆℃暟瓒婂ぇ瓒婂钩婊戯紙浠呭奖鍝嶆樉绀猴級銆? 琛ㄧず涓嶅钩婊戙€?,
                key="smooth_iterations",
            )
        # 璋冭瘯寮€鍏筹細鏄剧ず鏈钩婊戠殑鍘熷鎶樼嚎
        show_raw_route = st.checkbox(
            "鏄剧ず鍘熷缃戞牸鎶樼嚎锛堣皟璇曠敤锛?,
            value=False,
            help="浠呭紑鍙?璋冭瘯鐢細鏄剧ず鏈钩婊戠殑缃戞牸姝ヨ繘鎶樼嚎銆?,
            )

        if st.button("瑙勫垝璺嚎", use_container_width=True, type="primary"):
            if input_method == "缁忕含搴?:
                start_ij = planner_service.latlon_to_ij(env_ctx, start_lat, start_lon)
                goal_ij = planner_service.latlon_to_ij(env_ctx, goal_lat, goal_lon)
                if not start_ij or not goal_ij:
                    st.error("璧?缁堢偣缁忕含搴﹁秴鍑烘敮鎸佸尯鍩熴€?)
                    st.stop()
            else:
                start_ij, goal_ij = (si, sj), (gi, gj)
                # Validate grid indices to prevent out-of-bounds errors
                h, w = env_ctx.cost_da.shape[-2:]
                if not (0 <= start_ij[0] < h and 0 <= start_ij[1] < w and 0 <= goal_ij[0] < h and 0 <= goal_ij[1] < w):
                    st.error(f"鏍呮牸绱㈠紩瓒婄晫銆傛湁鏁堣寖鍥翠负 i: [0, {h-1}], j: [0, {w-1}]銆?)
                    st.stop()

            if start_ij == goal_ij:
                st.warning("璧风偣鍜岀粓鐐圭浉鍚屻€?)
                st.stop()



            st.session_state.planning_inputs = {"start_ij": start_ij, "goal_ij": goal_ij, "heuristic": heuristic, "allow_diagonal": allow_diagonal}

            with st.spinner("姝ｅ湪璁＄畻鑸嚎..."):
                route_result = planner_service.compute_route(env_ctx, start_ij, goal_ij, allow_diagonal, heuristic)
                if not route_result.reachable:
                    st.warning("鏈壘鍒板彲琛岃矾寰勩€?)
                summary = planner_service.summarize_route(route_result)
                # 闄勫姞楂樼骇妯″紡鍏冧俊鎭紝渚夸簬缁撴灉鍖烘樉绀轰笌璋冭瘯
                try:
                    attrs = getattr(env_ctx.cost_da, "attrs", {}) or {}
                    summary.update({
                        "fusion_mode_effective": attrs.get("fusion_mode_effective"),
                        "risk_agg_mode_effective": attrs.get("risk_agg_mode_effective", attrs.get("risk_agg_mode")),
                        "risk_agg_alpha": attrs.get("risk_agg_alpha"),
                        "escort_applied": bool(getattr(env_ctx, "escort_applied", False)),
                        "w_interact": attrs.get("w_interact"),
                    })
                except Exception:
                    pass
                cost_analysis = planner_service.analyze_route_cost(env_ctx, route_result)
                prior_adherence = planner_service.analyze_prior_adherence(env_ctx, route_result)

                eco_summary = None
                eco_mode = "simple"
                if route_result.reachable:
                    if eco_enabled:
                        try:
                            eco_summary = planner_service.evaluate_route_eco(route_result, env_ctx)
                            if not (getattr(eco_summary, "details", {}) or {}).get("ok", False):
                                st.warning("Eco 妯″潡涓嶅彲鐢ㄦ垨杩斿洖寮傚父锛屽凡鍥為€€涓哄熀浜庤窛绂荤殑浼扮畻銆?)
                                eco_summary = planner_service.estimate_eco_simple(route_result)
                                eco_mode = "simple"
                            else:
                                eco_mode = "eco"
                        except Exception:
                            st.warning("Eco 妯″潡璁＄畻澶辫触锛屽凡鍥為€€涓哄熀浜庤窛绂荤殑浼扮畻銆?)
                            eco_summary = planner_service.estimate_eco_simple(route_result)
                            eco_mode = "simple"
                    else:
                        eco_summary = planner_service.estimate_eco_simple(route_result)
                        eco_mode = "simple"

            st.session_state.route_result = route_result
            st.session_state.summary = summary
            st.session_state.cost_analysis = cost_analysis
            st.session_state.prior_adherence = prior_adherence
            st.session_state.eco_summary = eco_summary
            st.session_state.eco_mode = eco_mode
            st.rerun()

    # --- Main Panel ---
    st.subheader("鑸嚎鍦板浘")
    route_result = st.session_state.get("route_result")

    if _HAS_FOLIUM and env_ctx.has_latlon:
        # --- Initialize Map with a Stable View ---
        # Set a fixed initial view for the Arctic region and get the cached base map
        initial_center_lat, initial_center_lon, initial_zoom = 75.0, 0.0, 3
        base_map = get_base_map(initial_center_lat, initial_center_lon, initial_zoom)
        
        # Create a deep copy to modify for this specific render
        m = copy.deepcopy(base_map)

        # --- Fit Map to Route Bounds if a route exists ---
        if route_result and route_result.path_lonlat:
            # route_result.path_lonlat 缁熶竴涓?(lat, lon)
            route_bounds = [
                [min(p[0] for p in route_result.path_lonlat), min(p[1] for p in route_result.path_lonlat)],
                [max(p[0] for p in route_result.path_lonlat), max(p[1] for p in route_result.path_lonlat)]
            ]
            m.fit_bounds(route_bounds, padding=(20, 20))
        
        # Create a FeatureGroup to hold dynamic layers
        fg = folium.FeatureGroup(name="Dynamic Layers")

        # 灏忓伐鍏凤細GeoJSON [lon,lat] -> (lat,lon)锛屽苟瑙勮寖鍖栫粡搴﹀埌 [-180,180]
        def _lon180(lon: float) -> float:
            return ((float(lon) + 180.0) % 360.0) - 180.0
        def _geojson_coords_lonlat_to_latlon(coords):
            if isinstance(coords[0], (float, int)):
                lon, lat = coords
                return (float(lat), _lon180(float(lon)))
            else:
                return [_geojson_coords_lonlat_to_latlon(c) for c in coords]

        # === 鍥惧眰浣跨敤鐨勫奖鍍?bounds锛氭敼涓哄熀浜庣綉鏍艰寖鍥达紙涓嶅啀璋冪敤鏃х増鍑芥暟锛?===
        _grid_bounds = _compute_bounds_from_grid(env_ctx)
        if _grid_bounds is None:
            _grid_bounds = (50.0, 90.0, -180.0, 180.0)
        lat_min, lat_max, lon_min, lon_max = _grid_bounds
        bounds = [[lat_min, lon_min], [lat_max, lon_max]]

        # 鍘熸潵鐨勭煩褰唬鐮佹殏鏃跺仠鐢紝浣跨敤缁熶竴鐨勬湁闄愬€煎寘缁滆绠?        if False:
            folium.Rectangle(
                bounds=bounds,
                color='#0078A8',
                fill=True,
                fill_color='#0078A8',
                fill_opacity=0.1,
                tooltip="鏁版嵁鏈夋晥鑼冨洿"
            ).add_to(m)
            folium.Rectangle(
                bounds=bounds,
                color='#0078A8',
                fill=True,
                fill_color='#0078A8',
                fill_opacity=0.1,
                tooltip="鏁版嵁鏈夋晥鑼冨洿"
            ).add_to(fg)

        def _add_data_extent_rectangle_old(route_map, env_ctx):
            """[DEPRECATED] 鏃х増鐭╁舰缁樺埗鍑芥暟锛屼繚鐣欎互渚垮洖婧€?""
            pass

        # 浣跨敤缁熶竴鐨勬柊閫昏緫锛氫粎鍦ㄤ富 map 涓婄粯鍒朵竴娆¤摑鑹茬煩褰紙鍥寸粫璺嚎/缃戞牸/鍖楁瀬鍏滃簳锛?        _add_data_extent_rectangle(m, env_ctx, route_result)

        if show_sic and env_ctx.sic_da is not None:
            folium.raster_layers.ImageOverlay(image=env_ctx.sic_da.values, bounds=bounds, opacity=0.6, name='SIC', colormap=plt.cm.get_cmap('Blues')).add_to(fg)
        if show_cost and env_ctx.cost_da is not None:
            cost_normalized = (env_ctx.cost_da - np.nanmin(env_ctx.cost_da)) / (np.nanmax(env_ctx.cost_da) - np.nanmin(env_ctx.cost_da))
            folium.raster_layers.ImageOverlay(image=cost_normalized.values, bounds=bounds, opacity=0.6, name='Risk Cost', colormap=plt.cm.get_cmap('Reds')).add_to(fg)
        
        # --- Add Start/Goal Markers ---
        start_latlon, goal_latlon = None, None
        if input_method == "缁忕含搴?:
            start_latlon = [start_lat, start_lon]  # (lat, lon)
            goal_latlon = [goal_lat, goal_lon]
        else:  # "鏍呮牸绱㈠紩"
            try:
                if env_ctx.has_latlon:
                    start_latlon = planner_service.path_ij_to_lonlat(env_ctx, [(si, sj)])[0]  # (lat, lon)
                    goal_latlon = planner_service.path_ij_to_lonlat(env_ctx, [(gi, gj)])[0]
            except (IndexError, AttributeError):
                pass  # Gracefully ignore if conversion fails or env not ready

        if start_latlon:
            folium.Marker(location=[start_latlon[0], start_latlon[1]], icon=folium.Icon(color="green"), tooltip="璧风偣").add_to(fg)
        if goal_latlon:
            folium.Marker(location=[goal_latlon[0], goal_latlon[1]], icon=folium.Icon(color="red"), tooltip="缁堢偣").add_to(fg)
        
        # --- Add Route Polyline ---
        if show_route and route_result and route_result.path_lonlat:
            # 鍘熷璺緞 (lat, lon)
            route_lonlat_raw = [(float(p[0]), float(p[1])) for p in route_result.path_lonlat]
            # 浠呯敤浜庢樉绀虹殑骞虫粦锛氭湇鍔″眰鍑芥暟瑕佹眰杈撳叆涓?[(lon, lat)]
            if smooth_display:
                try:
                    raw_ll = [(lon, lat) for (lat, lon) in route_lonlat_raw]
                    disp_ll = planner_service.smooth_path_lonlat_for_display(raw_ll, iterations=smooth_iterations)
                    route_lonlat_display = [(lat, lon) for (lon, lat) in disp_ll]
                except Exception:
                    route_lonlat_display = route_lonlat_raw
            else:
                route_lonlat_display = route_lonlat_raw

            # 鐢?display 璺緞缁樺埗涓荤嚎
            folium.PolyLine(
                locations=[(lat, lon) for (lat, lon) in route_lonlat_display],
                color="#ff2f92",
                weight=4,
                opacity=0.9,
                smooth_factor=2.0,
            ).add_to(fg)

            # 鍙€夛細鍙犲姞鍘熷閿娇鎶樼嚎鐢ㄤ簬璋冭瘯
            if show_raw_route and route_lonlat_raw:
                folium.PolyLine(
                    locations=[(lat, lon) for (lat, lon) in route_lonlat_raw],
                    color="#888888",
                    weight=1.5,
                    opacity=0.6,
                    dash_array="4",
            ).add_to(fg)
        # --- Overlay Review Route (if any) ---
        route_review = st.session_state.get("route_review")
        if show_route and route_review and getattr(route_review, "path_lonlat", None):
            folium.PolyLine(locations=[(float(p[0]), float(p[1])) for p in route_review.path_lonlat], color="#005bbb", weight=3, opacity=0.9, tooltip="鍙嶉閲嶈鍒?).add_to(fg)

        # --- Add Historical Main Routes Layer ---
        if show_main_routes:
            import json, os
            main_routes_fg = folium.FeatureGroup(name="鍘嗗彶涓昏埅绾?, show=True)
            prior_data = None
            try:
                # 浼樺厛浣跨敤澶勭悊鍚庣殑 WGS84 鏂囦欢锛涜嫢涓嶅瓨鍦ㄥ垯浠庢湇鍔″眰鑾峰彇 GDF 骞惰浆鎹负 GeoJSON 鏄犲皠
                prior_path = "ArcticRoute/data_processed/prior/prior_centerlines_all_wgs84.geojson"
                if os.path.exists(prior_path):
                    with open(prior_path, "r", encoding="utf-8") as f:
                        prior_data = json.load(f)
                else:
                    main_routes_gdf = planner_service.load_prior_centerlines()
                    if main_routes_gdf is not None:
                        prior_data = main_routes_gdf.__geo_interface__
            except Exception:
                prior_data = None

            if prior_data:
                # 濡備富鑸嚎缁忓害涓?[0,360]锛屽湪缁樺埗鍓嶇粺涓€瑙勮寖鍒?[-180,180]锛屽悓鏃朵繚鎸?GeoJSON [lon,lat] 椤哄簭
                try:
                    import copy as _copy
                    def _wrap_coords(coords):
                        if isinstance(coords[0], (float, int)):
                            lon, lat = coords
                            return [((float(lon) + 180.0) % 360.0) - 180.0, float(lat)]
                        else:
                            return [_wrap_coords(c) for c in coords]
                    def _sample_lon(coords):
                        try:
                            if isinstance(coords[0], (float, int)):
                                return float(coords[0])
                            return _sample_lon(coords[0])
                        except Exception:
                            return None
                    need_wrap = False
                    if isinstance(prior_data, dict):
                        feats = prior_data.get('features') or []
                        for feat in feats[:3]:
                            geom = (feat or {}).get('geometry') or {}
                            coords = geom.get('coordinates')
                            lon0 = _sample_lon(coords) if coords is not None else None
                            if lon0 is not None and (lon0 > 180.0 + 1e-6 or lon0 < -360.0):
                                need_wrap = True
                                break
                        if need_wrap:
                            prior_data = _copy.deepcopy(prior_data)
                            for feat in prior_data.get('features', []):
                                geom = (feat or {}).get('geometry') or {}
                                coords = geom.get('coordinates')
                                if coords is not None:
                                    geom['coordinates'] = _wrap_coords(coords)
                except Exception:
                    pass

                folium.GeoJson(
                    prior_data,
                    name="Historical Main Routes",
                    style_function=lambda feature: {
                        "color": "#444444",
                        "weight": 1,
                        "opacity": 0.7,
                    },
                    tooltip="鍘嗗彶涓昏埅绾?,
                ).add_to(main_routes_fg)
                main_routes_fg.add_to(m)
        
        # Add the feature group to the map
        fg.add_to(m)
        folium.LayerControl().add_to(m)

        map_data = st_folium(m, key="planner_map", height=500, width="100%", returned_objects=["last_clicked"])

        if map_data and map_data.get("last_clicked"):
            clicked_lat = map_data["last_clicked"]["lat"]
            clicked_lon = map_data["last_clicked"]["lng"]
            selection_mode = st.session_state.get('point_selection_mode')

            if selection_mode:
                ij = planner_service.latlon_to_ij(env_ctx, clicked_lat, clicked_lon)
                if ij:
                    if selection_mode == 'start':
                        st.session_state.planner_input_method = "鏍呮牸绱㈠紩"
                        st.session_state.planner_si, st.session_state.planner_sj = ij[0], ij[1]
                        st.success(f"璧风偣宸叉洿鏂颁负: {ij}")
                    elif selection_mode == 'goal':
                        st.session_state.planner_input_method = "鏍呮牸绱㈠紩"
                        st.session_state.planner_gi, st.session_state.planner_gj = ij[0], ij[1]
                        st.success(f"缁堢偣宸叉洿鏂颁负: {ij}")
                    st.session_state.point_selection_mode = None # Reset mode
                    st.rerun()
                else:
                    st.warning("鐐瑰嚮浣嶇疆瓒呭嚭鏁版嵁鑼冨洿锛屾棤娉曢€夋嫨銆?)
    else:
        st.warning("鐜鏁版嵁缂哄皯缁忕含搴︿俊鎭垨 Folium 鏈畨瑁咃紝鏃犳硶鏄剧ず鍦板浘銆?)

    st.subheader("璺嚎鎽樿涓庢垚鏈垎鏋?)
    if route_result:
        summary = st.session_state.get("summary")
        eco_summary = st.session_state.get("eco_summary")

        # --- Tabs ---
        tabs_labels = ["瑙勫垝缁撴灉", "Pareto 瀵规瘮", "浜哄湪鍥炶矾 Review", "绋冲仴鎬т笌涓嶇‘瀹氭€?, "AI 瑙ｈ"]
        tab_main, tab_pareto, tab_review, tab_uncert, tab_ai = st.tabs(tabs_labels)

        with tab_main:
            if summary:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("鎬昏窛绂?(km)", f"{summary.get('distance_km', 0):.1f}")

                if eco_summary and eco_summary.fuel_total_t > 0:
                    c2.metric("鐕冩补娑堣€?(鍚?", f"{eco_summary.fuel_total_t:.2f}", help="鍩轰簬 Eco 妯″瀷浼扮畻")
                    c3.metric("CO2鎺掓斁 (鍚?", f"{eco_summary.co2_total_t:.2f}", help=f"鍩轰簬 Eco 妯″瀷浼扮畻锛岀噧娌规垚鏈害 ${eco_summary.cost_usd:,.0f} USD")
                else:
                    c2.metric("浼扮畻鐕冩补 (鍚?", f"{summary.get('estimated_fuel_ton', 0):.2f}", help="鍩轰簬璺濈鐨勭矖鐣ヤ及绠?)
                    c3.metric("CO2鎺掓斁 (鍚?", "鈥?, help="Eco 鍏抽棴鎴栦笉鍙敤鏃朵笉璁＄畻 CO鈧?鏄庣粏")
                
                c4.metric("涓昏埅绾胯创鍚堝害", f"{st.session_state.get('prior_adherence', 0.0):.2f}")

                # 妯″紡鎻愮ず
                eco_mode = st.session_state.get("eco_mode", "simple")
                eco_on_now = bool(st.session_state.get("eco_enabled_checkbox", True))
                if eco_mode == "eco" and eco_on_now:
                    st.caption("褰撳墠 Eco 妯″紡锛氬熀浜?Eco 妯″瀷浼扮畻銆?)
                elif not eco_on_now:
                    st.caption("锛堝凡鍏抽棴 Eco 妯″潡锛屽綋鍓嶄负鍩轰簬璺濈鐨勭矖鐣ヤ及绠楋級")
                else:
                    st.caption("锛圗co 妯″潡涓嶅彲鐢ㄦ垨寮傚父锛屽凡鍥為€€涓哄熀浜庤窛绂荤殑绮楃暐浼扮畻锛?)

            cost_analysis = st.session_state.get("cost_analysis")
            if cost_analysis:
                st.markdown("##### 璺嚎鎴愭湰鏋勬垚")
                df_cost = pd.DataFrame(list(cost_analysis.items()), columns=['椋庨櫓绫诲瀷', '鎴愭湰璐＄尞'])
                st.bar_chart(df_cost.set_index('椋庨櫓绫诲瀷'))

            # --- Export Route ---
            if route_result.reachable:
                geojson_data = planner_service.route_to_geojson(route_result)
                st.download_button(
                    label="瀵煎嚭涓?GeoJSON",
                    data=geojson_data,
                    file_name=f"route_{env_ctx.ym}.geojson",
                    mime="application/json",
                    use_container_width=True
                )

        with tab_pareto:
            if (ureg.is_advanced_enabled("enable_pareto_tab", True)):
                pts = planner_service.load_pareto_front(ym, scenario="default")
                if pts:
                    st.markdown("**Pareto 鍊欓€?*锛堝彧璇诲睍绀猴級")
                    df = pd.DataFrame(pts)
                    st.dataframe(df, use_container_width=True)
                    import json as _json
                    st.download_button("涓嬭浇 Pareto Front JSON", data=_json.dumps(pts, ensure_ascii=False, indent=2), file_name=f"pareto_front_{ym}_default.json", mime="application/json")
                else:
                    st.info("褰撳墠鏈壘鍒?Pareto 鍓嶆部 JSON锛坮eports/phaseG 鎴?d_stage/phaseG锛夈€?)
            else:
                st.info("鏈惎鐢?Pareto Tab銆?)

        with tab_review:
            if (ureg.is_advanced_enabled("enable_review_tab", True)):
                st.markdown("涓婁紶 feedback.jsonl锛屽簲鐢ㄧ害鏉熷悗閲嶆柊瑙勫垝锛堟渶灏忕ず渚嬶級")
                up = st.file_uploader("閫夋嫨 feedback.jsonl", type=["jsonl", "txt", "json"], help="楂樼骇鐢ㄦ埛鍙洿鎺ヤ笂浼?JSONL")
                # 涓婁紶鏂囦欢鐨勫揩鎹锋寜閽紙淇濇寔鍏煎锛?                if up and st.button("搴旂敤鍙嶉骞堕噸瑙勫垝锛堜娇鐢ㄤ笂浼犳枃浠讹級", use_container_width=True, key="btn_apply_fb_upload"):
                    import tempfile, pathlib
                    tmp = pathlib.Path(tempfile.gettempdir()) / f"feedback_{ym}.jsonl"
                    tmp.write_bytes(up.read())
                    try:
                        new_route = planner_service.apply_feedback_and_replan(
                            st.session_state.env_ctx,
                            route_result,
                            tmp,
                            allow_diagonal=st.session_state.planning_inputs.get("allow_diagonal", True),
                            heuristic=st.session_state.planning_inputs.get("heuristic", "manhattan"),
                        )
                        if new_route and new_route.reachable:
                            st.success("宸茬敓鎴愮害鏉熷悗鐨勬柊璺嚎锛堜笂浼犳枃浠讹級銆?)
                            st.session_state.route_review = new_route
                            base_sum = st.session_state.summary or {}
                            new_sum = planner_service.summarize_route(new_route)
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("**鍘熻矾绾?*")
                                st.json(base_sum)
                            with c2:
                                st.markdown("**鏂拌矾绾?*")
                                st.json(new_sum)
                            st.download_button("涓嬭浇鏂拌矾绾?GeoJSON", data=planner_service.route_to_geojson(new_route), file_name=f"route_{ym}_constrained.geojson", mime="application/json", use_container_width=True)
                        else:
                            st.warning("鏃犳硶鐢熸垚鏂拌矾绾裤€?)
                    except Exception as e:
                        st.error(f"閲嶈鍒掑け璐ワ細{e}")

                st.markdown("鈥斺€?鎴?鈥斺€?鍦ㄤ笅鏂瑰湴鍥句笂缁樺埗绂佽鍖?璧板粖鍚庝竴閿簲鐢細")
                # 鏋勫缓鍙粯鍒跺湴鍥?                try:
                    from folium.plugins import Draw  # type: ignore
                    review_map = copy.deepcopy(get_base_map(75.0, 0.0, 3))
                    try:
                        _lat_arr = np.asarray(env_ctx.lat_arr, dtype=float)
                        _lon_arr = np.asarray(env_ctx.lon_arr, dtype=float)
                        _lon_wrap = (((_lon_arr + 180.0) % 360.0) - 180.0)
                        lat_min, lat_max = float(np.nanmin(_lat_arr)), float(np.nanmax(_lat_arr))
                        lon_min, lon_max = float(np.nanmin(_lon_wrap)), float(np.nanmax(_lon_wrap))
                        lat_min, lat_max = max(-90.0, lat_min), min(90.0, lat_max)
                        lon_min, lon_max = max(-180.0, lon_min), min(180.0, lon_max)
                        bounds = [[lat_min, lon_min], [lat_max, lon_max]]
                    except Exception:
                        bounds = [[float(np.nanmin(env_ctx.lat_arr)), float(np.nanmin(env_ctx.lon_arr))], [float(np.nanmax(env_ctx.lat_arr)), float(np.nanmax(env_ctx.lon_arr))]]
                    folium.Rectangle(bounds=bounds, color='#0078A8', fill=True, fill_color='#0078A8', fill_opacity=0.1, tooltip="鏁版嵁鏈夋晥鑼冨洿").add_to(review_map)
                    Draw(
                        export=False,
                        filename="feedback.json",
                        draw_options={
                            "polyline": True,
                            "polygon": True,
                            "rectangle": True,
                            "circle": False,
                            "marker": False,
                            "circlemarker": False,
                        },
                        edit_options={"edit": True, "remove": True},
                    ).add_to(review_map)
                    draw_state = st_folium(review_map, key="review_draw_map", height=420, width="100%")
                except Exception:
                    draw_state = None

                def _extract_shapes(ds):
                    if not isinstance(ds, dict):
                        return []
                    if ds.get("all_drawings"):
                        return ds.get("all_drawings") or []
                    if ds.get("last_active_drawing"):
                        lad = ds.get("last_active_drawing")
                        return [lad] if lad else []
                    # 鏌愪簺鐗堟湰鍙兘鏀惧湪 "features"
                    if ds.get("features"):
                        return ds.get("features") or []
                    return []

                shapes = _extract_shapes(draw_state)
                st.caption(f"宸茬粯鍒跺嚑浣曟暟閲忥細{len(shapes) if shapes else 0}")

                # 鍗曟寜閽細浼樺厛浣跨敤涓婁紶鏂囦欢锛涘惁鍒欎娇鐢ㄧ粯鍒跺嚑浣?                if st.button("搴旂敤鍙嶉骞堕噸瑙勫垝锛堜笂浼犳垨缁樺埗锛?, use_container_width=True, key="btn_apply_fb_auto"):
                    import tempfile, pathlib
                    fb_path = None
                    try:
                        if up is not None:
                            tmp = pathlib.Path(tempfile.gettempdir()) / f"feedback_{ym}.jsonl"
                            tmp.write_bytes(up.read())
                            fb_path = tmp
                            fb_src = "upload"
                        else:
                            if shapes:
                                pack = planner_service.build_feedback_from_shapes(shapes)
                                tmp = pathlib.Path(tempfile.gettempdir()) / f"feedback_draw_{ym}.jsonl"
                                planner_service.write_feedback_jsonl(pack.get("records") or [], tmp)
                                fb_path = tmp
                                fb_src = "draw"
                    except Exception as e:
                        st.error(f"鏋勫缓鍙嶉澶辫触锛歿e}")
                        fb_path = None

                    if fb_path is None:
                        st.warning("璇蜂笂浼?feedback.jsonl 鎴栧湪鍦板浘涓婄粯鍒剁琛屽尯/璧板粖銆?)
                    else:
                        try:
                            new_route = planner_service.apply_feedback_and_replan(
                                st.session_state.env_ctx,
                                route_result,
                                fb_path,
                                allow_diagonal=st.session_state.planning_inputs.get("allow_diagonal", True),
                                heuristic=st.session_state.planning_inputs.get("heuristic", "manhattan"),
                            )
                            if new_route and new_route.reachable:
                                st.success(f"宸茬敓鎴愮害鏉熷悗鐨勬柊璺嚎锛堟潵婧愶細{'涓婁紶' if up is not None else '缁樺埗'}锛夈€?)
                                st.session_state.route_review = new_route
                                base_sum = st.session_state.summary or {}
                                new_sum = planner_service.summarize_route(new_route)
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.markdown("**鍘熻矾绾?*")
                                    st.json(base_sum)
                                with c2:
                                    st.markdown("**鏂拌矾绾?*")
                                    st.json(new_sum)
                                st.download_button("涓嬭浇鏂拌矾绾?GeoJSON", data=planner_service.route_to_geojson(new_route), file_name=f"route_{ym}_constrained.geojson", mime="application/json", use_container_width=True)
                            else:
                                st.warning("鏃犳硶鐢熸垚鏂拌矾绾裤€?)
                        except Exception as e:
                            st.error(f"閲嶈鍒掑け璐ワ細{e}")
            else:
                st.info("鏈惎鐢?Review Tab銆?)

        with tab_uncert:
            # 鏄剧ず鏈浣跨敤鐨勮仛鍚?铻嶅悎鍏冧俊鎭紙鍗犱綅锛?            try:
                attrs = getattr(st.session_state.env_ctx.cost_da, "attrs", {}) or {}
                rows = {k: attrs.get(k) for k in ["fusion_mode_effective", "risk_agg_mode", "risk_agg_alpha", "use_escort", "w_interact"]}
                st.json(rows)
            except Exception:
                st.info("鏃犲彲鐢ㄧ殑涓嶇‘瀹氭€?铻嶅悎鍏冧俊鎭€?)
        with tab_ai:
            st.subheader("AI 瑙ｈ鏈鑸嚎")
            rr = st.session_state.get("route_result")
            env_ctx = st.session_state.get("env_ctx")
            route_summary = st.session_state.get("summary")
            cost_breakdown = st.session_state.get("cost_analysis")
            eco_summary = st.session_state.get("eco_summary")
            eco_mode = st.session_state.get("eco_mode")
            prior_adherence = st.session_state.get("prior_adherence")
            prior_weight_val = float(st.session_state.get("prior_weight_slider", 0.0))

            if rr is None or env_ctx is None or route_summary is None:
                st.info("璇峰厛鍦ㄢ€滆鍒掔粨鏋溾€濇爣绛鹃〉瀹屾垚涓€娆℃垚鍔熺殑璺緞瑙勫垝锛岀劧鍚庡啀鐢熸垚 AI 瑙ｈ銆?)
            else:
                if "ai_explain_text" not in st.session_state:
                    st.session_state.ai_explain_text = None

                if st.session_state.ai_explain_text:
                    st.markdown(st.session_state.ai_explain_text)

                if st.button("鐢熸垚 AI 瑙ｈ", use_container_width=True):
                    with st.spinner("姝ｅ湪璋冪敤 AI 瑙ｉ噴鍣紝璇风◢鍊欌€?):
                        try:
                            payload = planner_service.build_ai_explain_payload(
                                env=env_ctx,
                                route=rr,
                                route_summary=route_summary,
                                cost_breakdown=cost_breakdown,
                                eco_summary=eco_summary,
                                prior_adherence=prior_adherence,
                                prior_weight=prior_weight_val,
                                eco_mode=eco_mode,
                                scenario="default",
                            )
                            text = ai_explainer.generate_route_explanation(payload)
                            st.session_state.ai_explain_text = text
                            st.markdown(text)
                        except Exception as e:
                            st.error(f"AI 瑙ｈ澶辫触锛歿e}")
    else:
        st.info("璇峰湪宸︿晶濉啓鍙傛暟骞剁偣鍑?**瑙勫垝璺嚎** 鎸夐挳浠ョ敓鎴愭憳瑕併€?)
