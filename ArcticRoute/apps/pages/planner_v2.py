from __future__ import annotations

import streamlit as st
import numpy as np
import pandas as pd

from ArcticRoute.core import planner_service
from ArcticRoute.apps.theme import inject_theme, read_theme_flag
from ArcticRoute.core.planner_service import RobustPlannerConfig, run_planning_pipeline_evidential_robust
from ArcticRoute.core.eco import vessel_profiles as eco_vessel_profiles


def render(ctx: dict | None = None) -> None:
    # 主题注入与标题
    inject_theme(read_theme_flag())
    st.title("ArcticRoute Planner v2")
    st.caption("多模态环境场下的北极航线智能规划 · 新版工作台（实验）")

    # 左右两列布局
    col_left, col_right = st.columns([1, 2])

    # 左侧控制面板
    with col_left:
        st.subheader("场景与时间")
        ym = st.text_input("环境时间 (ym)", value="202412", help="以 YYYYMM 形式输入，例如 202412")

        # 规划模式
        mode_labels = {
            "baseline": "经典模式（A* Baseline）",
            "evidential_robust": "高级模式（Evidential-Robust）",
        }
        planner_mode = st.selectbox(
            "规划模式 (Planner Mode)",
            options=list(mode_labels.keys()),
            index=0,
            format_func=lambda k: mode_labels.get(k, k),
        )

        st.markdown("---")
        st.subheader("起止点输入（经纬度）")
        # 默认经纬度（不预加载环境，避免页面初次渲染卡顿）
        start_lat_default, start_lon_default = 69.0, 33.0
        end_lat_default, end_lon_default = 70.5, 170.0
        # 提示：若后续需要，可在按下“规划路线”后，基于已加载的 env 再次自动校正端点
        start_lat = st.number_input("起点纬度 start_lat", min_value=60.0, max_value=85.0, value=float(start_lat_default), step=0.1)
        start_lon = st.number_input("起点经度 start_lon", min_value=-180.0, max_value=180.0, value=float(start_lon_default), step=0.1)
        end_lat = st.number_input("终点纬度 end_lat", min_value=60.0, max_value=85.0, value=float(end_lat_default), step=0.1)
        end_lon = st.number_input("终点经度 end_lon", min_value=-180.0, max_value=180.0, value=float(end_lon_default), step=0.1)

        # 船型选择
        try:
            all_profiles = eco_vessel_profiles.load_all_profiles() or {}
            vessel_keys = [k for k, v in all_profiles.items() if isinstance(v, dict) and (v.get("ui_visible", True))]
            vessel_keys.sort()
            def _v_label(k: str) -> str:
                v = all_profiles.get(k, {})
                name = v.get("display_name") or v.get("name") or k
                dwt = v.get("dwt")
                try:
                    return f"{name} ({float(dwt)/1000:.0f}k DWT)" if dwt is not None else str(name)
                except Exception:
                    return str(name)
        except Exception:
            vessel_keys = ["panamax"]
            all_profiles = {}
            def _v_label(k: str) -> str:
                return k
        vessel_key = st.selectbox(
            "船型 (vessel_profile)",
            options=vessel_keys or ["panamax"],
            index=(vessel_keys or ["panamax"]).index("panamax") if (vessel_keys and "panamax" in vessel_keys) else 0,
            format_func=_v_label,
            key="planner_v2_vessel_key",
            help="用于 ECO 与部分权重缩放的船型配置",
        )

        st.markdown("---")
        st.subheader("风险权重（简化版）")
        w_ice = st.slider("冰险权重 (w_ice)", 0.0, 2.0, 1.0, 0.1)
        prior_weight = st.slider("历史主航线权重 (prior_weight)", 0.0, 2.0, 0.5, 0.1)

        st.markdown("---")
        do_plan = st.button("规划路线", type="primary")

        # 将输入聚合为 params，右侧使用
        params = {
            "ym": str(ym),
            "start_lat": float(start_lat),
            "start_lon": float(start_lon),
            "end_lat": float(end_lat),
            "end_lon": float(end_lon),
            "w_ice": float(w_ice),
            "prior_weight": float(prior_weight),
            "planner_mode": str(planner_mode),
            "vessel_key": str(vessel_key),
        }
        st.caption("参数: " + str({k: v for k, v in params.items()}))

    # 右侧：计算与展示
    with col_right:
        if do_plan:
            # 先加载一次环境（balanced），给出成本构建提示
            try:
                with st.spinner("加载环境与构建成本场..."):
                    env_ctx = planner_service.load_environment(
                        ym=str(ym),
                        w_ice=float(w_ice),
                        w_accident=0.0,
                        prior_weight=float(prior_weight),
                        profile_name="balanced",
                        vessel_profile=str(vessel_key),
                    )
            except Exception as e:
                st.error(f"环境加载失败: {e}")
                return

            # 三种策略
            profiles = ["efficient", "balanced", "safe"]
            routes_info: list[dict] = []

            for prof in profiles:
                try:
                    env_ctx_p = planner_service.load_environment(
                        ym=str(ym),
                        w_ice=float(w_ice),
                        w_accident=0.0,
                        prior_weight=float(prior_weight),
                        profile_name=str(prof),
                        vessel_profile=str(vessel_key),
                    )
                except Exception as e:
                    routes_info.append({
                        "label": prof,
                        "route_result": None,
                        "summary": None,
                        "eco_summary": None,
                        "error": f"环境加载失败: {e}",
                        "color": None,
                    })
                    continue

                # 规划入口：优先严格端点函数
                try:
                    route_result = planner_service.compute_route_strict_from_latlon(
                        env_ctx=env_ctx_p,
                        start_lat=float(start_lat),
                        start_lon=float(start_lon),
                        end_lat=float(end_lat),
                        end_lon=float(end_lon),
                    )
                except Exception:
                    # 兜底：吸附到域与海面，再用 compute_route
                    try:
                        s_lat, s_lon, s_info = planner_service.snap_point_to_domain_and_ocean(
                            lat=float(start_lat), lon=float(start_lon), env_ctx=env_ctx_p
                        )
                        g_lat, g_lon, g_info = planner_service.snap_point_to_domain_and_ocean(
                            lat=float(end_lat), lon=float(end_lon), env_ctx=env_ctx_p
                        )
                        # latlon -> ij
                        try:
                            # planner_service.latlon_to_ij 可能不存在；此处复用 snap 返回的 grid_ij
                            s_ij = (int(s_info.get("grid_i")), int(s_info.get("grid_j")))
                            g_ij = (int(g_info.get("grid_i")), int(g_info.get("grid_j")))
                        except Exception:
                            s_ij = None
                            g_ij = None
                        if s_ij is None or g_ij is None:
                            raise RuntimeError("端点吸附失败，无法获取网格索引")
                        route_result = planner_service.compute_route(
                            env=env_ctx_p,
                            start_ij=s_ij,
                            goal_ij=g_ij,
                            allow_diagonal=True,
                            heuristic="euclidean",
                        )
                    except Exception as e:
                        routes_info.append({
                            "label": prof,
                            "route_result": None,
                            "summary": None,
                            "eco_summary": None,
                            "error": f"规划失败: {e}",
                            "color": None,
                        })
                        # 调试：记录失败信息
                        try:
                            st.write(f"[DEBUG] {prof} reachable=False, len(path_ij)=0, len(path_lonlat)=0, error={e}")
                        except Exception:
                            pass
                        continue

                # 汇总
                if not getattr(route_result, "reachable", False):
                    routes_info.append({
                        "label": prof,
                        "route_result": route_result,
                        "summary": None,
                        "eco_summary": None,
                        "error": "不可达",
                        "color": None,
                    })
                    # 调试：记录结果长度与可达性
                    try:
                        st.write(f"[DEBUG] {prof} reachable={getattr(route_result, 'reachable', None)}, "
                                 f"len(path_ij)={len(getattr(route_result, 'path_ij', []))}, "
                                 f"len(path_lonlat)={len(getattr(route_result, 'path_lonlat', []))}")
                    except Exception:
                        pass
                else:
                    try:
                        summary = planner_service.summarize_route(route_result)
                    except Exception:
                        summary = None
                    try:
                        eco_summary = planner_service.evaluate_route_eco(route_result, env_ctx_p)
                    except Exception:
                        eco_summary = None
                    routes_info.append({
                        "label": prof,
                        "route_result": route_result,
                        "summary": summary,
                        "eco_summary": eco_summary,
                        "error": None,
                        "color": None,
                    })
                    # 调试：记录结果长度与可达性（可达时）
                    try:
                        st.write(f"[DEBUG] {prof} reachable={getattr(route_result, 'reachable', None)}, "
                                 f"len(path_ij)={len(getattr(route_result, 'path_ij', []))}, "
                                 f"len(path_lonlat)={len(getattr(route_result, 'path_lonlat', []))}")
                    except Exception:
                        pass

            # 高级 Evidential-Robust 方案（仅叠加，不影响 baseline 逻辑）
            robust_result = None
            if str(planner_mode) == "evidential_robust":
                try:
                    robust_cfg = RobustPlannerConfig(
                        risk_agg_mode="cvar",
                        risk_agg_alpha=0.9,
                        fusion_mode="evidential",
                        allow_diagonal=True,
                        heuristic="euclidean",
                    )
                    robust_result = run_planning_pipeline_evidential_robust(
                        ym=str(ym),
                        start_lat=float(start_lat),
                        start_lon=float(start_lon),
                        end_lat=float(end_lat),
                        end_lon=float(end_lon),
                        profile_name="balanced",
                        robust_cfg=robust_cfg,
                    )
                except Exception as e:
                    st.warning(f"高级规划（Evidential-Robust）失败: {e}")
                    robust_result = None

            # 地图展示（pydeck）
            try:
                import pydeck as pdk
            except Exception as e:
                st.error(f"pydeck 导入失败，无法绘制地图: {e}")
                pdk = None

            if pdk is not None:
                st.subheader("规划结果地图（Pydeck）")
                path_data = []
                for info in routes_info:
                    r = info.get("route_result")
                    if r is None or not getattr(r, "reachable", False):
                        continue
                    coords = getattr(r, "path_lonlat", None) or []
                    if not coords:
                        continue
                    path_data.append({
                        "name": info.get("label", "route"),
                        "path": [[float(lon), float(lat)] for (lat, lon) in coords],
                    })

                # 叠加 Evidential-Robust 路线
                if pdk is not None and robust_result is not None:
                    try:
                        rr = robust_result.get("route") if isinstance(robust_result, dict) else None
                        if rr is not None and getattr(rr, "reachable", False):
                            coords = getattr(rr, "path_lonlat", None) or []
                            if coords:
                                path_data.append({
                                    "name": "Evidential-Robust",
                                    "path": [[float(lon), float(lat)] for (lat, lon) in coords],
                                })
                    except Exception:
                        pass

                # 调试：打印路径基本信息
                st.subheader("调试：路径数据 (DEBUG)")
                try:
                    st.write({
                        "num_routes": len(path_data),
                        "routes": [
                            {
                                "name": item.get("name"),
                                "num_points": len(item.get("path", [])),
                                "first": item.get("path", [None])[0],
                                "last": item.get("path", [None])[-1] if item.get("path") else None,
                            }
                            for item in path_data
                        ],
                    })
                except Exception:
                    pass

                if path_data:
                    all_points = [pt for item in path_data for pt in item["path"]]
                    avg_lon = float(np.mean([p[0] for p in all_points])) if all_points else float(params["start_lon"])
                    avg_lat = float(np.mean([p[1] for p in all_points])) if all_points else float(params["start_lat"])

                    # 视角调试信息
                    st.write({"DEBUG_view_state": {"lon": avg_lon, "lat": avg_lat}})

                    layer = pdk.Layer(
                        "PathLayer",
                        data=path_data,
                        get_path="path",
                        get_width=3,
                        get_color=[255, 100, 100],  # 更亮的颜色，方便观察
                        pickable=True,
                    )

                    view_state = pdk.ViewState(
                        longitude=avg_lon,
                        latitude=avg_lat,
                        zoom=3,
                        pitch=30,
                    )

                    st.pydeck_chart(pdk.Deck(
                        layers=[layer],
                        initial_view_state=view_state,
                        tooltip={"text": "{name}"},
                    ))
                else:
                    st.warning("没有可显示的路线。")

            # 摘要表
            rows = []
            for info in routes_info:
                label = info.get("label")
                r = info.get("route_result")
                s = info.get("summary")
                if r is None:
                    rows.append({
                        "方案": label,
                        "船型": str(vessel_key),
                        "可达": False,
                        "距离_km": None,
                        "总成本": None,
                        "预计燃油_t": None,
                        "预计CO2_t": None,
                    })
                    continue
                if not getattr(r, "reachable", False):
                    rows.append({
                        "方案": label,
                        "船型": str(vessel_key),
                        "可达": False,
                        "距离_km": None,
                        "总成本": None,
                        "预计燃油_t": None,
                        "预计CO2_t": None,
                    })
                    continue
                fuel = None
                co2 = None
                eco_summary = info.get("eco_summary")
                if eco_summary is not None:
                    try:
                        fuel = getattr(eco_summary, "fuel_total_t", None) or eco_summary.get("fuel_total_t")
                    except Exception:
                        try:
                            fuel = eco_summary.get("fuel_total_ton")
                        except Exception:
                            fuel = None
                    try:
                        co2 = getattr(eco_summary, "co2_total_t", None) or eco_summary.get("co2_total_t")
                    except Exception:
                        try:
                            co2 = eco_summary.get("co2_total_ton")
                        except Exception:
                            co2 = None
                rows.append({
                    "方案": label,
                    "船型": str(vessel_key),
                    "可达": True,
                    "距离_km": (s or {}).get("distance_km") if isinstance(s, dict) else None,
                    "总成本": (s or {}).get("cost_sum") if isinstance(s, dict) else None,
                    "预计燃油_t": fuel,
                    "预计CO2_t": co2,
                })

            if rows:
                df = pd.DataFrame(rows)
                st.subheader("路线摘要")
                st.dataframe(df, width="stretch")

            # 风险贡献（仅展示 balanced 方案）
            try:
                balanced_info = next((info for info in routes_info if info.get("label") == "balanced" and info.get("route_result") is not None and getattr(info["route_result"], "reachable", False)), None)
            except Exception:
                balanced_info = None
            if balanced_info is not None:
                try:
                    breakdown = planner_service.analyze_route_cost(env_ctx, balanced_info["route_result"])  # 使用最初 balanced 环境
                except Exception as e:
                    breakdown = None
                    st.warning(f"风险分解计算失败: {e}")
                if breakdown:
                    risk_components = (breakdown or {}).get("risk_components") or {}
                    if risk_components:
                        rows_risk = []
                        total_risk = sum([v for v in risk_components.values() if v is not None])
                        for k, v in risk_components.items():
                            if v is None or float(v) == 0.0:
                                continue
                            frac = (float(v) / float(total_risk)) if total_risk and total_risk > 0 else 0.0
                            rows_risk.append({
                                "风险维度": k,
                                "累积成本贡献": float(v),
                                "占比": float(frac),
                            })
                        if rows_risk:
                            df_risk = pd.DataFrame(rows_risk).sort_values("累积成本贡献", ascending=False)
                            st.subheader("风险贡献（balanced 方案）")
                            st.dataframe(df_risk, use_container_width=True)
                            try:
                                st.bar_chart(df_risk.set_index("风险维度")["累积成本贡献"])
                            except Exception:
                                pass
                        else:
                            st.info("当前成本构建下，沿线路暂无显著的风险贡献。")
                    else:
                        st.info("未获得有效的风险分解结果。")

                # 调试：原始 risk_components 结构
                try:
                    with st.expander("调试：原始 risk_components 结构 (DEBUG)", expanded=False):
                        try:
                            cost_components_attr = getattr(getattr(env_ctx, "cost_da", None), "attrs", {}).get("cost_components", None)
                        except Exception:
                            cost_components_attr = None
                        st.write({
                            "risk_components": (breakdown or {}).get("risk_components"),
                            "risk_components_normalized": (breakdown or {}).get("risk_components_normalized"),
                            "cost_components_attr": cost_components_attr,
                        })
                except Exception:
                    pass

            # 若全部不可达，给出错误提示
            try:
                if not any((info.get("route_result") and getattr(info.get("route_result"), "reachable", False)) for info in routes_info):
                    st.error("当前起止点与环境配置下，三种方案均不可达，请调整起止点或权重后重试。")
            except Exception:
                pass

            # 高级路线详情（可选）
            if robust_result is not None and isinstance(robust_result, dict):
                try:
                    rr = robust_result.get("route")
                    if rr is not None and getattr(rr, "reachable", False):
                        with st.expander("高级路线（Evidential-Robust）详情", expanded=False):
                            rsum = robust_result.get("summary") or {}
                            rmeta = robust_result.get("robust_meta") or {}
                            # ECO 使用选定船型（用 balanced env_ctx 评估）
                            try:
                                robust_eco = planner_service.evaluate_route_eco(rr, env_ctx)
                            except Exception:
                                robust_eco = None
                            st.write({
                                "distance_km": rsum.get("distance_km"),
                                "cost_sum": rsum.get("cost_sum"),
                                "eco_fuel_t": getattr(robust_eco, "fuel_total_t", None) if robust_eco else None,
                                "eco_co2_t": getattr(robust_eco, "co2_total_t", None) if robust_eco else None,
                            })
                            st.write({k: rmeta.get(k) for k in [
                                "fusion_mode_effective", "cost_risk_agg_mode_effective", "risk_agg_mode", "risk_agg_alpha", "fusion_mode"
                            ] if k in rmeta})
                except Exception:
                    pass
        else:
            st.info("请在左侧设置参数并点击 ‘规划路线’ 开始计算。")

