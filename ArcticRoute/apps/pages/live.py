from __future__ import annotations
import json
import time
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps import state as ui_state  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore
from ArcticRoute.apps.components.error_card import show_error_card  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("live", False):
        st.info("Live 页面已在配置中禁用（ui.pages.live=false）")
        return
    render_page_header("🛰️", "Live · 在线重规划", "Watcher + 一键 Replan")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 步骤：① 选择月份与场景 → ② 选择风险聚合与权重（本页最小化隐藏）→ ③ 点击 Replan now → ④ 查看新版路线与差异。
        - 前提：需要已有 risk_fused_*.nc 与先验层（最小可仅有 risk_fused_YYYYMM.nc）。
        - 产物：ArcticRoute/data_processed/routes/live/route_<scenario>_<ts>_v01.geojson + 对应 .ui_action.meta.json。
        - CLI 示例：
          - python -m ArcticRoute.api.cli route.replan --scenario nsr_wbound_smoke --live
          - python -m ArcticRoute.api.cli risk.nowcast --ym 202412
        """)
    # 前置数据检查：risk_fused 是否存在
    risk_dir = _repo_root()/"ArcticRoute"/"data_processed"/"risk"
    ym_guess = str(st.session_state.get("ym", "202412"))
    fused = risk_dir/f"risk_fused_{ym_guess}.nc"
    if not fused.exists():
        show_error_card("NO_RISK_DATA", "未找到融合风险层 risk_fused_*.nc", f"请先构建或融合风险层；示例：python -m ArcticRoute.api.cli risk.fuse --ym {ym_guess}", {"expected": str(fused)})
    scen_yaml = _repo_root()/"configs"/"scenarios.yaml"
    try:
        import yaml  # type: ignore
        _sc = yaml.safe_load(scen_yaml.read_text(encoding="utf-8")) if scen_yaml.exists() else {}
        _sc_ids = [s.get("id") for s in (_sc.get("scenarios") or []) if isinstance(s, dict)] or ["nsr_wbound_smoke"]
    except Exception:
        _sc_ids = ["nsr_wbound_smoke"]
    c1, c2, c3, c4 = st.columns(4)
    scenario_id = c1.selectbox("Scenario", options=_sc_ids, index=0)
    interval = int(c2.number_input("Watcher间隔(s)", value=300, step=60))
    do_live = c3.toggle("Live 模式", value=True)
    if c4.button("Replan now", width='stretch'):
        try:
            import subprocess as sp, sys
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "route.replan", "--scenario", scenario_id]
            if do_live:
                cmd.append("--live")
            res = sp.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                st.success("已触发重规划")
                # 记录 UI 动作元信息
                ui_state.write_action_meta2(
                    action="live.replan",
                    inputs={"scenario": scenario_id, "live": bool(do_live)},
                    outputs={}
                )
                st.code(res.stdout[-1200:], language="json")
            else:
                st.error(f"replan 失败：{res.returncode}")
                st.code(res.stderr[-1200:])
        except Exception as e:
            st.error(str(e))

    risk_dir = _repo_root()/"ArcticRoute"/"data_processed"/"risk"
    live_files = sorted(risk_dir.glob("risk_fused_live_*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
    if live_files:
        st.info(f"最新 live: {live_files[0].name} · mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(live_files[0].stat().st_mtime))}")
    else:
        st.caption("尚无 live 风险面，可运行：python -m ArcticRoute.api.cli risk.nowcast --ym YYYYMM")

    # 简易 diff：最新与上一个 live 路线
    routes_dir = _repo_root()/"ArcticRoute"/"data_processed"/"routes"/"live"
    cands = sorted(routes_dir.glob(f"route_{scenario_id}_*.geojson"), key=lambda p: p.stat().st_mtime, reverse=True)
    if len(cands) >= 2:
        try:
            a = json.loads(cands[0].read_text(encoding="utf-8"))
            b = json.loads(cands[1].read_text(encoding="utf-8"))
            def _coords(gj):
                return gj.get("features", [{}])[0].get("geometry", {}).get("coordinates") or []
            ca, cb = _coords(a), _coords(b)
            from ArcticRoute.core.route.metrics import compute_distance_km as _dkm  # REUSE
            pa = [(float(x[0]), float(x[1])) for x in ca]
            pb = [(float(x[0]), float(x[1])) for x in cb]
            da = _dkm(pa); db = _dkm(pb)
            st.metric("距离(km)", f"{da:.1f}", delta=(da-db))
        except Exception as e:
            st.caption(f"diff 计算失败：{e}")

