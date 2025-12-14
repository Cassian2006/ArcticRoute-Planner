from __future__ import annotations
import json
from pathlib import Path
import streamlit as st
from ArcticRoute.apps.registry import UIRegistry  # type: ignore
from ArcticRoute.apps.theme import inject_theme, read_theme_flag  # type: ignore
from ArcticRoute.apps.components.page_header import render_page_header  # type: ignore
from ArcticRoute.apps.components.error_card import show_error_card  # type: ignore


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def render(ctx: dict | None = None) -> None:
    inject_theme(read_theme_flag())
    if not UIRegistry().is_page_enabled("explain", False):
        st.info("Explain 页面已在配置中禁用（ui.pages.explain=false）")
        return
    render_page_header("🧭", "Explain · 路线解释", "生成 route_attr_* 并可视化分段贡献")
    with st.expander("使用说明 / Usage", expanded=False):
        st.markdown("""
        - 步骤：选择路线与月份 → 生成分段贡献 → 查看贡献条与积分误差。
        - 产物：ArcticRoute/reports/d_stage/phaseH/route_attr_{ym}_*.json
        - CLI 示例：
          - python -m ArcticRoute.api.cli route.explain --route ArcticRoute/data_processed/routes/route_202412_nsr_wbound_smoke_balanced.geojson --ym 202412
        """)
    st.markdown('<div class="section"><h2>构建解释</h2><p>选择 YYYYMM 与路线，生成解释产物。</p></div>', unsafe_allow_html=True)
    ym = st.text_input("月份 YYYYMM", value=str(st.session_state.get("ym", "202412")))
    route_path = st.text_input("路线 GeoJSON", value=(str(_repo_root()/"ArcticRoute"/"data_processed"/"routes"/f"route_{ym}_nsr_wbound_smoke_balanced.geojson")))
    if not route_path or not Path(route_path).exists():
        show_error_card("NO_ROUTE_SELECTED", "未选择或未找到路线 GeoJSON 文件", f"请在上方输入有效的路线文件路径，或先在 Compare 页面导出代表路线；CLI：python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound_smoke --ym {ym}", {"route_path": route_path})
        return
    out_dir = _repo_root()/"ArcticRoute"/"reports"/"d_stage"/"phaseH"
    c1, c2 = st.columns(2)
    if c1.button("生成解释 (route.explain)", width='stretch'):
        try:
            import subprocess, sys
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "route.explain", "--route", route_path, "--ym", ym, "--out", str(out_dir)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode == 0:
                st.success("已生成解释产物")
                # 记录 UI 动作元信息
                try:
                    from ArcticRoute.apps import state as ui_state  # type: ignore
                    ui_state.write_action_meta2(
                        action="explain.build",
                        inputs={"ym": ym, "route": route_path, "out": str(out_dir)},
                        outputs={}
                    )
                except Exception:
                    pass
                st.code(res.stdout[-1200:], language="json")
            else:
                st.error(f"route.explain 失败：{res.returncode}")
                st.code(res.stderr[-1200:])
        except Exception as e:
            st.error(str(e))
    attr_jsons = sorted(out_dir.glob(f"route_attr_{ym}_*.json"))
    if not attr_jsons:
        show_error_card("EXPLAIN_DATA_MISSING", "未找到解释产物 route_attr_*.json", f"请先点击上方按钮生成，或在 CLI 运行：python -m ArcticRoute.api.cli route.explain --route {route_path} --ym {ym}", {"expected_dir": str(out_dir)})
        return
    sel = st.selectbox("选择解释结果", options=[p.name for p in attr_jsons], index=0)
    p = out_dir/sel
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        # 最小可视化：显示每段贡献和总和校验
        segs = data.get("segments") or []
        target = float(data.get("objective_integral", float("nan")))
        total = 0.0
        rows = []
        for s in segs:
            val = float(s.get("total", 0.0))
            total += val
            rows.append({
                "idx": s.get("index"),
                "Risk": s.get("risk", 0.0),
                "Dist": s.get("dist", 0.0),
                "Prior": s.get("prior", 0.0),
                "Interact": s.get("interact", 0.0),
                "Congest": s.get("congest", 0.0),
                "Eco": s.get("eco", 0.0),
                "Total": val,
            })
        st.table(rows[:50])
        if target == target:  # not NaN
            err = abs(total - target) / max(1e-6, abs(target))
            st.metric("积分误差(≤2% 通过)", f"{err*100:.2f}%")
            if err <= 0.02:
                st.success("通过")
            else:
                st.warning("超过 2%：请检查风险层/路线匹配与权重")
        st.download_button("下载解释 JSON", data=p.read_bytes(), file_name=p.name, mime="application/json")
    except Exception as e:
        st.error(str(e))

