from __future__ import annotations

import pandas as pd
import streamlit as st

from arcticroute.core.pareto import pareto_front, solutions_to_dataframe
from arcticroute.core.pareto import ParetoSolution

def render_pareto_panel(solutions: list[ParetoSolution]):
    st.subheader("Pareto 多目标前沿（实验）")

    if not solutions:
        st.warning("没有可用候选解（solutions 为空）。")
        return

    default_fields = ["distance_km", "total_cost", "edl_uncertainty"]
    fields = st.multiselect(
        "目标维度（最小化）",
        options=["distance_km", "total_cost", "edl_risk", "edl_uncertainty"],
        default=default_fields,
    )
    if not fields:
        st.info("至少选择一个目标维度。")
        return

    front = pareto_front(solutions, fields=fields)
    df_all = solutions_to_dataframe(solutions)
    df_front = solutions_to_dataframe(front)

    st.caption(f"solutions={len(solutions)} → front={len(front)} ; fields={fields}")
    st.dataframe(df_front, use_container_width=True)

    # scatter (use streamlit native chart)
    cols = list(df_front.columns)
    x = st.selectbox("X", options=[c for c in cols if c in df_front.columns], index=cols.index("distance_km") if "distance_km" in cols else 0)
    y = st.selectbox("Y", options=[c for c in cols if c in df_front.columns], index=cols.index("total_cost") if "total_cost" in cols else 0)
    st.scatter_chart(df_front[[x, y]])

    # selection
    key = st.selectbox("选择一个前沿解", options=df_front["key"].tolist())
    chosen = next((s for s in solutions if s.key == key), None)
    if chosen is None:
        st.warning("未找到选中的解。")
        return

    st.markdown("**Objectives**")
    st.json(chosen.objectives)
    st.markdown("**Meta (weights/toggles)**")
    st.json(chosen.meta)

    # route points preview
    if chosen.route:
        pts = pd.DataFrame(chosen.route, columns=["lat", "lon"])
        st.markdown("**Route preview (points)**")
        st.dataframe(pts.head(50), use_container_width=True)
        try:
            st.map(pts, latitude="lat", longitude="lon")
        except Exception:
            pass

    st.download_button("下载 pareto_front.csv", data=df_front.to_csv(index=False), file_name="pareto_front.csv", mime="text/csv")
    st.download_button("下载 pareto_solutions.csv", data=df_all.to_csv(index=False), file_name="pareto_solutions.csv", mime="text/csv")


