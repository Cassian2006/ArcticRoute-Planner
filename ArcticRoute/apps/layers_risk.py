from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

try:
    import xarray as xr
except Exception:
    xr = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def _plot_da(da: "xr.DataArray", title: str):
    if xr is None or plt is None:
        return
    if "time" in da.dims:
        da = da.isel(time=0)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    da.plot.imshow(ax=ax, cmap="inferno", vmin=0.0, vmax=1.0, add_colorbar=True, cbar_kwargs={"label": da.name or "val"})
    ax.set_title(title)
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def render_fused_risk_layer(ym: str, risk_dir: Path):
    """在 Layers 页探测 risk_fused_<ym>.nc，显示：
    - Risk(fused) 主图
    - Uncertainty (RiskVar) 可选叠加
    - Phase F: R_interact 叠加（非破坏性）
    """
    if xr is None or plt is None:
        return

    p_rfuse = risk_dir / f"risk_fused_{ym}.nc"
    p_interact = risk_dir / f"R_interact_{ym}.nc"

    if p_rfuse.exists():
        st.session_state.setdefault("layer_rfuse_show", False)
        st.checkbox("Risk(fused)", key="layer_rfuse_show")
        if st.session_state.get("layer_rfuse_show"):
            try:
                with xr.open_dataset(p_rfuse) as ds:
                    _plot_da(ds["Risk"], "Risk (fused)")
                    # Phase I: 不确定性叠加（若存在 RiskVar）
                    if "RiskVar" in ds or "risk_var" in ds or "variance" in ds:
                        st.session_state.setdefault("layer_riskvar_show", False)
                        st.checkbox("Uncertainty (RiskVar)", key="layer_riskvar_show")
                        if st.session_state.get("layer_riskvar_show"):
                            vname = "RiskVar" if "RiskVar" in ds else ("risk_var" if "risk_var" in ds else "variance")
                            _plot_da(ds[vname].rename("RiskVar"), "Uncertainty (RiskVar)")
            except Exception as e:
                st.warning(f"无法渲染 Risk(fused)/RiskVar: {e}")

    # Phase F: 交互风险图层（默认隐藏）
    if p_interact.exists():
        st.session_state.setdefault("layer_rinter_show", False)
        st.checkbox("Interaction risk (R_interact)", key="layer_rinter_show")
        if st.session_state.get("layer_rinter_show"):
            try:
                with xr.open_dataset(p_interact) as ds:
                    var = "risk" if "risk" in ds else list(ds.data_vars)[0]
                    _plot_da(ds[var].rename("risk"), "Interaction risk (R_interact)")
            except Exception as e:
                st.warning(f"无法渲染 R_interact: {e}")
