from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import json
import numpy as np  # type: ignore

try:
    import streamlit as st  # type: ignore
    import matplotlib.pyplot as plt  # type: ignore
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    st = None  # type: ignore
    plt = None  # type: ignore
    xr = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
PRIOR_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "prior"
CENTER_DIR = REPO_ROOT / "reports" / "phaseE" / "center"


def _find_prior_nc(ym: str) -> Optional[Path]:
    # 优先选用 prior_corridor_selected_<YM>.nc；回退 transformer
    c1 = PRIOR_DIR / f"prior_corridor_selected_{ym}.nc"
    if c1.exists():
        return c1
    c2 = PRIOR_DIR / f"prior_transformer_{ym}.nc"
    if c2.exists():
        return c2
    return None


def _find_centerlines(ym: str) -> Optional[Path]:
    p = CENTER_DIR / f"prior_centerlines_{ym}.geojson"
    return p if p.exists() else None


def render_prior_layer(ym: str) -> None:
    """在当前页面绘制 P_prior 热力图并叠加中心线（若存在）。仅在被调用时渲染；文件缺失则友好提示。"""
    if st is None or xr is None or plt is None:
        return
    nc = _find_prior_nc(ym)
    gj = _find_centerlines(ym)
    st.subheader("Prior · P_prior 与中心线")
    if not nc and not gj:
        st.info("未找到 prior 栅格或中心线产物。请先运行 prior.export / prior.centerline。")
        return
    try:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if nc:
            with xr.open_dataset(nc) as ds:
                if "P_prior" in ds:
                    da = ds["P_prior"].load()
                    da.plot.imshow(ax=ax, cmap="viridis", vmin=0.0, vmax=1.0, add_colorbar=True, cbar_kwargs={"label": "P_prior"})
                    ax.set_title(f"P_prior · {ym}")
        # 叠加中心线（若存在），以像素索引空间近似：使用 lat/lon 时可能需坐标映射；此处简化为不坐标变换的可视叠加
        if gj:
            try:
                obj = json.loads(Path(gj).read_text(encoding="utf-8"))
                # 仅做提示与下载按钮，避免坐标系不一致导致的误导性叠加
                st.caption(f"已发现中心线：{gj.name}")
                with open(gj, "rb") as fh:
                    st.download_button("下载中心线 GeoJSON", data=fh.read(), file_name=gj.name, mime="application/geo+json")
            except Exception:
                pass
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
    except Exception as e:  # pragma: no cover - 容错绘制
        st.warning(f"Prior 图层渲染失败：{e}")


def route_params_prior(ym: str) -> None:
    """在 Route 参数区域注入先验相关 UI：
    - w_p 滑条：0..1，默认 0（不启用）；>0 时启用 PriorPenalty
    - prior 文件检测提示：未指定时自动发现 prior_corridor_selected_<YM>.nc / prior_transformer_<YM>.nc
    不改变核心逻辑，仅写入 session_state：prior_w_p, prior_path
    """
    if st is None:
        return
    st.markdown("**PriorPenalty 设置（可选）**")
    nc = _find_prior_nc(ym)
    st.session_state.setdefault("prior_w_p", 0.0)
    w_p = float(st.slider("w_p (PriorPenalty 权重)", 0.0, 1.0, float(st.session_state.get("prior_w_p", 0.0)), 0.05, help=">0 则在路由代价中叠加 w_p * PriorPenalty"))
    st.session_state["prior_w_p"] = w_p
    if nc:
        st.session_state["prior_path"] = str(nc)
        if w_p <= 0.0:
            st.caption(f"发现可用 prior：{nc.name}（当前 w_p=0 未启用）")
        else:
            st.caption(f"使用 prior：{nc.name} · w_p={w_p}")
    else:
        st.session_state.pop("prior_path", None)
        st.caption("未发现 prior_corridor_selected/transformer nc；将不启用先验。")


__all__ = ["render_prior_layer", "route_params_prior"]












