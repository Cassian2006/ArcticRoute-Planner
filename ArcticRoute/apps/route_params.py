from __future__ import annotations

import streamlit as st


def pick_risk_source(default: str = "ice") -> str:
    """在 Route 页提供“风险来源”选择器。
    返回值："ice" 或 "fused"；自动存入 session_state["risk_source"].
    Phase F: 暴露交互风险叠加权重滑条（默认 0，非破坏性）。
    Phase M: 增加绿色航行（CO₂）选项、权重与船型选择。
    """
    st.caption("选择路由使用的风险来源（缺失将自动回退到 ICE COST 并提示）")
    options = ["ice", "fused"]
    idx = 0 if default == "ice" else 1
    choice = st.selectbox("风险来源", options=options, index=idx, help="ice=ICE COST；fused=融合风险 Risk")
    st.session_state["risk_source"] = choice

    # Phase F: 交互风险叠加权重（影响融合与成本）
    st.session_state.setdefault("interact_weight", 0.0)
    iw = st.slider("交互风险权重 (interact_weight)", min_value=0.0, max_value=1.0, value=float(st.session_state.get("interact_weight", 0.0)), step=0.05, help="0 关闭叠加；>0 时纳入 R_interact")
    st.session_state["interact_weight"] = float(iw)

    # Phase M: 绿色航行（CO₂）
    st.session_state.setdefault("eco_enabled", False)
    eco_enabled = st.checkbox("Green Sailing (CO₂)", value=bool(st.session_state.get("eco_enabled", False)))
    st.session_state["eco_enabled"] = bool(eco_enabled)
    st.session_state.setdefault("w_e", 0.0)
    w_e = st.slider("w_e (ECO 权重)", min_value=0.0, max_value=1.0, value=float(st.session_state.get("w_e", 0.0)), step=0.05, help="权重越大越偏向低 CO₂ 走廊")
    st.session_state["w_e"] = float(w_e)
    # 船型（从配置读取失败时提供静态选项）
    try:
        import yaml  # type: ignore
        from pathlib import Path as _P
        cfg_path = _P(__file__).resolve().parents[2] / "ArcticRoute" / "config" / "eco.yaml"
        vopts = ["cargo_iceclass", "cargo_standard", "icebreaker", "fishing_small"]
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            classes = list(((cfg.get("eco") or {}).get("vessel_classes") or {}).keys())
            if classes:
                vopts = classes
    except Exception:
        vopts = ["cargo_iceclass", "cargo_standard", "icebreaker", "fishing_small"]
    vclass = st.selectbox("Vessel Class", options=vopts, index=max(0, vopts.index(st.session_state.get("vessel_class", vopts[0])) if st.session_state.get("vessel_class") in vopts else 0))
    st.session_state["vessel_class"] = vclass
    return choice


def pick_risk_aggregation(default_mode: str = "mean", default_alpha: float = 0.95) -> tuple[str, float]:
    """Phase I: 风险聚合模式与 α 控件。
    - mode: mean | q | cvar
    - alpha: 分位/CVaR 的 α（0.5–0.99 推荐）
    返回 (mode, alpha)，并写入 session_state["risk_agg"], ["alpha"].
    """
    st.caption("聚合模式控制仅在使用 fused 风险且存在方差 (RiskVar) 时生效；否则自动回退到均值。")
    modes = ["mean", "q", "cvar"]
    try:
        mi = modes.index(default_mode)
    except Exception:
        mi = 0
    mode = st.selectbox("风险聚合模式", options=modes, index=mi, help="mean=均值；q=分位；cvar=条件在险 (ES)")
    alpha = st.slider("α (用于分位/CVaR)", min_value=0.50, max_value=0.99, value=float(default_alpha), step=0.01)
    st.session_state["risk_agg"] = mode
    st.session_state["alpha"] = float(alpha)
    return mode, float(alpha)
