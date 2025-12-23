from __future__ import annotations

from datetime import datetime
from pathlib import Path
import time

import pandas as pd
import streamlit as st

from arcticroute.ui.cover_page import render_cover
from arcticroute.ui.planner_minimal import render_planner
from arcticroute.ui.ui_doctor import render_ui_doctor_banner
from arcticroute.ui.data_discovery import (
    build_search_dirs,
    discover_ais_density,
    discover_newenv_cmems,
    discover_static_assets,
)


def render_data_page() -> None:
    st.header("Data Availability")
    st.caption("Multi-source discovery for AIS/CMEMS/static assets.")

    data_root = st.session_state.get("data_root_override") or None
    manifest_path = st.session_state.get("static_assets_manifest") or None

    search_dirs = build_search_dirs(data_root=data_root, manifest_path=manifest_path)
    with st.expander(f"Search dirs ({len(search_dirs)})", expanded=False):
        if search_dirs:
            st.code("\n".join(str(d) for d in search_dirs))
        else:
            st.info("No search dirs found. Set ARCTICROUTE_DATA_ROOT or Data root.")

    st.subheader("AIS density")
    grid_sig = st.session_state.get("grid_signature")
    ais_df, ais_meta = discover_ais_density(search_dirs, grid_sig)
    if ais_meta.get("count", 0) > 0:
        st.caption(f"Found {ais_meta.get('count')} files | Latest: {ais_meta.get('latest_path', '')}")
        st.caption(f"Grid signature: {ais_meta.get('grid_signature', '')}")
        cols = [c for c in ["path", "grid_signature", "shape", "mtime", "match"] if c in ais_df.columns]
        st.dataframe(ais_df[cols] if cols else ais_df, use_container_width=True)
        if any("mismatch" in str(m) for m in ais_df.get("match", [])):
            st.warning(f"Grid mismatch detected. Current grid signature: {ais_meta.get('grid_signature', '')}")
    else:
        st.warning("No AIS density files found.")
        st.info("Put files under data_real/ais/density or data_real/ais/derived, or set ARCTICROUTE_DATA_ROOT.")

    st.subheader("CMEMS newenv (SIC/SIT/SWH/Drift)")
    newenv_dir = st.session_state.get("newenv_dir")
    if not newenv_dir and data_root:
        newenv_dir = str(Path(data_root) / "newenv")
    newenv_meta = discover_newenv_cmems(newenv_dir)
    if newenv_meta.get("files"):
        rows = []
        for key, meta in newenv_meta["files"].items():
            rows.append(
                {
                    "key": key,
                    "exists": meta.get("exists"),
                    "path": meta.get("path", ""),
                    "mtime": meta.get("mtime", ""),
                    "vars": ", ".join(meta.get("vars", []) or []),
                    "shape": meta.get("shape", ""),
                    "shape_match": meta.get("shape_match"),
                    "reason": meta.get("reason", ""),
                }
            )
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        if any(not r.get("exists") for r in rows):
            st.warning("Some CMEMS files are missing.")
            st.info("Run cmems_refresh_and_export or cmems_newenv_sync, and set ARCTICROUTE_DATA_ROOT.")
    else:
        st.warning("No CMEMS newenv metadata available.")
        if newenv_meta.get("reason"):
            st.info(f"Reason: {newenv_meta.get('reason')}")
        st.info("Expected files: ice_copernicus_sic.nc / ice_thickness.nc / wave_swh.nc / ice_drift.nc")

    st.subheader("Static assets manifest")
    if st.button("Rescan static assets", key="rescan_static_assets"):
        with st.spinner("Scanning static assets..."):
            st.session_state["static_assets_last_scan"] = time.time()
            assets_meta = discover_static_assets(manifest_path)
            st.session_state["static_assets_last_count"] = assets_meta.get("entries_count", 0)
    else:
        assets_meta = discover_static_assets(manifest_path)

    last_scan = st.session_state.get("static_assets_last_scan")
    last_count = st.session_state.get("static_assets_last_count")
    if last_scan:
        st.caption(f"Last scan: {datetime.fromtimestamp(last_scan).isoformat(timespec='seconds')}")
    if last_count is not None:
        st.caption(f"Entries: {last_count}")

    if assets_meta.get("exists"):
        st.info(f"Manifest entries: {assets_meta.get('entries_count', 0)}")
        st.caption(f"Path: {assets_meta.get('path', '')}")
        st.caption(f"Mtime: {assets_meta.get('mtime', '')}")
    else:
        st.warning("Static assets manifest not found.")
        reason = assets_meta.get("reason", "")
        if reason:
            st.info(f"Reason: {reason}")
        st.info("Set the manifest path in the sidebar or ARCTICROUTE_DATA_ROOT.")


def render_diag_page() -> None:
    st.header("Diagnostics")
    st.info("Diagnostics panels will be wired here in a later phase.")


PAGES = {
    "cover": ("Cover", render_cover),
    "planner": ("Planner", render_planner),
    "data": ("Data", render_data_page),
    "diag": ("Diagnostics", render_diag_page),
}


def _get_page() -> str:
    qp = None
    try:
        qp = st.query_params.get("page", None)
        if isinstance(qp, list):
            qp = qp[0] if qp else None
    except Exception:
        qp = None

    if qp in PAGES:
        st.session_state["nav_page"] = qp
        return qp

    if st.session_state.get("nav_page") in PAGES:
        return st.session_state["nav_page"]

    st.session_state["nav_page"] = "cover"
    return "cover"


def _set_page(page: str) -> None:
    st.session_state["nav_page"] = page
    try:
        st.query_params["page"] = page
    except Exception:
        pass


def main() -> None:
    if not st.session_state.get("_ar_page_config_set"):
        st.set_page_config(
            page_title="ArcticRoute UI",
            layout="wide",
            initial_sidebar_state="expanded",
        )
        st.session_state["_ar_page_config_set"] = True

    page = _get_page()

    with st.sidebar:
        st.title("ArcticRoute")
        render_ui_doctor_banner()
        labels = [PAGES[k][0] for k in PAGES]
        keys = list(PAGES.keys())
        idx = keys.index(page) if page in keys else 0
        choice = st.radio("Navigation", options=keys, index=idx, format_func=lambda k: PAGES[k][0])
        if choice != page:
            _set_page(choice)
            st.rerun()

    _, fn = PAGES[page]
    fn()


if __name__ == "__main__":
    main()
