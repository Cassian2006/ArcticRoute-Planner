from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import streamlit as st

@dataclass
class UIDoctorResult:
    ok: bool
    notes: list[str]

def run_ui_doctor() -> UIDoctorResult:
    notes: list[str] = []
    ok = True

    # 静态资产 manifest（如果你们用环境变量或固定路径）
    manifest = Path("arcticroute/ui/assets/static_assets_manifest.json")
    if not manifest.exists():
        notes.append("⚠️ 未发现静态资产清单（static_assets_manifest.json），港口/走廊/水深可能无法自动发现。")
        ok = False
    else:
        try:
            json.loads(manifest.read_text(encoding="utf-8", errors="ignore") or "{}")
        except Exception:
            notes.append("⚠️ 静态资产清单存在但无法解析为 JSON。")
            ok = False

    # CMEMS cache/newenv 常见目录
    cache_dir = Path("data/cmems_cache")
    newenv_dir = Path("data_processed/newenv")
    if not cache_dir.exists():
        notes.append("⚠️ CMEMS 缓存目录 data/cmems_cache 不存在（近实时数据会回退）。")
    if not newenv_dir.exists():
        notes.append("⚠️ newenv 目录不存在（CMEMS 同步可能不可用）。")

    # AIS density 常见目录（根据你们项目调整）
    ais_dir = Path("data_real/ais")
    if not ais_dir.exists():
        notes.append("⚠️ AIS 数据目录 data_real/ais 不存在（AIS 密度功能可能不可用）。")

    return UIDoctorResult(ok=ok, notes=notes)

def render_ui_doctor_banner() -> None:
    r = run_ui_doctor()
    if r.notes:
        with st.expander("🩺 启动体检", expanded=not r.ok):
            for n in r.notes:
                st.write(n)
            if r.ok:
                st.success("体检通过（存在一些可选项缺失提示）。")
            else:
                st.warning("体检未完全通过：部分功能可能回退/不可用。")


