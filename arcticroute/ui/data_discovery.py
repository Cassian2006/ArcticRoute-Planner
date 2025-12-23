from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import shutil

import os
import time

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScanResult:
    count: int
    examples: List[str]
    roots_used: List[str]


def _candidate_roots() -> List[Path]:
    """返回预期扫描的根目录（包含可能不存在的路径）。"""
    roots: List[Path] = []
    env_root = os.environ.get("ARCTICROUTE_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))

    roots.extend(
        [
            ROOT / "data",
            ROOT / "data_real",
            ROOT / "data_real" / "ais",
            ROOT / "data_processed" / "newenv",
            ROOT / "data" / "cmems_cache",
            ROOT / "data" / "static_assets",
        ]
    )
    return roots


def _iter_roots(candidates: List[Path]) -> List[Path]:
    """按照优先级返回实际存在的数据根目录列表。"""
    roots: List[Path] = []
    # 去重并仅保留存在的目录
    seen = set()
    for r in candidates:
        try:
            r = r.resolve()
        except Exception:
            continue
        if not r.exists():
            continue
        if r not in seen:
            seen.add(r)
            roots.append(r)
    return roots


def sync_newenv_from_env_root() -> Dict[str, Any]:
    """
    若仓库 newenv 缺少关键 NC，则尝试从 ARCTICROUTE_DATA_ROOT/**/newenv 复制。
    返回复制摘要，供 UI 显示。
    """
    env_root = os.environ.get("ARCTICROUTE_DATA_ROOT")
    dest_dir = ROOT / "data_processed" / "newenv"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_files = {
        "ice_copernicus_sic.nc": dest_dir / "ice_copernicus_sic.nc",
        "wave_swh.nc": dest_dir / "wave_swh.nc",
    }

    copied: List[tuple[str, str]] = []
    missing: List[str] = []

    if not env_root:
        return {"status": "error", "message": "未设置 ARCTICROUTE_DATA_ROOT，无法同步。", "copied": copied, "missing": list(target_files.keys())}

    env_root_path = Path(env_root)
    if not env_root_path.exists():
        return {"status": "error", "message": f"ARCTICROUTE_DATA_ROOT 不存在：{env_root_path}", "copied": copied, "missing": list(target_files.keys())}

    for name, dest in target_files.items():
        if dest.exists():
            continue
        src_path = None
        try:
            for p in env_root_path.rglob(name):
                if "newenv" in p.parts:
                    src_path = p
                    break
        except Exception:
            src_path = None

        if src_path is None:
            missing.append(name)
            continue
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_path, dest)
            copied.append((str(src_path), str(dest)))
        except Exception:
            missing.append(name)

    status = "copied" if copied else ("missing" if missing else "skipped")
    msg_parts = []
    if copied:
        msg_parts.append(f"已复制 {len(copied)} 个文件")
    if missing:
        msg_parts.append(f"缺少 {', '.join(missing)}")
    return {"status": status, "message": "; ".join(msg_parts) or "无操作", "copied": copied, "missing": missing}


def _glob_many(roots: List[Path], patterns: List[str]) -> ScanResult:
    hits: List[str] = []
    used_roots: List[str] = []
    for root in roots:
        any_hit = False
        for pat in patterns:
            for p in root.glob(pat):
                if not p.is_file():
                    continue
                if str(p) not in hits:
                    hits.append(str(p))
                    any_hit = True
        if any_hit:
            used_roots.append(str(root))
    return ScanResult(count=len(hits), examples=hits[:10], roots_used=used_roots)


def _sit_patterns() -> List[str]:
    """
    修正 SIT 识别：仅匹配 _sit_, sit-, sit. 或 ice_thickness。
    避免 accident_density_static 这类“static”误判。
    """
    return [
        "*_sit_*.nc",
        "*_sit-*.nc",
        "*_sit.nc",
        "*sit_*.nc",
        "*sit-*.nc",
        "*sit.nc",
        "*ice_thickness*.nc",
    ]


def scan_all() -> Dict[str, Any]:
    """
    扫描数据资产，返回结构化结果：
      - ais_density
      - sic
      - swh
      - sit
      - drift
      - static_assets
    每类包含：count / examples[:10] / roots_used
    """
    candidates = _candidate_roots()
    roots = _iter_roots(candidates)
    res: Dict[str, Any] = {
        "roots_requested": [str(r) for r in candidates],
        "roots_used": [str(r) for r in roots],
        "env": {
            "ARCTICROUTE_DATA_ROOT": os.environ.get("ARCTICROUTE_DATA_ROOT"),
        },
        "hits": {},
    }

    # AIS density
    res["hits"]["ais_density_nc"] = _glob_many(
        roots,
        ["**/*ais*dens*.nc", "**/*density*.nc"],
    ).__dict__

    # SIC
    res["hits"]["cmems_nc_sic"] = _glob_many(
        roots,
        ["**/*sic*.nc", "**/*siconc*.nc"],
    ).__dict__

    # SWH
    res["hits"]["cmems_nc_swh"] = _glob_many(
        roots,
        ["**/*swh*.nc", "**/*wave*height*.nc"],
    ).__dict__

    # SIT（使用更严格的文件名片段）
    res["hits"]["cmems_nc_sit"] = _glob_many(roots, [f"**/{pat}" for pat in _sit_patterns()]).__dict__

    # DRIFT
    res["hits"]["cmems_nc_drift"] = _glob_many(
        roots,
        ["**/*drift*.nc", "**/*ice_drift*.nc", "**/*uice*.nc", "**/*vice*.nc"],
    ).__dict__

    # 静态资产（geo/bathy/pdf 等交由 static_assets 自己管理）
    res["hits"]["static_assets"] = _glob_many(
        roots,
        [
            "**/*.geojson",
            "**/*ibcao*.nc",
            "**/*ibcao*.tif",
            "**/*bathym*.nc",
            "**/*depth*.tif",
            "**/*.pdf",
        ],
    ).__dict__

    return res


def render_data_discovery_panel() -> None:
    """
    在 Data 页中渲染数据发现面板：
      - 支持“重新扫描静态资产”按钮
      - 展示每类数据的 count / examples / roots_used
    """
    st.subheader("数据发现 / Data Discovery")

    if "scan_token" not in st.session_state:
        st.session_state["scan_token"] = time.time()

    if st.button("🔄 重新扫描 CMEMS 数据 / Rescan data assets"):
        st.session_state["scan_token"] = time.time()
        st.toast("已触发重新扫描", icon="🔍")

    if st.button("📦 同步 newenv (SIC/SWH)"):
        sync_result = sync_newenv_from_env_root()
        status = sync_result.get("status")
        msg = sync_result.get("message", "")
        if status == "copied":
            st.success(f"{msg}，来源→目标：{sync_result.get('copied')}")
        elif status == "missing":
            st.warning(msg or "未找到可复制的文件。")
        elif status == "error":
            st.error(msg or "同步失败")
        else:
            st.info(msg or "无需同步")

    with st.spinner("扫描中..."):
        t0 = time.time()
        snapshot = scan_all()
        elapsed = time.time() - t0

    hits = snapshot.get("hits") or {}
    total_hits = sum(int(info.get("count", 0)) for info in hits.values()) if isinstance(hits, dict) else 0
    st.success(f"扫描完成，用时 {elapsed:.2f} 秒，命中 {total_hits} 个文件。")
    st.toast(f"扫描完成：{total_hits} 个命中", icon="✅")

    roots_req = snapshot.get("roots_requested", [])
    roots_used = snapshot.get("roots_used") or []
    roots_used_set = set(roots_used)
    env_root_val = snapshot.get("env", {}).get("ARCTICROUTE_DATA_ROOT")
    st.caption(f"ARCTICROUTE_DATA_ROOT={env_root_val}")
    if roots_req:
        st.caption("扫描根目录（✅=使用 / ⚠=未找到）：")
        for r in roots_req:
            prefix = "✅" if r in roots_used_set else "⚠"
            st.code(f"{prefix} {r}", language="text")
    elif roots_used:
        st.caption("扫描根目录：")
        for r in roots_used:
            st.code(f"✅ {r}", language="text")
    else:
        st.warning("未找到任何可用的数据根目录。请检查 ARCTICROUTE_DATA_ROOT 或本地 data 目录。")

    for key, info in hits.items():
        with st.expander(f"{key} (count={info.get('count', 0)})", expanded=False):
            st.write("roots_used:", info.get("roots_used", []))
            examples = info.get("examples") or []
            if not examples:
                st.info("暂无示例文件。")
            else:
                st.write("examples（最多 10 条）：")
                for p in examples:
                    st.code(p, language="text")


