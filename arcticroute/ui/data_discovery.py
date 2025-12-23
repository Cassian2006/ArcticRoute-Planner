from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import os
import time

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ScanResult:
    count: int
    examples: List[str]
    roots_used: List[str]


def _iter_roots() -> List[Path]:
    """按照优先级返回要扫描的数据根目录列表。"""
    roots: List[Path] = []
    env_root = os.environ.get("ARCTICROUTE_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))

    # 仓库内固定目录
    roots.extend(
        [
            ROOT / "data",
            ROOT / "data_real",
            ROOT / "data_processed" / "newenv",
            ROOT / "data" / "cmems_cache",
            ROOT / "data" / "static_assets",
        ]
    )
    # 去重并仅保留存在的目录
    seen = set()
    out: List[Path] = []
    for r in roots:
        try:
            r = r.resolve()
        except Exception:
            continue
        if not r.exists():
            continue
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


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
    roots = _iter_roots()
    res: Dict[str, Any] = {
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

    if st.button("🔄 重新扫描数据资产 / Rescan data assets"):
        st.session_state["scan_token"] = time.time()
        st.toast("已重新扫描数据资产", icon="✅")

    with st.spinner("扫描中..."):
        t0 = time.time()
        snapshot = scan_all()
        elapsed = time.time() - t0

    st.success(f"扫描完成，用时 {elapsed:.2f} 秒。")

    roots_used = snapshot.get("roots_used", [])
    if roots_used:
        st.caption("扫描根目录：")
        for r in roots_used:
            st.code(r, language="text")

    hits = snapshot.get("hits", {})
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


