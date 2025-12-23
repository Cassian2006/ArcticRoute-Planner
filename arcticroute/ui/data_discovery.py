from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
import shutil

import os
import time

import streamlit as st


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DiscoveryItem:
    """单类数据的发现结果，便于 UI 展示。"""

    found_paths: List[str]
    selected_path: Optional[str]
    searched_paths: List[str]
    patterns: List[str]
    reason: str
    roots_used: List[str]


@dataclass
class ScanResult:
    count: int
    examples: List[str]
    roots_used: List[str]


def _parse_extra_paths() -> List[Path]:
    """从 ARCTICROUTE_EXTRA_DATA_PATHS 环境变量解析额外的搜索路径。"""
    raw = os.environ.get("ARCTICROUTE_EXTRA_DATA_PATHS", "")
    parts = [p for p in raw.split(os.pathsep) if p.strip()]
    return [Path(p.strip()) for p in parts]


def _candidate_roots() -> List[Path]:
    """返回预期扫描的根目录（包含可能不存在的路径）。"""
    roots: List[Path] = []

    # 1) 显式配置：ARCTICROUTE_EXTRA_DATA_PATHS，支持外部目录（; 分隔）
    roots.extend(_parse_extra_paths())

    # 2) 兼容已有的 ARCTICROUTE_DATA_ROOT
    env_root = os.environ.get("ARCTICROUTE_DATA_ROOT")
    if env_root:
        roots.append(Path(env_root))

    # 3) 项目内的常用目录（保持向后兼容）
    roots.extend(
        [
            ROOT / "data_processed" / "newenv",
            ROOT / "data" / "cmems_cache",
            ROOT / "data" / "static_assets",
            ROOT / "data_real",
            ROOT / "data_real" / "ais",
            ROOT / "data",
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


def _scan_category(
    roots: List[Path],
    patterns: List[str],
    *,
    forbid_tif: bool = False,
    note_when_tif: Optional[str] = None,
) -> DiscoveryItem:
    """
    在给定根目录中按 patterns 搜索文件，返回结构化结果。

    Args:
        roots: 实际存在的根目录列表
        patterns: 相对 glob 模式（支持 **）
        forbid_tif: 若为 True，则即便找到 tif 也不选用（仅报告）
        note_when_tif: 当仅命中 tif 时的原因说明
    """
    found: List[str] = []
    seen_tif = False
    used_roots: List[str] = []
    for root in roots:
        any_hit = False
        for pat in patterns:
            for p in root.rglob(pat):
                if not p.is_file():
                    continue
                if forbid_tif and p.suffix.lower() == ".tif":
                    any_hit = True
                    seen_tif = True
                    continue
                resolved = str(p.resolve())
                if resolved not in found:
                    found.append(resolved)
                    any_hit = True
        if any_hit:
            used_roots.append(str(root))

    selected: Optional[str] = None
    reason = ""
    if found:
        selected = found[0]
        reason = f"已按优先顺序选择 {selected}"
    elif forbid_tif and seen_tif:
        reason = note_when_tif or "仅找到 tif，但当前未启用 tif 读取。"
    else:
        reason = f"未找到匹配文件，期望文件名包含：{patterns}"

    return DiscoveryItem(
        found_paths=found,
        selected_path=selected,
        searched_paths=[str(r) for r in roots],
        patterns=patterns,
        reason=reason,
        roots_used=used_roots,
    )


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
            "ARCTICROUTE_EXTRA_DATA_PATHS": os.environ.get("ARCTICROUTE_EXTRA_DATA_PATHS"),
        },
        "hits": {},
        "items": {},
    }

    # AIS density（保留兼容信息）
    ais_item = _scan_category(roots, ["**/*ais*dens*.nc", "**/*density*.nc"])
    res["items"]["ais_density_nc"] = ais_item.__dict__
    res["hits"]["ais_density_nc"] = {
        "count": len(ais_item.found_paths),
        "examples": ais_item.found_paths[:10],
        "roots_used": ais_item.roots_used,
        "selected_path": ais_item.selected_path,
        "reason": ais_item.reason,
    }

    # SIC
    sic_item = _scan_category(roots, ["**/*sic*.nc", "**/ice_copernicus_sic.nc"])
    res["items"]["cmems_nc_sic"] = sic_item.__dict__
    res["hits"]["cmems_nc_sic"] = {
        "count": len(sic_item.found_paths),
        "examples": sic_item.found_paths[:10],
        "roots_used": sic_item.roots_used,
        "selected_path": sic_item.selected_path,
        "reason": sic_item.reason,
    }

    # SWH
    swh_item = _scan_category(roots, ["**/*swh*.nc", "**/wave_swh.nc"])
    res["items"]["cmems_nc_swh"] = swh_item.__dict__
    res["hits"]["cmems_nc_swh"] = {
        "count": len(swh_item.found_paths),
        "examples": swh_item.found_paths[:10],
        "roots_used": swh_item.roots_used,
        "selected_path": swh_item.selected_path,
        "reason": swh_item.reason,
    }

    # SIT（使用更严格的文件名片段）
    sit_item = _scan_category(roots, [f"**/{pat}" for pat in _sit_patterns()])
    res["items"]["cmems_nc_sit"] = sit_item.__dict__
    res["hits"]["cmems_nc_sit"] = {
        "count": len(sit_item.found_paths),
        "examples": sit_item.found_paths[:10],
        "roots_used": sit_item.roots_used,
        "selected_path": sit_item.selected_path,
        "reason": sit_item.reason,
    }

    # DRIFT
    drift_item = _scan_category(roots, ["**/*drift*.nc", "**/ice_drift*.nc"])
    res["items"]["cmems_nc_drift"] = drift_item.__dict__
    res["hits"]["cmems_nc_drift"] = {
        "count": len(drift_item.found_paths),
        "examples": drift_item.found_paths[:10],
        "roots_used": drift_item.roots_used,
        "selected_path": drift_item.selected_path,
        "reason": drift_item.reason,
    }

    # bathymetry（优先 NC，若仅有 tif 则说明未启用）
    bathy_item = _scan_category(
        roots,
        ["**/IBCAO*.nc", "**/ibcao*.nc", "**/IBCAO*.tif", "**/ibcao*.tif"],
        forbid_tif=True,
        note_when_tif="发现 tif 但当前未启用 tif 读取，请提供 IBCAO*.nc。",
    )
    res["items"]["bathymetry"] = bathy_item.__dict__
    res["hits"]["bathymetry"] = {
        "count": len(bathy_item.found_paths),
        "examples": bathy_item.found_paths[:10],
        "roots_used": bathy_item.roots_used,
        "selected_path": bathy_item.selected_path,
        "reason": bathy_item.reason,
    }

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


def summarize_discovery(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """将 scan_all 结果转成便于 UI/测试消费的摘要。"""
    items = snapshot.get("items", {}) if isinstance(snapshot, dict) else {}
    roots_used = snapshot.get("roots_used", [])

    def _entry(key: str, default_reason: str) -> Dict[str, Any]:
        info = items.get(key, {}) if isinstance(items, dict) else {}
        found_paths = info.get("found_paths") or []
        selected = info.get("selected_path")
        searched = info.get("searched_paths") or roots_used
        reason = info.get("reason") or default_reason
        patterns = info.get("patterns") or []
        return {
            "found": bool(found_paths),
            "selected_path": selected,
            "found_paths": found_paths,
            "searched_paths": searched,
            "reason": reason,
            "patterns": patterns,
        }

    summary = {
        "sic": _entry("cmems_nc_sic", "未找到 SIC 文件"),
        "swh": _entry("cmems_nc_swh", "未找到 SWH 文件"),
        "sit": _entry("cmems_nc_sit", "未找到 SIT 文件"),
        "drift": _entry("cmems_nc_drift", "未找到 Drift 文件"),
        "bathymetry": _entry("bathymetry", "未找到 IBCAO/Bathymetry 文件"),
        "ais": _entry("ais_density_nc", "未找到 AIS 密度文件"),
        "roots_used": roots_used,
    }
    return summary


def availability_flags(summary: Dict[str, Any]) -> Dict[str, bool]:
    """从摘要中提取可用性布尔值，供 UI 控制开关/禁用状态。"""
    return {
        "sic": bool(summary.get("sic", {}).get("found")),
        "swh": bool(summary.get("swh", {}).get("found")),
        "sit": bool(summary.get("sit", {}).get("found")),
        "drift": bool(summary.get("drift", {}).get("found")),
        "bathymetry": bool(summary.get("bathymetry", {}).get("found")),
        "ais": bool(summary.get("ais", {}).get("found")),
    }


def render_data_discovery_panel() -> None:
    """
    在 Data 页中渲染数据发现面板：
      - 支持“重新扫描静态资产”按钮
      - 展示每类数据的 count / examples / roots_used
    """
    st.subheader("数据发现 / Data Discovery")

    if st.button("🔄 重新扫描 CMEMS 数据 / Rescan data assets"):
        _ = scan_all()
        st.toast("扫描完成", icon="✅")
        st.rerun()

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

    summary = summarize_discovery(snapshot)
    _ = availability_flags(summary)
    total_hits = sum(len(info.get("found_paths", [])) for info in summary.values() if isinstance(info, dict))
    st.success(f"扫描完成，用时 {elapsed:.2f} 秒，命中 {total_hits} 个文件。")
    st.toast("扫描完成", icon="✅")

    roots_req = snapshot.get("roots_requested", [])
    roots_used = snapshot.get("roots_used") or []
    roots_used_set = set(roots_used)
    env_root_val = snapshot.get("env", {}).get("ARCTICROUTE_DATA_ROOT")
    extra_paths = snapshot.get("env", {}).get("ARCTICROUTE_EXTRA_DATA_PATHS")
    st.caption(f"ARCTICROUTE_DATA_ROOT={env_root_val} | ARCTICROUTE_EXTRA_DATA_PATHS={extra_paths}")
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

    label_map = {
        "sic": "SIC",
        "swh": "SWH",
        "sit": "SIT (ice_thickness)",
        "drift": "Drift",
        "bathymetry": "Bathymetry",
        "ais": "AIS Density",
    }

    for key, label in label_map.items():
        info = summary.get(key, {}) if isinstance(summary, dict) else {}
        found = info.get("found", False)
        status = "✅ 已找到" if found else "❌ 未找到"
        selected = info.get("selected_path")
        reason = info.get("reason")
        patterns = info.get("patterns") or []
        searched = info.get("searched_paths") or []
        st.markdown(f"**{label}** - {status}")
        if selected:
            st.code(f"selected: {selected}", language="text")
        if info.get("found_paths"):
            with st.expander("候选文件", expanded=False):
                for p in info["found_paths"][:10]:
                    st.code(p, language="text")
        if searched:
            with st.expander("搜索路径", expanded=False):
                for p in searched:
                    st.code(p, language="text")
        hint = f"期望文件名包含：{patterns}" if patterns else ""
        if not found:
            st.warning(f"{reason or '未找到匹配文件'}；{hint}", icon="⚠️")
        else:
            st.info(reason or "已选择首个匹配文件。", icon="✅")


