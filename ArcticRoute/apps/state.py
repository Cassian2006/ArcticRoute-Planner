from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _runtime_yaml() -> Path:
    return _repo_root() / "ArcticRoute" / "config" / "runtime.yaml"


def _safe_git_sha() -> Optional[str]:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=_repo_root(), capture_output=True, text=True, check=False)
        sha = (res.stdout or "").strip()
        return sha or None
    except Exception:
        return None


def _hash_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


@dataclass
class UIActionMeta:
    run_id: str
    git_sha: Optional[str]
    config_hash: Optional[str]
    action: str
    inputs: Dict[str, Any]


def ensure_outputs_dir() -> Path:
    out = _repo_root() / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    return out

def ensure_ui_dirs() -> Path:
    ui = ensure_outputs_dir() / "ui"
    (ui / "actions").mkdir(parents=True, exist_ok=True)
    (ui / "logs").mkdir(parents=True, exist_ok=True)
    return ui


def save_ui_state(state: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """
    将当前 UI 状态写入 outputs/.ui_state.json（或指定路径）。
    建议内容：月份、图层开关/色表、路由参数、控件映射后的内部名等。
    """
    if path is None:
        path = ensure_outputs_dir() / ".ui_state.json"
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        # 回退为紧凑写入，尽量不失败
        path.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    return path


def load_ui_state(path: Optional[Path] = None) -> Dict[str, Any]:
    if path is None:
        path = ensure_outputs_dir() / ".ui_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_action_meta(action: str, inputs: Dict[str, Any], path: Optional[Path] = None) -> Path:
    """
    旧版：写入 outputs/.ui_action.meta.json（保留以兼容历史逻辑）。
    """
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    meta = UIActionMeta(
        run_id=run_id,
        git_sha=_safe_git_sha(),
        config_hash=_hash_file(_runtime_yaml()),
        action=action,
        inputs=inputs,
    )
    if path is None:
        path = ensure_outputs_dir() / ".ui_action.meta.json"
    path.write_text(json.dumps(asdict(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_action_meta2(action: str, *, inputs: Dict[str, Any], outputs: Dict[str, Any] | None = None) -> Path:
    """
    新版：统一动作元信息落盘。
    - 目录：outputs/ui/actions/
    - 文件名：YYYYMMDD_HHMMSS_<action>_<shortid>.meta.json
    - 字段：ts, action, inputs, outputs, run_id, git_sha, config_hash
    """
    ui_dir = ensure_ui_dirs()
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    shortid = (hashlib.sha1(f"{ts}-{action}".encode("utf-8")).hexdigest())[:6]
    filename = f"{ts}_{action}_{shortid}.meta.json"
    payload = {
        "ts": ts,
        "action": action,
        "inputs": inputs,
        "outputs": outputs or {},
        "run_id": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "git_sha": _safe_git_sha(),
        "config_hash": _hash_file(_runtime_yaml()),
    }
    out_path = ui_dir/"actions"/filename
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


# ---- 辅助：从 Streamlit session_state 提取最小状态（可在 app_min/pages 调用） ----

def snapshot_from_session(session: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        # 全局
        "ym", "alpha", "time_index",
        # 路由参数
        "inp_si", "inp_sj", "inp_gi", "inp_gj", "inp_diag", "inp_heuristic",
        # 风险/聚合
        "risk_source", "risk_agg", "alpha_risk", "interact_weight",
        # ECO
        "eco_enabled", "w_e", "vessel_class",
        # 图层
        "layer_sic_show", "layer_sic_cmap", "layer_sic_auto", "layer_sic_vmin", "layer_sic_vmax",
        "layer_ice_show", "layer_ice_cmap", "layer_ice_auto", "layer_ice_vmin", "layer_ice_vmax",
    ]
    out: Dict[str, Any] = {}
    for k in keys:
        if k in session:
            out[k] = session[k]
    return out

