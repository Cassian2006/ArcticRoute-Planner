from __future__ import annotations

"""
Phase N · 重规划 Watcher（最小实现）

职责：
- 周期轮询：检查 live 风险面是否更新，或周期触发
- 依据规则与冷却期决定是否触发 route.replan（通过 CLI 子进程）
- 记录状态与事件日志至 outputs/live/<scenario>/{state.json, events.log}

约束：
- 最小实现不接入 AIS；current_pos 使用旧路线最近点占位
- 支持 --once 用于烟雾测试（只跑一轮）
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
RISK_DIR = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
LIVE_OUT = REPO_ROOT / "ArcticRoute" / "outputs" / "live"


def _latest_live_surface() -> Optional[Path]:
    cands = sorted(RISK_DIR.glob("risk_fused_live_*.nc"), key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0] if cands else None


@dataclass
class WatchState:
    last_update_ts: float = 0.0
    last_replan_ts: float = 0.0
    last_reason: str = ""
    last_live_path: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {
            "last_update_ts": self.last_update_ts,
            "last_replan_ts": self.last_replan_ts,
            "last_reason": self.last_reason,
            "last_live_path": self.last_live_path,
        }


def _load_rules(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {
            "replan": {
                "period_sec": 1800,
                "lookahead_nm": 50,
                "risk_threshold": 0.55,
                "risk_delta": 0.15,
                "interact_delta": 0.10,
                "eco_delta_pct": 5,
                "cool_down_sec": 900,
                "handover_nm": 8,
                "min_change_nm": 3,
            }
        }
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def run_watcher(scenario: str, interval: int = 300, rules_path: str = "configs/replan.yaml", once: bool = False) -> int:
    LIVE_OUT.mkdir(parents=True, exist_ok=True)
    scen_dir = LIVE_OUT / scenario
    scen_dir.mkdir(parents=True, exist_ok=True)
    state_path = scen_dir / "state.json"
    events_path = scen_dir / "events.log"

    # state
    if state_path.exists():
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            st = WatchState(
                last_update_ts=float(data.get("last_update_ts", 0.0) or 0.0),
                last_replan_ts=float(data.get("last_replan_ts", 0.0) or 0.0),
                last_reason=str(data.get("last_reason", "")),
                last_live_path=str(data.get("last_live_path", "")),
            )
        except Exception:
            st = WatchState()
    else:
        st = WatchState()

    rules = _load_rules(REPO_ROOT / rules_path)

    def log_event(msg: str) -> None:
        t = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(f"[{t}] {msg}\n")

    while True:
        live = _latest_live_surface()
        deltas = {
            "periodic": True,  # 周期检查
            "now_ts": time.time(),
            "has_new_surface": False,
        }
        if live is not None:
            mtime = live.stat().st_mtime
            if mtime > st.last_update_ts:
                deltas["has_new_surface"] = True
                st.last_update_ts = mtime
                st.last_live_path = str(live)
        # 简化决策：若有新 surface 或超过 period，则触发
        need = False
        reason = "stable"
        period = int((rules.get("replan") or {}).get("period_sec", 1800))
        if deltas["has_new_surface"]:
            need = True; reason = "new_surface"
        elif st.last_replan_ts <= 0 or (time.time() - st.last_replan_ts) >= max(60, period):
            need = True; reason = "periodic"

        if need:
            # 调用 CLI route.replan
            cmd = [
                os.fspath(REPO_ROOT / "venv" / "bin" / "python") if (REPO_ROOT / "venv" / "bin" / "python").exists() else os.fspath(os.environ.get("PYTHON", "python")),
                "-m", "ArcticRoute.api.cli", "route.replan",
                "--scenario", scenario, "--live",
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if proc.returncode == 0:
                    st.last_replan_ts = time.time()
                    st.last_reason = reason
                    log_event(f"replan ok · {reason}")
                else:
                    log_event(f"replan failed rc={proc.returncode} · {proc.stderr.strip()[:200]}")
            except Exception as e:  # noqa: BLE001
                log_event(f"replan exception · {e}")
            state_path.write_text(json.dumps(st.to_json(), ensure_ascii=False, indent=2), encoding="utf-8")
        if once:
            break
        time.sleep(max(5, int(interval)))
    return 0


__all__ = ["run_watcher"]

