from __future__ import annotations
import json, os, time, hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional

try:
    import yaml  # type: ignore
except Exception:
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
CFG_PATHS = [REPO_ROOT/"config"/"paper_profiles.yaml", ARCTIC_DIR/"config"/"paper_profiles.yaml"]
PAPER_DIR = ARCTIC_DIR / "reports" / "paper"
LOG_PATH = PAPER_DIR / "repro_log.json"


def _load_profiles() -> Dict[str, Any]:
    for p in CFG_PATHS:
        if p.exists() and yaml is not None:
            try:
                data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if isinstance(data, dict) and data.get("profiles"):
                    return data["profiles"]
            except Exception:
                continue
    return {}


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        sha = subprocess.check_output(["git","rev-parse","HEAD"], cwd=str(REPO_ROOT)).decode().strip()
        return sha
    except Exception:
        return None


def _hash_cfg(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


def _append_log(entry: Dict[str, Any]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    data: List[Dict[str, Any]] = []
    if LOG_PATH.exists():
        try:
            data = json.loads(LOG_PATH.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                data = []
        except Exception:
            data = []
    data.append(entry)
    LOG_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_profile(profile_id: str) -> Dict[str, Any]:
    profiles = _load_profiles()
    prof = profiles.get(profile_id)
    if not isinstance(prof, dict):
        raise ValueError(f"profile not found: {profile_id}")

    run_id = time.strftime("%Y%m%dT%H%M%S")
    entry = {
        "run_id": run_id,
        "profile": profile_id,
        "git": _git_sha(),
        "config_hash": _hash_cfg(prof),
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "steps": [],
    }

    months: List[str] = [str(x) for x in (prof.get("months") or [])]
    scenarios: List[str] = [str(x) for x in (prof.get("scenarios") or [])]
    figs: List[str] = [str(x) for x in (prof.get("figures") or [])]

    # 1) 触发必要的报告生成（REUSE 已有 report.build）
    from ArcticRoute.api.cli import handle_report_build  # REUSE
    import argparse
    for ym in months:
        # calibration/uncertainty/eco 按需要触发
        include: List[str] = []
        if any(x in figs for x in ("calibration",)):
            include.append("calibration")
        if any(x in figs for x in ("uncertainty",)):
            include.append("uncertainty")
        if any(x in figs for x in ("eco",)):
            include.append("eco")
        if include:
            args = argparse.Namespace(ym=ym, scenario=None, include=include)
            rc = handle_report_build(args)
            entry["steps"].append({"report.build": {"ym": ym, "include": include, "rc": rc}})
        # pareto 需要 scenario 扫描
        if any(x in figs for x in ("pareto", "attribution",)):
            for scen in scenarios:
                args = argparse.Namespace(ym=ym, scenario=scen, include=["pareto"])
                rc = handle_report_build(args)
                entry["steps"].append({"report.build": {"ym": ym, "scenario": scen, "include": ["pareto"], "rc": rc}})

    # 2) 生成图与表
    from ArcticRoute.paper import figures as PFIG
    from ArcticRoute.paper import tables as PTAB
    out_figs: List[str] = []
    out_tabs: List[str] = []
    for ym in months:
        if "calibration" in figs:
            out_figs.append(str(PFIG.fig_calibration(ym)))
        if "pareto" in figs:
            for scen in scenarios:
                out_figs.append(str(PFIG.fig_pareto(ym, scen)))
        if "attribution" in figs:
            for scen in scenarios:
                out_figs.append(str(PFIG.fig_attribution(ym, scen)))
        if "uncertainty" in figs:
            out_figs.append(str(PFIG.fig_uncertainty(ym)))
        if "eco" in figs:
            for scen in scenarios:
                out_figs.append(str(PFIG.fig_eco(ym, scen)))
        if "domain_bucket" in figs:
            out_figs.append(str(PFIG.fig_domain_bucket(ym)))
        if "ablation_grid" in figs:
            out_figs.append(str(PFIG.fig_ablation_grid(months, scenarios)))
        # tables
        out_tabs.append(str(PTAB.tab_metrics_summary(ym, scenarios)))
    # ablation table
    out_tabs.append(str(PTAB.tab_ablation(months, scenarios)))

    entry.update({
        "figures": out_figs,
        "tables": out_tabs,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    _append_log(entry)
    return entry






