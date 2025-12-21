from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Tuple


SIT_KEYWORDS = ("thickness", "sithick", "sit", "ice_thickness", "ice_thk")
DRIFT_KEYWORDS = (
    "drift",
    "velocity",
    "ice_u",
    "ice_v",
    "uice",
    "vice",
    "speed",
    "dx",
    "dy",
    "displacement",
)

SIT_FALLBACK_DATASET = "esa_obs-si_arc_phy-sit_nrt_l4-multi_P1D-m"
DRIFT_FALLBACK_DATASET = "cmems_sat-si_glo_drift_nrt_north_d"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _load_json(path: Path) -> dict | None:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def _parse_probe_text(path: Path, keywords: Iterable[str], min_vars: int) -> Tuple[str | None, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, []

    dataset_ids = re.findall(r'"dataset_id"\s*:\s*"([^"]+)"', text)
    dataset_id = dataset_ids[0] if dataset_ids else None

    var_hits: list[str] = []
    for key in keywords:
        pattern = re.compile(rf'"[^"]*"\s*:\s*"([^"]*{re.escape(key)}[^"]*)"', re.IGNORECASE)
        for match in pattern.findall(text):
            var_hits.append(match)

    # Deduplicate while preserving order.
    seen = set()
    vars_unique = []
    for v in var_hits:
        if v not in seen:
            vars_unique.append(v)
            seen.add(v)

    if dataset_id and len(vars_unique) >= min_vars:
        return dataset_id, vars_unique[: max(min_vars, 3)]
    return None, []


def _parse_describe_json(path: Path, keywords: Iterable[str], min_vars: int) -> Tuple[str | None, list[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None, []

    dataset_ids = re.findall(r'"dataset_id"\s*:\s*"([^"]+)"', text)
    dataset_id = dataset_ids[0] if dataset_ids else None

    var_names = re.findall(r'"short_name"\s*:\s*"([^"]+)"', text)
    if not var_names:
        var_names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    var_hits: list[str] = []
    for key in keywords:
        for name in var_names:
            if key.lower() in name.lower():
                var_hits.append(name)

    seen = set()
    vars_unique = []
    for v in var_hits:
        if v not in seen:
            vars_unique.append(v)
            seen.add(v)

    if dataset_id and len(vars_unique) >= min_vars:
        return dataset_id, vars_unique[: max(min_vars, 3)]
    return None, []


def _ensure_drift_uv(vars_found: list[str], json_path: Path) -> list[str]:
    if not json_path.exists():
        return vars_found
    try:
        text = json_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return vars_found
    names = re.findall(r'"short_name"\s*:\s*"([^"]+)"', text)
    for cand in ("dX_mean", "dY_mean"):
        if cand in names and cand not in vars_found:
            vars_found.append(cand)
    return vars_found


def _resolve_component(
    name: str,
    keywords: Iterable[str],
    min_vars: int,
    nextsim_available: bool,
    nextsim_reason: str,
    probe_path: Path,
    nextsim_json: Path,
    fallback_dataset: str,
    fallback_json: Path,
) -> dict:
    # Prefer nextsim if describe output is large enough and has keywords.
    if nextsim_available and nextsim_json.exists() and nextsim_json.stat().st_size >= 1000:
        dataset_id, vars_found = _parse_describe_json(nextsim_json, keywords, min_vars)
        if dataset_id and vars_found:
            return {
                "dataset_id": dataset_id,
                "variables": vars_found,
                "status": "ok",
                "reason": "",
                "source": "nextsim",
            }

    # If nextsim probe text works, allow it as fallback for variables.
    if nextsim_available:
        dataset_id, vars_found = _parse_probe_text(probe_path, keywords, min_vars)
        if dataset_id and vars_found:
            return {
                "dataset_id": dataset_id,
                "variables": vars_found,
                "status": "ok",
                "reason": "nextsim_probe_text_used",
                "source": "nextsim",
            }

    # Final fallback to NRT datasets.
    dataset_id, vars_found = _parse_describe_json(fallback_json, keywords, min_vars)
    if dataset_id is None:
        dataset_id = fallback_dataset
    if name == "drift":
        vars_found = _ensure_drift_uv(list(vars_found), fallback_json)
    return {
        "dataset_id": dataset_id,
        "variables": vars_found,
        "status": "ok" if vars_found else "skipped",
        "reason": "fallback_to_nrt",
        "source": "nrt",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Resolve CMEMS datasets for SIC/Wave/SIT/Drift")
    ap.add_argument(
        "--output",
        default="reports/cmems_resolved.json",
        help="Output JSON path",
    )
    ap.add_argument(
        "--probe",
        default="reports/cmems_sic_probe_nextsim.txt",
        help="Text probe output to parse (nextsim)",
    )
    ap.add_argument(
        "--nextsim-json",
        default="reports/cmems_nextsim_vars.json",
        help="Nextsim describe JSON output",
    )
    ap.add_argument(
        "--sit-json",
        default="reports/cmems_sit_nrt_vars.json",
        help="SIT NRT describe JSON output",
    )
    ap.add_argument(
        "--drift-json",
        default="reports/cmems_drift_nrt_vars.json",
        help="Drift NRT describe JSON output",
    )
    ap.add_argument(
        "--strategy",
        default="reports/cmems_strategy.json",
        help="Strategy JSON to determine nextsim availability",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    probe_path = Path(args.probe)
    nextsim_json = Path(args.nextsim_json)
    sit_json = Path(args.sit_json)
    drift_json = Path(args.drift_json)
    strategy = _load_json(Path(args.strategy)) or {}

    nextsim_available = bool(strategy.get("nextsim_available", False))
    nextsim_reason = str(strategy.get("nextsim_reason", "unknown"))

    existing = _load_json(out_path) or {}
    sit_res = _resolve_component(
        "sit",
        SIT_KEYWORDS,
        min_vars=1,
        nextsim_available=nextsim_available,
        nextsim_reason=nextsim_reason,
        probe_path=probe_path,
        nextsim_json=nextsim_json,
        fallback_dataset=SIT_FALLBACK_DATASET,
        fallback_json=sit_json,
    )
    drift_res = _resolve_component(
        "drift",
        DRIFT_KEYWORDS,
        min_vars=2,
        nextsim_available=nextsim_available,
        nextsim_reason=nextsim_reason,
        probe_path=probe_path,
        nextsim_json=nextsim_json,
        fallback_dataset=DRIFT_FALLBACK_DATASET,
        fallback_json=drift_json,
    )

    resolved = {
        "sic": existing.get("sic", {}),
        "wav": existing.get("wav", {}),
        "sit": sit_res,
        "drift": drift_res,
        "timestamp": _now_iso(),
    }

    strategy_out = Path(args.strategy)
    fallback_reason = []
    if sit_res.get("source") != "nextsim":
        fallback_reason.append("sit: fallback to nrt")
    if drift_res.get("source") != "nextsim":
        fallback_reason.append("drift: fallback to nrt")
    strategy = {
        "nextsim_available": nextsim_available,
        "nextsim_reason": nextsim_reason,
        "sit_source": sit_res.get("source"),
        "drift_source": drift_res.get("source"),
        "fallback_reason": "; ".join(fallback_reason) if fallback_reason else "",
        "timestamp": _now_iso(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    strategy_out.parent.mkdir(parents=True, exist_ok=True)
    strategy_out.write_text(json.dumps(strategy, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
