from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Iterable, Tuple


SIT_KEYWORDS = ("thickness", "sithick", "sit", "ice_thickness", "ice_thk")
DRIFT_KEYWORDS = ("drift", "velocity", "ice_u", "ice_v", "uice", "vice", "speed")


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


def _resolve_component(
    name: str,
    keywords: Iterable[str],
    min_vars: int,
    nextsim_available: bool,
    nextsim_reason: str,
    probe_path: Path,
) -> dict:
    if not nextsim_available:
        return {
            "dataset_id": None,
            "variables": [],
            "status": "skipped",
            "reason": f"nextsim_unavailable: {nextsim_reason}",
        }

    dataset_id, vars_found = _parse_probe_text(probe_path, keywords, min_vars)
    if dataset_id and vars_found:
        return {
            "dataset_id": dataset_id,
            "variables": vars_found,
            "status": "ok",
            "reason": "",
        }

    return {
        "dataset_id": None,
        "variables": [],
        "status": "skipped",
        "reason": f"{name}_nextsim_parse_failed",
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
        "--strategy",
        default="reports/cmems_strategy.json",
        help="Strategy JSON to determine nextsim availability",
    )
    args = ap.parse_args()

    out_path = Path(args.output)
    probe_path = Path(args.probe)
    strategy = _load_json(Path(args.strategy)) or {}

    nextsim_available = bool(strategy.get("nextsim_available", False))
    nextsim_reason = str(strategy.get("nextsim_reason", "unknown"))

    existing = _load_json(out_path) or {}
    resolved = {
        "sic": existing.get("sic", {}),
        "wav": existing.get("wav", {}),
        "sit": _resolve_component(
            "sit",
            SIT_KEYWORDS,
            min_vars=1,
            nextsim_available=nextsim_available,
            nextsim_reason=nextsim_reason,
            probe_path=probe_path,
        ),
        "drift": _resolve_component(
            "drift",
            DRIFT_KEYWORDS,
            min_vars=2,
            nextsim_available=nextsim_available,
            nextsim_reason=nextsim_reason,
            probe_path=probe_path,
        ),
        "timestamp": _now_iso(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(resolved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()
