#!/usr/bin/env python3
"""High-level repository consistency checks; historic helper.

@role: legacy
"""

"""
Repository consistency checks for ArcticRoute.

Validations:
1) Ensure config/runtime.yaml contains required keys used by api/cli.py.
2) Confirm tag alignment between route_*.geojson, route_on_risk_*.png, and run_report_*.json in outputs/.
3) For every run_report_*.json, verify referenced files exist and SHA-like fields are well-formed.

Results are written to logs/consistency_report.txt.
Exit status is non-zero when issues are detected.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIG_RUNTIME = PROJECT_ROOT / "config" / "runtime.yaml"
LOG_PATH = PROJECT_ROOT / "logs" / "consistency_report.txt"

SHA1_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_CONFIG_KEYS: Tuple[Tuple[str, ...], ...] = (
    ("data", "env_nc"),
    ("route", "start"),
    ("route", "goal"),
    ("outputs", "dir"),
)


def load_yaml(path: Path) -> Dict[str, object]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"failed to parse YAML at {path}: {exc}") from exc


def nested_get(data: Dict[str, object], path: Sequence[str]) -> Optional[object]:
    cur: Optional[object] = data
    for key in path:
        if isinstance(cur, dict) and key in cur:
            cur = cur[key]
        else:
            return None
    return cur


def check_config_keys(report: List[str], issues: List[str]) -> None:
    try:
        config = load_yaml(CONFIG_RUNTIME)
    except FileNotFoundError:
        issues.append(f"[CONFIG] missing runtime configuration: {CONFIG_RUNTIME}")
        return
    except RuntimeError as exc:
        issues.append(str(exc))
        return

    missing: List[str] = []
    for path in REQUIRED_CONFIG_KEYS:
        value = nested_get(config, path)
        if value in (None, "", {}):
            missing.append(".".join(path))
    if missing:
        issues.append(f"[CONFIG] runtime.yaml missing required keys: {', '.join(missing)}")

    # Detect potential mismatched section names (output vs outputs).
    if nested_get(config, ("outputs",)) is None and nested_get(config, ("output",)) is not None:
        report.append("[CONFIG] Found 'output' section but expected 'outputs'; CLI will ignore singular form.")


def collect_tag_map(
    glob_pattern: str,
    category_prefix: str,
) -> Dict[str, List[str]]:
    tag_map: Dict[str, List[str]] = {}
    for path in OUTPUTS_DIR.glob(glob_pattern):
        if not path.is_file():
            continue
        stem = path.stem
        if not stem.startswith(category_prefix):
            continue
        remainder = stem[len(category_prefix) :]
        if remainder.startswith("_"):
            remainder = remainder[1:]
        parts = remainder.split("_")
        if len(parts) < 2:
            tag_map.setdefault("<invalid>", []).append(stem)
            continue
        tag_candidate = "_".join(parts[-2:])
        if not re.fullmatch(r"\d{8}_\d{4,6}", tag_candidate):
            tag_map.setdefault("<invalid>", []).append(stem)
            continue
        base = "_".join(parts[:-2]) or "__root__"
        tag = tag_candidate
        tag_map.setdefault(base, []).append(tag)
    return tag_map


def check_output_tags(report: List[str], issues: List[str]) -> None:
    if not OUTPUTS_DIR.exists():
        report.append(f"[OUTPUTS] outputs directory absent ({OUTPUTS_DIR})")
        return

    route_tags = collect_tag_map("route_*.geojson", "route")
    risk_tags = collect_tag_map("route_on_risk_*.png", "route_on_risk")
    report_tags = collect_tag_map("run_report_*.json", "run_report")

    # Handle any invalid stems captured under "<invalid>".
    for stem in route_tags.get("<invalid>", []):
        issues.append(f"[OUTPUTS] route file has invalid tag format: {stem}")
    for stem in risk_tags.get("<invalid>", []):
        issues.append(f"[OUTPUTS] route_on_risk file has invalid tag format: {stem}")
    for stem in report_tags.get("<invalid>", []):
        issues.append(f"[OUTPUTS] run_report file has invalid tag format: {stem}")
    route_tags.pop("<invalid>", None)
    risk_tags.pop("<invalid>", None)
    report_tags.pop("<invalid>", None)

    all_bases = set(route_tags) | set(risk_tags) | set(report_tags)
    for base in sorted(all_bases):
        geo_tags = set(route_tags.get(base, []))
        png_tags = set(risk_tags.get(base, []))
        rpt_tags = set(report_tags.get(base, []))
        # Allow multiple historical runs; ensure every category shares identical tag sets if present.
        compare = [geo_tags, png_tags, rpt_tags]
        non_empty = [tags for tags in compare if tags]
        if not non_empty:
            continue
        reference = non_empty[0]
        mismatched = [tags for tags in non_empty[1:] if tags != reference]
        if mismatched:
            issues.append(
                f"[OUTPUTS] tag mismatch for '{base}': geojson={sorted(geo_tags)} "
                f"png={sorted(png_tags)} report={sorted(rpt_tags)}"
            )


def looks_like_path(value: str) -> bool:
    if value.startswith(("http://", "https://")):
        return False
    if any(sep in value for sep in ("/", "\\")):
        return True
    lower = value.lower()
    return lower.endswith((".nc", ".parquet", ".json", ".geojson", ".png", ".csv", ".txt", ".nc"))  # dup okay


def normalize_path(value: str) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def scan_report_for_paths(payload: object, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            yield from scan_report_for_paths(value, new_prefix)
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            new_prefix = f"{prefix}[{idx}]"
            yield from scan_report_for_paths(item, new_prefix)
    elif isinstance(payload, str):
        if looks_like_path(payload):
            yield prefix, payload


def scan_report_for_sha(payload: object, prefix: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(payload, dict):
        for key, value in payload.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            yield from scan_report_for_sha(value, new_prefix)
    elif isinstance(payload, list):
        for idx, item in enumerate(payload):
            new_prefix = f"{prefix}[{idx}]"
            yield from scan_report_for_sha(item, new_prefix)
    elif isinstance(payload, str):
        if SHA1_RE.fullmatch(payload):
            yield prefix, payload


def check_run_report_references(report: List[str], issues: List[str]) -> None:
    run_reports = sorted(OUTPUTS_DIR.rglob("run_report_*.json"))
    if not run_reports:
        report.append("[REPORT] no run_report_*.json files found")
        return

    for path in run_reports:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.append(f"[REPORT] failed to parse JSON {path}: {exc}")
            continue

        missing_paths: List[str] = []
        for location, raw_path in scan_report_for_paths(data):
            resolved = normalize_path(raw_path)
            if not resolved.exists():
                missing_paths.append(f"{location} -> {resolved}")

        bad_sha: List[str] = []
        for location, sha in scan_report_for_sha(data):
            if not SHA1_RE.fullmatch(sha):
                bad_sha.append(f"{location} -> {sha}")

        if missing_paths:
            issues.append(f"[REPORT] {path.relative_to(PROJECT_ROOT)} references missing files: {', '.join(missing_paths)}")
        if bad_sha:
            issues.append(f"[REPORT] {path.relative_to(PROJECT_ROOT)} has invalid SHA1 values: {', '.join(bad_sha)}")


def write_report(lines: List[str]) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    report_lines: List[str] = []
    issues: List[str] = []

    report_lines.append(f"Consistency check - {datetime.now():%Y-%m-%d %H:%M}")
    report_lines.append(f"Project root: {PROJECT_ROOT}")
    report_lines.append("")

    check_config_keys(report_lines, issues)
    check_output_tags(report_lines, issues)
    check_run_report_references(report_lines, issues)

    if issues:
        report_lines.append("Issues:")
        report_lines.extend(f"- {item}" for item in issues)
    else:
        report_lines.append("Issues: none detected")

    write_report(report_lines)
    print("\n".join(report_lines))
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
