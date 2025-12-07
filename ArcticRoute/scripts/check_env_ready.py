#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Check environment readiness for demos (dependencies, data presence, etc.).

@role: core
"""

"""
Validate .env configuration and writable project directories.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
LOG_DIR = PROJECT_ROOT / "logs"

KEYS_TO_PRINT = [
    "DEFAULT_STAC_SOURCE",
    "DEFAULT_MISSION",
    "LOG_LEVEL",
    "FORCE_HTTP_MOSAIC",
    "MPC_STAC_URL",
    "CDSE_STAC_URL",
    "CDSE_AUTH_MODE",
    "SAT_CACHE_DIR",
    "STAC_CACHE_DIR",
    "COG_DIR",
    "MOONSHOT_API_KEY",
    "MOONSHOT_MODEL",
    "AI_ENABLED",
]
DIR_KEYS = ["SAT_CACHE_DIR", "STAC_CACHE_DIR", "COG_DIR"]


def parse_env_file(path: Path) -> Dict[str, str]:
    env_data: Dict[str, str] = {}
    if not path.exists():
        return env_data
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        env_data[key.strip()] = value.strip()
    return env_data


def mask_value(key: str, value: str) -> str:
    if not value:
        return ""
    sensitive_tokens = ("TOKEN", "KEY", "SECRET", "PASSWORD")
    if any(token in key.upper() for token in sensitive_tokens):
        if len(value) <= 8:
            return "*" * len(value)
        return f"{value[:4]}***{value[-4:]}"
    return value


def check_directory(path: Path) -> Tuple[bool, str]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".env_ready_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True, "writable"
    except Exception as err:  # pragma: no cover - depends on FS permissions
        return False, f"not writable: {err}"


def main(argv: List[str] | None = None) -> int:
    env_values = parse_env_file(ENV_PATH)
    if not env_values:
        print(f"[ENV] .env missing at {ENV_PATH}")
    # merge into os.environ without overwriting existing
    for key, value in env_values.items():
        os.environ.setdefault(key, value)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    checks: List[Dict[str, str]] = []
    overall_status = "pass"

    def record_check(name: str, status: str, message: str) -> None:
        nonlocal overall_status
        checks.append({"name": name, "status": status, "message": message})
        if status.lower() != "pass":
            overall_status = "warn"

    print(f"[ENV] Loaded {len(env_values)} key(s) from {ENV_PATH}")
    for key in KEYS_TO_PRINT:
        value = env_values.get(key, "")
        print(f"  - {key} = {mask_value(key, value)}")

    default_source = env_values.get("DEFAULT_STAC_SOURCE", "")
    if default_source.upper() == "MPC":
        record_check("DEFAULT_STAC_SOURCE", "pass", "DEFAULT_STAC_SOURCE=MPC")
    elif default_source:
        record_check("DEFAULT_STAC_SOURCE", "warn", f"DEFAULT_STAC_SOURCE={default_source} (expected MPC)")
    else:
        record_check("DEFAULT_STAC_SOURCE", "warn", "DEFAULT_STAC_SOURCE missing")

    mpc_url = env_values.get("MPC_STAC_URL", "")
    if not mpc_url:
        record_check("MPC_STAC_URL", "warn", "MPC_STAC_URL missing")
    else:
        parsed = urlparse(mpc_url)
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            record_check("MPC_STAC_URL", "pass", f"MPC_STAC_URL={parsed.geturl()}")
        else:
            record_check("MPC_STAC_URL", "warn", f"MPC_STAC_URL invalid: {mpc_url}")

    for key in DIR_KEYS:
        raw_path = env_values.get(key, "")
        if not raw_path:
            record_check(key, "warn", f"{key} not set")
            continue
        dir_path = (PROJECT_ROOT / raw_path).resolve()
        ok, note = check_directory(dir_path)
        status = "pass" if ok else "warn"
        record_check(key, status, f"{dir_path} -> {note}")

    report = {
        "env_path": str(ENV_PATH),
        "status": overall_status,
        "checks": checks,
    }
    report_json = LOG_DIR / "env_ready_report.json"
    report_txt = LOG_DIR / "env_ready_report.txt"
    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"Env Ready Status: {overall_status}", ""]
    for entry in checks:
        lines.append(f"[{entry['status'].upper()}] {entry['name']}: {entry['message']}")
    report_txt.write_text("\n".join(lines), encoding="utf-8")
    print(f"[ENV] Report written to {report_json}")
    print(f"[ENV] Summary written to {report_txt}")
    return 0 if overall_status == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())

