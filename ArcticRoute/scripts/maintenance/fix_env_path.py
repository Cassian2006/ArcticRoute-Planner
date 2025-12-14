#!/usr/bin/env python3
"""Maintenance script to normalize environment path variables; kept for reference.

@role: legacy
"""

"""Utility to fix config env_nc path to latest env_clean.nc."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "runtime.yaml"


def find_latest_env(root: Path) -> Path | None:
    candidates = [p for p in root.rglob("env_clean.nc") if p.is_file()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def update_config(config_path: Path, new_value: str) -> tuple[str, str]:
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    data_block = data.setdefault("data", {})
    old_value = data_block.get("env_nc")
    data_block["env_nc"] = new_value
    config_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return (str(old_value) if old_value is not None else "", new_value)


def main() -> int:
    latest = find_latest_env(ROOT)
    if not latest:
        print("[FIX] env_clean.nc not found in repository; aborting.")
        return 1
    try:
        rel_value = latest.relative_to(ROOT).as_posix()
    except ValueError:
        rel_value = str(latest)

    old_value, new_value = update_config(CONFIG_PATH, rel_value)
    print(f"[FIX] runtime config: {CONFIG_PATH}")
    print(f"[FIX] old env_nc: {old_value}")
    print(f"[FIX] new env_nc: {new_value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
