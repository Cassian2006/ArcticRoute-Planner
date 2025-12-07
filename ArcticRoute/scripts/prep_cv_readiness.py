#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare inputs and sanity checks for CV readiness.

@role: pipeline
"""

"""
ArcticRoute CV readiness preflight checklist.

The script performs lightweight environment diagnostics required before
enabling the future computer-vision (satellite) pipeline. It keeps the
current behaviour intact and only reports findings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
REQUIRED_DIRS = [
    PROJECT_ROOT / "data_processed" / "sat_cache",
    PROJECT_ROOT / "data_processed" / "stac_cache",
    PROJECT_ROOT / "data_processed" / "cog",
    PROJECT_ROOT / "data" / "cog_dir",
]
ENV_KEYS = ["CDSE_USERNAME", "CDSE_PASSWORD", "MPC_SAS_TOKEN"]
DISK_THRESHOLD_BYTES = 50 * 1024 * 1024 * 1024  # 50 GiB


@dataclass
class CheckResult:
    name: str
    status: str
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.extra:
            payload["extra"] = self.extra
        return payload


def _import_optional(module_name: str) -> tuple[bool, Optional[str]]:
    try:
        __import__(module_name)
    except ModuleNotFoundError as err:
        return False, str(err)
    except Exception as err:  # pragma: no cover - defensive guard
        return False, str(err)
    return True, None


def _detect_cuda() -> CheckResult:
    try:
        import torch

        available = bool(torch.cuda.is_available())
        capability = torch.version.cuda or "unknown"
        status = "pass" if available else "warn"
        message = "CUDA available" if available else "CUDA unavailable"
        return CheckResult(
            name="cuda",
            status=status,
            message=message,
            extra={"torch": torch.__version__, "cuda_version": capability},
        )
    except ModuleNotFoundError:
        return CheckResult(
            name="cuda",
            status="warn",
            message="PyTorch not installed; CUDA availability unknown",
        )
    except Exception as err:  # pragma: no cover - best effort
        return CheckResult(
            name="cuda",
            status="warn",
            message=f"CUDA probe failed: {err}",
        )


def _check_disk_space(root: Path) -> CheckResult:
    try:
        usage = shutil.disk_usage(str(root))
    except Exception as err:  # pragma: no cover - filesystem edge case
        return CheckResult(
            name="disk_space",
            status="warn",
            message=f"Unable to determine disk space: {err}",
        )
    free_gib = usage.free / (1024 ** 3)
    status = "pass" if usage.free >= DISK_THRESHOLD_BYTES else "warn"
    message = f"Free space: {free_gib:.1f} GiB"
    return CheckResult(
        name="disk_space",
        status=status,
        message=message + (" (OK)" if status == "pass" else " (less than 50 GiB)"),
        extra={
            "total_gib": usage.total / (1024 ** 3),
            "used_gib": usage.used / (1024 ** 3),
            "free_gib": free_gib,
        },
    )


def _check_dependencies() -> List[CheckResult]:
    modules = [
        ("rasterio", "rasterio"),
        ("pystac-client", "pystac_client"),
        ("stackstac", "stackstac"),
        ("rioxarray", "rioxarray"),
        ("shapely", "shapely"),
        ("scikit-image", "skimage"),
        ("folium", "folium"),
        ("streamlit", "streamlit"),
    ]
    results: List[CheckResult] = []
    for display_name, import_name in modules:
        ok, err = _import_optional(import_name)
        status = "pass" if ok else "warn"
        msg = "available" if ok else f"missing ({err.splitlines()[0]})"
        results.append(
            CheckResult(
                name=f"dependency:{display_name}",
                status=status,
                message=msg,
            )
        )
    return results


def _ensure_directory(path: Path) -> CheckResult:
    try:
        path.mkdir(parents=True, exist_ok=True)
        test_file = path / ".write_test"
        with test_file.open("w", encoding="utf-8") as handle:
            handle.write("ok")
        test_file.unlink(missing_ok=True)
        return CheckResult(
            name=f"directory:{path.relative_to(PROJECT_ROOT)}",
            status="pass",
            message="exists and writable",
        )
    except PermissionError as err:
        return CheckResult(
            name=f"directory:{path.relative_to(PROJECT_ROOT)}",
            status="warn",
            message=f"permission issue: {err}",
        )
    except Exception as err:  # pragma: no cover - edge filesystem scenario
        return CheckResult(
            name=f"directory:{path.relative_to(PROJECT_ROOT)}",
            status="warn",
            message=f"failed to prepare: {err}",
        )


def _parse_env_file(env_path: Path) -> Dict[str, str]:
    values: Dict[str, str] = {}
    if not env_path.exists():
        return values
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def _check_credentials(env_map: Dict[str, str]) -> List[CheckResult]:
    results: List[CheckResult] = []
    for key in ENV_KEYS:
        value = os.environ.get(key) or env_map.get(key)
        if value:
            status = "pass"
            message = "configured"
        else:
            status = "warn"
            message = "missing (set in .env or environment)"
        results.append(CheckResult(name=f"credential:{key}", status=status, message=message))
    return results


def _aggregate_status(checks: Iterable[CheckResult]) -> str:
    final_status = "pass"
    for check in checks:
        if check.status == "warn":
            final_status = "warn"
    return final_status


def generate_report() -> Dict[str, Any]:
    checks: List[CheckResult] = []

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    checks.append(
        CheckResult(
            name="python",
            status="pass",
            message=f"Python {python_version} ({platform.system()} {platform.release()})",
        )
    )
    checks.append(_detect_cuda())
    checks.append(_check_disk_space(PROJECT_ROOT))
    checks.extend(_check_dependencies())
    for directory in REQUIRED_DIRS:
        checks.append(_ensure_directory(directory))

    env_values = _parse_env_file(PROJECT_ROOT / ".env")
    checks.extend(_check_credentials(env_values))

    summary_status = _aggregate_status(checks)
    return {
        "project_root": str(PROJECT_ROOT),
        "status": summary_status,
        "checks": [check.to_dict() for check in checks],
    }


def write_reports(report: Dict[str, Any]) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    text_path = LOG_DIR / "cv_readiness_report.txt"
    json_path = LOG_DIR / "cv_readiness_report.json"

    lines: List[str] = [
        "=== ArcticRoute CV Readiness Report ===",
        f"Project root: {report['project_root']}",
        f"Overall status: {report['status'].upper()}",
        "",
    ]
    for item in report["checks"]:
        extra = item.get("extra")
        extra_repr = f" | extra={extra}" if extra else ""
        lines.append(f"[{item['status'].upper():4}] {item['name']}: {item['message']}{extra_repr}")
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ArcticRoute CV readiness preflight")
    parser.parse_args(argv)  # placeholder for future switches

    report = generate_report()
    write_reports(report)

    print("=== CV Readiness ===")
    for check in report["checks"]:
        print(f"[{check['status']}] {check['name']}: {check['message']}")
    print(f"Overall status: {report['status']}")

    return 0 if report["status"] == "pass" else 0


if __name__ == "__main__":
    sys.exit(main())
