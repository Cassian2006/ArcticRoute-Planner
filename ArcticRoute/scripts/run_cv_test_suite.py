#!/usr/bin/env python3
"""Run a lightweight CV validation suite and summarize artifacts for debugging.

@role: analysis
"""

"""End-to-end CV validation harness."""
from __future__ import annotations

import argparse
import os
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
import yaml

# --- bootstrap imports ---
try:
    from ArcticRoute.scripts._modpath import ensure_path, get_cli_mod
except ModuleNotFoundError:
    HERE = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(HERE, "..", "..")),
        os.path.abspath(os.path.join(HERE, "..")),
        HERE,
    ]
    for candidate in candidates:
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    from scripts._modpath import ensure_path, get_cli_mod  # type: ignore[import]

ensure_path()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CFG = PROJECT_ROOT / "config" / "runtime.yaml"
DEFAULT_TIDX = 0
DEFAULT_GAMMA = 0.0
DEFAULT_ALPHAS = "0.01,0.1,0.25,0.5,0.75,0.9,0.99"
ERROR_LOG_PATH = PROJECT_ROOT / "logs" / "cv_test_suite_error.log"
LOG_ENTRIES: List[str] = []


@dataclass
class StepStatus:
    name: str
    status: str
    detail: str


class StepFailure(RuntimeError):
    pass


def _load_runtime_dict(cfg_path: Path) -> dict:
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def _resolve_project_path(value: Optional[str]) -> Optional[Path]:
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _infer_date_from_env(env_path: Path, tidx: int) -> str:
    with xr.open_dataset(env_path) as ds:
        if "time" not in ds.coords:
            raise ValueError("environment dataset lacks 'time' coordinate")
        times = ds["time"].values
        if times.size == 0:
            raise ValueError("environment dataset has empty time axis")
        index = max(0, min(int(tidx), times.size - 1))
        value = np.datetime_as_string(times[index], unit="D")
    return str(value)


def _run_command(cmd: List[str], name: str, *, allow_failure: bool = False) -> subprocess.CompletedProcess:
    global LOG_ENTRIES
    print(f"[SUITE] {name}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    entry_lines = [
        f"== {name} ==",
        f"CMD: {' '.join(cmd)}",
        f"Return code: {proc.returncode}",
        "STDOUT:",
        proc.stdout.rstrip() if proc.stdout else "(empty)",
        "STDERR:",
        proc.stderr.rstrip() if proc.stderr else "(empty)",
    ]
    LOG_ENTRIES.append("\n".join(entry_lines))
    if proc.stdout:
        print(proc.stdout, end="")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="")
    if proc.returncode != 0 and not allow_failure:
        raise StepFailure(f"{name} failed with exit code {proc.returncode}")
    return proc


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_alpha_sweep(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "alpha_ice": float(row["alpha_ice"]),
                    "total_cost": float(row["total_cost"]),
                    "mean_risk": float(row["mean_risk"]),
                    "max_risk": float(row["max_risk"]),
                    "distance": float(row["distance"]),
                    "elapsed_s": float(row["elapsed_s"]),
                }
            )
    return rows


def main(argv: List[str] | None = None) -> int:
    LOG_ENTRIES.clear()

    parser = argparse.ArgumentParser(description="Run automated CV validation pipeline.")
    parser.add_argument("--cfg", default=str(DEFAULT_CFG), help="Runtime config file (default: config/runtime.yaml)")
    parser.add_argument("--tidx", type=int, default=DEFAULT_TIDX, help="Time index for planner runs (default: 0)")
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA, help="Gamma override for planner runs (default: 0)")
    parser.add_argument(
        "--alphas",
        default=DEFAULT_ALPHAS,
        help="Comma-separated alpha_ice list for sweep (default: 0.01,...,0.99)",
    )
    args = parser.parse_args(argv)

    cfg_path = Path(args.cfg)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    if not cfg_path.exists():
        print(f"[SUITE] Config file not found: {cfg_path}")
        return 1

    try:
        runtime_config = _load_runtime_dict(cfg_path)
    except Exception as err:
        print(f"[SUITE] Failed to parse config: {err}")
        return 1

    data_block = runtime_config.get("data") or {}
    env_path = _resolve_project_path(data_block.get("env_nc"))
    mission_hint_raw = (runtime_config.get("predictor_params") or {}).get("cv_sat", {}).get("mission")
    mission_hint = str(mission_hint_raw).upper() if mission_hint_raw else None

    ctx: Dict[str, object] = {
        "cfg_path": cfg_path,
        "tidx": args.tidx,
        "gamma": args.gamma,
        "alphas": args.alphas,
        "artifacts": [],
        "config": runtime_config,
    }
    cli_module = get_cli_mod()
    step_records: List[StepStatus] = []
    failed = False
    failure_message: Optional[str] = None
    cache_failed_reason: Optional[str] = None
    env_ok = False

    # Step 1: Environment readiness
    try:
        _run_command([sys.executable, "scripts/check_env_ready.py"], "Environment check")
        env_report_path = PROJECT_ROOT / "logs" / "env_ready_report.json"
        if not env_report_path.exists():
            raise StepFailure("env_ready_report.json missing after environment check.")
        env_report = _load_json(env_report_path)
        ctx["env_report"] = env_report
        if env_report.get("status") != "pass":
            raise StepFailure("Environment readiness did not pass (MPC configuration may be incomplete).")
        step_records.append(StepStatus("Environment check", "pass", "Environment readiness OK."))
        env_ok = True
    except StepFailure as err:
        step_records.append(StepStatus("Environment check", "fail", str(err)))
        failed = True
        failure_message = str(err)

    # Step 2: Export ice cache
    ice_cache_nc = PROJECT_ROOT / "data_processed" / "cv_cache" / "ice_prob_latest.nc"
    ice_cache_json = PROJECT_ROOT / "data_processed" / "cv_cache" / "ice_prob_latest.json"
    if env_ok:
        def _attempt_export() -> Tuple[bool, Optional[str]]:
            base_cmd = [
                sys.executable,
                "scripts/export_ice_cache.py",
                "--cfg",
                str(cfg_path),
                "--tidx",
                str(args.tidx),
            ]
            proc = _run_command(base_cmd, "Export ice cache", allow_failure=True)
            if proc.returncode == 0:
                return True, None
            reason = f"exit code {proc.returncode}"
            if proc.stderr:
                reason = f"{reason}: {proc.stderr.strip()}"
            if env_path is None:
                return False, reason
            try:
                date_hint = _infer_date_from_env(env_path, args.tidx)
            except Exception as date_err:
                return False, f"{reason}; retry unavailable ({date_err})"
            retry_cmd = list(base_cmd)
            retry_cmd.extend(["--date", date_hint])
            if mission_hint:
                retry_cmd.extend(["--mission", mission_hint])
            suffix = f" --mission {mission_hint}" if mission_hint else ""
            print(f"[SUITE] Export retry with --date {date_hint}{suffix}")
            retry_proc = _run_command(retry_cmd, "Export ice cache (retry)", allow_failure=True)
            if retry_proc.returncode == 0:
                return True, None
            retry_reason = f"exit code {retry_proc.returncode}"
            if retry_proc.stderr:
                retry_reason = f"{retry_reason}: {retry_proc.stderr.strip()}"
            return False, f"{reason}; retry failed: {retry_reason}"

        if env_path is None or not env_path.exists():
            cache_failed_reason = "data.env_nc missing or path invalid"
            step_records.append(StepStatus("Export ice cache", "fail", cache_failed_reason))
            failed = True
        else:
            export_ok, export_reason = _attempt_export()
            if export_ok and ice_cache_nc.exists() and ice_cache_json.exists():
                ctx["ice_cache_json"] = _load_json(ice_cache_json)
                ctx["artifacts"].extend([ice_cache_nc, ice_cache_json])
                step_records.append(StepStatus("Export ice cache", "pass", "Ice probability cache exported."))
            else:
                cache_failed_reason = export_reason or "Ice probability cache export failed."
                ctx["cache_failed_reason"] = cache_failed_reason
                step_records.append(StepStatus("Export ice cache", "fail", cache_failed_reason))
                failed = True
    else:
        step_records.append(StepStatus("Export ice cache", "skipped", "Skipped due to environment failure."))

    # Step 3: Planner runs
    runs: Dict[str, dict] = {}
    planner_ok = False
    planner_notes: List[str] = []
    if env_ok:
        try:
            run_defs = [
                ("cv_probe", 0.0),
                ("cv_blend", 0.5),
            ]
            fallback_tags: List[str] = []
            for tag, alpha in run_defs:
                cmd = [
                    sys.executable,
                    "-m",
                    cli_module,
                    "plan",
                    "--cfg",
                    str(cfg_path),
                    "--predictor",
                    "cv_sat",
                    "--alpha-ice",
                    str(alpha),
                    "--tidx",
                    str(args.tidx),
                    "--gamma",
                    str(args.gamma),
                    "--tag",
                    tag,
                ]
                proc = _run_command(cmd, f"Planner run ({tag})", allow_failure=True)
                predictor_used = "cv_sat"
                if proc.returncode != 0:
                    fallback_reason = proc.stderr.strip() if proc.stderr else f"exit code {proc.returncode}"
                    planner_notes.append(f"{tag} fallback to env_nc ({fallback_reason})")
                    fallback_tags.append(tag)
                    fallback_cmd = [
                        sys.executable,
                        "-m",
                        cli_module,
                        "plan",
                        "--cfg",
                        str(cfg_path),
                        "--predictor",
                        "env_nc",
                        "--alpha-ice",
                        str(alpha),
                        "--tidx",
                        str(args.tidx),
                        "--gamma",
                        str(args.gamma),
                        "--tag",
                        tag,
                    ]
                    _run_command(fallback_cmd, f"Planner fallback ({tag})")
                    predictor_used = "env_nc"

                report_path = PROJECT_ROOT / "outputs" / f"run_report_{tag}.json"
                geojson_path = PROJECT_ROOT / "outputs" / f"route_{tag}.geojson"
                png_path = PROJECT_ROOT / "outputs" / f"route_on_risk_{tag}.png"
                for artifact in (report_path, geojson_path, png_path):
                    if not artifact.exists():
                        raise StepFailure(f"Missing planner artifact: {artifact}")

                report = _load_json(report_path)
                predictor_used = report.get("predictor", predictor_used)
                if predictor_used != "cv_sat":
                    planner_notes.append(f"{tag} predictor={predictor_used}")
                actual_alpha = report.get("alpha_ice")
                if actual_alpha is not None and abs(float(actual_alpha) - alpha) > 1e-6:
                    planner_notes.append(f"{tag} alpha mismatch (expected {alpha}, got {actual_alpha})")

                runs[tag] = {
                    "report": report,
                    "paths": {
                        "report": report_path,
                        "geojson": geojson_path,
                        "png": png_path,
                    },
                }
                ctx["artifacts"].extend([report_path, geojson_path, png_path])

            ctx["runs"] = runs
            detail = "Baseline and blended plans completed."
            status_label = "pass"
            if planner_notes:
                detail += " " + " ".join(planner_notes)
                status_label = "warn"
            if fallback_tags:
                failed = True
            step_records.append(StepStatus("Planner runs", status_label, detail))
            planner_ok = True
        except StepFailure as err:
            step_records.append(StepStatus("Planner runs", "fail", str(err)))
            failed = True
            failure_message = str(err)
    else:
        step_records.append(StepStatus("Planner runs", "skipped", "Skipped due to environment failure."))

    # Step 4: Compare artifacts
    compare_docs = [
        PROJECT_ROOT / "docs" / "compare_metrics.md",
        PROJECT_ROOT / "docs" / "compare_routes.png",
        PROJECT_ROOT / "docs" / "ice_prob_hist.png",
    ]
    if env_ok and planner_ok:
        try:
            base_report = PROJECT_ROOT / "outputs" / "run_report_cv_probe.json"
            blend_report = PROJECT_ROOT / "outputs" / "run_report_cv_blend.json"
            _run_command(
                [
                    sys.executable,
                    "scripts/compare_ice_blend.py",
                    "--base",
                    str(base_report),
                    "--blend",
                    str(blend_report),
                ],
                "Compare runs",
            )
            for path in compare_docs:
                if not path.exists():
                    raise StepFailure(f"Comparison artifact missing: {path}")
            ctx["artifacts"].extend(compare_docs)
            step_records.append(StepStatus("Comparison", "pass", "Comparison artifacts generated."))
        except StepFailure as err:
            step_records.append(StepStatus("Comparison", "fail", str(err)))
            failed = True
            failure_message = str(err)
    elif env_ok:
        step_records.append(StepStatus("Comparison", "skipped", "Skipped because planner runs failed."))
    else:
        step_records.append(StepStatus("Comparison", "skipped", "Skipped due to environment failure."))

    # Step 5: Alpha sweep
    sweep_outputs = [
        PROJECT_ROOT / "outputs" / "alpha_sweep.csv",
        PROJECT_ROOT / "docs" / "alpha_sweep_cost.png",
        PROJECT_ROOT / "docs" / "alpha_sweep_risk.png",
    ]
    alpha_rows: Optional[List[dict]] = None
    if env_ok:
        proc = _run_command(
            [
                sys.executable,
                "scripts/batch_alpha_sweep.py",
                "--cfg",
                str(cfg_path),
                "--gamma",
                str(args.gamma),
                "--tidx",
                str(args.tidx),
                "--alphas",
                args.alphas,
            ],
            "Alpha sweep",
            allow_failure=True,
        )
        if proc.returncode == 0:
            try:
                for path in sweep_outputs:
                    if not path.exists():
                        raise StepFailure(f"Alpha sweep artifact missing: {path}")
                alpha_rows = _read_alpha_sweep(sweep_outputs[0])
                ctx["alpha_sweep_rows"] = alpha_rows
                ctx["artifacts"].extend(sweep_outputs)
                step_records.append(StepStatus("Alpha sweep", "pass", "Alpha sweep completed."))
            except StepFailure as err:
                step_records.append(StepStatus("Alpha sweep", "fail", str(err)))
                failed = True
                failure_message = str(err)
        else:
            fallback_detail = "Alpha sweep failed; fallback baseline CSV generated (plots skipped)."
            fallback_row: dict
            baseline = runs.get("cv_probe") if isinstance(runs, dict) else None
            if baseline:
                report = baseline.get("report") or {}
                distance_km = float(report.get("geodesic_length_m", float("nan"))) / 1000.0
                fallback_row = {
                    "alpha_ice": float(report.get("alpha_ice", 0.0) or 0.0),
                    "total_cost": float(report.get("total_cost", float("nan"))),
                    "mean_risk": float(report.get("mean_risk", float("nan"))),
                    "max_risk": float(report.get("max_risk", float("nan"))),
                    "distance": distance_km,
                    "elapsed_s": float(report.get("elapsed_s", float("nan"))),
                }
            else:
                fallback_row = {
                    "alpha_ice": 0.0,
                    "total_cost": float("nan"),
                    "mean_risk": float("nan"),
                    "max_risk": float("nan"),
                    "distance": float("nan"),
                    "elapsed_s": float("nan"),
                }
            csv_path = sweep_outputs[0]
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            with csv_path.open("w", newline="", encoding="utf-8") as fh:
                fieldnames = ["alpha_ice", "total_cost", "mean_risk", "max_risk", "distance", "elapsed_s"]
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(fallback_row)
            alpha_rows = [fallback_row]
            ctx["alpha_sweep_rows"] = alpha_rows
            ctx["artifacts"].append(csv_path)
            step_records.append(StepStatus("Alpha sweep", "warn", fallback_detail))
            failed = True
    else:
        step_records.append(StepStatus("Alpha sweep", "skipped", "Skipped due to environment failure."))

    # Persist log output
    ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    ERROR_LOG_PATH.write_text("\n\n".join(LOG_ENTRIES) + "\n", encoding="utf-8")
    ctx["artifacts"].append(ERROR_LOG_PATH)

    # Summary
    summary_path = PROJECT_ROOT / "logs" / "cv_test_summary.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    env_report = ctx.get("env_report")
    ice_summary = ctx.get("ice_cache_json")
    runs_data = ctx.get("runs")
    alpha_rows = ctx.get("alpha_sweep_rows") if alpha_rows is None else alpha_rows

    def fmt(value: Optional[float], precision: int = 3) -> str:
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            return "n/a"
        return f"{value:.{precision}f}"

    lines: List[str] = ["# CV Test Suite Summary", ""]

    lines.append("## Step Status")
    lines.append("")
    lines.append("| Step | Status | Notes |")
    lines.append("| --- | --- | --- |")
    for record in step_records:
        lines.append(f"| {record.name} | {record.status.upper()} | {record.detail} |")
    lines.append("")

    lines.append("## Environment")
    if isinstance(env_report, dict):
        lines.append(f"- Status: **{env_report.get('status', 'unknown')}**")
        for entry in env_report.get("checks") or []:
            if entry.get("name") in {"DEFAULT_STAC_SOURCE", "MPC_STAC_URL"}:
                lines.append(
                    f"- {entry.get('name')}: {entry.get('status', 'N/A').upper()} - {entry.get('message', '')}"
                )
    else:
        lines.append("- Environment report unavailable (step failed).")
    lines.append("")

    lines.append("## Ice Probability Cache")
    if isinstance(ice_summary, dict):
        lines.append(
            "- Stats: mean={mean:.4f}, max={max:.4f}, coverage={coverage:.2f}%, valid={valid:.2f}%".format(
                mean=ice_summary.get("mean", float("nan")),
                max=ice_summary.get("max", float("nan")),
                coverage=ice_summary.get("coverage_pct", float("nan")),
                valid=ice_summary.get("valid_ratio_pct", float("nan")),
            )
        )
        lines.append(f"- SHA1: `{ice_summary.get('sha1', 'n/a')}`")
        lines.append(f"- Cache NC: `{ice_summary.get('path', 'n/a')}`")
    elif cache_failed_reason:
        lines.append(f"- Status: **FAILED** ({cache_failed_reason}) - histogram skipped.")
    else:
        lines.append("- Cache export unavailable.")
    lines.append("")

    lines.append("## Planner Comparison")
    if isinstance(runs_data, dict) and "cv_probe" in runs_data and "cv_blend" in runs_data:
        probe = runs_data["cv_probe"]["report"]
        blend = runs_data["cv_blend"]["report"]
        metrics = [
            ("total_cost", "Total cost", 3, 1.0),
            ("mean_risk", "Mean risk", 4, 1.0),
            ("max_risk", "Max risk", 4, 1.0),
            ("geodesic_length_m", "Distance (km)", 3, 0.001),
        ]
        lines.append("")
        lines.append("| Metric | cv_probe | cv_blend | diff (blend - probe) |")
        lines.append("| --- | --- | --- | --- |")
        for key, label, precision, scale in metrics:
            probe_val = probe.get(key)
            blend_val = blend.get(key)
            probe_fmt = fmt(probe_val * scale if probe_val is not None else None, precision)
            blend_fmt = fmt(blend_val * scale if blend_val is not None else None, precision)
            if probe_val is None or blend_val is None:
                delta_fmt = "n/a"
            else:
                delta_fmt = fmt((blend_val - probe_val) * scale, precision)
            lines.append(f"| {label} | {probe_fmt} | {blend_fmt} | {delta_fmt} |")
        lines.append("")
        lines.append(
            f"- Reports: `{_relative(runs_data['cv_probe']['paths']['report'])}`, "
            f"`{_relative(runs_data['cv_blend']['paths']['report'])}`"
        )
        lines.append(
            f"- Route plots: `{_relative(runs_data['cv_probe']['paths']['png'])}`, "
            f"`{_relative(runs_data['cv_blend']['paths']['png'])}`"
        )
    else:
        lines.append("- Planner comparison unavailable.")
    lines.append("")

    lines.append("## Alpha Sweep")
    if isinstance(alpha_rows, list) and alpha_rows:
        best = min(alpha_rows, key=lambda row: row["total_cost"])
        lines.append(
            "- Best alpha_ice={alpha:.2f} with total_cost={cost:.3f}, mean_risk={mean:.4f}".format(
                alpha=best["alpha_ice"],
                cost=best["total_cost"],
                mean=best["mean_risk"],
            )
        )
        lines.append("- Metrics CSV: `outputs/alpha_sweep.csv`")
        if all(path.exists() for path in sweep_outputs[1:]):
            lines.append("- Cost trend: ![Cost](../docs/alpha_sweep_cost.png)")
            lines.append("- Risk trend: ![Risk](../docs/alpha_sweep_risk.png)")
        else:
            lines.append("- Cost/Risk plots unavailable (fallback run).")
    else:
        lines.append("- Alpha sweep unavailable.")
    lines.append("")

    lines.append("## Logs")
    lines.append(f"- Detailed command output: `{_relative(ERROR_LOG_PATH)}`")
    lines.append("")

    lines.append("## Artifacts")
    artifacts = ctx.get("artifacts", [])
    if artifacts:
        seen = set()
        for path in artifacts:
            rel = _relative(Path(path))
            if rel in seen:
                continue
            lines.append(f"- `{rel}`")
            seen.add(rel)
    else:
        lines.append("- No artifacts recorded.")
    lines.append("")

    if failure_message:
        lines.append("## Failure")
        lines.append("")
        lines.append(f"- {failure_message}")
        lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[SUITE] Summary written to {summary_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
