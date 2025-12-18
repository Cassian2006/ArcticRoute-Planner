"""
Phase 12: 对照实验套件（CMEMS × POLARIS）

执行 4 组对照实验：
  A: demo_env + POLARIS OFF
  B: demo_env + POLARIS ON
  C: cmems_latest(本地缓存/可回退) + POLARIS OFF
  D: cmems_latest(本地缓存/可回退) + POLARIS ON

产出：
- summary.csv: 汇总所有场景的指标
- 每个场景的独立产物文件夹
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclass(frozen=True)
class Scenario:
    key: str
    env_source: str  # "demo" or "cmems_latest"
    polaris_enabled: bool


def _write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_demo_summary_txt(p: Path) -> Dict[str, Any]:
    """
    Parse summary.txt produced by demo_end_to_end.
    Keep it robust: if fields missing, return partial.
    """
    out: Dict[str, Any] = {}
    if not p.exists():
        return out
    txt = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in txt:
        line = line.strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        out[k.strip().lower().replace(" ", "_")] = v.strip()
    return out


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _collect_metrics(run_dir: Path) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}

    # cost_breakdown.json (preferred)
    cb = run_dir / "cost_breakdown.json"
    if cb.exists():
        try:
            obj = json.loads(cb.read_text(encoding="utf-8", errors="ignore"))
            # Common keys (best-effort)
            metrics["distance_km"] = _safe_float(
                obj.get("distance_km")
                or obj.get("distance")
                or obj.get("distanceKm")
                or obj.get("total_distance_km")
            )
            metrics["total_cost"] = _safe_float(
                obj.get("total_cost") or obj.get("cost_total") or obj.get("total")
            )
            # EDL
            metrics["edl_risk"] = _safe_float(obj.get("edl_risk"))
            metrics["edl_uncertainty"] = _safe_float(obj.get("edl_uncertainty"))
            # Component sums if present
            comps = obj.get("components") or {}
            if isinstance(comps, dict):
                for k in [
                    "ice_risk",
                    "wave_risk",
                    "ais_risk",
                    "ais_density",
                    "ais_corridor",
                    "base_distance",
                    "ice_resistance",
                ]:
                    if k in comps:
                        metrics[k] = _safe_float(comps.get(k))
        except Exception as e:
            metrics["cost_breakdown_parse_error"] = str(e)

    # polaris_diagnostics.csv (if exists)
    pd = run_dir / "polaris_diagnostics.csv"
    if pd.exists():
        # Keep lightweight: count levels by scanning text
        txt = pd.read_text(encoding="utf-8", errors="ignore").splitlines()
        if txt:
            header = txt[0].split(",")
            lvl_idx = header.index("level") if "level" in header else -1
            rio_idx = header.index("rio") if "rio" in header else -1
            levels = []
            rios = []
            for row in txt[1:]:
                cols = row.split(",")
                if lvl_idx >= 0 and lvl_idx < len(cols):
                    levels.append(cols[lvl_idx].strip())
                if rio_idx >= 0 and rio_idx < len(cols):
                    try:
                        rios.append(float(cols[rio_idx]))
                    except Exception:
                        pass
            n = max(1, len(levels))
            metrics["polaris_points"] = len(levels)
            metrics["polaris_special_fraction"] = (
                sum(1 for x in levels if x == "special") / n
            )
            metrics["polaris_elevated_fraction"] = (
                sum(1 for x in levels if x == "elevated") / n
            )
            if rios:
                metrics["polaris_rio_min"] = float(min(rios))
                metrics["polaris_rio_mean"] = float(np.mean(rios))

    # summary.txt (fallback)
    s = run_dir / "summary.txt"
    metrics.update(_parse_demo_summary_txt(s))

    return metrics


def _run_demo_end_to_end(
    run_dir: Path,
    env_source: str,
    polaris_enabled: bool,
    bbox: list[float] | None = None,
    days: int = 2,
    seed: int = 42,
) -> str:
    """
    调用 demo_end_to_end.py 脚本（通过 python -m 方式）。
    返回状态：ok, error
    """
    try:
        cmd = [
            sys.executable,
            "-m",
            "scripts.demo_end_to_end",
            "--outdir",
            str(run_dir),
            "--env-source",
            env_source,
            "--polaris-enabled" if polaris_enabled else "--polaris-disabled",
            "--seed",
            str(seed),
            "--days",
            str(days),
        ]
        
        if bbox:
            cmd.extend(["--bbox"] + [str(x) for x in bbox])
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        
        # 保存 stdout 和 stderr
        (run_dir / "run_stdout.txt").write_text(result.stdout, encoding="utf-8")
        (run_dir / "run_stderr.txt").write_text(result.stderr, encoding="utf-8")
        
        if result.returncode == 0:
            return "ok"
        else:
            (run_dir / "run_error.txt").write_text(
                f"Return code: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
                encoding="utf-8",
            )
            return "error"
    except subprocess.TimeoutExpired:
        (run_dir / "run_error.txt").write_text("Timeout after 300s", encoding="utf-8")
        return "error"
    except Exception as e:
        (run_dir / "run_error.txt").write_text(str(e), encoding="utf-8")
        return "error"


def _sync_cmems_to_newenv() -> tuple[bool, Optional[str]]:
    """
    尝试同步 CMEMS 最新缓存到 newenv。
    返回 (成功?, 错误信息)
    """
    try:
        script_path = Path("scripts/cmems_newenv_sync.py")
        if not script_path.exists():
            return False, "cmems_newenv_sync.py not found (graceful skip)"
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            return True, None
        else:
            return False, f"Return code {result.returncode}: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Timeout after 120s"
    except Exception as e:
        return False, str(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--outdir", default="reports/ablation", help="Output root directory"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        default=None,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
    )
    ap.add_argument("--days", type=int, default=2)
    args = ap.parse_args()

    out_root = Path(args.outdir)
    out_root.mkdir(parents=True, exist_ok=True)

    scenarios = [
        Scenario("A_demo_no_polaris", "demo", False),
        Scenario("B_demo_polaris", "demo", True),
        Scenario("C_cmems_no_polaris", "cmems_latest", False),
        Scenario("D_cmems_polaris", "cmems_latest", True),
    ]

    # record suite config
    _write_json(
        out_root / "suite_config.json",
        {
            "seed": args.seed,
            "bbox": args.bbox,
            "days": args.days,
            "scenarios": [
                {"key": s.key, "env_source": s.env_source, "polaris_enabled": s.polaris_enabled}
                for s in scenarios
            ],
            "notes": "CMEMS uses local cache/newenv sync when possible; should gracefully fallback if unavailable.",
        },
    )

    rows = []
    for sc in scenarios:
        print(f"\n{'='*60}")
        print(f"Running scenario: {sc.key}")
        print(f"  env_source: {sc.env_source}")
        print(f"  polaris_enabled: {sc.polaris_enabled}")
        print(f"{'='*60}")

        run_dir = out_root / sc.key
        run_dir.mkdir(parents=True, exist_ok=True)

        # If CMEMS scenario: attempt to sync latest cache to newenv (best-effort)
        cmems_sync_ok = False
        cmems_sync_err = None
        if sc.env_source == "cmems_latest":
            print("  [CMEMS] Attempting to sync latest cache to newenv...")
            cmems_sync_ok, cmems_sync_err = _sync_cmems_to_newenv()
            if cmems_sync_ok:
                print("  [CMEMS] Sync successful")
            else:
                print(f"  [CMEMS] Sync failed: {cmems_sync_err}")
                print("  [CMEMS] Will proceed with existing data (graceful fallback)")

        # run demo end-to-end
        print(f"  Running demo_end_to_end.py...")
        status = _run_demo_end_to_end(
            run_dir=run_dir,
            env_source=sc.env_source,
            polaris_enabled=sc.polaris_enabled,
            bbox=args.bbox,
            days=args.days,
            seed=args.seed,
        )
        print(f"  Status: {status}")

        metrics = _collect_metrics(run_dir)
        row = {
            "key": sc.key,
            "env_source": sc.env_source,
            "polaris_enabled": sc.polaris_enabled,
            "env_source_requested": sc.env_source,
            "polaris_enabled_requested": sc.polaris_enabled,
            "status": status,
            "cmems_sync_ok": cmems_sync_ok,
            "cmems_sync_err": cmems_sync_err,
            **metrics,
        }
        rows.append(row)

        _write_json(run_dir / "run_meta.json", row)

    # write summary.csv (simple, no pandas)
    keys = sorted({k for r in rows for k in r.keys()})
    csv_lines = [",".join(keys)]
    for r in rows:
        csv_lines.append(
            ",".join(
                ""
                if r.get(k) is None
                else str(r.get(k)).replace("\n", " ").replace(",", ";")
                for k in keys
            )
        )
    (out_root / "summary.csv").write_text(
        "\n".join(csv_lines) + "\n", encoding="utf-8"
    )

    print(f"\n{'='*60}")
    print(f"[Phase12] Wrote {out_root/'summary.csv'}")
    print(f"{'='*60}")
    for r in rows:
        print(f"  {r['key']}: {r['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

