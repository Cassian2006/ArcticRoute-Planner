from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Dict, Optional

def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())

def _run(cmd: list[str], timeout: int = 30) -> tuple[int, str, str]:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr

def detect_polarroute_cli() -> Dict[str, Any]:
    # Heuristic: polarroute CLI name might vary; keep minimal.
    candidates = [
        ["polarroute", "--help"],
        ["polarroute-cli", "--help"],
    ]
    for c in candidates:
        try:
            rc, out, err = _run(c, timeout=15)
            if rc == 0 and ("usage" in (out+err).lower() or "polar" in (out+err).lower()):
                return {"ok": True, "cmd": c[0], "rc": rc, "stdout": out[:800], "stderr": err[:800]}
        except Exception as e:
            continue
    return {"ok": False}

def detect_pipeline_cli(pipeline_dir: Optional[str]) -> Dict[str, Any]:
    if not pipeline_dir:
        return {"ok": False, "reason": "pipeline_dir_not_set"}
    d = Path(pipeline_dir)
    if not d.exists():
        return {"ok": False, "reason": "pipeline_dir_missing", "pipeline_dir": pipeline_dir}
    # Prefer existing doctor script if present
    try:
        rc, out, err = _run([sys.executable, "-m", "scripts.polarroute_pipeline_doctor", "--pipeline-dir", str(d)], timeout=30)
        ok = (rc == 0)
        return {"ok": ok, "rc": rc, "stdout": out[-1200:], "stderr": err[-1200:], "pipeline_dir": str(d)}
    except Exception as e:
        return {"ok": False, "reason": f"doctor_failed: {e}", "pipeline_dir": str(d)}

def choose_backend(mode: str, pipeline_dir: Optional[str], vessel_mesh: Optional[str], route_config: Optional[str]) -> Dict[str, Any]:
    """
    mode:
      auto | astar | polarroute_pipeline | polarroute_external
    """
    meta: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "requested_mode": mode,
        "planner_used": None,
        "planner_mode": None,
        "fallback_reason": None,
        "pipeline_dir": pipeline_dir,
        "external_vessel_mesh": vessel_mesh,
        "external_route_config": route_config,
    }

    if mode == "astar":
        meta["planner_used"] = "astar"
        meta["planner_mode"] = "astar"
        return meta

    if mode == "polarroute_pipeline":
        pipe = detect_pipeline_cli(pipeline_dir)
        if pipe.get("ok"):
            meta["planner_used"] = "polarroute"
            meta["planner_mode"] = "pipeline"
            return meta
        meta["planner_used"] = "astar"
        meta["planner_mode"] = "astar"
        meta["fallback_reason"] = f"pipeline_unavailable: {pipe.get('reason') or pipe.get('rc')}"
        return meta

    if mode == "polarroute_external":
        # Require files
        if not vessel_mesh or not route_config:
            meta["planner_used"] = "astar"
            meta["planner_mode"] = "astar"
            meta["fallback_reason"] = "external_files_missing"
            return meta
        if not Path(vessel_mesh).exists() or not Path(route_config).exists():
            meta["planner_used"] = "astar"
            meta["planner_mode"] = "astar"
            meta["fallback_reason"] = "external_files_not_found"
            return meta
        # If CLI present, use it; otherwise fallback
        cli = detect_polarroute_cli()
        if cli.get("ok"):
            meta["planner_used"] = "polarroute"
            meta["planner_mode"] = "external"
            return meta
        meta["planner_used"] = "astar"
        meta["planner_mode"] = "astar"
        meta["fallback_reason"] = "polarroute_cli_unavailable"
        return meta

    # auto
    pipe = detect_pipeline_cli(pipeline_dir)
    if pipe.get("ok"):
        meta["planner_used"] = "polarroute"
        meta["planner_mode"] = "pipeline"
        return meta

    # external as second choice if both files provided
    if vessel_mesh and route_config and Path(vessel_mesh).exists() and Path(route_config).exists():
        cli = detect_polarroute_cli()
        if cli.get("ok"):
            meta["planner_used"] = "polarroute"
            meta["planner_mode"] = "external"
            meta["fallback_reason"] = f"pipeline_unavailable: {pipe.get('reason') or pipe.get('rc')}"
            return meta

    meta["planner_used"] = "astar"
    meta["planner_mode"] = "astar"
    meta["fallback_reason"] = f"pipeline_unavailable: {pipe.get('reason') or pipe.get('rc')}"
    return meta

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="auto", choices=["auto","astar","polarroute_pipeline","polarroute_external"])
    ap.add_argument("--pipeline-dir", default=os.environ.get("POLARROUTE_PIPELINE_DIR", None))
    ap.add_argument("--vessel-mesh", default=None)
    ap.add_argument("--route-config", default=None)
    ap.add_argument("--out-json", default="reports/polarroute_selection.json")
    args = ap.parse_args()

    meta = choose_backend(args.mode, args.pipeline_dir, args.vessel_mesh, args.route_config)
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()

