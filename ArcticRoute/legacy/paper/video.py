from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
PAPER_DIR = ARCTIC_DIR / "reports" / "paper"
VID_DIR = PAPER_DIR / "videos"


def _write_meta(out: Path, logical_id: str, inputs: List[str]) -> None:
    try:
        meta = {
            "logical_id": logical_id,
            "inputs": inputs,
            "run_id": time.strftime("%Y%m%dT%H%M%S"),
            "config_hash": hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode("utf-8")).hexdigest(),
        }
        out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def make_timeline(layer_paths: List[Path], out: Path, fps: int = 4, fmt: str = "mp4") -> Path:
    # REUSE report.animate.animate_layers
    from ArcticRoute.core.reporting.animate import animate_layers  # REUSE
    VID_DIR.mkdir(parents=True, exist_ok=True)
    out = VID_DIR / out.name if not out.is_absolute() else out
    out.parent.mkdir(parents=True, exist_ok=True)
    res = animate_layers(layer_paths, out, fps=fps, side_by_side=(len(layer_paths) > 1), overlay_routes=None, fmt=fmt)
    _write_meta(res, res.name, [str(p) for p in layer_paths])
    return res


def make_route_compare(baseline: Path, candidate: Path, out: Path, ym: Optional[str] = None, fmt: str = "mp4") -> Path:
    # 使用风险层作为背景，叠加两条路线；若缺失则占位
    VID_DIR.mkdir(parents=True, exist_ok=True)
    out = VID_DIR / out.name if not out.is_absolute() else out
    # 背景层：risk_fused_ym.nc（如果传了 ym）
    layers: List[Path] = []
    if ym:
        risk_nc = ARCTIC_DIR / "data_processed" / "risk" / f"risk_fused_{ym}.nc"
        if risk_nc.exists():
            layers.append(risk_nc)
    # REUSE animate with overlay routes
    try:
        from ArcticRoute.core.reporting.animate import animate_layers  # REUSE
        res = animate_layers(layers, out, fps=1, side_by_side=False, overlay_routes=[baseline, candidate], fmt=fmt)
    except Exception:
        # 写占位文件
        out.write_text(f"route_compare placeholder: {baseline} vs {candidate}\n", encoding="utf-8")
        res = out
    _write_meta(res, res.name, [str(baseline), str(candidate)] + [str(p) for p in layers])
    return res


__all__ = ["make_timeline", "make_route_compare"]






