from __future__ import annotations

# Phase H | Animations
# animate_layers(layers: List[Path], out: Path, fps=4, side_by_side=False, overlay_routes: Optional[List[Path]]=None)

import json
import os
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

try:
    import xarray as xr  # type: ignore
except Exception:  # pragma: no cover
    xr = None  # type: ignore

try:
    import imageio.v2 as imageio  # type: ignore
except Exception:  # pragma: no cover
    imageio = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"


def _frames_from_nc(path: Path) -> List[np.ndarray]:
    if xr is None:
        raise RuntimeError("xarray required")
    ds = xr.open_dataset(path)
    # pick first var
    var = ds[list(ds.data_vars)[0]]
    if "time" in var.dims and int(var.sizes.get("time", 0)) > 1:
        arr = np.asarray(var.values, dtype=float)
        frames = [np.squeeze(arr[i]) for i in range(arr.shape[0])]
    else:
        arr = np.asarray((var.isel(time=0).values if "time" in var.dims else var.values), dtype=float)
        frames = [np.squeeze(arr)]
    try:
        ds.close()
    except Exception:
        pass
    # normalize to 0-255
    outs: List[np.ndarray] = []
    for f in frames:
        A = np.asarray(f, dtype=float)
        finite = A[np.isfinite(A)]
        if finite.size:
            lo, hi = np.nanpercentile(finite, 1), np.nanpercentile(finite, 99)
            if hi > lo:
                B = (np.clip(A, lo, hi) - lo) / (hi - lo)
            else:
                B = np.zeros_like(A)
        else:
            B = np.zeros_like(A)
        outs.append((B * 255).astype(np.uint8))
    return outs


def animate_layers(layers: List[Path], out: Path, fps: int = 4, side_by_side: bool = False, overlay_routes: Optional[List[Path]] = None, fmt: str = "gif") -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames: List[np.ndarray] = []
    for p in layers:
        if not p or not Path(p).exists():
            continue
        fs = _frames_from_nc(Path(p))
        if side_by_side:
            # 对齐高度，水平拼接
            if not frames:
                frames = [f for f in fs]
            else:
                # zip 对齐帧数；若不同长度，截断到最小
                n = min(len(frames), len(fs))
                frames = [np.hstack([frames[i][:, :, None], fs[i][:, :, None]]).squeeze() for i in range(n)]
        else:
            frames.extend(fs)
    if not frames:
        # 保障至少有一个空白帧
        frames = [np.zeros((64, 64), dtype=np.uint8)]
    # 写动图（mp4/gif）
    if imageio is None:
        # 退化为用 matplotlib 写单帧 PNG
        if plt is not None:
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(frames[0], cmap="inferno")
            fig.tight_layout()
            fig.savefig(out.with_suffix(".png"), dpi=120)
            plt.close(fig)
            outp = out.with_suffix(".png")
        else:
            # 写原始 NPY
            np.save(out.with_suffix(".npy"), frames[0])
            outp = out.with_suffix(".npy")
    else:
        try:
            if fmt == "mp4":
                imageio.mimsave(str(out), frames, fps=max(1, int(fps)))
            else:
                imageio.mimsave(str(out), frames, duration=max(0.01, 1.0 / max(1, int(fps))))
            outp = out
        except Exception:
            # 回退 GIF
            fallback = out.with_suffix(".gif")
            imageio.mimsave(str(fallback), frames, duration=max(0.01, 1.0 / max(1, int(fps))))
            outp = fallback
    # meta
    try:
        with open(str(outp) + ".meta.json", "w", encoding="utf-8") as f:
            import time, json, subprocess, hashlib
            def _git_sha():
                try:
                    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
                except Exception:
                    return "unknown"
            cfg_hash = hashlib.sha256(json.dumps([str(p) for p in layers], ensure_ascii=False).encode("utf-8")).hexdigest()[:16]
            json.dump({
                "logical_id": Path(outp).name,
                "inputs": [str(p) for p in layers],
                "run_id": time.strftime("%Y%m%dT%H%M%S"),
                "git_sha": _git_sha(),
                "config_hash": cfg_hash,
            }, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return outp


__all__ = ["animate_layers"]

