#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare baseline vs ice-blend runs and generate docs/plots.

@role: analysis
"""

"""
Compare two planner runs (baseline vs ice-blend) and generate documentation artifacts.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    from skimage.filters import threshold_otsu
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    threshold_otsu = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
ICE_PROB_CACHE = ROOT / "data_processed" / "cv_cache" / "ice_prob_latest.nc"
DEFAULT_DOCS_DIR = ROOT / "docs"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _derive_tag(path: Path) -> Optional[str]:
    stem = path.stem
    if stem.startswith("run_report_"):
        return stem[len("run_report_") :]
    return None


def _load_route(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if not path.exists():
        return None
    try:
        data = _load_json(path)
    except Exception:
        return None
    features = data.get("features")
    if not isinstance(features, list) or not features:
        return None
    geometry = features[0].get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "LineString":
        return None
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        return None
    lon = []
    lat = []
    for item in coordinates:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            lon.append(float(item[0]))
            lat.append(float(item[1]))
    if not lon:
        return None
    return np.array(lon, dtype=np.float64), np.array(lat, dtype=np.float64)


def _load_report_bundle(run_report_path: Path) -> Dict[str, Optional[object]]:
    run_data = _load_json(run_report_path)
    tag = _derive_tag(run_report_path)
    parent = run_report_path.parent
    geojson_path = parent / f"route_{tag}.geojson" if tag else None
    png_path = parent / f"route_on_risk_{tag}.png" if tag else None
    bundle = {
        "report": run_data,
        "geojson_path": geojson_path if geojson_path and geojson_path.exists() else None,
        "png_path": png_path if png_path and png_path.exists() else None,
        "tag": tag,
    }
    return bundle


def _metric_value(data: dict, keys: Sequence[str]) -> Optional[float]:
    current: object = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _format_number(value: Optional[float]) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    if abs(value) >= 1e6:
        return f"{value/1e6:.3f}M"
    if abs(value) >= 1e3:
        return f"{value/1e3:.3f}K"
    return f"{value:.3f}"


def _ensure_docs_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_markdown_table(path: Path, rows: Iterable[Tuple[str, str, str, str]]) -> None:
    lines = [
        "# Ice Blend Comparison",
        "",
        "| Metric | Base | Blend | Δ (Blend-Base) |",
        "| --- | --- | --- | --- |",
    ]
    for metric, base, blend, delta in rows:
        lines.append(f"| {metric} | {base} | {blend} | {delta} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _optionally_warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def _load_cached_ice_prob_values() -> Optional[Tuple[np.ndarray, Dict[str, float], Path]]:
    if not ICE_PROB_CACHE.exists():
        return None
    try:
        with xr.open_dataset(ICE_PROB_CACHE) as ds:
            if "ice_prob" not in ds:
                _optionally_warn(f"ice_prob variable missing from {ICE_PROB_CACHE}")
                return None
            array = np.asarray(ds["ice_prob"].values, dtype=np.float32)
    except Exception as err:  # pragma: no cover - IO heavy
        _optionally_warn(f"Failed to read cached ice_prob from {ICE_PROB_CACHE}: {err}")
        return None

    finite = np.isfinite(array)
    finite_count = int(np.count_nonzero(finite))
    if finite_count == 0:
        _optionally_warn("Cached ice_prob contains no finite values; histogram skipped.")
        return None

    values = array[finite]
    total = array.size if array.size else 1
    stats = {
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
        "coverage_pct": float(np.mean(values > 0.5) * 100.0),
        "valid_ratio_pct": float(finite_count / total * 100.0),
        "sample_count": float(values.size),
    }
    return values, stats, ICE_PROB_CACHE


def _plot_route_overlay(
    base_coords: Optional[Tuple[np.ndarray, np.ndarray]],
    blend_coords: Optional[Tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    has_any = False
    if base_coords is not None:
        ax.plot(base_coords[0], base_coords[1], color="gray", linewidth=2.0, label="Base route")
        has_any = True
    if blend_coords is not None:
        ax.plot(blend_coords[0], blend_coords[1], color="red", linewidth=2.0, label="Blend route")
        has_any = True
    if not has_any:
        fig.text(0.5, 0.5, "No routes available", ha="center", va="center")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Route Comparison")
    if has_any:
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_ice_prob_hist(values: np.ndarray, stats: Dict[str, float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = 100
    ax.hist(values, bins=bins, color="#4472C4", alpha=0.85)
    threshold = None
    if threshold_otsu is not None and values.size > 0:
        try:
            threshold = float(threshold_otsu(values))
        except Exception:
            threshold = None
    if threshold is not None and np.isfinite(threshold):
        ax.axvline(threshold, color="red", linestyle="--", label=f"Otsu threshold = {threshold:.3f}")
        ax.legend()
    sample_count = int(round(stats.get("sample_count", float(values.size))))
    annotation = f"samples={sample_count}"
    if threshold is not None and np.isfinite(threshold):
        annotation += f"\notsu={threshold:.3f}"
    ax.text(0.98, 0.95, annotation, ha="right", va="top", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("Ice Probability")
    ax.set_ylabel("Frequency")
    ax.set_title("Ice Probability - Global Histogram")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compare baseline vs ice blend planner runs.")
    parser.add_argument("--base", required=True, help="Path to baseline run_report JSON")
    parser.add_argument("--blend", required=True, help="Path to ice blend run_report JSON")
    parser.add_argument("--docs-dir", default=str(DEFAULT_DOCS_DIR), help="Output directory for docs (default: docs/)")
    args = parser.parse_args(argv)

    base_path = Path(args.base).resolve()
    blend_path = Path(args.blend).resolve()
    docs_dir = _ensure_docs_dir(Path(args.docs_dir).resolve())

    if not base_path.exists():
        parser.error(f"Base run report not found: {base_path}")
    if not blend_path.exists():
        parser.error(f"Blend run report not found: {blend_path}")

    base_bundle = _load_report_bundle(base_path)
    blend_bundle = _load_report_bundle(blend_path)

    metrics = [
        ("Total Cost", ("total_cost",)),
        ("Mean Risk", ("mean_risk",)),
        ("Max Risk", ("max_risk",)),
        ("Distance (km)", ("geodesic_length_m",)),
        ("Fuel Proxy", ("fuel_proxy",)),
    ]

    rows = []
    for label, key_path in metrics:
        base_val = _metric_value(base_bundle["report"], key_path)
        blend_val = _metric_value(blend_bundle["report"], key_path)
        if key_path == ("geodesic_length_m",):
            base_fmt = _format_number(base_val / 1000.0 if base_val is not None else None)
            blend_fmt = _format_number(blend_val / 1000.0 if blend_val is not None else None)
            delta_val = (blend_val - base_val) / 1000.0 if base_val is not None and blend_val is not None else None
        else:
            base_fmt = _format_number(base_val)
            blend_fmt = _format_number(blend_val)
            delta_val = blend_val - base_val if base_val is not None and blend_val is not None else None
        rows.append((label, base_fmt, blend_fmt, _format_number(delta_val)))

    metrics_md = docs_dir / "compare_metrics.md"
    _write_markdown_table(metrics_md, rows)
    print(f"[INFO] Metrics table written to {metrics_md}")

    base_route = (
        _load_route(base_bundle["geojson_path"])
        if base_bundle["geojson_path"]
        else None
    )
    blend_route = (
        _load_route(blend_bundle["geojson_path"])
        if blend_bundle["geojson_path"]
        else None
    )

    compare_routes_png = docs_dir / "compare_routes.png"
    _plot_route_overlay(base_route, blend_route, compare_routes_png)
    print(f"[INFO] Route comparison saved to {compare_routes_png}")

    cache_result = _load_cached_ice_prob_values()
    if cache_result is None:
        _optionally_warn(
            "No cached ice_prob field found; run "
            "`python scripts/export_ice_cache.py --cfg config/runtime.yaml --tidx 0 --date YYYY-MM-DD --mission S2` "
            "to export cache (histogram skipped)."
        )
    else:
        values, stats, cache_path = cache_result
        hist_path = docs_dir / "ice_prob_hist.png"
        _plot_ice_prob_hist(values, stats, hist_path)
        sample_count = int(round(stats.get("sample_count", float(values.size))))
        print(
            "[INFO] Ice probability histogram saved to {path} "
            "(mean={mean:.4f}, max={max:.4f}, coverage={cov:.2f}%, valid={valid:.2f}%, samples={samples}, source={src})".format(
                path=hist_path,
                mean=stats.get("mean", float("nan")),
                max=stats.get("max", float("nan")),
                cov=stats.get("coverage_pct", float("nan")),
                valid=stats.get("valid_ratio_pct", float("nan")),
                samples=sample_count,
                src=cache_path,
            )
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

