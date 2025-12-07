from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Literal, Sequence
from contextlib import suppress

import numpy as np
import streamlit as st
import xarray as xr

try:  # optional dependencies for hotspot map
    import pandas as pd  # type: ignore
    import pydeck as pdk  # type: ignore
except Exception:  # pragma: no cover - optional
    pd = None
    pdk = None

try:  # optional dependencies for map overlays
    import folium  # type: ignore
    from streamlit_folium import st_folium  # type: ignore
except Exception:  # pragma: no cover - optional
    folium = None  # type: ignore
    st_folium = None  # type: ignore

try:
    import rasterio
except Exception:  # pragma: no cover - optional
    rasterio = None  # type: ignore

try:
    from matplotlib import cm
except Exception:  # pragma: no cover - optional
    cm = None  # type: ignore

try:
    from skimage.filters import threshold_otsu  # type: ignore
except Exception:  # pragma: no cover - optional
    threshold_otsu = None  # type: ignore

from api.cli import (
    load_yaml_file,
    model_dump,
    run_plan,
    validate_runtime_config,
)
from ai.planner_advisor import llm_advice, rule_advice
from ai.schema import AdvisorInput, advisor_input_from_files
from ai.explainer import explain_single
from scripts import build_cache_index as cache_index_script
from scripts import cleanup_cache as cache_cleanup_script
from ArcticRoute.exceptions import ArcticRouteError
from logging_config import get_logger, get_recent_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CFG = PROJECT_ROOT / "config" / "runtime.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

logger = get_logger(__name__)


def render_recent_logs(container, *, limit: int = 200) -> None:
    lines = get_recent_output(limit)
    if not lines:
        container.info("No log output captured yet.")
        return
    container.code("\n".join(lines), language="text")


def render_error_block(container, error: ArcticRouteError) -> None:
    container.error(f"Planner failed ({error.code}).")
    detail_lines = [
        f"code: {error.code}",
        f"message: {error.message}",
    ]
    if error.detail:
        detail_lines.append(f"detail: {error.detail}")
    container.code("\n".join(detail_lines), language="text")


def human_bytes(size_bytes: float | int) -> str:
    value = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


@st.cache_data(show_spinner=False)
def load_config(cfg_path: Path) -> dict:
    raw = load_yaml_file(cfg_path)
    runtime_model = validate_runtime_config(raw)
    return model_dump(runtime_model)


@st.cache_data(ttl=60, show_spinner=False)
def load_cache_index() -> dict:
    return cache_index_script.build_index(write_json=True)


@st.cache_resource(show_spinner=False)
def detect_time_length(env_path: Path) -> int:
    with xr.open_dataset(env_path) as ds:
        if "time" not in ds.dims:
            return 1
        return int(ds.dims["time"])


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def find_latest_run_report() -> Optional[Path]:
    candidates = sorted(OUTPUTS_DIR.rglob("run_report_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_latest_run_report() -> Tuple[Optional[Path], Optional[Dict[str, Any]]]:
    report_path = find_latest_run_report()
    if not report_path:
        return None, None
    try:
        report_data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return report_path, None
    return report_path, report_data


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    valid = array[np.isfinite(array)]
    if valid.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    lo, hi = np.percentile(valid, (2, 98))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        scaled = np.zeros(array.shape, dtype=np.float32)
    else:
        scaled = np.clip((array - lo) / (hi - lo), 0.0, 1.0)
    scaled[~np.isfinite(array)] = 0.0
    return (scaled * 255).astype(np.uint8)


def _search_dirs_for_mosaic() -> Sequence[Path]:
    return [
        PROJECT_ROOT / "data_processed",
        PROJECT_ROOT / "data_processed" / "sat_cache",
        PROJECT_ROOT / "outputs",
    ]


def find_latest_satellite_path() -> Optional[Tuple[Path, float]]:
    best: Optional[Tuple[Path, float]] = None
    for base in _search_dirs_for_mosaic():
        if not base.exists():
            continue
        for path in base.rglob("sat_mosaic_*.tif"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if best is None or mtime > best[1]:
                best = (path, mtime)
    return best


@st.cache_data(show_spinner=False)
def prepare_satellite_bundle(path_str: str, mtime: float) -> Dict[str, Any]:
    if rasterio is None:
        raise RuntimeError("rasterio is required for satellite overlays.")
    path = Path(path_str)
    with rasterio.open(path) as src:
        data = src.read()
        bounds = src.bounds
        descriptions = [
            (desc or f"BAND{idx}").upper()
            for idx, desc in enumerate(src.descriptions, start=1)
        ]
    band_lookup = {desc: idx for idx, desc in enumerate(descriptions)}

    def pick_band(*names: str) -> Optional[np.ndarray]:
        for name in names:
            key = name.upper()
            if key in band_lookup:
                return data[band_lookup[key]]
        return None

    red = pick_band("B04", "RED")
    green = pick_band("B03", "GREEN")
    blue = pick_band("B02", "BLUE")

    mosaic_image: Optional[np.ndarray] = None
    if red is not None and green is not None and blue is not None:
        r = _normalize_to_uint8(red)
        g = _normalize_to_uint8(green)
        b = _normalize_to_uint8(blue)
        mosaic_image = np.dstack([r, g, b])

    ice_stats: Optional[Dict[str, float]] = None
    ice_image: Optional[np.ndarray] = None
    nir = pick_band("B08", "NIR")
    if green is not None and nir is not None:
        denom = green + nir
        denom = np.where(np.abs(denom) > 1e-6, denom, np.nan)
        ndwi = (green - nir) / denom
        valid = np.isfinite(ndwi)
        if np.count_nonzero(valid):
            ndwi_valid = ndwi[valid]
            if threshold_otsu is not None:
                try:
                    threshold = float(threshold_otsu(ndwi_valid))
                except Exception:
                    threshold = float(np.nanmean(ndwi_valid))
            else:
                threshold = float(np.nanmean(ndwi_valid))
            mask = np.full(ndwi.shape, np.nan, dtype=np.float32)
            mask[valid] = (ndwi_valid > threshold).astype(np.float32)
            finite_mask = np.nan_to_num(mask, nan=0.0, posinf=0.0, neginf=0.0)
            mean_prob = float(np.nanmean(mask))
            max_prob = float(np.nanmax(mask))
            coverage_pct = float(np.nanmean(mask > 0.5) * 100.0)
            ice_stats = {
                "mean": mean_prob,
                "max": max_prob,
                "coverage_pct": coverage_pct,
            }
            if cm is not None:
                cmap = cm.get_cmap("viridis")
                rgba = cmap(np.clip(finite_mask, 0.0, 1.0))
                ice_image = (rgba[..., :3] * 255).astype(np.uint8)
            else:
                grey = np.clip(finite_mask, 0.0, 1.0)
                ice_image = np.dstack([(grey * 255).astype(np.uint8)] * 3)

    return {
        "path": path,
        "mtime": mtime,
        "bounds": [[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        "mosaic_image": mosaic_image,
        "ice_image": ice_image,
        "ice_stats": ice_stats,
    }


def get_latest_satellite_bundle() -> Optional[Dict[str, Any]]:
    latest = find_latest_satellite_path()
    if latest is None:
        return None
    path, mtime = latest
    try:
        bundle = prepare_satellite_bundle(str(path), mtime)
    except Exception:
        return None
    return bundle


def render_satellite_overlays(container, bundle: Optional[Dict[str, Any]], show_mosaic: bool, show_ice: bool) -> None:
    if not show_mosaic and not show_ice:
        return
    if bundle is None:
        container.warning("No satellite mosaic found. Run the planner with predictor=cv_sat to generate one.")
        return
    if folium is None or st_folium is None:
        container.warning("Install folium and streamlit-folium to view satellite overlays.")
        return

    bounds = bundle["bounds"]
    south, west = bounds[0]
    north, east = bounds[1]
    center_lat = (south + north) / 2.0
    center_lon = (west + east) / 2.0

    fmap = folium.Map(location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB Positron")
    added_layer = False

    if show_mosaic:
        mosaic_img = bundle.get("mosaic_image")
        if mosaic_img is not None:
            folium.raster_layers.ImageOverlay(
                image=mosaic_img,
                bounds=[[south, west], [north, east]],
                name="Satellite Mosaic",
                opacity=0.4,
            ).add_to(fmap)
            added_layer = True
        else:
            container.warning("Satellite mosaic image unavailable for overlay.")

    if show_ice:
        ice_img = bundle.get("ice_image")
        if ice_img is not None:
            folium.raster_layers.ImageOverlay(
                image=ice_img,
                bounds=[[south, west], [north, east]],
                name="Ice Probability",
                opacity=0.4,
            ).add_to(fmap)
            added_layer = True
        else:
            container.warning("Ice probability overlay unavailable; run the planner with predictor=cv_sat.")

    if added_layer:
        folium.LayerControl().add_to(fmap)
        st_folium(fmap, width=700, height=500, returned_objects=[])



def build_advisor_input(beta: float, gamma: float, p_exp: float, beta_a: float) -> Tuple[AdvisorInput, Optional[Path]]:
    report_path, report_data = load_latest_run_report()
    run_tag: Optional[str] = None
    if report_path:
        stem = report_path.stem
        if stem.startswith("run_report_"):
            run_tag = stem[len("run_report_") :]
    parameters = {
        "beta": beta,
        "gamma": gamma,
        "p": p_exp,
        "beta_a": beta_a,
    }
    try:
        advisor_input = advisor_input_from_files(parameters=parameters, run_report_tag=run_tag)
    except Exception:
        percentiles = report_data.get("risk_env_percentiles") if report_data else {}
        if not isinstance(percentiles, dict):
            percentiles = {}
        advisor_input = AdvisorInput(
            beta=beta,
            gamma=gamma,
            p=p_exp,
            beta_a=beta_a,
            risk_env_percentiles={k: float(v) for k, v in percentiles.items() if isinstance(v, (int, float))},
            recent_metrics=None,
        )
    return advisor_input, report_path


@dataclass
class RunVariant:
    tag: str
    variant: Literal["base", "advised"]
    suffix: str
    report_path: Path
    data: Dict[str, Any]
    png_path: Path
    geojson_path: Path
    timestamp: float

    @property
    def beta_a(self) -> Optional[float]:
        value = self.data.get("beta_a")
        if isinstance(value, (int, float)):
            return float(value)
        return None


COMPARE_METRICS: Dict[str, Dict[str, Any]] = {
    "total_cost": {"label": "Total Cost", "decimals": 0, "better": "lower"},
    "mean_risk": {"label": "Mean Risk", "decimals": 3, "better": "lower"},
    "geodesic_length_km": {"label": "Geodesic Length (km)", "decimals": 1, "better": "lower", "unit": " km"},
    "nearest_accident_km.mean": {"label": "Nearest Accident Distance (km)", "decimals": 1, "better": "lower", "unit": " km"},
}

COMPARE_PARAMETER_FIELDS: list[tuple[str, str]] = [
    ("beta", "Beta"),
    ("gamma", "Gamma"),
    ("p", "Risk Exponent (p)"),
    ("beta_a", "Accident Weight (beta_a)"),
    ("tidx", "Time Index"),
    ("time_step_nodes", "Time-step Layers"),
    ("coarsen", "Coarsen Factor"),
    ("speed_knots", "Cruise Speed (knots)"),
    ("start", "Start (lat, lon)"),
    ("goal", "Goal (lat, lon)"),
    ("accident_mode", "Accident Mode"),
]


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _candidate_report_paths() -> list[Path]:
    items: list[tuple[float, Path]] = []
    seen: set[Path] = set()
    for base_dir in {OUTPUTS_DIR, REPO_ROOT / "outputs"}:
        if not base_dir.exists():
            continue
        for path in base_dir.rglob("run_report_*.json"):
            if path in seen:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            items.append((mtime, path))
            seen.add(path)
    items.sort(key=lambda item: item[0], reverse=True)
    return [path for _, path in items]


def _load_run_variant(report_path: Path, variant: Literal["base", "advised"], tag: str, suffix: str) -> Optional[RunVariant]:
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    try:
        timestamp = report_path.stat().st_mtime
    except OSError:
        timestamp = 0.0
    png_path = report_path.with_name(f"route_on_risk_{suffix}.png")
    geojson_path = report_path.with_name(f"route_{variant}_{tag}.geojson") if tag else report_path.with_name(f"route_{suffix}.geojson")
    if not geojson_path.exists():
        geojson_path = report_path.with_name(f"route_{suffix}.geojson")
    return RunVariant(
        tag=tag or variant,
        variant=variant,
        suffix=suffix,
        report_path=report_path,
        data=data,
        png_path=png_path,
        geojson_path=geojson_path,
        timestamp=timestamp,
    )


def find_latest_base_advised_pair() -> Optional[Tuple[RunVariant, RunVariant]]:
    buckets: Dict[str, Dict[Literal["base", "advised"], RunVariant]] = {}
    for report_path in _candidate_report_paths():
        stem = report_path.stem
        if not stem.startswith("run_report_"):
            continue
        suffix = stem[len("run_report_") :]
        for variant in ("base", "advised"):
            prefix = f"{variant}_"
            if not suffix.startswith(prefix):
                continue
            tag = suffix[len(prefix) :]
            entry = _load_run_variant(report_path, variant, tag, suffix)
            if entry is None:
                break
            bucket = buckets.setdefault(tag, {})
            if variant not in bucket:
                bucket[variant] = entry
            break
    latest: Optional[Tuple[RunVariant, RunVariant]] = None
    latest_ts = -1.0
    for bucket in buckets.values():
        if "base" in bucket and "advised" in bucket:
            pair_time = max(bucket["base"].timestamp, bucket["advised"].timestamp)
            if pair_time > latest_ts:
                latest_ts = pair_time
                latest = (bucket["base"], bucket["advised"])
    return latest


def extract_compare_metrics(data: Dict[str, Any]) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {
        "total_cost": _to_float(data.get("total_cost")),
        "mean_risk": _to_float(data.get("mean_risk")),
        "geodesic_length_km": None,
        "nearest_accident_km.mean": None,
    }
    geo_m = _to_float(data.get("geodesic_length_m"))
    if geo_m is not None:
        metrics["geodesic_length_km"] = geo_m / 1000.0
    nearest = data.get("nearest_accident_km")
    if isinstance(nearest, dict):
        metrics["nearest_accident_km.mean"] = _to_float(nearest.get("mean"))
    return metrics


def _load_route_properties(variant: RunVariant) -> Dict[str, Any]:
    if variant.geojson_path.exists():
        try:
            payload = json.loads(variant.geojson_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        features = payload.get("features")
        if isinstance(features, list) and features:
            first = features[0]
            if isinstance(first, dict):
                props = first.get("properties")
                if isinstance(props, dict):
                    return props
    return {}


def extract_variant_parameters(variant: RunVariant) -> Dict[str, Any]:
    params = _load_route_properties(variant)
    if "beta_a" not in params and variant.beta_a is not None:
        params["beta_a"] = variant.beta_a
    return params


def format_parameter_value(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        if abs(value) >= 1000:
            formatted = f"{value:,.0f}"
        elif abs(value) >= 1:
            formatted = f"{value:,.2f}"
        else:
            formatted = f"{value:.3f}"
        return formatted.rstrip("0").rstrip(".")
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, (list, tuple)):
        items = ", ".join(format_parameter_value(item) for item in value)
        return f"[{items}]"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def build_parameter_rows(base_params: Dict[str, Any], advised_params: Dict[str, Any]) -> list[Dict[str, str]]:
    rows: list[Dict[str, str]] = []
    for key, label in COMPARE_PARAMETER_FIELDS:
        rows.append(
            {
                "Parameter": label,
                "Base": format_parameter_value(base_params.get(key)),
                "Advised": format_parameter_value(advised_params.get(key)),
            }
        )
    return rows


def format_metric_value(value: Optional[float], decimals: int, unit: str = "") -> str:
    if value is None:
        return "--"
    formatted = f"{value:,.{decimals}f}"
    if formatted.startswith("-0"):
        formatted = formatted.replace("-0", "0", 1)
    return f"{formatted}{unit}"


def format_difference(base: Optional[float], advised: Optional[float], *, decimals: int, unit: str = "", better: str = "lower") -> str:
    if base is None or advised is None:
        return "→ --"
    diff = advised - base
    tolerance = 10 ** (-(decimals + 2))
    if abs(diff) <= tolerance:
        return "→ 0"
    if better == "lower":
        arrow = "↓" if diff < 0 else "↑"
    else:
        arrow = "↑" if diff > 0 else "↓"
    diff_str = f"{diff:+,.{decimals}f}"
    if diff_str.startswith("-0"):
        diff_str = diff_str.replace("-0", "0", 1)
    if unit:
        diff_str = f"{diff_str}{unit}"
    percent = ""
    if base not in (None, 0):
        percent = f" ({diff / base:+.1%})"
    return f"{arrow} {diff_str}{percent}"


def build_summary_lines(base_metrics: Dict[str, Optional[float]], advised_metrics: Dict[str, Optional[float]]) -> list[str]:
    lines: list[str] = []
    for key, meta in COMPARE_METRICS.items():
        label = meta["label"]
        base_value = base_metrics.get(key)
        advised_value = advised_metrics.get(key)
        if base_value is None or advised_value is None:
            lines.append(f"- {label}: insufficient data for comparison.")
            continue
        diff = advised_value - base_value
        decimals = int(meta.get("decimals", 2))
        unit = str(meta.get("unit", ""))
        tolerance = 10 ** (-(decimals + 2))
        if abs(diff) <= tolerance:
            lines.append(f"- {label}: unchanged at {format_metric_value(advised_value, decimals, unit)}.")
            continue
        direction = "decreased" if diff < 0 else "increased"
        better = meta.get("better", "lower")
        improved = diff < 0 if better == "lower" else diff > 0
        effect = "improving" if improved else "worsening"
        diff_amount = format_metric_value(abs(diff), decimals, unit)
        percent = ""
        if base_value:
            percent = f" ({abs(diff) / base_value:.1%})"
        lines.append(f"- {label}: advised route {direction} by {diff_amount}{percent}, {effect} the outcome.")
    return lines


def build_compare_markdown(
    base: RunVariant,
    advised: RunVariant,
    base_metrics: Dict[str, Optional[float]],
    advised_metrics: Dict[str, Optional[float]],
    table_rows: list[Dict[str, str]],
    parameter_rows: list[Dict[str, str]],
) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "# Route Comparison Report",
        "",
        f"Generated: {now}",
        "",
        "## Parameters",
        f"- Base tag: `{base.tag}`",
        f"- Advised tag: `{advised.tag}`",
    ]
    if parameter_rows:
        lines.extend(
            [
                "",
                "| Parameter | Base | Advised |",
                "| --- | --- | --- |",
            ]
        )
        for row in parameter_rows:
            lines.append(f"| {row['Parameter']} | {row['Base']} | {row['Advised']} |")
    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Metric | Base | Advised | Delta |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in table_rows:
        lines.append(f"| {row['Metric']} | {row['Base']} | {row['Advised']} | {row['Delta']} |")
    lines.extend(
        [
            "",
            "## Summary",
        ]
    )
    lines.extend(build_summary_lines(base_metrics, advised_metrics))
    for variant in (base, advised):
        try:
            rel_path = variant.png_path.relative_to(REPO_ROOT)
        except ValueError:
            rel_path = variant.png_path
        if variant.png_path.exists():
            lines.extend(
                [
                    "",
                    f"![{variant.variant.capitalize()} route]({rel_path.as_posix()})",
                ]
            )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    st.set_page_config(page_title="ArcticRoute Planner", layout="wide")
    st.title("ArcticRoute Streamlit Planner")
    st.markdown("Run via: `streamlit run ArcticRoute/apps/app_basic.py`")

    cfg_path = DEFAULT_CFG
    config = load_config(cfg_path)

    env_path_cfg = config.get("data", {}).get("env_nc")
    env_path = resolve_path(env_path_cfg) if env_path_cfg else PROJECT_ROOT / "data_processed" / "env_clean.nc"
    time_len = detect_time_length(env_path)

    default_beta = float(config.get("cost", {}).get("beta", 3.0))
    default_gamma = float(config.get("behavior", {}).get("gamma", 0.0))
    default_p = float(config.get("cost", {}).get("p", 1.0))
    default_beta_a = float(config.get("cost", {}).get("beta_accident", 0.0))
    default_tidx = int(config.get("run", {}).get("tidx", 0))
    default_time_step = int(config.get("run", {}).get("time_step_nodes", 0))
    default_start = config.get("route", {}).get("start", "75,10")
    default_goal = config.get("route", {}).get("goal", "70,20")
    default_coarsen = int(config.get("crop", {}).get("coarsen", 1))
    default_predictor = str(config.get("predictor", "env_nc")).lower()
    if default_predictor not in {"env_nc", "cv_sat"}:
        default_predictor = "env_nc"
    default_alpha_ice = float(np.clip(config.get("alpha_ice", 0.0), 0.0, 1.0))
    accident_density_path = REPO_ROOT / "data_processed" / "accident_density_static.nc"
    accident_data_available = accident_density_path.exists()
    accident_enabled_default = (default_beta_a > 0.0) or accident_data_available
    beta_a_enabled_default = default_beta_a if default_beta_a > 0 else 0.3

    state_defaults = {
        "ui_beta": default_beta,
        "ui_gamma": default_gamma,
        "ui_p": default_p,
        "ui_tidx": min(default_tidx, max(time_len - 1, 0)),
        "ui_beta_a": beta_a_enabled_default if accident_enabled_default else 0.0,
        "ui_speed": 12,
        "ui_time_steps": default_time_step,
        "ui_start": default_start,
        "ui_goal": default_goal,
        "ui_coarsen": default_coarsen if default_coarsen in (1, 2, 3) else 1,
        "ui_show_hotspots": False,
        "ui_use_llm": False,
        "ui_accident_enabled": accident_enabled_default,
        "ui_beta_a_enabled_value": beta_a_enabled_default,
        "ui_predictor": default_predictor,
        "ui_alpha_ice": default_alpha_ice,
        "ui_show_sat_overlay": False,
        "ui_show_ice_overlay": False,
    }
    for key, value in state_defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("ai_explanation", None)
    st.session_state.setdefault("ai_notice", None)

    with st.sidebar:
        st.header("Parameter Controls")
        beta = st.slider("Beta", min_value=1.0, max_value=8.0, value=float(st.session_state["ui_beta"]), step=0.1, key="ui_beta")
        gamma = st.slider("Gamma", min_value=0.0, max_value=0.8, value=float(st.session_state["ui_gamma"]), step=0.05, key="ui_gamma")
        p_exp = st.slider("p (risk exponent)", min_value=0.8, max_value=1.6, value=float(st.session_state["ui_p"]), step=0.05, key="ui_p")
        tidx = st.slider("Time index", min_value=0, max_value=max(time_len - 1, 0), value=int(st.session_state["ui_tidx"]), key="ui_tidx")
        predictor_options = ["env_nc", "cv_sat"]
        predictor_choice = st.selectbox(
            "Predictor",
            predictor_options,
            index=predictor_options.index(st.session_state["ui_predictor"]) if st.session_state["ui_predictor"] in predictor_options else 0,
            key="ui_predictor",
        )
        alpha_ice = st.slider(
            "alpha_ice (risk / ice blend)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state["ui_alpha_ice"]),
            step=0.05,
            key="ui_alpha_ice",
        )
        show_sat_overlay = st.checkbox(
            "Show satellite mosaic overlay",
            value=st.session_state["ui_show_sat_overlay"],
            key="ui_show_sat_overlay",
        )
        show_ice_overlay = st.checkbox(
            "Show ice probability overlay",
            value=st.session_state["ui_show_ice_overlay"],
            key="ui_show_ice_overlay",
        )
        accident_enabled = st.checkbox(
            "启用事故风险（beta_a>0）",
            value=st.session_state["ui_accident_enabled"],
            key="ui_accident_enabled",
        )
        if accident_enabled:
            enabled_value = st.session_state.get("ui_beta_a_enabled_value", beta_a_enabled_default)
            if st.session_state.get("ui_beta_a", 0.0) <= 0.0 and enabled_value > 0.0:
                st.session_state["ui_beta_a"] = float(enabled_value)
            beta_a = st.slider(
                "beta_a (accident weight)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state["ui_beta_a"]),
                step=0.05,
                key="ui_beta_a",
            )
            st.session_state["ui_beta_a_enabled_value"] = beta_a if beta_a > 0 else st.session_state.get("ui_beta_a_enabled_value", beta_a_enabled_default)
        else:
            st.slider(
                "beta_a (accident weight)",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                key="ui_beta_a_disabled",
                disabled=True,
            )
            beta_a = 0.0
            st.session_state["ui_beta_a"] = beta_a
        speed_knots = st.slider("Speed (knots)", min_value=6, max_value=20, value=int(st.session_state["ui_speed"]), step=1, key="ui_speed")
        time_step_nodes = st.slider("Time-step layers", min_value=0, max_value=20, value=int(st.session_state["ui_time_steps"]), step=1, key="ui_time_steps")
        start_txt = st.text_input("Start (lat,lon)", value=st.session_state["ui_start"], key="ui_start")
        goal_txt = st.text_input("Goal (lat,lon)", value=st.session_state["ui_goal"], key="ui_goal")
        coarsen = st.selectbox("Coarsen factor", options=[1, 2, 3], index=[1, 2, 3].index(st.session_state["ui_coarsen"]), key="ui_coarsen")
        use_llm_flag = st.checkbox("Use LLM", value=st.session_state["ui_use_llm"], key="ui_use_llm")
        ai_advice_clicked = st.button("AI Advice", width='stretch')
        explain_clicked = st.button("Generate Explanation", width='stretch')
        show_hotspots = st.checkbox("Show hotspot overlays", value=st.session_state["ui_show_hotspots"], key="ui_show_hotspots")
        run_button = st.button("Run Planner", width='stretch')

    if ai_advice_clicked:
        try:
            advisor_input, _ = build_advisor_input(beta, gamma, p_exp, beta_a)
        except Exception as err:
            advisor_input = AdvisorInput(
                beta=beta,
                gamma=gamma,
                p=p_exp,
                beta_a=beta_a,
                risk_env_percentiles={},
                recent_metrics=None,
            )
            advice = rule_advice(advisor_input)
            st.session_state["ai_notice"] = ("warning", f"AI advice failed; using rule-based fallback: {err}")
        else:
            try:
                if use_llm_flag:
                    advice, _ = llm_advice(advisor_input)
                else:
                    advice = rule_advice(advisor_input)
            except Exception as err:
                advice = rule_advice(advisor_input)
                st.session_state["ai_notice"] = ("warning", f"AI advice failed; using rule-based fallback: {err}")
            else:
                from_llm = getattr(advice, "_from_llm", False)
                if use_llm_flag and not from_llm:
                    st.session_state["ai_notice"] = ("warning", "LLM 调用失败，已使用规则建议。")
                else:
                    st.session_state["ai_notice"] = ("success", "AI Advice已应用。")
        if "advice" in locals():
            st.session_state["ui_beta"] = float(advice.beta)
            st.session_state["ui_gamma"] = float(advice.gamma)
            st.session_state["ui_p"] = float(advice.p)
            advised_beta_a = float(advice.beta_a)
            st.session_state["ui_beta_a"] = advised_beta_a
            if advised_beta_a > 0:
                st.session_state["ui_beta_a_enabled_value"] = advised_beta_a
                st.session_state["ui_accident_enabled"] = True
            else:
                st.session_state["ui_accident_enabled"] = False
        st.experimental_rerun()

    if explain_clicked:
        report_path, report_data = load_latest_run_report()
        if not report_path or not isinstance(report_data, dict):
            st.session_state["ai_notice"] = ("warning", "No valid run report found; unable to generate explanation.")
            st.session_state["ai_explanation"] = None
        else:
            params = {
                "beta": beta,
                "gamma": gamma,
                "p": p_exp,
                "beta_a": beta_a,
            }
            try:
                explanation = explain_single(report_data, params, use_llm=use_llm_flag)
            except Exception as err:
                explanation = explain_single(report_data, params, use_llm=False)
                from_llm = False
                st.session_state["ai_notice"] = ("warning", f"Explanation generation failed; template used: {err}")
            else:
                from_llm = getattr(explanation, "_from_llm", False)
                if use_llm_flag and not from_llm:
                    st.session_state["ai_notice"] = ("warning", "LLM explanation failed; template used instead.")
                else:
                    st.session_state["ai_notice"] = ("success", "Explanation generated.")
            st.session_state["ai_explanation"] = {
                "markdown": explanation.markdown,
                "bullets": explanation.bullets,
                "file_name": f"{report_path.stem}_explanation.md" if report_path else "explanation.md",
                "source_llm": from_llm,
            }
        st.experimental_rerun()

    plan_tab, compare_tab = st.tabs(["Planner", "Compare"])

    with plan_tab:
        col_left, col_right = st.columns([2, 1])
        result_slot = col_left.container()
        log_expander = col_left.expander("Execution Logs (last 200 lines)", expanded=False)
        log_placeholder = log_expander.empty()
        summary_slot = col_right.container()

        summary_slot.subheader("Current Parameters")
        summary_params_container = summary_slot.container()
        summary_warning_container = summary_slot.container()

        latest_report_path, latest_report_data = load_latest_run_report()
        overlay_bundle = get_latest_satellite_bundle()
        overlay_path = overlay_bundle.get("path") if overlay_bundle else None
        overlay_ice_stats = overlay_bundle.get("ice_stats") if overlay_bundle else None
        report_ice_stats = None
        if isinstance(latest_report_data, dict):
            report_ice_stats = latest_report_data.get("ice_prob_stats")
        ice_stats_display = overlay_ice_stats or report_ice_stats

        def build_summary(ice_stats_value, overlay_path_value):
            summary = {
                "predictor": predictor_choice,
                "alpha_ice": alpha_ice,
                "beta": beta,
                "gamma": gamma,
                "p": p_exp,
                "tidx": tidx,
                "beta_a": beta_a,
                "speed_knots": speed_knots,
                "time_step_nodes": time_step_nodes,
                "start": start_txt,
                "goal": goal_txt,
                "coarsen": coarsen,
                "show_hotspots": show_hotspots,
                "show_satellite_overlay": show_sat_overlay,
                "show_ice_overlay": show_ice_overlay,
                "accident_enabled": accident_enabled,
            }
            if overlay_path_value is not None:
                try:
                    summary["latest_mosaic"] = str(overlay_path_value.relative_to(PROJECT_ROOT))
                except ValueError:
                    summary["latest_mosaic"] = str(overlay_path_value)
            if isinstance(ice_stats_value, dict):
                summary["ice_prob_stats"] = ice_stats_value
            if latest_report_path:
                try:
                    summary["latest_report"] = str(latest_report_path.relative_to(PROJECT_ROOT))
                except ValueError:
                    summary["latest_report"] = str(latest_report_path)
            return summary

        def should_warn(ice_stats_value):
            return ice_stats_value is None and (
                predictor_choice == "cv_sat"
                or alpha_ice > 0.0
                or show_ice_overlay
            )

        summary_params_container.write(build_summary(ice_stats_display, overlay_path))
        if should_warn(ice_stats_display):
            summary_warning_container.warning(
                "Ice probability statistics unavailable. Run the planner with predictor=cv_sat."
            )
        notice = st.session_state.get("ai_notice")
        if notice:
            level, message = notice
            if level == "warning":
                summary_slot.warning(message)
            elif level == "success":
                summary_slot.success(message)
            else:
                summary_slot.info(message)
            st.session_state["ai_notice"] = None

        explanation_state = st.session_state.get("ai_explanation")
        if isinstance(explanation_state, dict):
            summary_slot.subheader("AI Explanation")
            summary_slot.markdown(explanation_state.get("markdown", ""))
            bullets = explanation_state.get("bullets") or []
            bullet_md = "\n".join(f"- {item}" for item in bullets) if bullets else ""
            if bullet_md:
                summary_slot.markdown(bullet_md)
            download_content = explanation_state.get("markdown", "")
            if bullet_md:
                download_content += "\n\n" + bullet_md
            file_name = explanation_state.get("file_name", "explanation.md")
            summary_slot.download_button(
                "Download Explanation (Markdown)",
                data=download_content.encode("utf-8"),
                file_name=file_name,
                mime="text/markdown",
            )

        if show_hotspots:
            hotspots_path = REPO_ROOT / "outputs" / "acc_hotspots.geojson"
            if hotspots_path.exists():
                if pd is None or pdk is None:
                    summary_slot.warning("Install pandas and pydeck to show hotspot overlays.")
                else:
                    try:
                        hotspots = json.loads(hotspots_path.read_text(encoding="utf-8"))
                        coords = [
                            {"lat": feature["geometry"]["coordinates"][1], "lon": feature["geometry"]["coordinates"][0], "value": feature["properties"].get("value", 1.0)}
                            for feature in hotspots.get("features", [])
                            if feature.get("geometry", {}).get("type") == "Point"
                        ]
                        if coords:
                            df = pd.DataFrame(coords)
                            layer = pdk.Layer(
                                "HeatmapLayer",
                                data=df,
                                get_position="[lon, lat]",
                                get_weight="value",
                                radiusPixels=40,
                            )
                            view_state = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=3)
                            summary_slot.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Hotspot {value}"}))
                        else:
                            summary_slot.info("No hotspots found for current threshold.")
                    except Exception as err:  # pragma: no cover
                        summary_slot.warning(f"Unable to render hotspot overlay: {err}")
        else:
            summary_slot.info("Hotspot overlay not found; run the export script first.")

    with summary_slot.expander("缓存管理", expanded=False):
        if st.button("刷新索引", key="cache_refresh_button"):
            load_cache_index.clear()
            st.session_state.pop("cache_cleanup_result", None)
            st.experimental_rerun()

        index_payload = load_cache_index()
        total_bytes = index_payload.get("total_bytes", 0)
        st.caption(f"当前缓存总量：{human_bytes(total_bytes)} · 文件数：{index_payload.get('total_files', 0)}")
        entries = index_payload.get("entries", [])
        if entries:
            if pd is not None:
                df_cache = pd.DataFrame(entries)
                df_cache["size_mb"] = df_cache["size_bytes"] / (1024 ** 2)
                if "modified" in df_cache:
                    with suppress(Exception):
                        df_cache["modified"] = pd.to_datetime(df_cache["modified"], errors="coerce")
                st.dataframe(
                    df_cache.sort_values("modified", ascending=False, na_position="last"),
                    width='stretch',
                )
            else:
                st.json(entries)
        else:
            st.info("缓存目录为空。")

        st.markdown("---")

        col_days, col_total = st.columns(2)
        days_value = col_days.number_input(
            "删除超过 N 天的缓存",
            min_value=0.0,
            value=30.0,
            step=1.0,
            key="cache_days_input",
        )
        max_total_value = col_total.number_input(
            "保留最大容量 (GB)",
            min_value=0.0,
            value=5.0,
            step=0.5,
            key="cache_max_total_input",
        )

        col_preview, col_execute = st.columns(2)
        preview_clicked = col_preview.button("清理预览", width='stretch', key="cache_preview_btn")
        execute_clicked = col_execute.button(
            "执行清理", width='stretch', key="cache_execute_btn", type="primary"
        )

        def _apply_cleanup(dry_run: bool) -> None:
            days_param = days_value if days_value > 0 else None
            max_total_param = max_total_value if max_total_value > 0 else None
            if days_param is None and max_total_param is None:
                st.warning("请设置清理条件（超过 N 天或最大容量）。")
                return
            result = cache_cleanup_script.cleanup_cache(
                days=days_param,
                max_total_gb=max_total_param,
                dry_run=dry_run,
            )
            st.session_state["cache_cleanup_result"] = result
            if not result["dry_run"]:
                load_cache_index.clear()
            st.experimental_rerun()

        if preview_clicked:
            _apply_cleanup(dry_run=True)
        if execute_clicked:
            _apply_cleanup(dry_run=False)

        cleanup_result = st.session_state.get("cache_cleanup_result")
        if cleanup_result:
            title = "清理预览" if cleanup_result["dry_run"] else "清理完成"
            st.subheader(title)
            reclaimed = human_bytes(cleanup_result["total_reclaimed_bytes"])
            remaining = human_bytes(cleanup_result["remaining_bytes"])
            st.write(
                f"拟删除 {cleanup_result['removed_count']} 个文件，预计释放 {reclaimed}。"
                if cleanup_result["dry_run"]
                else f"已删除 {cleanup_result['removed_count']} 个文件，释放 {reclaimed}，剩余 {remaining}。"
            )
            if cleanup_result["entries"]:
                if pd is not None:
                    df_removed = pd.DataFrame(cleanup_result["entries"])
                    df_removed["size_mb"] = df_removed["size_bytes"] / (1024 ** 2)
                    if "modified" in df_removed:
                        with suppress(Exception):
                            df_removed["modified"] = pd.to_datetime(df_removed["modified"], errors="coerce")
                    st.dataframe(
                        df_removed.sort_values("modified", ascending=False, na_position="last"),
                        width='stretch',
                    )
                else:
                    st.json(cleanup_result["entries"])
            else:
                st.info("无需删除缓存文件。")

    if run_button:
        with st.spinner("正在执行规划..."):
                try:
                    overrides = {
                        "beta": beta,
                        "gamma": gamma,
                        "p": p_exp,
                        "tidx": tidx,
                        "beta_accident": beta_a,
                        "time_step_nodes": time_step_nodes,
                        "start": start_txt,
                        "goal": goal_txt,
                        "coarsen": coarsen,
                        "predictor": predictor_choice,
                        "alpha_ice": alpha_ice,
                    }
                    execution = run_plan(
                        config,
                        overrides=overrides,
                        tag="streamlit",
                        output_dir=DEFAULT_OUTPUT_DIR,
                    )
                    result_slot.success("Planner run completed.")
                    if execution.png_path.exists():
                        result_slot.image(str(execution.png_path), caption=execution.png_path.name, use_column_width=True)
                    else:
                        result_slot.warning("PNG output not found.")

                    report_data = None
                    report_ice_stats = None
                    if execution.report_path.exists():
                        latest_report_path = execution.report_path
                        try:
                            report_text = execution.report_path.read_text(encoding="utf-8")
                            report_data = json.loads(report_text)
                        except Exception as err:  # pragma: no cover - streamlit ui
                            result_slot.warning(f"Failed to parse run report: {err}")
                            report_data = None
                        if isinstance(report_data, dict):
                            latest_report_data = report_data
                            report_ice_stats = report_data.get("ice_prob_stats")
                            result_slot.subheader("Run Metrics")
                            result_slot.json(report_data)
                        else:
                            result_slot.warning("Run report content is invalid.")
                    else:
                        result_slot.warning("Run report not found.")

                    overlay_bundle = get_latest_satellite_bundle()
                    overlay_path = overlay_bundle.get("path") if overlay_bundle else None
                    overlay_ice_stats = overlay_bundle.get("ice_stats") if overlay_bundle else None
                    ice_stats_display = overlay_ice_stats or report_ice_stats

                    summary_params_container.empty()
                    summary_params_container.write(build_summary(ice_stats_display, overlay_path))
                    summary_warning_container.empty()
                    if should_warn(ice_stats_display):
                        summary_warning_container.warning("Ice probability statistics unavailable. Run the planner with predictor=cv_sat.")
                    download_cols = result_slot.columns(3)
                    files = [
                        ("GeoJSON", execution.geojson_path, "application/geo+json"),
                        ("PNG", execution.png_path, "image/png"),
                        ("JSON", execution.report_path, "application/json"),
                    ]
                    for idx, (label, path, mime) in enumerate(files):
                        if path.exists():
                            with open(path, "rb") as fh:
                                download_cols[idx].download_button(
                                    label=f"涓嬭浇 {label}",
                                    data=fh.read(),
                                    file_name=path.name,
                                    mime=mime,
                                )
                        else:
                            download_cols[idx].write(f"{label} 涓嶅彲鐢?")
                except Exception as err:
                    result_slot.error(f"Planner failed: {err}")

        if show_sat_overlay or show_ice_overlay:
            render_satellite_overlays(result_slot, overlay_bundle, show_sat_overlay, show_ice_overlay)
        elif overlay_bundle is None and predictor_choice == "cv_sat":
            result_slot.warning("No satellite mosaic found. Run the planner with predictor=cv_sat to generate one.")

    with compare_tab:
        comparison = find_latest_base_advised_pair()
        if not comparison:
            st.info("未找到 base/advised 结果对，先运行规划生成对应的 run_report。")
        else:
            base_variant, advised_variant = comparison
            base_metrics = extract_compare_metrics(base_variant.data)
            advised_metrics = extract_compare_metrics(advised_variant.data)
            base_params = extract_variant_parameters(base_variant)
            advised_params = extract_variant_parameters(advised_variant)
            parameter_rows = build_parameter_rows(base_params, advised_params)

            variant_cols = st.columns(2)
            for col, variant, metrics in (
                (variant_cols[0], base_variant, base_metrics),
                (variant_cols[1], advised_variant, advised_metrics),
            ):
                col.subheader(f"{variant.variant.capitalize()} · {variant.tag}")
                if variant.png_path.exists():
                    col.image(str(variant.png_path), caption=variant.png_path.name, use_column_width=True)
                else:
                    col.warning("PNG 未找到，请先运行 planner 生成输出。")
                col.markdown("**重点指标**")
                metric_rows = []
                for key, meta in COMPARE_METRICS.items():
                    decimals = int(meta.get("decimals", 2))
                    unit = str(meta.get("unit", ""))
                    metric_rows.append(
                        {
                            "Metric": meta["label"],
                            "Value": format_metric_value(metrics.get(key), decimals, unit),
                        }
                    )
                col.table(metric_rows)
                beta_val = variant.beta_a
                caption_parts = []
                try:
                    rel_report = variant.report_path.relative_to(REPO_ROOT)
                except ValueError:
                    rel_report = variant.report_path
                caption_parts.append(str(rel_report))
                if beta_val is not None:
                    caption_parts.append(f"beta_a={beta_val:.2f}")
                col.caption(" | ".join(caption_parts))

            if parameter_rows:
                st.subheader("参数对比")
                st.table(parameter_rows)

            st.subheader("指标对比")
            table_rows: list[Dict[str, str]] = []
            for key, meta in COMPARE_METRICS.items():
                decimals = int(meta.get("decimals", 2))
                unit = str(meta.get("unit", ""))
                base_value = base_metrics.get(key)
                advised_value = advised_metrics.get(key)
                table_rows.append(
                    {
                        "Metric": meta["label"],
                        "Base": format_metric_value(base_value, decimals, unit),
                        "Advised": format_metric_value(advised_value, decimals, unit),
                        "Delta": format_difference(
                            base_value,
                            advised_value,
                            decimals=decimals,
                            unit=unit,
                            better=meta.get("better", "lower"),
                        ),
                    }
                )
            st.table(table_rows)

            summary_lines = build_summary_lines(base_metrics, advised_metrics)
            st.markdown("**结论小结**")
            st.markdown("\n".join(summary_lines))

            export_clicked = st.button("导出对比 Markdown", width='stretch')
            if export_clicked:
                markdown_content = build_compare_markdown(
                    base_variant,
                    advised_variant,
                    base_metrics,
                    advised_metrics,
                    table_rows,
                    parameter_rows,
                )
                output_path = REPO_ROOT / "docs" / "compare_report.md"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(markdown_content, encoding="utf-8")
                try:
                    relative_path = output_path.relative_to(REPO_ROOT)
                except ValueError:
                    relative_path = output_path
                st.success(f"已导出至 {relative_path}")

    col_left, col_right = st.columns([2, 1])
    result_slot = col_left.container()
    log_expander = col_left.expander("执行日志（最近 200 行）", expanded=False)
    log_placeholder = log_expander.empty()
    summary_slot = col_right.container()

    summary_slot.subheader("Current Parameters")
    summary_slot.write(
        {
            "beta": beta,
            "gamma": gamma,
            "p": p_exp,
            "tidx": tidx,
            "beta_a": beta_a,
            "speed_knots": speed_knots,
            "time_step_nodes": time_step_nodes,
            "start": start_txt,
            "goal": goal_txt,
            "coarsen": coarsen,
            "show_hotspots": show_hotspots,
        }
    )

    notice = st.session_state.get("ai_notice")
    if notice:
        level, message = notice
        if level == "warning":
            summary_slot.warning(message)
        elif level == "success":
            summary_slot.success(message)
        else:
            summary_slot.info(message)
        st.session_state["ai_notice"] = None

    explanation_state = st.session_state.get("ai_explanation")
    if isinstance(explanation_state, dict):
        summary_slot.subheader("AI Explanation")
        summary_slot.markdown(explanation_state.get("markdown", ""))
        bullets = explanation_state.get("bullets") or []
        bullet_md = "\n".join(f"- {item}" for item in bullets) if bullets else ""
        if bullet_md:
            summary_slot.markdown(bullet_md)
        download_content = explanation_state.get("markdown", "")
        if bullet_md:
            download_content += "\n\n" + bullet_md
        file_name = explanation_state.get("file_name", "explanation.md")
        summary_slot.download_button(
            "Download Explanation (Markdown)",
            data=download_content.encode("utf-8"),
            file_name=file_name,
            mime="text/markdown",
        )

    if show_hotspots:
        hotspots_path = REPO_ROOT / "outputs" / "acc_hotspots.geojson"
        if hotspots_path.exists():
            if pd is None or pdk is None:
                summary_slot.warning("Install pandas and pydeck to show hotspot overlays.")
            else:
                try:
                    hotspots = json.loads(hotspots_path.read_text(encoding="utf-8"))
                    coords = [
                        {"lat": feature["geometry"]["coordinates"][1], "lon": feature["geometry"]["coordinates"][0], "value": feature["properties"].get("value", 1.0)}
                        for feature in hotspots.get("features", [])
                        if feature.get("geometry", {}).get("type") == "Point"
                    ]
                    if coords:
                        df = pd.DataFrame(coords)
                        layer = pdk.Layer(
                            "HeatmapLayer",
                            data=df,
                            get_position="[lon, lat]",
                            get_weight="value",
                            radiusPixels=40,
                        )
                        view_state = pdk.ViewState(latitude=float(df["lat"].mean()), longitude=float(df["lon"].mean()), zoom=3)
                        summary_slot.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": "Hotspot {value}"}))
                    else:
                        summary_slot.info("No hotspots found for current threshold.")
                except Exception as err:  # pragma: no cover - fallback for missing deps
                    summary_slot.warning(f"Unable to render hotspot overlay: {err}")
        else:
            summary_slot.info("Hotspot overlay not found; run the export script first.")

    if run_button:
        with st.spinner("正在执行规划..."):
            try:
                overrides = {
                    "beta": beta,
                    "gamma": gamma,
                    "p": p_exp,
                    "tidx": tidx,
                    "beta_accident": beta_a,
                    "time_step_nodes": time_step_nodes,
                    "start": start_txt,
                    "goal": goal_txt,
                    "coarsen": coarsen,
                }
                execution = run_plan(
                    config,
                    overrides=overrides,
                    tag="streamlit",
                    output_dir=DEFAULT_OUTPUT_DIR,
                )
                result_slot.success("规划完成！")
                if execution.png_path.exists():
                    result_slot.image(str(execution.png_path), caption=execution.png_path.name, use_column_width=True)
                else:
                    result_slot.warning("PNG 输出未找到。")

                if execution.report_path.exists():
                    report_data = json.loads(execution.report_path.read_text(encoding="utf-8"))
                    result_slot.subheader("运行指标")
                    result_slot.json(report_data)
                else:
                    result_slot.warning("运行报告未找到。")

                download_cols = result_slot.columns(3)
                files = [
                    ("GeoJSON", execution.geojson_path, "application/geo+json"),
                    ("PNG", execution.png_path, "image/png"),
                    ("JSON", execution.report_path, "application/json"),
                ]
                for idx, (label, path, mime) in enumerate(files):
                    if path.exists():
                        with open(path, "rb") as fh:
                            download_cols[idx].download_button(
                                label=f"下载 {label}",
                                data=fh.read(),
                                file_name=path.name,
                                mime=mime,
                            )
                    else:
                        download_cols[idx].write(f"{label} 不可用")
            except ArcticRouteError as err:
                logger.exception("Planner failed with business error")
                render_error_block(result_slot, err)
            except Exception as err:
                logger.exception("Planner failed with unexpected error")
                wrapped = ArcticRouteError("ARC-UI-000", "Planner execution failed", detail=str(err))
                render_error_block(result_slot, wrapped)
        render_recent_logs(log_placeholder)
    else:
        render_recent_logs(log_placeholder)


if __name__ == "__main__":
    main()
