from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

from ArcticRoute.core.route.scan import run_scan  # REUSE
from ArcticRoute.api.cli import _load_yaml, _resolve_path  # REUSE cli helpers

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseI"


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _min_by(points: List[Dict[str, Any]], key: str, default: float) -> Dict[str, Any]:
    if not points:
        return {}
    return min(points, key=lambda d: float(d.get(key, default)))


def _svg_scatter(xs: List[float], ys: List[float], labels: List[str], colors: List[str], title: str, width: int = 480, height: int = 300) -> str:
    if not xs or not ys:
        return ""
    pad = 40
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    if xmax - xmin <= 1e-9:
        xmax = xmin + 1.0
    if ymax - ymin <= 1e-9:
        ymax = ymin + 1.0
    def sx(x: float) -> float:
        return pad + (x - xmin) / (xmax - xmin) * (width - 2 * pad)
    def sy(y: float) -> float:
        return height - pad - (y - ymin) / (ymax - ymin) * (height - 2 * pad)
    items = []
    # axes
    items.append(f"<line x1='{pad}' y1='{height-pad}' x2='{width-pad}' y2='{height-pad}' stroke='black' stroke-width='1' />")
    items.append(f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height-pad}' stroke='black' stroke-width='1' />")
    # ticks
    for t in range(5):
        x = pad + t*(width-2*pad)/4
        y = height - pad
        items.append(f"<line x1='{x}' y1='{y}' x2='{x}' y2='{y+5}' stroke='black' stroke-width='1' />")
        y2 = pad + t*(height-2*pad)/4
        items.append(f"<line x1='{pad}' y1='{y2}' x2='{pad-5}' y2='{y2}' stroke='black' stroke-width='1' />")
    # points
    for x, y, c in zip(xs, ys, colors):
        items.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='3.5' fill='{c}' opacity='0.85' />")
    # legend
    legend_y = pad
    for i, lab in enumerate(labels):
        items.append(f"<rect x='{width-140}' y='{legend_y+18*i}' width='10' height='10' fill='{colors[i]}' />")
        items.append(f"<text x='{width-125}' y='{legend_y+18*i+9}' font-size='12'>{lab}</text>")
    items.append(f"<text x='{width/2:.0f}' y='18' text-anchor='middle' font-size='14'>{title}</text>")
    return f"<svg width='{width}' height='{height}' xmlns='http://www.w3.org/2000/svg'>{''.join(items)}</svg>"


def build_compare(ym: str, scenario_id: str, alpha: float = 0.9) -> str:
    """生成 robust_<ym>_<scenario>.html：比较 mean vs cvar@alpha vs robust。
    - 直接调用 run_scan 计算 mean 与 cvar 两组候选（export_top=1）。
    - 鲁棒路线假定已通过 route.robust 生成；若未生成，仅展示 mean/cvar。
    """
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    grid = _resolve_path("configs/scenarios.yaml")
    cfg = _load_yaml(grid)
    scenario = next((s for s in (cfg.get("scenarios") or []) if s.get("id") == scenario_id), None)
    if scenario is None:
        raise ValueError(f"Scenario '{scenario_id}' not found in {grid}")

    res_mean = run_scan(scenario=scenario, ym=ym, risk_source="fused", risk_agg="mean", alpha=alpha, export_top=1)
    res_cvar = run_scan(scenario=scenario, ym=ym, risk_source="fused", risk_agg="cvar", alpha=alpha, export_top=1)

    robust_geo = REPORT_DIR / f"route_{ym}_{scenario_id}_robust.geojson"
    robust_exists = robust_geo.exists()

    j_mean = _read_json(REPO_ROOT / res_mean.get("front", ""))
    j_cvar = _read_json(REPO_ROOT / res_cvar.get("front", ""))

    best_mean = _min_by(j_mean.get("points", []), "risk_integral", 1e18)
    best_cvar = _min_by(j_cvar.get("points", []), "risk_integral", 1e18)

    robust_props: Dict[str, Any] = {}
    if robust_exists:
        try:
            data_r = json.loads(robust_geo.read_text(encoding="utf-8"))
            robust_props = (data_r.get("features") or [{}])[0].get("properties", {})
        except Exception:
            robust_props = {}

    # 构造 Summary 表格
    def row(label: str, props: Dict[str, Any]) -> str:
        if not props:
            return f"<tr><td>{label}</td><td colspan='6'><em>N/A</em></td></tr>"
        return (
            "<tr>"
            f"<td>{label}</td>"
            f"<td>{props.get('risk_agg','')}</td>"
            f"<td>{props.get('alpha','')}</td>"
            f"<td>{props.get('distance_km','')}</td>"
            f"<td>{props.get('risk_integral','')}</td>"
            f"<td>{props.get('es_value', '')}</td>"
            f"<td>{props.get('samples', '')}</td>"
            "</tr>"
        )

    table = [
        "<table border='1' cellpadding='4' cellspacing='0'>",
        "<tr><th>variant</th><th>agg</th><th>alpha</th><th>distance_km</th><th>risk_integral</th><th>ES</th><th>samples</th></tr>",
        row("mean", best_mean),
        row("cvar", best_cvar),
        row("robust", robust_props),
        "</table>",
    ]

    # 准备散点图（distance vs risk），各策略取代表点
    xs: List[float] = []
    ys: List[float] = []
    labs: List[str] = []
    cols: List[str] = []
    def add_point(props: Dict[str, Any], lab: str, col: str):
        try:
            xs.append(float(props.get("distance_km", "nan")))
            ys.append(float(props.get("risk_integral", "nan")))
            labs.append(lab)
            cols.append(col)
        except Exception:
            pass
    if best_mean:
        add_point(best_mean, "mean", "#4c78a8")
    if best_cvar:
        add_point(best_cvar, "cvar", "#f58518")
    if robust_props:
        add_point(robust_props, "robust", "#54a24b")

    svg = _svg_scatter(xs, ys, labs, cols, title="Distance vs Risk (representatives)") if xs and ys else ""

    cand_mean = (j_mean.get("routes", {}) or {}).get("candidates", [])
    cand_cvar = (j_cvar.get("routes", {}) or {}).get("candidates", [])

    html_lines = [
        "<html><head><meta charset='utf-8'><title>Robust Compare</title></head><body>",
        f"<h1>Robust Compare {ym} / {scenario_id}</h1>",
        f"<p>alpha={alpha:.2f}</p>",
        "<h2>Summary</h2>",
        *table,
        "<h2>Scatter</h2>",
        svg or "<p><em>Not enough data for scatter.</em></p>",
        "<h2>Mean candidates (top)</h2>",
        f"<pre>{json.dumps(cand_mean[:3], ensure_ascii=False, indent=2)}</pre>",
        "<h2>CVaR candidates (top)</h2>",
        f"<pre>{json.dumps(cand_cvar[:3], ensure_ascii=False, indent=2)}</pre>",
    ]
    if robust_exists:
        html_lines += [
            "<h2>Robust route (ES-selected)</h2>",
            f"<p>GeoJSON: {robust_geo.as_posix()}</p>",
        ]
    else:
        html_lines += [
            "<h2>Robust route</h2>",
            "<p><em>Not found. Run route.robust to generate.</em></p>",
        ]
    html_lines += ["</body></html>"]

    out_html = REPORT_DIR / f"robust_{ym}_{scenario_id}.html"
    out_html.write_text("\n".join(html_lines), encoding="utf-8")
    return str(out_html)


__all__ = ["build_compare"]
