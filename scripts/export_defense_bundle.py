from __future__ import annotations

import csv
import json
import tempfile
import zipfile
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


def _get_attr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _get_components(route: Any) -> dict:
    if route is None:
        return {}
    comps = _get_attr(route, "cost_components", {}) or {}
    notes = _get_attr(route, "notes", {}) or {}
    breakdown = None
    if isinstance(notes, dict):
        breakdown = notes.get("breakdown")
    if breakdown is not None and getattr(breakdown, "component_totals", None):
        comps = breakdown.component_totals
    return comps or {}


def _risk_cost(components: dict) -> float:
    keys = [
        "ice_risk",
        "wave_risk",
        "ais_density",
        "edl_risk",
        "edl_uncertainty_penalty",
        "ice_class_soft",
        "ice_class_hard",
    ]
    total = 0.0
    for k in keys:
        v = components.get(k)
        if v is None:
            continue
        try:
            total += float(v)
        except Exception:
            continue
    return total


def _distance_of(route: Any) -> float | None:
    dist = _get_attr(route, "distance_km")
    if dist is None:
        dist = _get_attr(route, "approx_length_km")
    return dist


def _coords_lonlat(route: Any) -> list[list[float]]:
    coords = _get_attr(route, "path_lonlat") or _get_attr(route, "coords") or []
    out = []
    for pt in coords:
        try:
            lat, lon = pt
            out.append([float(lon), float(lat)])
        except Exception:
            continue
    return out


def build_defense_bundle(
    scenario_id: str,
    routes_info: Iterable[Any],
    env_meta: dict,
    eval_summary: dict | None = None,
) -> Path:
    """
    根据当前场景与三条方案，生成一个 zip 文件，包含：
    - summary.csv: 三条方案的距离、成本、各风险分量
    - routes.geojson: 三条路线的 GeoJSON 折线
    - kpi_report.md: 文本形式的 KPI & EDL 评估摘要
    返回 zip 文件的路径（位于 reports/defense_bundle 下）。
    """
    safe_id = scenario_id or "unknown"
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = Path("reports") / "defense_bundle"
    output_dir.mkdir(parents=True, exist_ok=True)

    route_list = list(routes_info) if not isinstance(routes_info, dict) else list(routes_info.values())

    def _total_cost_of(route: Any) -> float | None:
        val = _get_attr(route, "total_cost")
        if val is not None:
            return val
        comps = _get_components(route)
        if comps:
            try:
                return float(sum(v for v in comps.values() if v is not None))
            except Exception:
                return None
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        summary_path = tmp_path / "summary.csv"
        routes_path = tmp_path / "routes.geojson"
        report_path = tmp_path / "kpi_report.md"

        # summary.csv
        fieldnames = [
            "scenario_id",
            "mode",
            "label",
            "reachable",
            "distance_km",
            "total_cost",
            "risk_cost",
            "edl_risk_cost",
            "edl_uncertainty_cost",
            "ice_cost",
            "wave_cost",
            "ice_class_soft",
            "ice_class_hard",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for route in route_list:
                reachable = bool(_get_attr(route, "reachable", False))
                comps = _get_components(route)
                risk_cost = _risk_cost(comps)
                writer.writerow(
                    {
                        "scenario_id": safe_id,
                        "mode": _get_attr(route, "mode"),
                        "label": _get_attr(route, "label"),
                        "reachable": reachable,
                        "distance_km": _distance_of(route),
                        "total_cost": _total_cost_of(route),
                        "risk_cost": risk_cost,
                        "edl_risk_cost": comps.get("edl_risk"),
                        "edl_uncertainty_cost": comps.get("edl_uncertainty_penalty"),
                        "ice_cost": comps.get("ice_risk"),
                        "wave_cost": comps.get("wave_risk"),
                        "ice_class_soft": comps.get("ice_class_soft"),
                        "ice_class_hard": comps.get("ice_class_hard"),
                    }
                )

        # routes.geojson
        features = []
        for route in route_list:
            coords = _coords_lonlat(route)
            if not coords:
                continue
            comps = _get_components(route)
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "scenario_id": safe_id,
                        "mode": _get_attr(route, "mode"),
                        "label": _get_attr(route, "label"),
                        "distance_km": _distance_of(route),
                        "total_cost": _total_cost_of(route),
                        "risk_cost": _risk_cost(comps),
                        "edl_risk_cost": comps.get("edl_risk"),
                        "edl_uncertainty_cost": comps.get("edl_uncertainty_penalty"),
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": coords,
                    },
                }
            )

        geojson = {"type": "FeatureCollection", "features": features}
        routes_path.write_text(json.dumps(geojson, ensure_ascii=False, indent=2), encoding="utf-8")

        # kpi_report.md
        baseline = next((r for r in route_list if _get_attr(r, "mode") == "efficient"), None)
        safe_route = next((r for r in route_list if _get_attr(r, "mode") == "edl_safe"), None)
        robust_route = next((r for r in route_list if _get_attr(r, "mode") == "edl_robust"), None)

        def _delta_line(target: Any) -> str:
            if baseline is None or target is None:
                return "缺少基准或目标方案。"
            d0 = _distance_of(baseline) or 0.0
            d1 = _distance_of(target)
            c0 = _total_cost_of(baseline) or 0.0
            c1 = _total_cost_of(target)
            risk0 = _risk_cost(_get_components(baseline))
            risk1 = _risk_cost(_get_components(target))
            parts: list[str] = []
            if d0 > 0 and d1 is not None:
                parts.append(f"距离变化 {((d1 - d0) / d0 * 100):+.1f}%")
            if c0 > 0 and c1 is not None:
                parts.append(f"成本变化 {((c1 - c0) / c0 * 100):+.1f}%")
            if risk0 > 0 and risk1 is not None:
                parts.append(f"风险降低 {(risk0 - risk1) / risk0 * 100:.1f}%")
            return "；".join(parts) if parts else "缺少指标。"

        lines = [
            f"# KPI 报告 | 场景：{safe_id}",
            f"- 生成时间：{datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            f"- 环境元信息：{json.dumps(env_meta, ensure_ascii=False)}",
            "",
            "## 基线与对比",
            f"- edl_safe：{_delta_line(safe_route)}",
            f"- edl_robust：{_delta_line(robust_route)}",
            "",
        ]
        if eval_summary:
            lines.append("## EDL 评估摘要")
            if is_dataclass(eval_summary):
                eval_summary = asdict(eval_summary)
            lines.append("```json")
            lines.append(json.dumps(eval_summary, ensure_ascii=False, indent=2))
            lines.append("```")
        report_path.write_text("\n".join(lines), encoding="utf-8")

        zip_path = output_dir / f"{safe_id}_defense_bundle_{timestamp}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(summary_path, arcname=summary_path.name)
            zf.write(routes_path, arcname=routes_path.name)
            zf.write(report_path, arcname=report_path.name)

    return zip_path
