from __future__ import annotations

# Phase H | Audit page
# - collect_meta(paths) -> dict: 递归收集 .meta.json 并合并
# - render_audit_html(meta, out_html)

import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = REPO_ROOT / "ArcticRoute" / "reports" / "d_stage" / "phaseH"


def _iter_meta_files(root_paths: Iterable[Path]) -> List[Path]:
    outs: List[Path] = []
    for p in root_paths:
        p = Path(p)
        if p.is_file() and p.suffix == ".json" and p.name.endswith(".meta.json"):
            outs.append(p)
        elif p.is_file() and p.suffix in {".json", ".html", ".png", ".gif", ".mp4", ".geojson"}:
            m = Path(str(p) + ".meta.json")
            if m.exists():
                outs.append(m)
        elif p.is_dir():
            for dp, _, files in os.walk(p):
                for fn in files:
                    if fn.endswith(".meta.json"):
                        outs.append(Path(dp) / fn)
    return outs


def collect_meta(paths: List[Path]) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for m in _iter_meta_files(paths):
        try:
            obj = json.loads(m.read_text(encoding="utf-8"))
            lid = obj.get("logical_id") or m.name
            out[lid] = obj
        except Exception:
            continue
    return out


def _parse_ym_from_name(name: str) -> str:
    # 期望 audit_{ym}_{scenario}.html
    base = Path(name).stem
    parts = base.split("_")
    if len(parts) >= 3 and parts[0] == "audit":
        return parts[1]
    return ""


def _load_multimodal_manifest(ym: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not ym:
        return rows
    try:
        for fp in sorted(REPORT_DIR.glob(f"route_attr_{ym}_*.json")):
            try:
                obj = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            segs = obj.get("segments") or []
            for r in segs:
                mode = r.get("mode", "sea")
                dist_km = float(r.get("distance_m", 0.0)) / 1000.0
                # 关键信息列：risk/prior/interact/distance/transfer（若有）
                entry = {
                    "file": fp.name,
                    "mode": str(mode),
                    "i": str(r.get("i", 0)),
                    "distance_km": f"{dist_km:.3f}",
                    "c_risk": f"{float(next((v for k,v in r.items() if k.startswith('c_') and 'risk' in k), 0.0)):.3f}",
                    "c_prior": f"{float(next((v for k,v in r.items() if k.startswith('c_') and 'prior' in k), 0.0)):.3f}",
                    "c_interact": f"{float(next((v for k,v in r.items() if k.startswith('c_') and 'interact' in k), 0.0)):.3f}",
                    "c_distance": f"{float(next((v for k,v in r.items() if k.startswith('c_') and 'distance' in k), 0.0)):.3f}",
                    "c_transfer": f"{float(r.get('c_transfer', 0.0)):.3f}",
                }
                rows.append(entry)
    except Exception:
        return rows
    return rows


def render_audit_html(meta: Dict[str, dict], out_html: Path) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    # 扁平列表
    rows_simple: List[str] = []
    for lid, m in sorted(meta.items()):
        inputs = m.get("inputs") or []
        run_id = m.get("run_id", "")
        sha = m.get("git_sha", "")
        cfg = m.get("config_hash", "")
        rows_simple.append(f"<tr><td>{lid}</td><td>{run_id}</td><td>{sha}</td><td>{cfg}</td><td><pre>{json.dumps(inputs, ensure_ascii=False)}</pre></td></tr>")

    ym = _parse_ym_from_name(out_html.name)
    manifest = _load_multimodal_manifest(ym)
    rows_manifest = []
    for r in manifest:
        rows_manifest.append(
            f"<tr><td>{r['file']}</td><td>{r['i']}</td><td>{r['mode']}</td>"
            f"<td>{r['distance_km']}</td><td>{r['c_risk']}</td><td>{r['c_prior']}</td><td>{r['c_interact']}</td><td>{r['c_distance']}</td><td>{r['c_transfer']}</td></tr>"
        )

    html = [
        "<html><head><meta charset='utf-8'><title>Audit</title>",
        "<style>body{font-family:Arial} table{border-collapse:collapse} td,th{border:1px solid #ddd;padding:6px} h2{margin-top:24px}</style>",
        "</head><body>",
        "<h1>ArcticRoute Phase H Audit</h1>",
        "<h2>Artifacts</h2>",
        "<table>",
        "<thead><tr><th>logical_id</th><th>run_id</th><th>git_sha</th><th>config_hash</th><th>inputs</th></tr></thead>",
        "<tbody>",
        *rows_simple,
        "</tbody></table>",
    ]
    if rows_manifest:
        html += [
            "<h2>Multimodal Route Manifest</h2>",
            "<table>",
            "<thead><tr><th>file</th><th>seg</th><th>mode</th><th>distance_km</th><th>c_risk</th><th>c_prior</th><th>c_interact</th><th>c_distance</th><th>c_transfer</th></tr></thead>",
            "<tbody>",
            *rows_manifest,
            "</tbody></table>",
        ]
    html.append("</body></html>")

    out_html.write_text("\n".join(html), encoding="utf-8")
    # 写 meta
    try:
        with open(str(out_html) + ".meta.json", "w", encoding="utf-8") as f:
            import time
            json.dump({"logical_id": out_html.name, "inputs": list(meta.keys()), "run_id": time.strftime("%Y%m%dT%H%M%S")}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return out_html


__all__ = ["collect_meta", "render_audit_html"]



