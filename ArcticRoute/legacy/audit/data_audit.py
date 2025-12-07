# REUSE: 体检任务引入，可回退
# 说明：尽量复用现有 audit/health 能力，仅做扫描与汇总，不改核心算法。
from __future__ import annotations
import json, os, math, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import xarray as xr  # type: ignore
except Exception:
    xr = None  # type: ignore
try:
    import pandas as pd  # type: ignore
except Exception:
    pd = None  # type: ignore

# 日志
try:
    from logging_config import get_logger
except Exception:  # pragma: no cover
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)
logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC = REPO_ROOT / "ArcticRoute"
OUT_DIR = REPO_ROOT / "reports" / "audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- 工具 -----------------

def _exists_nonempty(p: Path) -> bool:
    return p.exists() and (p.is_file() and p.stat().st_size > 0 or p.is_dir())


def _safe_open_nc(p: Path):
    if xr is None or not p.exists():
        return None
    try:
        return xr.open_dataset(p)
    except Exception:
        return None


def _safe_read_parquet(p: Path):
    if pd is None or not p.exists():
        return None
    try:
        return pd.read_parquet(p)
    except Exception:
        return None


# ----------------- 检查实现 -----------------

def check_ais_features(ym: str) -> Dict[str, Any]:
    base = REPO_ROOT / "data_processed" / "ais"
    if not base.exists():
        base = ARCTIC / "data_processed" / "ais"
    feat_dir = REPO_ROOT / "data_processed" / "features"
    if not feat_dir.exists():
        feat_dir = ARCTIC / "data_processed" / "features"
    
    findings: List[str] = []
    status = "unknown"
    rows = {}

    tracks = base / f"tracks_{ym}.parquet"
    segidx = base / f"segment_index_{ym}.parquet"
    dens = feat_dir / f"ais_density_{ym}.nc"
    f_nc = feat_dir / f"features_{ym}.nc"

    tracks_exist = _exists_nonempty(tracks)
    segidx_exist = _exists_nonempty(segidx)
    dens_exist = _exists_nonempty(dens)
    f_nc_exist = _exists_nonempty(f_nc)

    if not any([tracks_exist, segidx_exist, dens_exist, f_nc_exist]):
        status = "missing_optional"
        findings.append("本月未生成 AIS 原始轨迹，仅影响该月的先验训练/更新，不影响当前 demo")
        return {"status": status, "issues": findings, "stats": rows}

    tracks_missing = not tracks_exist
    if tracks_exist:
        df = _safe_read_parquet(tracks)
        n = int(len(df)) if df is not None else -1
        rows["tracks_rows"] = n
        if n <= 0:
            findings.append("tracks 行数<=0")
    else:
        findings.append("缺少 tracks_*.parquet")

    if segidx_exist:
        df = _safe_read_parquet(segidx)
        n = int(len(df)) if df is not None else -1
        rows["segment_index_rows"] = n
        if n <= 0:
            findings.append("segment_index 行数<=0")
    else:
        findings.append("缺少 segment_index_*.parquet")
        
    if pd is not None and tracks_exist:
        df = _safe_read_parquet(tracks)
        if df is not None and {'lat','lon'}.issubset(df.columns):
            bad = df[(df['lat'].abs().gt(90)) | (df['lon'].abs().gt(180)) | ((df['lat']==0) & (df['lon']==0))]
            rows["bad_latlon"] = int(len(bad))
            if len(bad) > 0:
                findings.append("存在异常经纬度")

    for nc in [dens, f_nc]:
        if _exists_nonempty(nc):
            ds = _safe_open_nc(nc)
            if ds is not None:
                try:
                    v = list(ds.data_vars)[0]
                except Exception:
                    v = None
                if v is not None:
                    da = ds[v]
                    import numpy as np  # type: ignore
                    arr = da.values
                    m = float(np.nanmean(arr)) if np.isfinite(arr).any() else float('nan')
                    s = float(np.nanstd(arr)) if np.isfinite(arr).any() else float('nan')
                    nz = np.isfinite(arr).sum()
                    tot = float(arr.size)
                    nan_ratio = 1.0 - (nz / tot) if tot > 0 else 1.0
                    rows[str(nc.name)+"_mean"] = m
                    rows[str(nc.name)+"_std"] = s
                    rows[str(nc.name)+"_nan_ratio"] = nan_ratio
                    if (abs(s) < 1e-12) or math.isclose(s, 0.0, abs_tol=1e-12):
                        findings.append(f"{nc.name} 方差≈0")
                    if nan_ratio > 0.5:
                        findings.append(f"{nc.name} NaN比例>50%")
        else:
            findings.append(f"缺少 {nc.name}")

    if (tracks_missing or not segidx_exist) and (dens_exist or f_nc_exist):
        status = "missing_optional"
    else:
        status = "ok" if not findings else "suspect"
        
    return {"status": status, "issues": findings, "stats": rows}


def check_prior(ym: str) -> Dict[str, Any]:
    risk_dir = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
    prior_nc = risk_dir / f"prior_penalty_{ym}.nc"
    # 接受等价产物：prior_transformer_{ym}.nc 内的 PriorPenalty/P_prior
    prior_transformer_nc = REPO_ROOT / "ArcticRoute" / "data_processed" / "prior" / f"prior_transformer_{ym}.nc"
    # 中心线：接受 reports/phaseE/center 与 data_processed/prior/centerlines 两种
    cl_geo1 = REPO_ROOT / "ArcticRoute" / "reports" / "phaseE" / "center" / f"prior_centerlines_{ym}.geojson"
    cl_geo2 = REPO_ROOT / "ArcticRoute" / "data_processed" / "prior" / f"prior_centerlines_{ym}.geojson"
    corr_sel = risk_dir / f"prior_corridor_selected_{ym}.nc"

    findings: List[str] = []
    stats: Dict[str, Any] = {}

    def _collect_prior_stats(ds) -> None:
        import numpy as np
        v = None
        if "PriorPenalty" in ds.variables:
            v = "PriorPenalty"
        elif "prior_penalty" in ds.variables:
            v = "prior_penalty"
        elif "P_prior" in ds.variables:
            # 可从 P_prior 推导 PriorPenalty
            arr_p = ds["P_prior"].values.astype(float)
            arr = (1.0 - arr_p)
            stats["prior_std"] = float(np.nanstd(arr))
            p_prior = 1.0 - arr
            stats["p_prior_mean"] = float(np.nanmean(p_prior))
            nz = np.isfinite(arr).sum(); tot = arr.size
            stats["prior_coverage"] = float(nz / tot) if tot>0 else 0.0
            if np.nanmin(arr) < -1e-3 or np.nanmax(arr) > 1 + 1e-3:
                findings.append("PriorPenalty 范围异常")
            return
        elif ds.data_vars:
            v = list(ds.data_vars)[0]
        if v is None:
            findings.append("prior 栅格缺少可识别变量")
            return
        arr = ds[v].values.astype(float)
        import numpy as np
        if np.nanmin(arr) < 0 - 1e-3 or np.nanmax(arr) > 1 + 1e-3:
            findings.append("PriorPenalty 范围异常")
        stats["prior_std"] = float(np.nanstd(arr))
        p_prior = 1.0 - arr
        if np.nanmin(p_prior) < -1e-3 or np.nanmax(p_prior) > 1 + 1e-3:
            findings.append("P_prior ∉[0,1]")
        nz = np.isfinite(arr).sum(); tot = arr.size
        stats["prior_coverage"] = float(nz / tot) if tot>0 else 0.0
        stats["p_prior_mean"] = float(np.nanmean(p_prior))

    # 读取 prior
    prior_found = False
    if _exists_nonempty(prior_nc) and xr is not None:
        ds = _safe_open_nc(prior_nc)
        if ds is not None:
            _collect_prior_stats(ds)
            prior_found = True
    elif _exists_nonempty(prior_transformer_nc) and xr is not None:
        ds = _safe_open_nc(prior_transformer_nc)
        if ds is not None:
            _collect_prior_stats(ds)
            prior_found = True
        else:
            findings.append("无法打开 prior_transformer_*.nc")

    if not prior_found:
        stats["prior_penalty_status"] = "planned_disabled"
        findings.append("缺少 prior_penalty/transformer_*.nc (planned_disabled, 当前版本仅使用概率地图 P_prior)")

    # 中心线存在性
    if not (_exists_nonempty(cl_geo1) or _exists_nonempty(cl_geo2)):
        findings.append("缺少 prior_centerlines_*.geojson")
    # 检查 corridor selected 文件，若缺失则标记为 planned_disabled
    if not _exists_nonempty(corr_sel):
        stats["corridor_selected_status"] = "planned_disabled"
        findings.append("缺少 prior_corridor_selected_*.nc (planned_disabled)")
    else:
        stats["corridor_selected_status"] = "ok"

    non_disabled_findings = [f for f in findings if "planned_disabled" not in f]
    status = "suspect" if non_disabled_findings else "ok"
    return {"status": status, "issues": findings, "stats": stats}


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

def check_risk_and_fusion(ym: str) -> Dict[str, Any]:
    rdir = REPO_ROOT / "ArcticRoute" / "data_processed" / "risk"
    findings: List[str] = []
    stats: Dict[str, Any] = {}
    
    risk_fuse_config_path = REPO_ROOT / "configs" / f"risk_fuse_{ym}.yaml"
    risk_fuse_config = _load_yaml_config(risk_fuse_config_path)
    use_wave = risk_fuse_config.get("use_wave", True)

    to_check = {
        "R_ice": [rdir / f"R_ice_eff_{ym}.nc"],
        "R_wave": [rdir / f"R_wave_{ym}.nc"],
        "R_acc": [rdir / f"R_acc_{ym}.nc", rdir / f"risk_accident_{ym}.nc"],
        "risk_fused": [rdir / f"risk_fused_{ym}.nc"],
    }

    is_sparse = False

    for name, paths in to_check.items():
        found_path = None
        for p in paths:
            if _exists_nonempty(p):
                found_path = p
                break

        if not found_path:
            if name == "R_wave" and not use_wave:
                stats[name] = {"status": "planned_disabled"}
                findings.append(f"{paths[0].name} is planned_disabled (use_wave=false)")
            else:
                findings.append(f"缺少 {paths[0].name}")
            continue

        p = found_path
        ds = _safe_open_nc(p)
        if ds is None:
            findings.append(f"无法打开 {p.name}")
            continue
        try:
            v = "risk" if "risk" in ds.variables else (list(ds.data_vars)[0] if ds.data_vars else None)
            if v is None:
                findings.append(f"{p.name} 无变量")
                continue
            da = ds[v]
            import numpy as np
            arr = da.values.astype(float)
            mn, mx = float(np.nanmin(arr)), float(np.nanmax(arr))
            mu, sd = float(np.nanmean(arr)), float(np.nanstd(arr))
            nz = np.isfinite(arr).sum(); tot = arr.size
            nan_ratio = 1.0 - (nz / tot) if tot>0 else 1.0
            stats[name] = {"min": mn, "max": mx, "mean": mu, "std": sd, "nan_ratio": nan_ratio}
            if mn < -1e-3 or mx > 1 + 1e-3:
                findings.append(f"{name} 超出[0,1]")
            if nan_ratio > 0.5:
                findings.append(f"{name} NaN比例>50%")
            if sd < 1e-6:
                if name in ["R_ice", "R_acc"]:
                    is_sparse = True
                    findings.append(f"{name} 方差过低≈0. 该月风险信号极弱，仅适合结构演示")
                else:
                    findings.append(f"{name} 方差过低≈0")
        except Exception:
            findings.append(f"读取 {p.name} 失败")

    other_findings = [f for f in findings if "方差过低" not in f and "planned_disabled" not in f]
    if is_sparse and not other_findings:
        status = "data_sparse"
    else:
        status = "ok" if not findings else "suspect"

    return {"status": status, "issues": findings, "stats": stats}


def check_eco_and_routes(ym: str) -> Dict[str, Any]:
    findings: List[str] = []
    stats: Dict[str, Any] = {}
    # ECO
    eco_dir = ARCTIC / "data_processed" / "eco"
    if eco_dir.exists():
        # 不强制文件名，仅记录存在性
        stats["eco_files"] = len(list(eco_dir.glob("*.nc")))
        if stats["eco_files"] <= 0:
            findings.append("eco 目录存在但无文件")
    # 路由
    rdir = ARCTIC / "data_processed" / "routes"
    cand = list(rdir.glob(f"route_{ym}_*.geojson"))
    stats["routes_count"] = len(cand)
    if not cand:
        findings.append("缺少代表性路线 GeoJSON")
    else:
        import json as _json
        ok_cnt = 0
        for p in cand[:3]:
            try:
                gj = _json.loads(p.read_text(encoding='utf-8'))
                coords = (gj.get("features", [{}])[0].get("geometry", {}).get("coordinates") or [])
                ok_cnt += 1 if (isinstance(coords, list) and len(coords) >= 2) else 0
            except Exception:
                pass
        if ok_cnt == 0:
            findings.append("路线坐标不可解析/点数不足")
    return {"status": ("ok" if not findings else "suspect"), "issues": findings, "stats": stats}


def check_reports(ym: str) -> Dict[str, Any]:
    # 复用现有 realdata 报告
    out = {
        "status": "unknown",
        "issues": [],
        "stats": {},
    }
    real_j = REPO_ROOT / "reports" / "audit" / "realdata_report.json"
    if real_j.exists():
        try:
            payload = json.loads(real_j.read_text(encoding='utf-8'))
            out["stats"]["realdata_checked"] = int(payload.get("summary", {}).get("checked", 0))
            out["status"] = "ok"
        except Exception:
            out["issues"].append("realdata_report.json 无法解析")
            out["status"] = "suspect"
    else:
        out["issues"].append("缺少 realdata_report.json")
        out["status"] = "suspect"
    return out


def render_html(payload: Dict[str, Any]) -> str:
    def _sec(name: str, item: Dict[str, Any]) -> str:
        import html
        issues = ''.join(f"<li>{html.escape(str(x))}</li>" for x in item.get('issues', []))
        st = item.get('status', 'unknown')
        stats = item.get('stats', {})
        stats_rows = ''.join(f"<tr><td>{html.escape(str(k))}</td><td>{html.escape(str(v))}</td></tr>" for k,v in stats.items())
        return f"<h2>{html.escape(name)} — {html.escape(st)}</h2><ul>{issues}</ul><table><tbody>{stats_rows}</tbody></table>"
    html_doc = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Data Audit</title>
<style>body{{font-family:Arial,sans-serif}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body>
<h1>产物体检报告</h1>
{_sec('AIS / 特征', payload.get('ais_features', {}))}
{_sec('先验层', payload.get('prior', {}))}
{_sec('风险与融合', payload.get('risk_fuse', {}))}
{_sec('绿色航行/路由', payload.get('eco_routes', {}))}
{_sec('报告', payload.get('reports', {}))}
{_sec('UI 烟雾', payload.get('ui', {}))}
</body></html>
"""
    return html_doc


def run_ui_smoke_section() -> Dict[str, Any]:
    # 复用 ui.smoke
    try:
        import sys
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from ArcticRoute.apps.diagnostics.ui_smoke import run_ui_smoke  # REUSE
        res = run_ui_smoke(profile="default")
        results = res.get('results', [])
        
        DISABLED_CODES = {"NO_PARETO", "EXPLAIN_DATA_MISSING", "NO_RISK_DATA", "SCENARIO_MISSING", "REPLAN_FAIL", "REPORT_BUILD_FAIL"}

        ok_pages = [r['page'] for r in results if r.get('ok')]
        broken_pages = {r['page']: r.get('error_code', 'UNKNOWN') for r in results if not r.get('ok') and r.get('error_code') not in DISABLED_CODES}
        disabled_pages = {r['page']: r.get('error_code', 'UNKNOWN') for r in results if not r.get('ok') and r.get('error_code') in DISABLED_CODES}

        issues = [f"{page} is broken: {code}" for page, code in broken_pages.items()]
        issues.extend([f"{page} is disabled (missing data): {code}" for page, code in disabled_pages.items()])

        status = "broken" if broken_pages else "ok"
        
        stats = {
            "ok_pages": ok_pages,
            "broken_pages": broken_pages,
            "disabled_pages": disabled_pages,
            "json_report": res.get('json'),
            "html_report": res.get('html')
        }
        
        return {"status": status, "issues": issues, "stats": stats}
    except Exception as e:
        return {"status": "suspect", "issues": [f"ui.smoke 运行失败: {e}"], "stats": {}}


def run_data_audit(ym: str = "202412") -> Dict[str, Any]:
    payload = {
        "ais_features": check_ais_features(ym),
        "prior": check_prior(ym),
        "risk_fuse": check_risk_and_fusion(ym),
        "eco_routes": check_eco_and_routes(ym),
        "reports": check_reports(ym),
        "ui": run_ui_smoke_section(),
    }
    # 汇总状态
    st_map = {k: v.get('status','unknown') for k,v in payload.items()}
    status_counter: Dict[str,int] = {}
    for s in st_map.values():
        status_counter[s] = status_counter.get(s, 0) + 1
    payload["summary"] = {"module_status": st_map, "bucket": status_counter}

    jpath = OUT_DIR / "data_audit.json"
    hpath = OUT_DIR / "data_audit.html"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    hpath.write_text(render_html(payload), encoding='utf-8')
    logger.info("data audit -> %s, %s", jpath, hpath)
    return payload


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", default="202412")
    args = ap.parse_args()
    run_data_audit(str(args.ym))





