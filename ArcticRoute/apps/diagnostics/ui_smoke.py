# REUSE
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
AR_DIR = REPO_ROOT / "ArcticRoute"


@dataclass
class PageResult:
    page: str
    ok: bool
    error_code: Optional[str] = None
    error_msg: Optional[str] = None
    hint: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        out = {"page": self.page, "ok": self.ok}
        if self.error_code:
            out["error_code"] = self.error_code
        if self.error_msg:
            out["error_msg"] = self.error_msg
        if self.hint:
            out["hint"] = self.hint
        if self.details is not None:
            out["details"] = self.details
        return out


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_runtime() -> Dict[str, Any]:
    # 支持 ArcticRoute/config/runtime.yaml 与 config/runtime.yaml 两种位置
    for p in [AR_DIR/"config"/"runtime.yaml", REPO_ROOT/"config"/"runtime.yaml"]:
        data = _load_yaml(p)
        if data:
            return data
    return {}


def _get_default_ym_scen() -> tuple[str, str]:
    # 尝试从产物推断 ym；否则回退 202412
    ym = None
    risk_dir = AR_DIR/"data_processed"/"risk"
    if risk_dir.exists():
        cands = sorted(risk_dir.glob("risk_fused_*.nc"))
        if cands:
            name = cands[-1].stem  # risk_fused_YYYYMM
            parts = name.split("_")
            if len(parts) >= 3:
                ym = parts[2]
    if ym is None:
        ym = "202412"
    # 场景：优先 configs/scenarios.yaml
    scen = "nsr_wbound_smoke"
    sc_path = REPO_ROOT/"configs"/"scenarios.yaml"
    sc = _load_yaml(sc_path)
    if sc:
        arr = sc.get("scenarios") or []
        if isinstance(arr, list) and arr:
            sid = arr[0].get("id") if isinstance(arr[0], dict) else None
            if isinstance(sid, str) and sid:
                scen = sid
    return ym, scen


def _write_reports(rows: List[PageResult]) -> Dict[str, str]:
    dev_dir = REPO_ROOT/"reports"/"dev"
    dev_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jpath = dev_dir/"ui_smoke_result.json"
    payload = {"ts": ts, "results": [r.to_dict() for r in rows]}
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # HTML（人类可读）
    hpath = dev_dir/"ui_smoke_result.html"
    def _row(r: PageResult) -> str:
        status = "✅ OK" if r.ok else "❌ FAIL"
        code = r.error_code or "-"
        msg = (r.error_msg or "").replace("<", "&lt;").replace(">", "&gt;")
        hint = (r.hint or "").replace("<", "&lt;").replace(">", "&gt;")
        return f"<tr><td>{r.page}</td><td>{status}</td><td>{code}</td><td>{msg}</td><td>{hint}</td></tr>"
    rows_html = "\n".join(_row(r) for r in rows)
    html = f"""
    <html><head><meta charset='utf-8'><title>UI Smoke Result</title>
    <style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:6px 10px}}th{{background:#f5f5f5}}</style>
    </head><body>
      <h1>UI 自检结果</h1>
      <p>时间：{ts}</p>
      <table>
        <tr><th>页面</th><th>状态</th><th>错误码</th><th>错误信息</th><th>建议</th></tr>
        {rows_html}
      </table>
    </body></html>
    """
    hpath.write_text(html, encoding="utf-8")
    return {"json": str(jpath), "html": str(hpath)}


def run_ui_smoke(profile: str = "default") -> Dict[str, Any]:  # noqa: ARG001
    """执行 UI 多页后端冒烟。返回结果字典并写入 JSON/HTML。"""
    rt = _read_runtime()
    pages_cfg = ((rt.get("ui") or {}).get("pages") or {}) if isinstance(rt, dict) else {}
    ym, scen = _get_default_ym_scen()

    results: List[PageResult] = []

    # Live
    if pages_cfg.get("live", True):
        try:
            import sys, subprocess as sp
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "route.replan", "--scenario", scen]
            # 倾向使用 fused 以减少依赖；不强制 --live
            proc = sp.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            if proc.returncode == 0:
                results.append(PageResult(page="Live", ok=True))
            else:
                results.append(PageResult(page="Live", ok=False, error_code="REPLAN_FAIL", error_msg=proc.stderr.strip()[:400], hint="检查 risk_fused_{ym}.nc 是否存在；或运行: python -m ArcticRoute.api.cli risk.fuse --ym "+ym))
        except Exception as e:
            results.append(PageResult(page="Live", ok=False, error_code="REPLAN_ERROR", error_msg=str(e), hint="确认 configs/scenarios.yaml 存在且包含场景"))

    # Reports（小样本或最小 include）
    if pages_cfg.get("reports", True):
        try:
            import sys, subprocess as sp
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "report.build", "--ym", ym, "--include", "pareto"]
            proc = sp.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            if proc.returncode == 0:
                results.append(PageResult(page="Reports", ok=True))
            else:
                results.append(PageResult(page="Reports", ok=False, error_code="REPORT_BUILD_FAIL", error_msg=proc.stderr.strip()[:400], hint="查看 reports/dev/ui_smoke_result.html 或直接运行同命令重试"))
        except Exception as e:
            results.append(PageResult(page="Reports", ok=False, error_code="REPORT_ERROR", error_msg=str(e), hint="确认 route.scan 产物与依赖存在"))

    # Compare（加载 pareto front）
    if pages_cfg.get("compare", True):
        pf = AR_DIR/"reports"/"d_stage"/"phaseG"/f"pareto_front_{ym}_{scen}.json"
        if pf.exists():
            results.append(PageResult(page="Compare", ok=True))
        else:
            results.append(PageResult(page="Compare", ok=False, error_code="NO_PARETO", error_msg=str(pf), hint=f"先运行: python -m ArcticRoute.api.cli route.scan --scenario {scen} --ym {ym}"))

    # Explain（尝试加载上一条 explain 产物）
    if pages_cfg.get("explain", True):
        phaseH = AR_DIR/"reports"/"d_stage"/"phaseH"
        cands = sorted(phaseH.glob(f"route_attr_{ym}_*.json"))
        if cands:
            results.append(PageResult(page="Explain", ok=True))
        else:
            results.append(PageResult(page="Explain", ok=False, error_code="EXPLAIN_DATA_MISSING", error_msg="route_attr_* 缺失", hint="先在 Compare/Routes 选择一条路线，或运行: python -m ArcticRoute.api.cli route.explain --route <path> --ym "+ym))

    # Review（检查依赖/做一次 dry-run 近似：仅验证 schema 可用与风险层存在）
    if pages_cfg.get("review", True):
        try:
            # 依赖：schema.py 可导入 + 风险层存在 + 场景存在
            from ArcticRoute.core.feedback import schema as _schema  # type: ignore  # noqa: F401
            risk = AR_DIR/"data_processed"/"risk"/f"risk_fused_{ym}.nc"
            sc_path = REPO_ROOT/"configs"/"scenarios.yaml"
            if not risk.exists():
                results.append(PageResult(page="Review", ok=False, error_code="NO_RISK_DATA", error_msg=str(risk), hint=f"先运行: python -m ArcticRoute.api.cli risk.fuse --ym {ym}"))
            elif not sc_path.exists():
                results.append(PageResult(page="Review", ok=False, error_code="SCENARIO_MISSING", error_msg=str(sc_path), hint="确认 configs/scenarios.yaml 存在且含目标场景"))
            else:
                results.append(PageResult(page="Review", ok=True))
        except Exception as e:
            results.append(PageResult(page="Review", ok=False, error_code="FEEDBACK_INVALID", error_msg=str(e), hint="用 core/feedback/schema.py 校验 jsonl 格式"))

    # Health
    if pages_cfg.get("health", True):
        try:
            # 直接复用 CLI 入口以写出报告
            import sys, subprocess as sp
            cmd = [sys.executable, "-m", "ArcticRoute.api.cli", "health.check", "--out", "reports/health/health_latest.json"]
            proc = sp.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
            if proc.returncode == 0:
                results.append(PageResult(page="Health", ok=True))
            else:
                results.append(PageResult(page="Health", ok=False, error_code="HEALTH_FAIL", error_msg=proc.stderr.strip()[:400], hint="查看 reports/health/ 下 json/html 详情"))
        except Exception as e:
            results.append(PageResult(page="Health", ok=False, error_code="HEALTH_ERROR", error_msg=str(e), hint="确认 ArcticRoute.api.health 可导入"))

    outs = _write_reports(results)
    summary = {
        "ok": all(r.ok for r in results),
        "results": [r.to_dict() for r in results],
        **outs,
    }
    return summary

