#!/usr/bin/env python
from __future__ import annotations

import json
import sys
import re
import os
import itertools
from dataclasses import dataclass, asdict
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from datetime import datetime


# 尽量零耦合：不引入项目内部重型模块，仅做文件/路径与轻量解析

try:
    import click  # type: ignore
except Exception:  # 可选依赖：若无 click，后续提供 argparse 退化入口
    click = None  # type: ignore

ARCTICROUTE_DIR = Path(__file__).resolve().parents[1]
PKG_DIR = ARCTICROUTE_DIR  # 与规范命名保持一致
REPO_ROOT = ARCTICROUTE_DIR.parent
# 可选环境变量覆盖仓库根
try:
    import os as _os
    _env_root = _os.environ.get("ARCTICROUTE_ROOT")
    if _env_root:
        rr = Path(_env_root).expanduser().resolve()
        if rr.exists():
            REPO_ROOT = rr
except Exception:
    pass
REPORTS_DIR = REPO_ROOT / "reports" / "health"  # legacy summary outputs
PHASEB_REPORT_MD = os.path.join(str(REPO_ROOT), "reports", "health", "phaseB_precheck.md")


@dataclass
class CheckItem:
    name: str
    ok: bool
    detail: str = ""
    status: str = ""  # 可选: "OK" | "WARN" | "FAIL"，用于表达更细粒度状态（如日志格式）


@dataclass
class HealthReport:
    timestamp: str
    repo_root: str
    checks: List[CheckItem]
    extras: Dict[str, Any] | None = None  # 扩展信息（磁盘、数据目录、git、最近运行）

    @property
    def all_ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def to_json(self) -> Dict[str, Any]:
        payload = {
            "timestamp": self.timestamp,
            "repo_root": self.repo_root,
            "all_ok": self.all_ok,
            "checks": [asdict(c) for c in self.checks],
        }
        if self.extras:
            payload["extras"] = self.extras
        return payload

    def to_markdown(self) -> str:
        lines: List[str] = []
        lines.append(f"# ArcticRoute 健康检查汇总 ({self.timestamp})")
        lines.append("")
        lines.append(f"Repo: {self.repo_root}")
        lines.append(f"总体状态: {'OK' if self.all_ok else 'FAILED'}")
        lines.append("")
        lines.append("## 详细检查项")
        for c in self.checks:
            status = "✅ OK" if c.ok else "❌ FAIL"
            lines.append(f"- {status} {c.name}")
            if c.detail:
                lines.append(f"  - {c.detail}")
        lines.append("")
        return "\n".join(lines)


def _read_version() -> CheckItem:
    # 优先 egg-info/PKG-INFO
    pkg_info = REPO_ROOT / "arcticroute.egg-info" / "PKG-INFO"
    version = None
    if pkg_info.exists():
        try:
            for line in pkg_info.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("Version:"):
                    version = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass
    # 备选 pyproject.toml
    if version is None:
        pyproj = REPO_ROOT / "pyproject.toml"
        if pyproj.exists():
            try:
                m = re.search(r"^version\s*=\s*\"([^\"]+)\"", pyproj.read_text(encoding="utf-8", errors="ignore"), re.M)
                if m:
                    version = m.group(1)
            except Exception:
                pass
    ok = version is not None
    detail = f"version={version}" if version else "未能解析版本 (PKG-INFO/pyproject.toml)"
    return CheckItem(name="版本信息", ok=ok, detail=detail)


def _check_paths() -> CheckItem:
    # 关键路径存在性与可读性
    candidates = [
        REPO_ROOT / "ArcticRoute" / "config" / "runtime.yaml",
        REPO_ROOT / "configs" / "p1_pipeline_202412.yaml",
        REPO_ROOT / "configs" / "p1_route_202412.yaml",
        REPO_ROOT / "configs" / "p1_cost_202412.yaml",
        REPO_ROOT / "data" / "samples" / "sat_demo.tif",
        REPO_ROOT / "data" / "samples" / "ais_demo.geojson",
        REPO_ROOT / "data" / "samples" / "coastline_stub.geojson",
        REPO_ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc",
        REPO_ROOT / "outputs",
        REPO_ROOT / "reports",
    ]
    missing: List[str] = []
    for p in candidates:
        if not p.exists():
            missing.append(str(p.relative_to(REPO_ROOT)))
    ok = len(missing) == 0
    detail = "全部存在" if ok else ("缺失: " + ", ".join(missing))
    return CheckItem(name="关键路径", ok=ok, detail=detail)


def _check_grid() -> CheckItem:
    # 不引入 xarray 等重依赖：仅做文件存在与非空检查
    nc = REPO_ROOT / "ArcticRoute" / "data_processed" / "env_clean.nc"
    if not nc.exists():
        return CheckItem(name="环境网格 (env_clean.nc)", ok=False, detail="缺失")
    size = nc.stat().st_size
    ok = size > 0
    return CheckItem(name="环境网格 (env_clean.nc)", ok=ok, detail=f"size={size} bytes")


def _check_p1_products() -> CheckItem:
    # 以现有样例文件/输出作为健康度信号
    expected = [
        REPO_ROOT / "outputs" / "summary_202412.json",
        REPO_ROOT / "outputs" / "route.geojson",
        REPO_ROOT / "outputs" / "route_on_risk.png",
        REPO_ROOT / "reports" / "health" / "health_check_202412.json",
    ]
    missing = [str(p.relative_to(REPO_ROOT)) for p in expected if not p.exists()]
    ok = len(missing) == 0
    detail = "全部存在" if ok else ("缺失: " + ", ".join(missing))
    return CheckItem(name="P1 产物可用性", ok=ok, detail=detail)


# 日志格式兼容检查：结构化 -> OK；纯文本(run_id/ym) -> WARN；其余 -> FAIL
_STRUCT_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}.*\|\s+(INFO|DEBUG|WARNING|ERROR)\s+\|\s+ArcticRoute\.[\w\.]+\s+\|")
_PLAIN_PAT = re.compile(r"(run_id=\d{8}T\d{6})|(\sym=\d{6,8}\b)")

def check_log_format(paths: List[str]) -> Dict[str, Any]:
    import itertools
    checked: List[str] = []
    first_line: Optional[str] = None
    # 默认：即便缺失也判定为 WARN，不阻塞
    status = "WARN"
    reason = "missing"
    for p in paths:
        fp = Path(p)
        if not fp.exists() or fp.is_dir():
            continue
        try:
            with fp.open("r", encoding="utf-8", errors="ignore") as fh:
                lines = list(itertools.islice(fh, 200))
                if not lines:
                    continue
                if first_line is None:
                    first_line = lines[0].strip()
                if any(_STRUCT_PAT.search(ln) for ln in lines):
                    return {"status": "OK", "reason": "structured", "checked": [str(fp)], "example": (first_line or "")}
                if any(_PLAIN_PAT.search(ln) for ln in lines):
                    status = "WARN"; reason = "plain"; checked.append(str(fp))
                    # 不中断，继续寻找是否有结构化日志
        except Exception:
            continue
    return {"status": status, "reason": reason, "checked": checked, "example": (first_line or "")}

def gather_log_candidates() -> List[str]:
    cwd = Path.cwd()
    cwp = cwd.parent
    cand: List[Path] = [
        cwd / "outputs" / "pipeline_runs.log",
        REPO_ROOT / "outputs" / "pipeline_runs.log",
        cwp / "outputs" / "pipeline_runs.log",
        REPO_ROOT / "cache_index.json",
        cwd / "cache_index.json",
        cwp / "cache_index.json",
    ]
    # 环境变量覆盖
    for k in ("ARCTICROUTE_LOG_FILE", "ARCTICROUTE_CACHE_INDEX"):
        v = os.environ.get(k)
        if v:
            cand.append(Path(v))
    seen: set[str] = set()
    out: List[str] = []
    for p in cand:
        sp = str(p)
        if sp not in seen:
            seen.add(sp)
            out.append(sp)
    return out


def _check_log_format() -> CheckItem:
    # 统一使用候选收集器
    log_candidates = gather_log_candidates()
    res = check_log_format(log_candidates)
    status = res.get("status")
    reason = res.get("reason")
    example = res.get("example", "")
    if status == "OK":
        return CheckItem(name="日志格式", ok=True, detail=f"status=OK structured; example='{example[:120]}'", status="OK")
    if status == "WARN":
        # missing 或 plain 均为 WARN，不阻塞
        why = "plain" if reason == "plain" else "missing"
        return CheckItem(name="日志格式", ok=True, detail=f"status=WARN {why}; 建议：统一 formatter 为结构化; example='{example[:120]}'", status="WARN")
    return CheckItem(name="日志格式", ok=False, detail="status=FAIL", status="FAIL")


def _extras_payload() -> Dict[str, Any]:
    # 磁盘信息（仓库所在分区）
    try:
        total, used, free = shutil.disk_usage(str(REPO_ROOT))
        disk = {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "usage": round(used / total, 4) if total else None,
        }
    except Exception:
        disk = {}
    # 数据目录
    data_dirs = []
    for rel in ["outputs", "reports", "ArcticRoute/data_processed"]:
        p = REPO_ROOT / rel
        data_dirs.append({"path": str(p), "exists": p.exists(), "writable": os.access(p, os.W_OK) if p.exists() else False})
    # git sha
    git_sha = None
    try:
        git_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), timeout=2).decode().strip()
    except Exception:
        pass
    # 最近运行（来自 cache/index/cache_index.json）
    last_runs: List[Dict[str, Any]] = []
    try:
        from ArcticRoute.cache.index_util import INDEX_FILE  # type: ignore  # REUSE
        idxp = Path(INDEX_FILE)
        if idxp.exists():
            obj = json.loads(idxp.read_text(encoding="utf-8"))
            arts = list(obj.get("artifacts") or [])
            arts.sort(key=lambda x: str(x.get("ts", "")), reverse=True)
            for it in arts[:10]:
                last_runs.append({
                    "ts": it.get("ts"),
                    "kind": it.get("kind"),
                    "path": it.get("path"),
                    "ym": (it.get("attrs") or {}).get("ym") or (it.get("attrs") or {}).get("month"),
                })
    except Exception:
        pass
    return {"disk": disk, "data_dirs": data_dirs, "git_sha": git_sha, "last_runs": last_runs}


def run_health_checks() -> HealthReport:
    checks = [
        _read_version(),
        _check_paths(),
        _check_grid(),
        _check_p1_products(),
        _check_log_format(),
    ]
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    extras = _extras_payload()
    return HealthReport(timestamp=ts, repo_root=str(REPO_ROOT), checks=checks, extras=extras)


def write_reports(report: HealthReport) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "summary.json").write_text(
        json.dumps(report.to_json(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS_DIR / "summary.md").write_text(report.to_markdown(), encoding="utf-8")


def _print_console_summary(report: HealthReport) -> None:
    if report.all_ok:
        has_warn = any(getattr(c, 'status', '').upper() == 'WARN' for c in report.checks)
        headline = "PASSED with warnings" if has_warn else "PASSED"
    else:
        headline = "FAILED"
    print(f"ArcticRoute 健康检查: {headline}")
    for c in report.checks:
        s = (getattr(c, 'status', '') or ("OK" if c.ok else "FAIL")).upper()
        tag = 'OK' if s == 'OK' else ('WARN' if s == 'WARN' else ('FAIL' if not c.ok else 'OK'))
        print(f"- [{tag}] {c.name}: {c.detail}")


# =============== Phase B 预检实现 =================

@dataclass
class PhaseBItem:
    name: str
    status: str  # OK | WARN | FAIL
    detail: str = ""


def _phaseb_check_grid_spec() -> PhaseBItem:
    import json as _json
    path = os.path.join(str(REPO_ROOT), "ArcticRoute", "config", "grid_spec.json")
    if not os.path.exists(path):
        return PhaseBItem(name="grid_spec.json 存在且可读", status="WARN", detail=f"缺失: {path}")
    try:
        txt = open(path, "r", encoding="utf-8").read()
        _json.loads(txt)
        return PhaseBItem(name="grid_spec.json 存在且可读", status="OK", detail=os.path.basename(path))
    except Exception as e:  # noqa: BLE001
        return PhaseBItem(name="grid_spec.json 存在且可读", status="FAIL", detail=f"无法解析为 JSON: {e}")


def _phaseb_check_paths() -> PhaseBItem:
    data_root = os.path.join(str(REPO_ROOT), "data")
    ais_dir = os.path.join(str(REPO_ROOT), "ArcticRoute", "data_raw", "ais")
    missing = []
    for p in (data_root, ais_dir):
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        # 缺项仅 WARN
        return PhaseBItem(name="数据根目录与 data_raw/ais 存在", status="WARN", detail="缺失: " + ", ".join(missing))
    return PhaseBItem(name="数据根目录与 data_raw/ais 存在", status="OK", detail=f"data={data_root}; ais={ais_dir}")


def _phaseb_check_flags() -> PhaseBItem:
    import yaml  # type: ignore
    path = os.path.join(str(REPO_ROOT), "configs", "feature_flags.yaml")
    if not os.path.exists(path):
        return PhaseBItem(name="feature_flags 三模块默认关闭", status="WARN", detail=f"缺失: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        bad = []
        for k in ("prior_module", "risk_module", "congest_module"):
            v = bool(cfg.get(k, False))
            if v:
                bad.append(k)
        if bad:
            return PhaseBItem(name="feature_flags 三模块默认关闭", status="FAIL", detail=f"为 true: {', '.join(bad)}")
        return PhaseBItem(name="feature_flags 三模块默认关闭", status="OK", detail="均为 false")
    except Exception as e:  # noqa: BLE001
        return PhaseBItem(name="feature_flags 三模块默认关闭", status="FAIL", detail=f"解析失败: {e}")


def _phaseb_build_md(ts: str, items: List[PhaseBItem]) -> str:
    lines: List[str] = []
    lines.append(f"# Phase B 预检（只读） | {ts}")
    lines.append("")
    # 汇总
    status_levels = {"OK": 0, "WARN": 1, "FAIL": 2}
    worst = max((status_levels.get(it.status, 1) for it in items), default=0)
    overall = [k for k, v in status_levels.items() if v == worst][0]
    lines.append(f"总体: {overall}")
    lines.append("")
    lines.append("## 检查项")
    for it in items:
        tag = "✅ OK" if it.status == "OK" else ("⚠️ WARN" if it.status == "WARN" else "❌ FAIL")
        lines.append(f"- {tag} {it.name}")
        if it.detail:
            lines.append(f"  - {it.detail}")
    lines.append("")
    return "\n".join(lines)


def _phaseb_write_report(md_text: str, run_id: str, items: List[PhaseBItem], dry_run: bool) -> None:
    # 仅在非 dry-run 写盘，并登记工件
    if dry_run:
        return
    # 确保目录
    out_dir = os.path.join(str(REPO_ROOT), "reports", "health")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "phaseB_precheck.md")
    with open(out_path, "w", encoding="utf-8") as fw:
        fw.write(md_text)
    try:
        from ArcticRoute.cache.index_util import register_artifact
        register_artifact(run_id=run_id, kind="health_phaseB_precheck", path=out_path, attrs={})
    except Exception:
        # 登记失败不影响主流程
        pass


def run_phaseb_precheck(dry_run: bool = True) -> int:
    ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    items = [
        _phaseb_check_grid_spec(),
        _phaseb_check_paths(),
        _phaseb_check_flags(),
    ]
    md = _phaseb_build_md(ts, items)
    _phaseb_write_report(md, run_id=run_id, dry_run=dry_run)
    # 控制台摘要
    print("Phase B 预检 (只读):")
    for it in items:
        print(f"- [{it.status}] {it.name}: {it.detail}")
    # 返回码：FAIL 存在则 2；否则 0（包含 WARN）
    has_fail = any(it.status == "FAIL" for it in items)
    return 2 if has_fail else 0


# =============== CLI 定义（优先 click, 退化 argparse）=================

def _cli_impl(dry_run: bool) -> int:
    report = run_health_checks()
    # dry-run：不进行任何非报告类副作用（本工具无副作用），仍写出报告
    write_reports(report)
    _print_console_summary(report)
    return 0 if report.all_ok else 1


if click is not None:
    @click.group(name="health")
    def health_group() -> None:
        """ArcticRoute 独立健康检查工具"""

    @health_group.command(name="check")
    @click.option("--dry-run", is_flag=True, default=False, help="仅检查并输出汇总（仍会写入报告）")
    def click_check(dry_run: bool) -> None:
        sys.exit(_cli_impl(dry_run=dry_run))

    # Phase B 预检子命令
    @health_group.command(name="preb")
    @click.option("--dry-run", is_flag=True, default=False, help="只运行检查，默认不写盘；去掉 --dry-run 才写入报告并登记工件")
    def click_preb(dry_run: bool) -> None:
        rc = run_phaseb_precheck(dry_run=dry_run)
        sys.exit(rc)

    # 新增：初始化日志文件输出（不改历史命令）
    @health_group.command(name="init-logging")
    @click.option("--structured/--plain", "structured", default=True, help="是否使用结构化 formatter")
    def click_init_logging(structured: bool) -> None:
        try:
            from logging_config import configure_logging
        except Exception as e:  # noqa: BLE001
            click.echo(f"无法导入 logging_config.configure_logging: {e}")
            sys.exit(2)
        configure_logging(structured=structured)
        click.echo(f"已配置文件日志输出 (structured={structured}) -> outputs/pipeline_runs.log")

    # 允许以 `health.check` 形式直接调用
    @click.command(name="health.check")
    @click.option("--dry-run", is_flag=True, default=False, help="仅检查并输出汇总（仍会写入报告）")
    def click_health_dot_check(dry_run: bool) -> None:
        sys.exit(_cli_impl(dry_run=dry_run))

    # 允许以 `health.preb` 形式直接调用
    @click.command(name="health.preb")
    @click.option("--dry-run", is_flag=True, default=False, help="只运行检查，默认不写盘；去掉 --dry-run 才写入报告并登记工件")
    def click_health_dot_preb(dry_run: bool) -> None:
        sys.exit(run_phaseb_precheck(dry_run=dry_run))

    # 暴露统一入口名，兼容 python -m ArcticRoute.api.health health.check / health.preb
    def cli() -> None:  # noqa: D401
        """CLI 入口。示例: python -m ArcticRoute.api.health health.check --dry-run"""
        argv = sys.argv[1:]
        # 兼容别名形式：将前缀 health.check / health.preb 直接派发
        if argv[:1] == ["health.check"]:
            from click import Command
            cmd: Command = click_health_dot_check  # type: ignore
            cmd.main(args=argv[1:], prog_name="health.check", standalone_mode=False)
            return
        if argv[:1] == ["health.preb"]:
            from click import Command
            cmd: Command = click_health_dot_preb  # type: ignore
            cmd.main(args=argv[1:], prog_name="health.preb", standalone_mode=False)
            return
        # 否则走分组（支持 `health check`、`health preb`、`health init-logging`）
        health_group()

    # 为 api 代理兼容
    main = cli  # type: ignore

else:
    # 无 click 依赖时，提供简化的 argparse 接口，保持基本使用能力
    import argparse

    def main(argv: Optional[Sequence[str]] = None) -> int:
        parser = argparse.ArgumentParser(prog="ArcticRoute.api.health", description="ArcticRoute 健康检查 (no-click)")
        sub = parser.add_subparsers(dest="cmd")

        p_check = sub.add_parser("health.check", help="运行健康检查")
        p_check.add_argument("--dry-run", action="store_true", help="仅检查并输出汇总（仍会写入报告）")

        p_preb = sub.add_parser("health.preb", help="Phase B 预检（只读）")
        p_preb.add_argument("--dry-run", action="store_true", help="只运行检查，默认不写盘；去掉 --dry-run 才写入报告并登记工件")

        ns = parser.parse_args(list(argv) if argv is not None else None)
        if ns.cmd == "health.check":
            return _cli_impl(dry_run=bool(ns.dry_run))
        if ns.cmd == "health.preb":
            return run_phaseb_precheck(dry_run=bool(ns.dry_run))
        parser.print_help()
        return 2

    def cli() -> None:
        sys.exit(main())


if __name__ == "__main__":
    # 允许 python -m ArcticRoute.api.health health.check --dry-run
    if "cli" in globals() and callable(globals()["cli"]):
        cli()  # type: ignore[misc]
    else:
        sys.exit(main())

