# REUSE: 体检任务引入，可回退
# 说明：该脚本仅做静态扫描与报告生成，不修改核心逻辑。
from __future__ import annotations
import json
import re
import sys
import html
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

try:
    from logging_config import get_logger  # repo 根路径运行时可用
except Exception:  # pragma: no cover
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
REPORT_DIR = REPO_ROOT / "reports" / "audit"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 扫描模式
SCAN_PATTERNS = [
    r"\bTODO\b",
    r"\bFIXME\b",
    r"\bXXX\b",
    r"\braise\s+NotImplementedError\b",
    r"\bpass\b",
    r"未完成",
    r"占位",
    r"stub",
]
SCAN_REGEX = re.compile("|".join(SCAN_PATTERNS))

# 模块归类前缀
MODULE_BUCKETS = [
    ("core/", "core"),
    ("io/", "io"),
    ("apps/", "apps"),
    ("api/", "api"),
    ("config/", "config"),
]

@dataclass
class Finding:
    path: str
    line: int
    text: str
    kind: str  # 命中关键字
    module: str

@dataclass
class CliRefIssue:
    place: str  # e.g. handler name
    ref: str    # import path or callable
    reason: str
    severity: str  # broken/suspect

@dataclass
class InterfaceIssue:
    path: str
    line: int
    text: str
    severity: str


def _guess_module(rel_path: str) -> str:
    for prefix, name in MODULE_BUCKETS:
        if rel_path.replace("\\", "/").startswith(f"ArcticRoute/{prefix}"):
            return name
    # 其它统一划入 core-like 或 unknown
    if rel_path.replace("\\", "/").startswith("ArcticRoute/"):
        return rel_path.split("/", 2)[1] if "/" in rel_path else "unknown"
    return "unknown"


def scan_repo() -> List[Finding]:
    findings: List[Finding] = []
    for p in REPO_ROOT.rglob("*.py"):
        # 跳过缓存与第三方
        rel = str(p.relative_to(REPO_ROOT))
        if any(seg in rel for seg in ("__pycache__", ".venv", "site-packages")):
            continue
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(content.splitlines(), start=1):
            m = SCAN_REGEX.search(line)
            if m:
                findings.append(Finding(
                    path=rel,
                    line=i,
                    text=line.strip()[:400],
                    kind=m.group(0),
                    module=_guess_module(rel),
                ))
    return findings


def group_findings(findings: List[Finding]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for f in findings:
        groups.setdefault(f.module, []).append(asdict(f))
    return groups


def parse_cli_refs() -> List[CliRefIssue]:
    """解析 api/cli.py：
    - 检查 handler 内的 import 路径是否存在（importlib.find_spec）
    - 简易检查：子命令在 main 中是否有分发（静态查找）
    仅做标记，不做修改。
    """
    import importlib.util

    cli_path = ARCTIC_DIR / "api" / "cli.py"
    issues: List[CliRefIssue] = []
    if not cli_path.exists():
        return issues
    txt = cli_path.read_text(encoding="utf-8", errors="ignore")

    # 粗略解析：查找 "from ArcticRoute.... import X" 与 "import ArcticRoute...."
    imp_re = re.compile(r"^(from\s+(ArcticRoute[\w\.]*)\s+import\s+([\w\.,\s]+))|^(import\s+(ArcticRoute[\w\.]*(?:\s+as\s+\w+)?))", re.M)
    for m in imp_re.finditer(txt):
        mod = None
        if m.group(2):
            mod = m.group(2)
        elif m.group(5):
            mod = m.group(5).split(" as ")[0]
        if not mod:
            continue
        try:
            spec = importlib.util.find_spec(mod)
            if spec is None:
                issues.append(CliRefIssue(place="cli.py", ref=mod, reason="模块不可导入", severity="broken"))
        except Exception as e:
            issues.append(CliRefIssue(place="cli.py", ref=mod, reason=f"导入异常:{e}", severity="suspect"))

    # 子命令分发检查：parser.add_parser("name") 是否在 main 分支中处理
    cmd_re = re.compile(r"subparsers\.add_parser\(\"([^\"]+)\"", re.M)
    cmds = set(cmd_re.findall(txt))
    # main 分发分支
    dispatch_re = re.compile(r"if\s+args\.command\s*==\s*\"([^\"]+)\"", re.M)
    dispatched = set(dispatch_re.findall(txt))
    for c in sorted(cmds):
        if c not in dispatched:
            issues.append(CliRefIssue(place="build_parser", ref=c, reason="子命令未在 main() 分发", severity="suspect"))

    return issues


def inspect_interfaces() -> List[InterfaceIssue]:
    """检查接口约定中是否有明显的未实现入口（pass/NotImplementedError）。
    主要扫描 core/interfaces.py 以及 core 下的 *interfaces*.py。
    """
    issues: List[InterfaceIssue] = []
    candidates = [ARCTIC_DIR / "core" / "interfaces.py"] + list((ARCTIC_DIR / "core").rglob("*interfaces*.py"))
    for p in candidates:
        if not p.exists() or not p.is_file():
            continue
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for i, line in enumerate(txt.splitlines(), start=1):
            if re.search(r"raise\s+NotImplementedError|\bpass\b", line):
                sev = "suspect"
                # 如果在 ABC 抽象方法内则允许，简单启发：出现 @abstractmethod 的附近视为允许
                window = "\n".join(txt.splitlines()[max(0, i-5): i+5])
                if "@abstractmethod" in window:
                    continue
                issues.append(InterfaceIssue(path=str(p.relative_to(REPO_ROOT)), line=i, text=line.strip(), severity=sev))
    return issues


def summarize_status(findings: List[Finding], cli_issues: List[CliRefIssue], iface_issues: List[InterfaceIssue]) -> Dict[str, str]:
    """模块状态：ok/suspect/broken/unknown"""
    status: Dict[str, str] = {}
    # 初始 unknown
    for _, name in MODULE_BUCKETS:
        status[name] = "unknown"
    # 若无命中且无问题则 ok
    groups = group_findings(findings)
    for mod in status.keys():
        has_findings = len(groups.get(mod, [])) > 0
        has_broken = any(i.severity == "broken" for i in cli_issues)
        has_suspect = any(i.severity == "suspect" for i in cli_issues) or any(True for _ in iface_issues)
        if has_broken:
            status[mod] = "broken"
        elif has_suspect or has_findings:
            status[mod] = "suspect"
        else:
            status[mod] = "ok"
    return status


def render_html(payload: Dict[str, Any]) -> str:
    rows = []
    for mod, items in (payload.get("groups") or {}).items():
        for it in items:
            rows.append(f"<tr><td>{html.escape(mod)}</td><td>{html.escape(it['path'])}</td><td>{it['line']}</td><td>{html.escape(it['kind'])}</td><td><pre style='margin:0'>{html.escape(it['text'])}</pre></td></tr>")
    cli_rows = []
    for it in (payload.get("cli_issues") or []):
        cli_rows.append(f"<tr><td>{html.escape(it['place'])}</td><td>{html.escape(it['ref'])}</td><td>{html.escape(it['severity'])}</td><td>{html.escape(it['reason'])}</td></tr>")
    iface_rows = []
    for it in (payload.get("interface_issues") or []):
        iface_rows.append(f"<tr><td>{html.escape(it['path'])}</td><td>{it['line']}</td><td>{html.escape(it['severity'])}</td><td><pre style='margin:0'>{html.escape(it['text'])}</pre></td></tr>")

    mod_rows = []
    for k, v in (payload.get("module_status") or {}).items():
        mod_rows.append(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>")

    html_doc = f"""
<!doctype html>
<html><head><meta charset='utf-8'><title>Code Audit</title>
<style>body{{font-family:Arial,sans-serif}} table{{border-collapse:collapse;width:100%}} td,th{{border:1px solid #ddd;padding:6px}} th{{background:#f5f5f5}}</style>
</head><body>
<h1>代码体检报告</h1>
<h2>模块状态</h2>
<table><thead><tr><th>模块</th><th>状态</th></tr></thead><tbody>{''.join(mod_rows)}</tbody></table>
<h2>扫描命中</h2>
<table><thead><tr><th>模块</th><th>文件</th><th>行</th><th>命中</th><th>片段</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>CLI 引用问题</h2>
<table><thead><tr><th>位置</th><th>引用</th><th>级别</th><th>原因</th></tr></thead><tbody>{''.join(cli_rows)}</tbody></table>
<h2>接口未实现线索</h2>
<table><thead><tr><th>文件</th><th>行</th><th>级别</th><th>片段</th></tr></thead><tbody>{''.join(iface_rows)}</tbody></table>
</body></html>
"""
    return html_doc


def run_code_audit() -> Dict[str, Any]:
    findings = scan_repo()
    groups = group_findings(findings)
    cli_issues = parse_cli_refs()
    iface_issues = inspect_interfaces()
    module_status = summarize_status(findings, cli_issues, iface_issues)

    payload = {
        "summary": {
            "total_hits": len(findings),
            "files": len(set(f["path"] for xs in groups.values() for f in xs)),
            "cli_issues": len(cli_issues),
            "interface_issues": len(iface_issues),
        },
        "groups": groups,
        "cli_issues": [asdict(x) for x in cli_issues],
        "interface_issues": [asdict(x) for x in iface_issues],
        "module_status": module_status,
    }

    jpath = REPORT_DIR / "code_audit.json"
    hpath = REPORT_DIR / "code_audit.html"
    jpath.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    hpath.write_text(render_html(payload), encoding="utf-8")
    logger.info("code audit -> %s, %s", jpath, hpath)
    return payload


if __name__ == "__main__":
    run_code_audit()






