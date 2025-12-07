# REUSE: 体检任务引入，可回退
# 汇总 code_audit / data_audit / ui_smoke 结果，输出 Markdown 报告
from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

try:
    from logging_config import get_logger
except Exception:  # pragma: no cover
    import logging
    def get_logger(name: str):
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
        return logging.getLogger(name)

logger = get_logger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_OUT = REPO_ROOT / "docs" / "audit"
DOC_OUT.mkdir(parents=True, exist_ok=True)
AUDIT_REPORT_DIR = REPO_ROOT / "reports" / "audit"


def _load_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        return {}


def _status_count(map_obj: Dict[str, str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for v in (map_obj or {}).values():
        out[v] = out.get(v, 0) + 1
    return out


def build_summary_md(ym: str = "202412") -> Path:
    code_j = AUDIT_REPORT_DIR / "code_audit.json"
    data_j = AUDIT_REPORT_DIR / "data_audit.json"

    code = _load_json(code_j) if code_j.exists() else {}
    data = _load_json(data_j) if data_j.exists() else {}

    # 概要计数
    module_status = {}
    if code:
        module_status.update({f"code:{k}": v for k, v in (code.get("module_status") or {}).items()})
    if data:
        module_status.update({f"data:{k}": v for k, v in ((data.get("summary") or {}).get("module_status") or {}).items()})
    bucket = _status_count(module_status)
    total = sum(bucket.values())

    # Top10 问题（基于 data/code 的 issues/reasons 聚合）
    issues: List[str] = []
    for it in (code.get("cli_issues") or []):
        issues.append(f"CLI:{it.get('ref')} — {it.get('reason')} [{it.get('severity')}]")
    for it in (code.get("interface_issues") or []):
        issues.append(f"IF:{it.get('path')}@{it.get('line')} — {it.get('text')} [{it.get('severity')}]")
    # data 各节 issues
    for sec in ("ais_features","prior","risk_fuse","eco_routes","reports","ui"):
        obj = data.get(sec) or {}
        for m in (obj.get("issues") or []):
            issues.append(f"{sec}:{m}")
    top10 = issues[:10]

    # 推荐排查步骤（按问题类型给出模版建议）
    def _advise(t: str) -> str:
        if "risk_fused" in t and ("全零" in t or "方差" in t):
            return "检查融合输入通道与权重；若由 CLI 触发，请先在 CLI 层加校验避免导出全 0 栅格"
        if "PriorPenalty 范围" in t or "P_prior" in t:
            return "核对 prior.export 产物的归一化与取反逻辑，确认 [0,1] 边界裁剪"
        if t.startswith("CLI:"):
            return "确认 api/cli.py 引用模块路径是否存在；必要时在 CLI 增加 try/except 并在报告中标注"
        if t.startswith("IF:"):
            return "该接口可能未实现或留有占位；短期可在调用处加守卫，长期补齐实现"
        if "缺少" in t:
            return "确认流水线是否产出该文件；若本次不需要，应在 CLI 层增加可选分支避免强依赖"
        return "复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段"

    advices = [f"- {m}\n  - 建议：{_advise(m)}" for m in top10]

    # Markdown 输出
    lines = []
    lines.append(f"# ArcticRoute 全量体检报告 ({ym})\n")
    lines.append("## 概要\n")
    lines.append(f"共评估模块 {total}，状态分布：" + ", ".join([f"{k}:{v}" for k,v in bucket.items()]))
    lines.append("\n## 按模块状态\n")
    for k, v in module_status.items():
        lines.append(f"- {k}: {v}")
    lines.append("\n## 高优先级问题 TOP 10\n")
    for m in top10:
        lines.append(f"- {m}")
    lines.append("\n## 推荐排查步骤\n")
    lines.extend(advices)
    lines.append("\n## 运行命令示例\n")
    lines.append("```")
    lines.append("python -m ArcticRoute.api.cli audit.full --ym " + ym)
    lines.append("```")
    lines.append("\n## 需要人工确认的灰区问题\n")
    lines.append("- 某些数据异常可能源于原始数据质量，请结合原始来源再次确认\n")

    out_md = DOC_OUT / "ARCTICROUTE-FULL-AUDIT.md"
    out_md.write_text("\n".join(lines), encoding='utf-8')
    logger.info("summary -> %s", out_md)
    return out_md


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ym", default="202412")
    args = ap.parse_args()
    build_summary_md(ym=str(args.ym))






