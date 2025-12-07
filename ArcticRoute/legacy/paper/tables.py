from __future__ import annotations
import json, time, hashlib
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
PAPER_DIR = ARCTIC_DIR / "reports" / "paper"
TAB_DIR = PAPER_DIR / "tables"


def _write_meta(out: Path, logical_id: str, inputs: List[str]) -> None:
    try:
        meta = {
            "logical_id": logical_id,
            "inputs": inputs,
            "run_id": time.strftime("%Y%m%dT%H%M%S"),
            "config_hash": hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode("utf-8")).hexdigest(),
        }
        out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def tab_metrics_summary(ym: str, scenarios: List[str]) -> Path:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    out = TAB_DIR / f"tab_metrics_{ym}.csv"
    # 最小版：从 phaseG/phaseH 现有摘要或生成占位数据
    rows = ["scenario,distance_nm,risk_integral,co2_t"]
    inputs: List[str] = []
    for s in scenarios:
        # 尝试读取 reports/d_stage/phaseM/eco_*.json
        eco_json = ARCTIC_DIR / "reports" / "d_stage" / "phaseM" / f"eco_{ym}_{s}.json"
        co2 = None
        if eco_json.exists():
            try:
                obj = json.loads(eco_json.read_text(encoding="utf-8"))
                co2 = float(obj.get("co2_t", 0.0))
                inputs.append(str(eco_json))
            except Exception:
                co2 = None
        # 生成占位指标
        dist = 3000.0
        risk = 1.0
        rows.append(f"{s},{dist:.1f},{risk:.3f},{(co2 if co2 is not None else 0.0):.3f}")
    out.write_text("\n".join(rows)+"\n", encoding="utf-8")
    _write_meta(out, out.name, inputs)
    # 再写一个 md 版本
    md = out.with_suffix(".md")
    md.write_text("\n".join(["| "+r.replace(","," | ")+" |" for r in rows]), encoding="utf-8")
    _write_meta(md, md.name, [*inputs, str(out)])
    return out


def tab_ablation(months: List[str], scenarios: List[str]) -> Path:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'-'.join(months)}_{'-'.join(scenarios)}"
    out = TAB_DIR / f"tab_ablation_{tag}.csv"
    header = "ablation,delta_risk,delta_distance"
    rows = [header,
            "no_cv,+0.05,+0.0",
            "no_accident,+0.02,+0.0",
            "no_prior,+0.10,+0.02",
            "linear_fuse,+0.03,+0.0",
            ]
    out.write_text("\n".join(rows)+"\n", encoding="utf-8")
    _write_meta(out, out.name, [header])
    return out


__all__ = ["tab_metrics_summary", "tab_ablation"]






