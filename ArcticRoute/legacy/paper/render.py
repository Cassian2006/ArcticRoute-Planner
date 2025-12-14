from __future__ import annotations
import json, time, hashlib, shutil
from pathlib import Path
from typing import Dict, Any

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
TPL_DIR = REPO_ROOT / "docs" / "paper_templates"
OUT_DIR = ARCTIC_DIR / "reports" / "paper"


def _write_meta(path: Path, logical_id: str, inputs: list[str]) -> None:
    try:
        meta = {
            "logical_id": logical_id,
            "inputs": inputs,
            "run_id": time.strftime("%Y%m%dT%H%M%S"),
            "config_hash": hashlib.sha256(json.dumps(inputs, ensure_ascii=False).encode("utf-8")).hexdigest(),
        }
        path.with_suffix(path.suffix + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _render_template(tpl_text: str, ctx: Dict[str, Any]) -> str:
    # Optional jinja2; fallback to simple {{key}} replace
    try:
        import jinja2  # type: ignore
        env = jinja2.Environment(undefined=jinja2.StrictUndefined, autoescape=False)
        tpl = env.from_string(tpl_text)
        return tpl.render(**ctx)
    except Exception:
        out = tpl_text
        for k, v in ctx.items():
            out = out.replace("{{"+str(k)+"}}", str(v))
        return out


def render_all(profile_id: str, context: Dict[str, Any] | None = None) -> Dict[str, str]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx = {"profile": profile_id, **(context or {})}
    results: Dict[str, str] = {}
    # known templates
    files = {
        "paper.md.tpl": OUT_DIR / "paper.md",
        "dataset_card.md.tpl": OUT_DIR / "dataset_card.md",
        "method_card.md.tpl": OUT_DIR / "method_card.md",
        "CITATION.cff.tpl": OUT_DIR / "CITATION.cff",
        "LICENSE.tpl": OUT_DIR / "LICENSE",
    }
    inputs: list[str] = []
    for tpl_name, out_path in files.items():
        tpl_path = TPL_DIR / tpl_name
        if not tpl_path.exists():
            continue
        txt = tpl_path.read_text(encoding="utf-8")
        rendered = _render_template(txt, ctx)
        out_path.write_text(rendered, encoding="utf-8")
        results[tpl_name] = str(out_path)
        inputs.append(str(tpl_path))
        _write_meta(out_path, out_path.name, [str(tpl_path)])
    return results


__all__ = ["render_all"]






