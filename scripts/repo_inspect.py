#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Repo Inspector

- Walk repo tree, generate a readable markdown report for LLM handoff

- Extract python structure via AST (imports, classes, functions)

- Heuristically detect entrypoints (streamlit, __main__, cli)

- Avoid dumping secrets; flag suspicious patterns and mask
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_EXCLUDES = {
    ".git", ".hg", ".svn",
    "node_modules", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".venv", "venv", "env", ".env",
    "dist", "build", ".build",
    ".idea", ".vscode",
    ".DS_Store",
    "data", "datasets", "outputs", "results", "runs", "logs", "cache", ".cache",
}

BINARY_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".svg",
    ".pdf",
    ".nc", ".netcdf", ".h5", ".hdf5", ".zarr",
    ".tif", ".tiff", ".grib", ".grb", ".jp2",
    ".pth", ".pt", ".ckpt", ".onnx",
    ".zip", ".tar", ".gz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib",
}

TEXT_EXTS_HINT = {
    ".py", ".md", ".txt", ".toml", ".yaml", ".yml", ".json", ".ini", ".cfg",
    ".js", ".ts", ".tsx", ".jsx", ".html", ".css", ".scss",
    ".sh", ".bat", ".ps1",
    ".dockerfile", "dockerfile",
    ".sql",
}

SUSPICIOUS_SECRET_PATTERNS = [
    # generic tokens
    re.compile(r"(?i)\b(api[_-]?key|secret|token|access[_-]?key|private[_-]?key)\b\s*[:=]\s*[\"']?([A-Za-z0-9_\-\/\+=]{16,})"),
    # AWS
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    # GitHub token (classic/pat rough)
    re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b"),
    # OpenAI style (rough)
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
]


@dataclass
class FileInfo:
    path: str
    size_bytes: int
    sha1: str
    is_binary: bool
    lines: Optional[int] = None
    language: Optional[str] = None
    preview: Optional[str] = None
    python_imports: Optional[List[str]] = None
    python_defs: Optional[Dict[str, List[str]]] = None
    entrypoint_hints: Optional[List[str]] = None
    suspicious_secrets: Optional[List[str]] = None


def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def looks_binary(p: Path) -> bool:
    ext = p.suffix.lower()
    if ext in BINARY_EXTS:
        return True
    # sniff a little
    try:
        with p.open("rb") as f:
            head = f.read(2048)
        if b"\x00" in head:
            return True
    except Exception:
        return True
    return False


def guess_language(p: Path) -> Optional[str]:
    ext = p.suffix.lower()
    name = p.name.lower()
    if name == "dockerfile" or name.endswith(".dockerfile"):
        return "dockerfile"
    if ext == ".py":
        return "python"
    if ext in {".md"}:
        return "markdown"
    if ext in {".yml", ".yaml"}:
        return "yaml"
    if ext in {".json"}:
        return "json"
    if ext in {".toml"}:
        return "toml"
    if ext in {".js"}:
        return "javascript"
    if ext in {".ts"}:
        return "typescript"
    if ext in {".tsx"}:
        return "tsx"
    if ext in {".jsx"}:
        return "jsx"
    if ext in {".html"}:
        return "html"
    if ext in {".css", ".scss"}:
        return "css"
    if ext in {".sh"}:
        return "shell"
    return None


def safe_read_text(p: Path, max_bytes: int) -> Tuple[str, bool]:
    """return (text, truncated)"""
    data = p.read_bytes()
    truncated = False
    if len(data) > max_bytes:
        data = data[:max_bytes]
        truncated = True
    # try utf-8, then latin-1 fallback
    try:
        return data.decode("utf-8", errors="replace"), truncated
    except Exception:
        return data.decode("latin-1", errors="replace"), truncated


def extract_python_structure(src: str) -> Tuple[List[str], Dict[str, List[str]]]:
    imports: List[str] = []
    defs: Dict[str, List[str]] = {"classes": [], "functions": []}
    try:
        tree = ast.parse(src)
    except Exception:
        return imports, defs

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for alias in node.names:
                if mod:
                    imports.append(f"{mod}.{alias.name}")
                else:
                    imports.append(alias.name)

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            defs["classes"].append(node.name)
        elif isinstance(node, ast.FunctionDef):
            defs["functions"].append(node.name)

    imports = sorted(set(imports))
    return imports, defs


def entrypoint_hints(path: Path, src: str) -> List[str]:
    hints: List[str] = []
    low = src.lower()
    if "streamlit" in low or "st." in low:
        hints.append("streamlit_candidate")
    if "if __name__ == '__main__'" in src or 'if __name__ == "__main__"' in src:
        hints.append("__main__")
    if re.search(r"\b(click|typer|argparse)\b", low):
        hints.append("cli_candidate")
    if path.name in {"app.py", "main.py"}:
        hints.append("common_entry_filename")
    if "uvicorn" in low or "fastapi" in low:
        hints.append("api_candidate")
    return hints


def scan_suspicious_secrets(src: str) -> List[str]:
    found: List[str] = []
    for pat in SUSPICIOUS_SECRET_PATTERNS:
        for m in pat.finditer(src):
            s = m.group(0)
            # mask middle
            if len(s) > 12:
                masked = s[:6] + "…" + s[-4:]
            else:
                masked = s
            found.append(masked)
    # dedupe
    return sorted(set(found))


def should_exclude(rel_parts: Tuple[str, ...], excludes: set[str]) -> bool:
    for part in rel_parts:
        if part in excludes:
            return True
    return False


def build_tree_summary(root: Path, excludes: set[str]) -> List[Tuple[str, int]]:
    """Return list of (dirpath, total_bytes) for quick size hotspots."""
    dir_sizes: Dict[str, int] = {}
    for p in root.rglob("*"):
        try:
            rel = p.relative_to(root)
        except Exception:
            continue
        if should_exclude(rel.parts, excludes):
            continue
        if p.is_file():
            dirpath = str(rel.parent)
            dir_sizes[dirpath] = dir_sizes.get(dirpath, 0) + p.stat().st_size
    return sorted(dir_sizes.items(), key=lambda x: x[1], reverse=True)


def format_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n/ (1024 if unit=='KB' else 1024**2 if unit=='MB' else 1024**3):.2f}{unit}"
    return f"{n}B"


def generate_report(
    root: Path,
    out_md: Path,
    out_json: Path,
    max_file_bytes: int,
    preview_lines: int,
    preview_max_bytes: int,
    excludes: set[str],
) -> None:
    files: List[FileInfo] = []

    # Collect file infos
    for p in sorted(root.rglob("*")):
        try:
            rel = p.relative_to(root)
        except Exception:
            continue
        if should_exclude(rel.parts, excludes):
            continue
        if p.is_dir():
            continue

        try:
            size = p.stat().st_size
        except Exception:
            continue

        is_bin = looks_binary(p)

        info = FileInfo(
            path=str(rel).replace("\\", "/"),
            size_bytes=size,
            sha1=sha1_file(p) if size <= 50 * 1024 * 1024 else "skipped_large_sha1",
            is_binary=is_bin,
            language=guess_language(p),
        )

        # Only read text if small enough and not binary
        if (not is_bin) and size <= max_file_bytes:
            try:
                text, truncated = safe_read_text(p, preview_max_bytes)
                lines = text.splitlines()
                info.lines = len(lines)

                # preview
                preview = "\n".join(lines[:preview_lines])
                if truncated or len(lines) > preview_lines:
                    preview += "\n…(truncated)…"
                info.preview = preview

                # structures
                if info.language == "python":
                    imps, defs = extract_python_structure(text)
                    info.python_imports = imps
                    info.python_defs = defs
                    info.entrypoint_hints = entrypoint_hints(p, text)
                else:
                    # still do entrypoint hints loosely
                    info.entrypoint_hints = entrypoint_hints(p, text)

                # secrets scan (don't dump raw)
                sus = scan_suspicious_secrets(text)
                if sus:
                    info.suspicious_secrets = sus[:20]
            except Exception:
                pass

        files.append(info)

    # Identify key config files
    key_files = [
        "README.md",
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "Pipfile",
        "package.json",
        "Dockerfile",
        "docker-compose.yml",
        ".env.example",
        "streamlit_app.py",
        "app.py",
        "main.py",
    ]
    key_present = [kf for kf in key_files if (root / kf).exists()]

    # Entrypoints candidates
    entry_candidates = []
    for f in files:
        if f.entrypoint_hints:
            if any(h in f.entrypoint_hints for h in ["streamlit_candidate", "__main__", "cli_candidate", "api_candidate"]):
                entry_candidates.append((f.path, f.entrypoint_hints))

    # Hot dirs
    hot_dirs = build_tree_summary(root, excludes)[:20]

    # Write JSON manifest
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump([asdict(x) for x in files], f, ensure_ascii=False, indent=2)

    # Write Markdown report
    out_md.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_bytes = sum(f.size_bytes for f in files)
    total_files = len(files)
    text_files = sum(1 for f in files if not f.is_binary)
    bin_files = total_files - text_files

    def md(s: str) -> str:
        return s.replace("\r\n", "\n")

    report = []
    report.append(f"# Repo Report\n\nGenerated: `{now}`\n")
    report.append(f"- Root: `{root.resolve()}`\n")
    report.append(f"- Files: **{total_files}** (text {text_files}, binary/large {bin_files})\n")
    report.append(f"- Total size (scanned): **{format_bytes(total_bytes)}**\n")
    report.append(f"- Excludes: `{', '.join(sorted(excludes))}`\n")
    report.append(f"- Limits: max_file_bytes={max_file_bytes}, preview_lines={preview_lines}, preview_max_bytes={preview_max_bytes}\n\n")

    report.append("## Key Files Present\n")
    report.append("\n".join([f"- `{x}`" for x in key_present]) if key_present else "- (none found)")
    report.append("\n\n")

    report.append("## Size Hotspots (Top Dirs)\n")
    for d, sz in hot_dirs:
        report.append(f"- `{d or '.'}`: {format_bytes(sz)}")
    report.append("\n\n")

    report.append("## Entrypoint Candidates (Heuristic)\n")
    if entry_candidates:
        for pth, hints in entry_candidates[:40]:
            report.append(f"- `{pth}`  ➜  {', '.join(hints)}")
    else:
        report.append("- (none detected)")
    report.append("\n\n")

    report.append("## File Index (Metadata)\n")
    report.append("| Path | Size | Lines | Lang | Binary | SHA1 |\n|---|---:|---:|---|---|---|\n")
    for f in files[:2000]:
        report.append(
            f"| `{f.path}` | {format_bytes(f.size_bytes)} | {f.lines or ''} | {f.language or ''} | {str(f.is_binary)} | `{f.sha1}` |"
        )
    report.append("\n\n")

    report.append("## Detailed Summaries (Text & Small Files)\n")
    for f in files:
        if f.is_binary or f.preview is None:
            continue
        report.append(f"### `{f.path}`\n")
        report.append(f"- size: {format_bytes(f.size_bytes)}; lines: {f.lines}; lang: {f.language}\n")
        if f.python_imports:
            report.append(f"- python_imports: {', '.join(f.python_imports[:60])}\n")
        if f.python_defs:
            report.append(f"- python_defs: classes={f.python_defs.get('classes', [])}; functions={f.python_defs.get('functions', [])}\n")
        if f.entrypoint_hints:
            report.append(f"- entrypoint_hints: {', '.join(f.entrypoint_hints)}\n")
        if f.suspicious_secrets:
            report.append(f"- ⚠ suspicious_secrets (masked): {', '.join(f.suspicious_secrets)}\n")
        report.append("\n```text\n")
        report.append(md(f.preview))
        report.append("\n```\n\n")

    out_md.write_text("\n".join(report), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--out-md", default="reports/repo_report.md")
    ap.add_argument("--out-json", default="reports/repo_manifest.json")
    ap.add_argument("--max-file-bytes", type=int, default=200_000, help="only read text content for files <= this size")
    ap.add_argument("--preview-lines", type=int, default=80)
    ap.add_argument("--preview-max-bytes", type=int, default=120_000, help="max bytes to read per file for preview")
    ap.add_argument("--exclude", default="", help="comma-separated extra excludes")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    excludes = set(DEFAULT_EXCLUDES)
    if args.exclude.strip():
        excludes |= {x.strip() for x in args.exclude.split(",") if x.strip()}

    generate_report(
        root=root,
        out_md=Path(args.out_md),
        out_json=Path(args.out_json),
        max_file_bytes=args.max_file_bytes,
        preview_lines=args.preview_lines,
        preview_max_bytes=args.preview_max_bytes,
        excludes=excludes,
    )

    print(f"[OK] Wrote: {args.out_md}")
    print(f"[OK] Wrote: {args.out_json}")


if __name__ == "__main__":
    main()



