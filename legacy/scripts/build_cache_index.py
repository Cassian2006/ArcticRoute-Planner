from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import get_logger

try:
    from scripts.cache_common import discover_cache_roots, to_relative
except ImportError:  # pragma: no cover - fallback when executed as module
    from .cache_common import discover_cache_roots, to_relative  # type: ignore

logger = get_logger(__name__)

_TAG_MAP = {
    ".nc": "netcdf",
    ".json": "json",
    ".geojson": "geojson",
    ".tif": "geotiff",
    ".tiff": "geotiff",
    ".parquet": "parquet",
    ".pkl": "pickle",
    ".pickle": "pickle",
    ".pt": "torch",
    ".bin": "binary",
    ".npy": "numpy",
    ".npz": "numpy",
    ".csv": "csv",
    ".log": "log",
    ".txt": "text",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def _guess_tag(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _TAG_MAP:
        return _TAG_MAP[suffix]
    if suffix:
        return suffix.lstrip(".")
    return "data"


def _format_mtime(timestamp: float | None) -> Optional[str]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp).isoformat()


import re
import hashlib

_TYPE_HINTS = {
    "blocks": "block",
    "_blocks": "block",
    "merged": "merged",
    "ice_cost": "cost",
    "cost": "cost",
    "snapshot": "snapshot",
    "outputs": "summary",
    "reports": "report",
    "route": "route",
}

_YM_RE = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")


def _guess_type(path: Path) -> str:
    p = str(path).replace("\\", "/").lower()
    for k, v in _TYPE_HINTS.items():
        if f"/{k}/" in p or path.name.lower().startswith(k) or k in path.name.lower():
            return v
    # heuristics by suffix/name
    name = path.name.lower()
    if name.startswith("sic_fcst_"):
        return "merged"
    if name.startswith("ice_cost_"):
        return "cost"
    if name.startswith("summary_"):
        return "summary"
    if name.endswith(".geojson") and "route" in name:
        return "route"
    if name.endswith(".nc"):
        return "block" if "block" in p else "merged"
    return "block"


def _guess_ym(path: Path) -> Optional[str]:
    # from filename
    m = _YM_RE.search(path.name)
    if m:
        return m.group(1) + m.group(2)
    # from parents
    for parent in path.parents:
        m = _YM_RE.search(parent.name)
        if m:
            return m.group(1) + m.group(2)
    return None


def _maybe_hash(path: Path, enable: bool, max_size_mb: float) -> Optional[str]:
    if not enable:
        return None
    try:
        size = path.stat().st_size
        if size > max_size_mb * 1024 * 1024:
            return None
        h = hashlib.sha1()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _scan_cache_directory(directory: Path, *, do_hash: bool = False, max_hash_mb: float = 32.0) -> Tuple[List[dict], dict]:
    files: List[dict] = []
    total_size = 0
    latest_mtime: float | None = None
    count = 0
    errors: List[dict] = []

    for path in directory.rglob("*"):
        if not path.is_file():
            continue
        try:
            stat = path.stat()
        except OSError as err:
            logger.warning("Unable to stat %s: %s", path, err)
            errors.append({"path": to_relative(path), "error": str(err)})
            continue
        total_size += stat.st_size
        latest_mtime = max(latest_mtime or stat.st_mtime, stat.st_mtime)
        count += 1
        ym = _guess_ym(path)
        typ = _guess_type(path)
        run_id = None
        if path.suffix.lower() in (".json",):
            try:
                txt = path.read_text(encoding="utf-8")
                if txt:
                    obj = json.loads(txt)
                    run_id = obj.get("run_id") or obj.get("meta", {}).get("run_id")
            except Exception:
                pass
        entry = {
            "path": to_relative(path),
            "ym": ym,
            "type": typ,
            "size": stat.st_size,
            "mtime": _format_mtime(stat.st_mtime),
            "tag": _guess_tag(path),
        }
        if run_id:
            entry["run_id"] = run_id
        h = _maybe_hash(path, do_hash, max_hash_mb)
        if h:
            entry["hash"] = h
        files.append(entry)

    summary = {
        "path": to_relative(directory),
        "label": directory.name,
        "file_count": count,
        "total_bytes": total_size,
        "modified": _format_mtime(latest_mtime),
        "errors": errors,
    }
    return files, summary


def build_index(
    *,
    write_json: bool = True,
    output_path: Optional[Path | str] = None,
    extra_dirs: Optional[Iterable[str]] = None,
    base_dir: Optional[Path | str] = None,
    include_outputs: bool = True,
    include_reports: bool = True,
    do_hash: bool = False,
    max_hash_mb: float = 32.0,
) -> dict:
    """Collect cache metadata and optionally persist it to cache_index.json."""
    # assemble scan roots
    cache_roots: List[Path] = []
    if base_dir:
        b = Path(base_dir)
        if b.exists():
            cache_roots.append(b)
            m = b / "merged"
            if m.exists():
                cache_roots.append(m)
    # include project-level outputs and reports if requested
    if include_outputs:
        op = PROJECT_ROOT / "outputs"
        if op.exists():
            cache_roots.append(op)
    if include_reports:
        rp = PROJECT_ROOT / "reports"
        if rp.exists():
            cache_roots.append(rp)
    # discover generic caches as well
    cache_roots.extend(discover_cache_roots(extra_dirs))
    # de-dup
    seen = []
    uniq = []
    for p in cache_roots:
        r = p.resolve()
        if r not in seen and r.exists():
            seen.append(r)
            uniq.append(r)
    cache_roots = uniq

    logger.info("Scanning %s cache directories", len(cache_roots))

    all_files: List[dict] = []
    directory_summaries: List[dict] = []

    for directory in cache_roots:
        files, summary = _scan_cache_directory(directory, do_hash=do_hash, max_hash_mb=max_hash_mb)
        if not files:
            logger.debug("Cache directory %s is empty", directory)
        all_files.extend(files)
        directory_summaries.append(summary)

    total_bytes = sum(item["size_bytes"] for item in all_files)

    all_files.sort(key=lambda item: (item["directory"], item["path"]))
    directory_summaries.sort(key=lambda item: item["path"])

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "project_root": str(PROJECT_ROOT),
        "total_bytes": total_bytes,
        "total_files": len(all_files),
        "directories": directory_summaries,
        "entries": all_files,
    }

    if write_json:
        destination = Path(output_path) if output_path else PROJECT_ROOT / "cache_index.json"
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(
            "Wrote cache index to %s (entries=%s, total=%.2f MB)",
            destination,
            len(all_files),
            total_bytes / (1024 ** 2) if total_bytes else 0.0,
        )

    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ArcticRoute cache index.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Override destination for cache_index.json (defaults to project root).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Do not write cache_index.json; useful when only inspecting output.",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        default=[],
        help="Additional cache directory to include (can be specified multiple times).",
    )
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Emit the generated JSON payload to stdout.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = build_index(
        write_json=not args.no_write,
        output_path=args.output,
        extra_dirs=args.extra_dir,
    )

    if args.print_json:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    main()
