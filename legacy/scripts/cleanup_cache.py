from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from logging_config import get_logger

try:
    from scripts.cache_common import discover_cache_roots, to_relative
except ImportError:  # pragma: no cover
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


def _collect_cache_files(roots: Iterable[Path]) -> List[dict]:
    entries: List[dict] = []
    for directory in roots:
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            try:
                stat = path.stat()
            except OSError as err:
                logger.warning("Unable to stat %s: %s", path, err)
                continue
            entries.append(
                {
                    "path": path,
                    "size_bytes": stat.st_size,
                    "mtime": stat.st_mtime,
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    "tag": _guess_tag(path),
                    "directory": directory,
                }
            )
    return entries


def cleanup_cache(
    *,
    days: Optional[float] = None,
    max_total_gb: Optional[float] = None,
    dry_run: bool = True,
    extra_dirs: Optional[Iterable[str]] = None,
) -> dict:
    """Apply cache cleanup policies and optionally delete files."""
    if days is None and max_total_gb is None:
        raise ValueError("Specify at least one of --days or --max-total-gb")

    roots = discover_cache_roots(extra_dirs)
    files = _collect_cache_files(roots)

    total_bytes = sum(item["size_bytes"] for item in files)
    current_bytes = total_bytes
    removal_plan: Dict[Path, dict] = {}

    logger.info("Evaluating %s cache files across %s directories", len(files), len(roots))

    if days is not None and days > 0:
        cutoff = datetime.now() - timedelta(days=days)
        for entry in files:
            if entry["modified"] < cutoff:
                removal_plan.setdefault(entry["path"], entry)
        logger.info(
            "Marked %s file(s) older than %.1f day(s) for deletion",
            len(removal_plan),
            days,
        )

    if max_total_gb is not None and max_total_gb > 0:
        limit_bytes = max_total_gb * (1024 ** 3)
        current_bytes = sum(
            entry["size_bytes"] for entry in files if entry["path"] not in removal_plan
        )
        if current_bytes > limit_bytes:
            sorted_candidates = sorted(
                (entry for entry in files if entry["path"] not in removal_plan),
                key=lambda item: (item["modified"], item["path"].name),
            )
            for entry in sorted_candidates:
                if current_bytes <= limit_bytes:
                    break
                removal_plan.setdefault(entry["path"], entry)
                current_bytes -= entry["size_bytes"]
            logger.info(
                "Marked additional files to satisfy max-total-gb constraint (target %.2f GB)",
                max_total_gb,
            )

    removal_list = sorted(
        removal_plan.values(),
        key=lambda item: (item["modified"], item["path"].name),
    )

    reclaimed = sum(entry["size_bytes"] for entry in removal_list)
    remaining_bytes = total_bytes - reclaimed

    if dry_run:
        logger.info(
            "Dry-run: would remove %s files totalling %.2f MB",
            len(removal_list),
            reclaimed / (1024 ** 2) if reclaimed else 0.0,
        )
    else:
        for entry in removal_list:
            path = entry["path"]
            try:
                path.unlink(missing_ok=True)
            except Exception as err:  # pragma: no cover - defensive
                logger.error("Failed to delete %s: %s", path, err)
            else:
                logger.info(
                    "Removed %s (%.2f MB)",
                    to_relative(path),
                    entry["size_bytes"] / (1024 ** 2) if entry["size_bytes"] else 0.0,
                )
        _remove_empty_directories({entry["directory"] for entry in removal_list})

    result = {
        "dry_run": dry_run,
        "applied_days": days,
        "applied_max_total_gb": max_total_gb,
        "cache_roots": [to_relative(directory) for directory in roots],
        "removed_count": len(removal_list),
        "total_files": len(files),
        "initial_bytes": total_bytes,
        "remaining_bytes": remaining_bytes if not dry_run else total_bytes - reclaimed,
        "total_reclaimed_bytes": reclaimed,
        "entries": [
            {
                "path": to_relative(entry["path"]),
                "size_bytes": entry["size_bytes"],
                "modified": entry["modified"].isoformat(),
                "tag": entry["tag"],
            }
            for entry in removal_list
        ],
    }
    return result


def _remove_empty_directories(directories: Iterable[Path]) -> None:
    unique = {path for path in directories if path.exists()}
    ordered = sorted(unique, key=lambda p: len(p.resolve().parts), reverse=True)
    for directory in ordered:
        try:
            directory.rmdir()
        except OSError:
            continue
        else:
            logger.debug("Removed empty directory %s", to_relative(directory))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean cache directories.")
    parser.add_argument("--days", type=float, help="Delete cache files older than N days.")
    parser.add_argument(
        "--max-total-gb",
        type=float,
        help="Ensure total cache size is below this threshold (in gigabytes).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview deletions without removing files.",
    )
    parser.add_argument(
        "--extra-dir",
        action="append",
        default=[],
        help="Additional cache directory to include (may be used multiple times).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        payload = cleanup_cache(
            days=args.days,
            max_total_gb=args.max_total_gb,
            dry_run=args.dry_run or (args.days is None and args.max_total_gb is None),
            extra_dirs=args.extra_dir,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    logger.info(
        "Cleanup summary: removed=%s (dry_run=%s) reclaimed=%.2f MB",
        payload["removed_count"],
        payload["dry_run"],
        payload["total_reclaimed_bytes"] / (1024 ** 2) if payload["total_reclaimed_bytes"] else 0.0,
    )
    return 0


if __name__ == "__main__":
    main()
