#!/usr/bin/env python3
"""Repository hygiene utility; historical helper (not part of pipeline).

@role: legacy
"""

"""
Repo housekeeping utility for ArcticRoute.

Default behaviour is a dry-run that inspects the project tree, reports missing / extra items,
and describes move / rename / delete operations that would be applied to enforce the standard layout.

Usage:
    python scripts/maintenance/repo_tidy.py               # dry-run
    python scripts/maintenance/repo_tidy.py --apply       # execute planned actions
    python scripts/maintenance/repo_tidy.py --keep 3      # retain three latest output batches
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
LOG_DIR = PROJECT_ROOT / "logs"
DEFAULT_DRYRUN_LOG = LOG_DIR / "repo_tidy_dryrun.txt"
MANIFEST_PATH = PROJECT_ROOT / "docs" / "MANIFEST_CURRENT.md"

REQUIRED_PATHS: Tuple[Path, ...] = (
    PROJECT_ROOT / "apps" / "app_basic.py",
    PROJECT_ROOT / "web" / "route_viewer.html",
    PROJECT_ROOT / "config" / "runtime.yaml",
    PROJECT_ROOT / "config" / "scenarios.yaml",
    PROJECT_ROOT / "config" / "env.yaml",
    PROJECT_ROOT / "data_download" / "download_era5.py",
    PROJECT_ROOT / "docs" / "README.md",
    PROJECT_ROOT / "docs" / "data_sources.md",
    PROJECT_ROOT / "notebooks",
    PROJECT_ROOT / "scripts",
    PROJECT_ROOT / "outputs",
    PROJECT_ROOT / "data_processed",
    PROJECT_ROOT / "data_raw" / "era5",
    PROJECT_ROOT / "data_raw" / "ais",
    PROJECT_ROOT / "outputs" / "overlays",
)

EXPECTED_TOP_LEVEL_DIRS = {
    "api",
    "apps",
    "config",
    "core",
    "data_download",
    "data_raw",
    "data_processed",
    "docs",
    "io",
    "logs",
    "notebooks",
    "outputs",
    "scripts",
    "tests",
    "web",
}

EXPECTED_TOP_LEVEL_FILES = {
    ".gitignore",
    "requirements.txt",
    "__init__.py",
}

DATA_RAW_ALLOWED_DIRS = {"ais", "era5", "incidents"}
DATA_RAW_KEYWORDS = (
    ("ais", "ais"),
    ("track", "ais"),
    ("era", "era5"),
    ("grib", "era5"),
    ("nc", "era5"),
    ("incident", "incidents"),
    ("accident", "incidents"),
)

DATA_PROCESSED_ALLOWED_DIRS = {"env", "corridor", "incidents", "accidents", "routes", "ais", "archive", "overlays"}
DATA_PROCESSED_KEYWORDS = (
    ("env", "env"),
    ("corr", "corridor"),
    ("incident", "incidents"),
    ("accident", "accidents"),
    ("density", "accidents"),
    ("route", "routes"),
    ("ais", "ais"),
)

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUT_ARCHIVE_DIR = OUTPUTS_DIR / "archive"
OVERLAY_KEYWORDS = ("overlay", "overlays", "bounds")
OUTPUT_PREFIXES_REQUIRING_TAG = ("route_on_risk", "route", "run_report", "acc_calib")
ARCHIVE_EXCLUDE = {"metrics.csv", "acc_calib.csv"}
CLEANUP_DIR_NAMES = {"__pycache__", ".ipynb_checkpoints"}

MANIFEST_TARGETS: Tuple[Tuple[str, Path, Sequence[str], Sequence[str]], ...] = (
    ("Processed Data", PROJECT_ROOT / "data_processed", ("*.nc", "*.parquet", "*.csv", "*.zarr", "*.zarr.zip"), ()),
    ("Outputs (active)", OUTPUTS_DIR, ("*.geojson", "*.json", "*.png", "*.csv"), ("archive",)),
)

MANIFEST_MINIMAL_SET: Tuple[Path, ...] = (
    PROJECT_ROOT / "config" / "runtime.yaml",
    PROJECT_ROOT / "config" / "scenarios.yaml",
    PROJECT_ROOT / "config" / "env.yaml",
    PROJECT_ROOT / "data_processed" / "env" / "env_clean.nc",
    PROJECT_ROOT / "data_processed" / "corridor" / "corridor_prob.nc",
    PROJECT_ROOT / "data_processed" / "incidents" / "incidents_clean.parquet",
    PROJECT_ROOT / "data_processed" / "ais" / "ais_aligned.parquet",
    PROJECT_ROOT / "scripts" / "run_scenarios.py",
    PROJECT_ROOT / "scripts" / "route_astar_min.py",
    PROJECT_ROOT / "apps" / "app_basic.py",
    PROJECT_ROOT / "web" / "route_viewer.html",
)


@dataclass
class PlannedOp:
    """Description of an operation required to tidy the repository."""

    kind: str
    source: Path
    destination: Optional[Path] = None
    reason: str = ""
    executed: bool = False
    error: Optional[str] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ArcticRoute repository tidy helper")
    parser.add_argument("--apply", action="store_true", help="execute planned operations")
    parser.add_argument("--keep", type=int, default=None, metavar="N", help="retain the latest N output batches")
    parser.add_argument(
        "--archive-tag",
        dest="archive_tag",
        type=str,
        default=None,
        help="tag used for outputs/archive/<tag> when archiving (default: timestamp)",
    )
    parser.add_argument(
        "--manifest",
        action="store_true",
        help=f"generate current manifest at {MANIFEST_PATH.relative_to(PROJECT_ROOT)}",
    )
    return parser.parse_args()


def timestamp_tag(ts: Optional[float] = None, *, include_seconds: bool = False) -> str:
    fmt = "%Y%m%d_%H%M%S" if include_seconds else "%Y%m%d_%H%M"
    return datetime.fromtimestamp(ts or datetime.now().timestamp()).strftime(fmt)


def ensure_unique_destination(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    counter = 1
    while True:
        candidate = dest.with_name(f"{stem}__{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def is_tagged(stem: str) -> bool:
    return bool(re.search(r"_[0-9]{8}_[0-9]{4,6}$", stem))


def detect_output_prefix(name: str) -> Optional[str]:
    lowered = name.lower()
    for prefix in OUTPUT_PREFIXES_REQUIRING_TAG:
        if lowered.startswith(prefix):
            return prefix
    return None


def classify_via_keywords(name: str, keywords: Iterable[Tuple[str, str]]) -> Optional[str]:
    lowered = name.lower()
    for needle, target in keywords:
        if needle in lowered:
            return target
    return None


class RepoTidy:
    def __init__(
        self,
        *,
        apply_changes: bool,
        keep: Optional[int],
        archive_tag: Optional[str],
        write_manifest: bool = False,
    ) -> None:
        self.apply_changes = apply_changes
        self.keep = keep if keep is not None and keep >= 0 else None
        self.archive_tag = archive_tag or timestamp_tag(include_seconds=False)
        self.write_manifest = write_manifest
        self.actions: List[PlannedOp] = []
        self.missing: List[PlannedOp] = []
        self.notes: List[str] = []
        self.unresolved: List[str] = []

    def run(self) -> None:
        self._ensure_log_dir()
        self._check_required_paths()
        self._check_top_level_inventory()
        self._tidy_data_dir(PROJECT_ROOT / "data_raw", DATA_RAW_ALLOWED_DIRS, DATA_RAW_KEYWORDS)
        self._tidy_data_dir(PROJECT_ROOT / "data_processed", DATA_PROCESSED_ALLOWED_DIRS, DATA_PROCESSED_KEYWORDS)
        self._capture_overlay_assets()
        self._enforce_output_tags()
        if self.keep is not None:
            self._plan_output_archival()
        self._plan_cleanup_dirs()

    def _ensure_log_dir(self) -> None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def _check_required_paths(self) -> None:
        for path in REQUIRED_PATHS:
            if not path.exists():
                self.missing.append(PlannedOp(kind="missing", source=path, reason="required item missing"))

    def _check_top_level_inventory(self) -> None:
        for entry in PROJECT_ROOT.iterdir():
            if entry.is_dir():
                if entry.name in CLEANUP_DIR_NAMES:
                    continue
                if entry.name.startswith(".") and entry.name not in {".pytest_cache"}:
                    continue
                if entry.name not in EXPECTED_TOP_LEVEL_DIRS:
                    self.notes.append(f"unexpected directory at top-level: {entry}")
            elif entry.is_file():
                if entry.name in EXPECTED_TOP_LEVEL_FILES:
                    continue
                if entry.name.startswith("."):
                    continue
                self.notes.append(f"unexpected file at top-level: {entry}")

    def _tidy_data_dir(
        self,
        root: Path,
        allowed_dirs: Iterable[str],
        keywords: Iterable[Tuple[str, str]],
    ) -> None:
        if not root.exists():
            self.missing.append(PlannedOp(kind="missing", source=root, reason="data directory not found"))
            return
        allowed = {name.lower() for name in allowed_dirs}
        for entry in root.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.is_dir():
                normalized = entry.name.lower()
                if normalized in allowed:
                    continue
                target = classify_via_keywords(entry.name, keywords)
                if not target:
                    # look into children for hints
                    target = self._classify_child_entries(entry, keywords, allowed)
                if target and target in allowed:
                    target_dir = root / target
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for child in entry.iterdir():
                        destination = ensure_unique_destination(target_dir / child.name)
                        self.actions.append(
                            PlannedOp(
                                kind="move",
                                source=child,
                                destination=destination,
                                reason=f"relocate under data directory `{target}`",
                            )
                        )
                    self.actions.append(
                        PlannedOp(kind="delete", source=entry, reason="remove empty placeholder after relocation")
                    )
                else:
                    self.unresolved.append(f"cannot determine target for {entry}")
            elif entry.is_file():
                target = classify_via_keywords(entry.name, keywords)
                if not target:
                    target = self._classify_by_suffix(entry, allowed)
                if target and target in allowed:
                    destination = ensure_unique_destination(root / target / entry.name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    self.actions.append(
                        PlannedOp(kind="move", source=entry, destination=destination, reason=f"move into `{target}`")
                    )
                else:
                    self.unresolved.append(f"unclassified data file: {entry}")

    def _classify_child_entries(
        self, directory: Path, keywords: Iterable[Tuple[str, str]], allowed: Iterable[str]
    ) -> Optional[str]:
        for child in directory.iterdir():
            target = classify_via_keywords(child.name, keywords)
            if target:
                return target
            suffix_target = self._classify_by_suffix(child, allowed)
            if suffix_target:
                return suffix_target
        return None

    @staticmethod
    def _classify_by_suffix(path: Path, allowed: Iterable[str]) -> Optional[str]:
        allowed_set = {item.lower() for item in allowed}
        suffix = path.suffix.lower()
        name_lower = path.name.lower()
        if "env" in allowed_set and suffix in {".nc", ".netcdf"} and "env" in name_lower:
            return "env"
        if "corridor" in allowed_set and suffix in {".nc"} and ("corridor" in name_lower or "corr" in name_lower):
            return "corridor"
        if "accidents" in allowed_set and suffix in {".nc"} and (
            "accident" in name_lower or "density" in name_lower or "hotspot" in name_lower
        ):
            return "accidents"
        if "incidents" in allowed_set and suffix in {".parquet"} and (
            "incident" in name_lower or "accident" in name_lower
        ):
            return "incidents"
        if "ais" in allowed_set and suffix in {".parquet"} and "ais" in name_lower:
            return "ais"
        if "routes" in allowed_set and suffix in {".geojson", ".json"} and "route" in name_lower:
            return "routes"
        if "era5" in allowed_set and suffix in {".nc", ".grib", ".grb", ".netcdf"}:
            return "era5"
        if suffix in {".csv", ".txt"}:
            if "ais" in name_lower or "track" in name_lower:
                return "ais"
            if "incident" in name_lower or "accident" in name_lower:
                return "incidents"
        if suffix in {".parquet"}:
            if "incident" in name_lower or "accident" in name_lower:
                return "incidents"
        return None

    def _capture_overlay_assets(self) -> None:
        overlays_dir = OUTPUTS_DIR / "overlays"
        overlays_dir.mkdir(parents=True, exist_ok=True)
        for path in OUTPUTS_DIR.rglob("*"):
            if not path.is_file():
                continue
            if "overlays" in path.relative_to(OUTPUTS_DIR).parts:
                continue
            lowered = path.name.lower()
            if any(keyword in lowered for keyword in OVERLAY_KEYWORDS):
                destination = ensure_unique_destination(overlays_dir / path.name)
                self.actions.append(
                    PlannedOp(
                        kind="move",
                        source=path,
                        destination=destination,
                        reason="collect overlay/bounds assets under outputs/overlays",
                    )
                )

    def _enforce_output_tags(self) -> None:
        if not OUTPUTS_DIR.exists():
            self.missing.append(PlannedOp(kind="missing", source=OUTPUTS_DIR, reason="outputs directory not found"))
            return
        for file_path in OUTPUTS_DIR.iterdir():
            if file_path.is_dir():
                continue
            prefix = detect_output_prefix(file_path.stem)
            if not prefix:
                continue
            if is_tagged(file_path.stem):
                continue
            tag = timestamp_tag(file_path.stat().st_mtime, include_seconds=False)
            new_name = f"{file_path.stem}_{tag}{file_path.suffix}"
            destination = ensure_unique_destination(file_path.with_name(new_name))
            self.actions.append(
                PlannedOp(
                    kind="rename",
                    source=file_path,
                    destination=destination,
                    reason="append run timestamp tag generated from mtime",
                )
            )

    def _iter_output_files_for_archival(self) -> Iterable[Path]:
        if not OUTPUTS_DIR.exists():
            return []
        for path in OUTPUTS_DIR.rglob("*"):
            if not path.is_file():
                continue
            try:
                rel = path.relative_to(OUTPUTS_DIR)
            except ValueError:
                continue
            if rel.parts and rel.parts[0] == "archive":
                continue
            if rel.name in ARCHIVE_EXCLUDE:
                continue
            yield path

    def _plan_output_archival(self) -> None:
        if self.keep is None or self.keep < 0:
            return
        rename_sources = {action.source for action in self.actions if action.kind == "rename"}
        files_by_category: Dict[str, List[Path]] = {}
        for path in self._iter_output_files_for_archival():
            if path in rename_sources:
                # Skip items that will be renamed during this run; they can be archived on the next pass.
                continue
            rel = path.relative_to(OUTPUTS_DIR)
            if len(rel.parts) > 1:
                category = "/".join(rel.parts[:-1])
            else:
                prefix = detect_output_prefix(path.stem) or path.stem.lower()
                category = prefix
            files_by_category.setdefault(category, []).append(path)

        for category, paths in files_by_category.items():
            if category in {"", "other"}:
                continue
            ordered = sorted(paths, key=lambda p: p.stat().st_mtime, reverse=True)
            archive_candidates = ordered[self.keep :]
            if not archive_candidates:
                continue
            archive_root = OUTPUT_ARCHIVE_DIR / self.archive_tag / category
            for candidate in archive_candidates:
                destination = ensure_unique_destination(archive_root / candidate.name)
                self.actions.append(
                    PlannedOp(
                        kind="move",
                        source=candidate,
                        destination=destination,
                        reason=f"archive older output for category `{category}`",
                    )
                )

    def _plan_cleanup_dirs(self) -> None:
        # outputs/tmp special case
        tmp_dir = OUTPUTS_DIR / "tmp"
        if tmp_dir.exists():
            self.actions.append(PlannedOp(kind="delete", source=tmp_dir, reason="remove temporary outputs directory"))
        for entry in PROJECT_ROOT.rglob("*"):
            if not entry.is_dir():
                continue
            if entry.name in CLEANUP_DIR_NAMES:
                self.actions.append(
                    PlannedOp(kind="delete", source=entry, reason=f"remove cache directory `{entry.name}`")
                )

    def execute(self) -> None:
        if not self.apply_changes:
            return
        for action in self.actions:
            try:
                if action.kind in {"move", "rename"} and action.destination is not None:
                    action.destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(action.source), str(action.destination))
                elif action.kind == "delete":
                    if action.source.is_dir():
                        shutil.rmtree(action.source, ignore_errors=False)
                    elif action.source.exists():
                        action.source.unlink()
                action.executed = True
            except Exception as err:  # pragma: no cover - defensive
                action.executed = False
                action.error = str(err)

    def render_report(self) -> str:
        lines: List[str] = []
        mode = "APPLY" if self.apply_changes else "DRY-RUN"
        lines.append(f"Repo tidy report ({mode}) - {datetime.now():%Y-%m-%d %H:%M}")
        lines.append(f"Project root : {PROJECT_ROOT}")
        lines.append(f"Repo root    : {REPO_ROOT}")
        if self.keep is not None:
            lines.append(f"Outputs keep : latest {self.keep} batches per category")
            lines.append(f"Archive tag  : {self.archive_tag}")
        lines.append("")

        move_count = sum(1 for act in self.actions if act.kind in {"move", "rename"})
        delete_count = sum(1 for act in self.actions if act.kind == "delete")
        fail_count = sum(1 for act in self.actions if act.error)
        lines.append("Summary")
        lines.append(f"- Planned move/rename : {move_count}")
        lines.append(f"- Planned deletions   : {delete_count}")
        lines.append(f"- Missing items       : {len(self.missing)}")
        lines.append(f"- Notes               : {len(self.notes)}")
        lines.append(f"- Unresolved          : {len(self.unresolved)}")
        if self.apply_changes:
            lines.append(f"- Execution errors    : {fail_count}")
        lines.append("")

        if self.missing:
            lines.append("Missing items:")
            for miss in self.missing:
                lines.append(f"  - {miss.source} :: {miss.reason}")
            lines.append("")

        if self.actions:
            lines.append("Planned operations:")
            for action in self.actions:
                status = ""
                if self.apply_changes:
                    if action.error:
                        status = f" [FAILED: {action.error}]"
                    elif action.executed:
                        status = " [OK]"
                if action.destination:
                    lines.append(f"  - {action.kind.upper()}: {action.source} -> {action.destination} ({action.reason}){status}")
                else:
                    lines.append(f"  - {action.kind.upper()}: {action.source} ({action.reason}){status}")
            lines.append("")

        if self.notes:
            lines.append("Notes:")
            for note in self.notes:
                lines.append(f"  - {note}")
            lines.append("")

        if self.unresolved:
            lines.append("Unresolved items (manual review suggested):")
            for item in self.unresolved:
                lines.append(f"  - {item}")
            lines.append("")

        return "\n".join(lines)

    def generate_manifest(self) -> Path:
        docs_dir = MANIFEST_PATH.parent
        docs_dir.mkdir(parents=True, exist_ok=True)
        sections: List[str] = []
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        sections.append(f"# ArcticRoute Manifest\n\nGenerated: {now_str}\n")

        for label, base, patterns, exclude_parts in MANIFEST_TARGETS:
            sections.append(f"## {label}")
            items = self._collect_manifest_items(base, patterns, exclude_parts)
            if not items:
                sections.append("_No matching files found._\n")
                continue
            sections.append("| Path | Size | Modified | SHA1 |")
            sections.append("| --- | ---: | --- | --- |")
            for rel_path, size_bytes, mtime_dt, sha1 in items:
                sections.append(
                    f"| {rel_path.as_posix()} | {self._format_size(size_bytes)} | {mtime_dt:%Y-%m-%d %H:%M} | {sha1} |"
                )
            sections.append("")

        sections.append("## Minimal Run Set (suggested)")
        for path in MANIFEST_MINIMAL_SET:
            rel = path.relative_to(PROJECT_ROOT)
            if path.exists():
                sections.append(f"- [x] {rel.as_posix()}")
            else:
                sections.append(f"- [ ] {rel.as_posix()} (missing)")
        sections.append("")

        content = "\n".join(sections).rstrip() + "\n"
        MANIFEST_PATH.write_text(content, encoding="utf-8")
        return MANIFEST_PATH

    def _collect_manifest_items(
        self, base: Path, patterns: Sequence[str], exclude_parts: Sequence[str]
    ) -> List[Tuple[Path, int, datetime, str]]:
        if not base.exists():
            return []
        items: Dict[Path, Tuple[int, datetime, str]] = {}
        exclude_set = {part.lower() for part in exclude_parts}
        for pattern in patterns:
            for path in base.rglob(pattern):
                if not path.is_file():
                    continue
                if exclude_set and any(part.lower() in exclude_set for part in path.relative_to(base).parts):
                    continue
                rel_path = path.relative_to(PROJECT_ROOT)
                stat = path.stat()
                items[rel_path] = (stat.st_size, datetime.fromtimestamp(stat.st_mtime), self._sha1(path))
        return [
            (rel_path, *items[rel_path])
            for rel_path in sorted(items, key=lambda p: (len(p.parts), p.as_posix()))
        ]

    @staticmethod
    def _sha1(path: Path) -> str:
        hasher = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _format_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 ** 2:
            return f"{size / 1024:.1f} KiB"
        if size < 1024 ** 3:
            return f"{size / (1024 ** 2):.1f} MiB"
        return f"{size / (1024 ** 3):.2f} GiB"


def main() -> int:
    args = parse_args()
    tidy = RepoTidy(
        apply_changes=args.apply,
        keep=args.keep,
        archive_tag=args.archive_tag,
        write_manifest=args.manifest,
    )
    tidy.run()
    tidy.execute()

    log_path = (
        LOG_DIR / f"repo_tidy_apply_{timestamp_tag(include_seconds=True)}.txt"
        if args.apply
        else DEFAULT_DRYRUN_LOG
    )
    report = tidy.render_report()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(report, encoding="utf-8")
    print(report)
    if args.manifest:
        manifest_path = tidy.generate_manifest()
        print(f"[MANIFEST] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
