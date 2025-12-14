from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import xarray as xr


MAX_DEPTH = 4
MAX_ITEMS = 4


def format_preview(items: Sequence[str], max_items: int = MAX_ITEMS) -> str:
    if not items:
        return "-"
    if len(items) > max_items:
        return ", ".join(items[:max_items]) + ", ..."
    return ", ".join(items)


def iter_nc_files(base: Path, max_depth: int = MAX_DEPTH) -> Iterable[Path]:
    """Yield .nc files under base, stopping when depth exceeds max_depth."""
    for dirpath, dirnames, filenames in os.walk(base):
        current = Path(dirpath)
        depth = len(current.relative_to(base).parts)
        if depth >= max_depth:
            dirnames[:] = []  # stop descending further
        for fname in filenames:
            if fname.lower().endswith(".nc"):
                yield current / fname


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def candidate_data_roots() -> list[Path]:
    roots: list[Path] = []

    env = os.getenv("ARCTICROUTE_DATA_ROOT")
    if env:
        roots.append(Path(env))

    project_root = get_project_root()
    roots.append(project_root.parent / "ArcticRoute_data_backup")
    roots.append(project_root / "data_real")

    deduped: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        key = str(r.resolve())
        if key not in seen:
            deduped.append(r)
            seen.add(key)
    return deduped


def hint_for_path(path: Path) -> str | None:
    name = path.name.lower()
    if "env_clean" in name or "grid_spec" in name:
        return "grid"
    if "land_mask" in name or "landmask" in name:
        return "landmask"
    if "ice_thick" in name:
        return "ice_thick"
    if "wave" in name or "swh" in name:
        return "wave"
    if "sic" in name:
        return "sic"
    return None


def describe_nc(path: Path) -> tuple[str, str, str]:
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        with xr.open_dataset(path, decode_times=False) as ds:
            var_names = format_preview(list(ds.data_vars))
            dim_names = format_preview(list(ds.dims))
        return f"{size_mb:.2f}MB", var_names, dim_names
    except Exception as exc:  # noqa: BLE001
        return f"{size_mb:.2f}MB", f"error: {exc}", "-"


def inspect_root(root: Path) -> None:
    status = "exists" if root.exists() else "missing"
    print(f"[ROOT] {root} ({status})")
    if not root.exists():
        return

    targets = [root / "data_processed", root / "data_processed" / "newenv"]
    for target in targets:
        if not target.exists():
            print(f"[DIR] {target} (missing)")
            continue

        print(f"[DIR] {target} (scanning)")
        for nc_path in iter_nc_files(target):
            rel_path = nc_path.relative_to(root)
            size_info, var_info, dim_info = describe_nc(nc_path)
            hint = hint_for_path(nc_path)
            hint_prefix = f"[HINT:{hint}] " if hint else ""
            print(
                f"{hint_prefix}[NC] {rel_path.as_posix()}  "
                f"size={size_info}  vars={var_info}  dims={dim_info}"
            )


def inspect_real_data_layout() -> None:
    for root in candidate_data_roots():
        inspect_root(root)


if __name__ == "__main__":
    inspect_real_data_layout()
