from __future__ import annotations
import argparse, json, shutil, time
from pathlib import Path
from typing import Optional, Dict, Any, List


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def find_latest_nc(cache_dir: Path, patterns: List[str]) -> Optional[Path]:
    cands: List[Path] = []
    for pat in patterns:
        cands.extend(cache_dir.rglob(pat))
    cands = [p for p in cands if p.is_file()]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0]


def resolve_default_dirs(repo_root: Path) -> tuple[Path, Path]:
    # cache dir candidates
    cache_dir = repo_root / "data" / "cmems_cache"
    if not cache_dir.exists():
        cache_dir = repo_root / "data" / "cmems" / "cache"
    # newenv dir candidates
    newenv_dir = repo_root / "data_processed" / "newenv"
    return cache_dir, newenv_dir


def sync_to_newenv(
    cache_dir: Optional[str] = None,
    newenv_dir: Optional[str] = None,
    write_index: bool = True,
    index_path: str = "reports/cmems_newenv_index.json",
    dry_run: bool = False,
) -> Dict[str, Any]:
    repo_root = Path(".").resolve()
    default_cache, default_newenv = resolve_default_dirs(repo_root)
    cache = Path(cache_dir).resolve() if cache_dir else default_cache.resolve()
    newenv = Path(newenv_dir).resolve() if newenv_dir else default_newenv.resolve()
    newenv.mkdir(parents=True, exist_ok=True)

    sic_nc = find_latest_nc(cache, patterns=["*sic*.nc", "*ice*conc*.nc", "*siconc*.nc"])
    swh_nc = find_latest_nc(cache, patterns=["*swh*.nc", "*wav*.nc", "*wave*height*.nc"])

    meta: Dict[str, Any] = {
        "timestamp": _now_iso(),
        "cache_dir": str(cache),
        "newenv_dir": str(newenv),
        "sic_src": str(sic_nc) if sic_nc else None,
        "swh_src": str(swh_nc) if swh_nc else None,
        "sic_dst": None,
        "swh_dst": None,
        "dry_run": dry_run,
        "ok": False,
        "reason": None,
    }

    if not cache.exists():
        meta["reason"] = "cache_dir_missing"
        if write_index:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(index_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    if sic_nc is None and swh_nc is None:
        meta["reason"] = "no_nc_found_in_cache"
        if write_index:
            Path(index_path).parent.mkdir(parents=True, exist_ok=True)
            Path(index_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return meta

    # canonical filenames expected by pipeline
    if sic_nc is not None:
        sic_dst = newenv / "ice_copernicus_sic.nc"
        meta["sic_dst"] = str(sic_dst)
        if not dry_run:
            shutil.copy2(sic_nc, sic_dst)
    if swh_nc is not None:
        swh_dst = newenv / "wave_swh.nc"
        meta["swh_dst"] = str(swh_dst)
        if not dry_run:
            shutil.copy2(swh_nc, swh_dst)

    meta["ok"] = True
    meta["reason"] = "synced_from_cache"
    if write_index:
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        Path(index_path).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--newenv-dir", default=None)
    ap.add_argument("--index-path", default="reports/cmems_newenv_index.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    meta = sync_to_newenv(
        cache_dir=args.cache_dir,
        newenv_dir=args.newenv_dir,
        index_path=args.index_path,
        dry_run=args.dry_run,
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
