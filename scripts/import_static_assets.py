from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------
# Matching rules (enhanced)
# ---------------------------
# Each asset_id has:
# - exact_names: preferred exact filename matches (case-insensitive)
# - must_tokens: candidate path/name must contain ALL tokens (case-insensitive)
# - extensions: allowed extensions
# - optional: whether it's optional (missing won't fail)
#
# Selection scoring:
# 1) exact name match highest
# 2) contains all tokens
# 3) larger size preferred (to avoid tiny wrong files)
#
ASSET_SPECS: Dict[str, Dict[str, Any]] = {
    "bathymetry_ibcao_v4_200m_nc": {
        "optional": False,
        "extensions": [".nc", ".netcdf"],
        "exact_names": ["IBCAO_v4_200m.nc"],
        "must_tokens": ["ibcao", "v4", "200m"],
    },
    "bathymetry_ibcao_v5_1_2025_depth_400m_tif": {
        "optional": True,  # optional because we can use nc for shallow penalty
        "extensions": [".tif", ".tiff"],
        "exact_names": ["ibcao_v5_1_2025_depth_400m.tif"],
        "must_tokens": ["ibcao", "400m"],
    },
    "ports_world_port_index_geojson": {
        "optional": False,
        "extensions": [".geojson", ".json"],
        "exact_names": ["World_Port_Index.geojson", "world_port_index.geojson"],
        "must_tokens": ["port", "index"],
    },
    "corridors_shipping_hydrography_geojson": {
        "optional": False,
        "extensions": [".geojson", ".json"],
        "exact_names": [],  # intentionally empty; names vary a lot
        "must_tokens": ["shipping", "hydrography"],  # most stable tokens
    },
    "rules_pub150_pdf": {
        "optional": False,
        "extensions": [".pdf"],
        "exact_names": ["pub150bk.pdf"],
        "must_tokens": ["pub", "150"],
    },
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_files_recursive(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file():
            yield p


def norm(s: str) -> str:
    return s.lower().replace(" ", "").replace("-", "_")


def token_hit(path: Path, tokens: List[str]) -> bool:
    hay = (path.name + " " + str(path)).lower()
    return all(t.lower() in hay for t in tokens)


def is_ext_ok(path: Path, exts: List[str]) -> bool:
    return path.suffix.lower() in [e.lower() for e in exts]


@dataclass
class Candidate:
    path: Path
    size: int
    exact_hit: bool
    token_hit: bool

    @property
    def score(self) -> Tuple[int, int, int]:
        # exact_hit first, token_hit second, size third
        return (1 if self.exact_hit else 0, 1 if self.token_hit else 0, self.size)


def pick_best_candidate(asset_id: str, candidates: List[Candidate]) -> Optional[Candidate]:
    if not candidates:
        return None
    # sort descending by score
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0]


def discover_candidates(src: Path, asset_id: str) -> List[Candidate]:
    spec = ASSET_SPECS[asset_id]
    exts: List[str] = spec["extensions"]
    exact_names: List[str] = [n.lower() for n in spec.get("exact_names", [])]
    must_tokens: List[str] = spec.get("must_tokens", [])

    out: List[Candidate] = []
    for p in iter_files_recursive(src):
        if not is_ext_ok(p, exts):
            continue
        exact_hit = p.name.lower() in exact_names if exact_names else False
        tok_hit = token_hit(p, must_tokens) if must_tokens else True
        # for assets with empty exact_names, rely on tokens
        if exact_names:
            ok = exact_hit or tok_hit
        else:
            ok = tok_hit
        if not ok:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        out.append(Candidate(path=p, size=int(size), exact_hit=exact_hit, token_hit=tok_hit))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source folder containing raw downloaded assets (recursive scan)")
    ap.add_argument("--dst", required=True, help="destination folder inside repo (gitignored), e.g. data/static_assets")
    ap.add_argument("--manifest", required=True, help="manifest output path, e.g. env/static_assets_manifest.json")
    ap.add_argument("--dry-run", action="store_true", help="only print selection, do not copy/write manifest")
    args = ap.parse_args()

    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    manifest = Path(args.manifest).expanduser().resolve()

    if not src.exists():
        print(f"[import_static_assets] src not found: {src}", file=sys.stderr)
        return 2

    dst.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    missing_required: List[str] = []
    entries: Dict[str, Any] = {}
    selection_debug: Dict[str, Any] = {}

    for asset_id, spec in ASSET_SPECS.items():
        cands = discover_candidates(src, asset_id)
        best = pick_best_candidate(asset_id, cands)
        selection_debug[asset_id] = {
            "optional": bool(spec.get("optional", False)),
            "candidates": [
                {
                    "path": str(c.path),
                    "bytes": c.size,
                    "exact_hit": c.exact_hit,
                    "token_hit": c.token_hit,
                }
                for c in sorted(cands, key=lambda x: x.score, reverse=True)[:10]
            ],
            "selected": str(best.path) if best else None,
        }

        if best is None:
            if not spec.get("optional", False):
                missing_required.append(asset_id)
            continue

        fname = best.path.name
        dp = dst / fname

        if args.dry_run:
            print(f"[dry-run] {asset_id} -> {best.path}")
        else:
            shutil.copy2(best.path, dp)
            st = dp.stat()
            entries[asset_id] = {
                "asset_id": asset_id,
                "filename": fname,
                "relpath": str(Path(args.dst) / fname).replace("\\", "/"),
                "abspath_hint": str(dp),
                "bytes": int(st.st_size),
                "mtime": int(st.st_mtime),
                "sha256": sha256_file(dp),
            }

    out = {
        "schema": "arcticroute.static_assets.v1",
        "root_rel": str(Path(args.dst)).replace("\\", "/"),
        "entries": entries,
        "missing_required": missing_required,
        "debug": selection_debug,
    }

    if args.dry_run:
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return 0 if not missing_required else 2

    manifest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing_required:
        print("[import_static_assets] missing required asset_ids:", file=sys.stderr)
        for aid in missing_required:
            print("  -", aid, file=sys.stderr)
        print(f"[import_static_assets] wrote manifest (partial): {manifest}", file=sys.stderr)
        return 2

    print(f"[import_static_assets] OK. wrote manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
