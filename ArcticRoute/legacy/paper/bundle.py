from __future__ import annotations
import json, hashlib, time, os, zipfile
from pathlib import Path
from typing import Dict, Any, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
ARCTIC_DIR = REPO_ROOT / "ArcticRoute"
REPORT_DIR = ARCTIC_DIR / "reports" / "paper"
RELEASE_DIR = REPO_ROOT / "outputs" / "release"


def _sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _collect_assets() -> List[Path]:
    assets: List[Path] = []
    if REPORT_DIR.exists():
        for sub in ("figures", "tables", "videos"):
            p = REPORT_DIR / sub
            if p.exists():
                assets.extend(sorted([q for q in p.rglob("*") if q.is_file() and not q.name.endswith(".zip")]))
        # docs
        for name in ("paper.md", "dataset_card.md", "method_card.md", "CITATION.cff", "LICENSE"):
            q = REPORT_DIR / name
            if q.exists():
                assets.append(q)
    return assets


def build_bundle(profile_id: str, tag: str) -> Dict[str, Any]:
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%S")
    prefix = f"arcticroute_repro_{tag}"
    zip_path = RELEASE_DIR / f"{prefix}.zip"

    assets = _collect_assets()
    manifest: Dict[str, Any] = {
        "profile": profile_id,
        "run_id": run_id,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "entries": [],
    }
    sums_lines: List[str] = []

    # Write zip
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # add assets
        for p in assets:
            rel = p.relative_to(REPO_ROOT)
            zf.write(p, arcname=str(rel))
            sha = _sha256_file(p)
            manifest["entries"].append({"logical_id": p.name, "path": str(rel), "sha256": sha, "size": p.stat().st_size})
            sums_lines.append(f"{sha}  {rel}\n")
        # write MANIFEST.json and SHA256SUMS.txt inside zip
        man_json = json.dumps(manifest, ensure_ascii=False, indent=2)
        zf.writestr("MANIFEST.json", man_json)
        zf.writestr("SHA256SUMS.txt", "".join(sums_lines))

    # also write adjacent meta
    (zip_path.parent / f"{zip_path.name}.meta.json").write_text(json.dumps({
        "logical_id": zip_path.name,
        "inputs": [str(p.relative_to(REPO_ROOT)) for p in assets],
        "manifest_count": len(manifest["entries"]),
        "profile": profile_id,
        "tag": tag,
        "run_id": run_id,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"zip": str(zip_path), "entries": len(manifest["entries"]) }


def check_bundle(bundle_path: Path) -> Dict[str, Any]:
    bundle_path = Path(bundle_path)
    if not bundle_path.exists():
        raise FileNotFoundError(str(bundle_path))
    # open and read manifest + sums
    with zipfile.ZipFile(bundle_path, "r") as zf:
        names = set(zf.namelist())
        if "MANIFEST.json" not in names:
            raise ValueError("MANIFEST.json missing in bundle")
        if "SHA256SUMS.txt" not in names:
            raise ValueError("SHA256SUMS.txt missing in bundle")
        manifest = json.loads(zf.read("MANIFEST.json").decode("utf-8"))
        sums_txt = zf.read("SHA256SUMS.txt").decode("utf-8")
        # basic verify: count >= 1, hashes format
        ok = True
        reasons: List[str] = []
        entries = manifest.get("entries", [])
        if not isinstance(entries, list) or len(entries) == 0:
            ok = False
            reasons.append("no entries in manifest")
        # verify hash lines
        for line in sums_txt.splitlines():
            if "  " not in line:
                ok = False
                reasons.append("bad SHA256SUMS format")
                break
        # Optionally spot-check at most 3 files by extracting to temp and hashing
        import tempfile
        import shutil
        with tempfile.TemporaryDirectory() as td:
            to_check = [e for e in entries[:3] if isinstance(e, dict) and e.get("path") in names]
            for e in to_check:
                p = e.get("path")
                sha = e.get("sha256")
                target = Path(td) / Path(p).name
                with open(target, "wb") as f:
                    f.write(zf.read(p))
                real = _sha256_file(target)
                if real != sha:
                    ok = False
                    reasons.append(f"hash mismatch for {p}")
        # environment snapshot
        import platform, sys
        env = {
            "python": sys.version.split(" ")[0],
            "platform": platform.platform(),
        }
        return {"ok": ok, "reasons": reasons, "entries": len(entries), "env": env}


__all__ = ["build_bundle", "check_bundle"]






