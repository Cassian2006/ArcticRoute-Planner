"""import_static_assets.py
从外部下载目录复制静态资产到仓库本地的 data/static_assets 目录，并生成清单 manifest。

使用方法示例：

python -m scripts.import_static_assets \
  --src "C:\\Users\\me\\Downloads\\xinshuju" \
  --dst "data/static_assets" \
  --manifest "env/static_assets_manifest.json"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any

# asset_id -> 原始文件名
ASSETS: dict[str, str] = {
    "bathymetry_ibcao_v4_200m_nc": "IBCAO_v4_200m.nc",
    "bathymetry_ibcao_v5_1_2025_depth_400m_tif": "ibcao_v5_1_2025_depth_400m.tif",
    "ports_world_port_index_geojson": "World_Port_Index.geojson",
    "corridors_shipping_hydrography_geojson": "Shipping_and_Hydrography_-6814444480993915291.geojson",
    "rules_pub150_pdf": "pub150bk.pdf",
}


@dataclass
class ManifestEntry:
    asset_id: str
    filename: str
    relpath: str
    abspath_hint: str
    bytes: int
    mtime: int
    sha256: str

    def as_json(self) -> Dict[str, Any]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """计算文件的 sha256 值。"""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(src: Path, dst: Path) -> tuple[dict[str, Any], list[str]]:
    """复制文件并生成 manifest 结构体，返回 (entries, missing)"""
    entries: dict[str, Any] = {}
    missing: list[str] = []

    for aid, fname in ASSETS.items():
        sp = src / fname
        if not sp.exists():
            missing.append(str(sp))
            continue

        # 目标路径保持文件名不变
        dp = dst / fname
        dp.parent.mkdir(parents=True, exist_ok=True)
        # 使用 copy2 保留元数据
        shutil.copy2(sp, dp)

        st = dp.stat()
        entry = ManifestEntry(
            asset_id=aid,
            filename=fname,
            relpath=str((Path("data") / "static_assets" / fname).as_posix()),
            abspath_hint=str(dp),
            bytes=st.st_size,
            mtime=int(st.st_mtime),
            sha256=sha256_file(dp),
        )
        entries[aid] = entry.as_json()

    manifest_dict: dict[str, Any] = {
        "schema": "arcticroute.static_assets.v1",
        "root_rel": "data/static_assets",
        "entries": entries,
        "missing": missing,
    }
    return manifest_dict, missing


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="源目录，包含下载好的静态资产文件")
    parser.add_argument("--dst", required=True, help="目标目录，例如 data/static_assets (gitignored)")
    parser.add_argument("--manifest", required=True, help="manifest 输出路径，例如 env/static_assets_manifest.json")

    args = parser.parse_args()
    src = Path(args.src).expanduser().resolve()
    dst = Path(args.dst).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()

    if not src.exists():
        print(f"[import_static_assets] 源目录不存在: {src}", file=sys.stderr)
        return 1

    dst.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_dict, missing = build_manifest(src, dst)

    manifest_path.write_text(json.dumps(manifest_dict, ensure_ascii=False, indent=2), encoding="utf-8")

    if missing:
        print("[import_static_assets] 缺失文件:", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print(f"[import_static_assets] manifest 已写入（缺失部分留空）：{manifest_path}")
        # 2 表示缺失文件
        return 2

    print(f"[import_static_assets] 成功，manifest 写入: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

