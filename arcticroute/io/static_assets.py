from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


MANIFEST_ENV = "ARCTICROUTE_STATIC_ASSETS_MANIFEST"
DEFAULT_MANIFEST_CANDIDATES = [
    Path("configs/static_assets_manifest.json"),
    Path("configs/static_assets_manifest.yaml"),
    Path("configs/static_assets_manifest.yml"),
    Path("data/static_assets_manifest.json"),
    Path("data/static_assets_manifest.yaml"),
    Path("data/static_assets_manifest.yml"),
    Path("static_assets_manifest.json"),
    Path("static_assets_manifest.yaml"),
    Path("static_assets_manifest.yml"),
]


@dataclass(frozen=True)
class StaticAssetRecord:
    asset_id: str
    path: Path | None
    raw_path: str | None = None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_yaml(path: Path) -> Any:
    if yaml is None:
        raise RuntimeError("PyYAML not available")
    return yaml.safe_load(path.read_text(encoding="utf-8-sig"))


def _read_manifest(path: Path) -> Any:
    if path.suffix.lower() in {".yaml", ".yml"}:
        return _load_yaml(path)
    return _load_json(path)


def _normalize_assets(data: Any, manifest_path: Path) -> tuple[dict[str, StaticAssetRecord], list[str], list[str]]:
    mapping: dict[str, StaticAssetRecord] = {}
    unknown_ids: list[str] = []
    warnings: list[str] = []

    def _resolve(raw: str | None) -> Path | None:
        if not raw:
            return None
        p = Path(raw)
        if not p.is_absolute():
            p = (manifest_path.parent / p).resolve()
        return p

    assets: Iterable[Any]
    if isinstance(data, dict):
        if "assets" in data and isinstance(data["assets"], list):
            assets = data["assets"]
        elif "assets" in data and isinstance(data["assets"], dict):
            assets = []
            for key, value in data["assets"].items():
                if isinstance(value, dict):
                    entry = dict(value)
                    entry.setdefault("id", key)
                    assets.append(entry)
                elif isinstance(value, str):
                    assets.append({"id": key, "path": value})
        elif "items" in data and isinstance(data["items"], list):
            assets = data["items"]
        elif all(isinstance(v, str) for v in data.values()):
            assets = [
                {"id": k, "path": v}
                for k, v in data.items()
            ]
        else:
            assets = []
            warnings.append("manifest format not recognized")
    elif isinstance(data, list):
        assets = data
    else:
        assets = []
        warnings.append("manifest format not recognized")

    for entry in assets:
        if not isinstance(entry, dict):
            continue
        asset_id = (
            entry.get("id")
            or entry.get("asset_id")
            or entry.get("logical_id")
            or entry.get("name")
            or entry.get("key")
        )
        raw_path = (
            entry.get("path")
            or entry.get("local_path")
            or entry.get("file")
            or entry.get("filepath")
            or entry.get("uri")
            or entry.get("url")
        )
        if not asset_id:
            unknown_ids.append("<missing-id>")
            continue
        mapping[str(asset_id)] = StaticAssetRecord(
            asset_id=str(asset_id),
            path=_resolve(raw_path) if raw_path else None,
            raw_path=str(raw_path) if raw_path is not None else None,
        )

    return mapping, unknown_ids, warnings


def _find_manifest_path() -> Path | None:
    import os

    env_value = os.getenv(MANIFEST_ENV)
    if env_value:
        return Path(env_value)

    for candidate in DEFAULT_MANIFEST_CANDIDATES:
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=1)
def _load_manifest_cached() -> tuple[Path | None, dict[str, StaticAssetRecord], list[str], list[str]]:
    manifest_path = _find_manifest_path()
    if manifest_path is None or not manifest_path.exists():
        return None, {}, [], ["manifest not found"]
    try:
        data = _read_manifest(manifest_path)
    except Exception as exc:  # pragma: no cover - defensive
        return manifest_path, {}, [], [f"failed to read manifest: {exc}"]

    mapping, unknown_ids, warnings = _normalize_assets(data, manifest_path)
    return manifest_path, mapping, unknown_ids, warnings


def get_static_asset_path(asset_id: str) -> Path | None:
    _, mapping, _, _ = _load_manifest_cached()
    record = mapping.get(asset_id)
    if record is None:
        return None
    return record.path


def static_assets_summary(
    *,
    required_ids: Iterable[str] | None = None,
    optional_ids: Iterable[str] | None = None,
) -> dict[str, Any]:
    manifest_path, mapping, unknown_ids, warnings = _load_manifest_cached()
    required_ids = list(required_ids or [])
    optional_ids = list(optional_ids or [])

    checks: dict[str, dict[str, Any]] = {}
    for asset_id in required_ids + optional_ids:
        record = mapping.get(asset_id)
        path = record.path if record else None
        exists = bool(path and path.exists())
        checks[asset_id] = {
            "path": str(path) if path else None,
            "exists": exists,
            "optional": asset_id in optional_ids,
        }

    missing_required = [aid for aid in required_ids if not checks.get(aid, {}).get("exists", False)]
    missing_optional = [aid for aid in optional_ids if not checks.get(aid, {}).get("exists", False)]

    known_ids = set(required_ids + optional_ids)
    unknown_assets = sorted(set([aid for aid in mapping.keys() if aid not in known_ids] + unknown_ids))

    return {
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_loaded": manifest_path is not None and bool(mapping),
        "checks": checks,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "unknown_assets": unknown_assets,
        "warnings": warnings,
    }
