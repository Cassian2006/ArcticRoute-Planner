#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Query Sentinel STAC catalogues and optionally validate asset access.

@role: pipeline
"""

"""
Query Sentinel STAC catalogues and optionally validate asset download paths.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT.parent))

from ArcticRoute.io.stac_ingest import (  # noqa: E402
    STACAuthError,
    build_asset_access_params,
    download_asset_preview,
    stac_search_sat,
    write_stac_results,
)


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


def _parse_bbox(text: str) -> Sequence[float]:
    parts = [float(part) for part in text.replace(",", " ").split()]
    if len(parts) != 4:
        raise ValueError("bbox must supply four numeric values (N, W, S, E).")
    north, west, south, east = parts
    min_lon = min(west, east)
    max_lon = max(west, east)
    min_lat = min(south, north)
    max_lat = max(south, north)
    # STAC expects [min_lon, min_lat, max_lon, max_lat]
    return [min_lon, min_lat, max_lon, max_lat]


def _collect_asset_refs(items: List[dict]) -> List[Dict[str, Dict[str, str]]]:
    collected: List[Dict[str, Dict[str, str]]] = []
    for item in items:
        item_id = item.get("id")
        assets = item.get("assets") or {}
        hrefs = {
            name: data.get("href", "")
            for name, data in assets.items()
            if isinstance(data, dict) and data.get("href")
        }
        collected.append({"id": item_id, "assets": hrefs})
    return collected


def _pick_validation_assets(items: List[dict], mission: str) -> List[str]:
    hrefs: List[str] = []
    preferred_names = {
        "S2": ("B04", "B03", "visual", "B08"),
        "S1": ("vv", "vh", "sigma0_vv"),
    }
    names = preferred_names.get(mission.upper(), ())
    for item in items:
        assets = item.get("assets") or {}
        for candidate in names:
            asset = assets.get(candidate)
            if isinstance(asset, dict) and asset.get("href"):
                hrefs.append(asset["href"])
                break
        if len(hrefs) >= 2:
            break
    return hrefs


def main(argv: List[str] | None = None) -> int:
    _load_env_file()
    default_source = os.getenv("DEFAULT_STAC_SOURCE", "CDSE")
    default_mpc_url = os.getenv("MPC_STAC_URL", "https://planetarycomputer.microsoft.com/api/stac/v1")

    parser = argparse.ArgumentParser(description="Query STAC and list Sentinel assets.")
    parser.add_argument("--bbox", required=True, help="Bounding box N,W,S,E (degrees)")
    parser.add_argument("--date", dest="date", help="Datetime or interval (e.g. 2023-07-15)")
    parser.add_argument("--mission", default="S2", choices=["S1", "S2"], help="Mission identifier")
    parser.add_argument("--limit", type=int, default=10, help="Maximum items to fetch")
    parser.add_argument("--source", default=default_source, help="Catalogue source (CDSE or MPC)")
    parser.add_argument(
        "--stac-url",
        help="Override STAC root URL (defaults to MPC_STAC_URL or CDSE endpoint)",
    )
    parser.add_argument("--lazy", action="store_true", help="Skip asset download validation")
    parser.add_argument(
        "--max-cloud",
        type=float,
        help="Optional maximum cloud cover percentage for Sentinel-2 queries.",
    )
    args = parser.parse_args(argv)

    bbox = _parse_bbox(args.bbox)
    source = args.source.upper()
    mission = args.mission.upper()
    stac_url = args.stac_url
    if source == "MPC" and not stac_url:
        stac_url = default_mpc_url

    max_cloud = args.max_cloud

    try:
        items = stac_search_sat(
            bbox=bbox,
            date=args.date,
            mission=mission,
            source=source,
            limit=args.limit,
            stac_url=stac_url,
            max_cloud=max_cloud,
        )
    except STACAuthError as err:
        print(str(err))
        return 1
    except RuntimeError as err:
        print(f"[STAC] Query failed: {err}")
        return 1

    collected = _collect_asset_refs(items)
    payload = {
        "source": source,
        "mission": mission,
        "bbox": bbox,
        "datetime": args.date,
        "limit": args.limit,
        "count": len(collected),
        "max_cloud": max_cloud,
        "items": collected,
    }
    write_stac_results(mission, args.date, payload)

    if not collected:
        print("[STAC] No items returned for the specified query.")
        return 0

    print("[STAC] Asset hrefs:")
    for item in collected:
        print(f"  Item {item.get('id')}:")
        for name, href in item["assets"].items():
            print(f"    - {name}: {href}")

    if args.lazy:
        return 0

    validation_hrefs = _pick_validation_assets(items, mission)
    if not validation_hrefs:
        print("[STAC] No candidate assets found for validation download.")
        return 0

    for href in validation_hrefs:
        headers, auth = build_asset_access_params(href)
        try:
            chunk = download_asset_preview(href, headers=headers, auth=auth)
        except STACAuthError as err:
            print(err)
            return 2
        except RuntimeError as err:
            print(f"[STAC] Asset validation failed: {err}")
            return 2
        else:
            size = len(chunk)
            auth_mode = "Bearer" if headers.get("Authorization", "").startswith("Bearer") else ("Basic" if auth else "None")
            print(f"[STAC] Validated asset {href} (received {size} bytes, auth mode={auth_mode})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
