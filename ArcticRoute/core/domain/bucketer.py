# REUSE: Inspired by discussions in phase-L-domain-adapt.md

import logging
from typing import Dict, Any, Tuple, Optional

import pandas as pd

try:
    import geopandas as gpd  # type: ignore
    _HAS_GPD = True
except Exception:  # pragma: no cover
    gpd = None  # type: ignore
    _HAS_GPD = False

# Placeholder for config loading
# from ArcticRoute.config import get_config

logger = logging.getLogger(__name__)


class Bucketer:
    """Determines the domain bucket based on context (region, season, vessel)."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the bucketer with domain definitions.

        Args:
            config: A dictionary, typically from a YAML file, containing
                    'regions_geojson_path', 'season_rules', and 'vessel_map'.
        """
        self.region_defs = config.get("regions")
        self.regions_gdf = None
        if _HAS_GPD and not self.region_defs:
            try:
                self.regions_gdf = gpd.read_file(config["regions_geojson_path"])  # type: ignore[arg-type]
            except Exception:
                self.regions_gdf = None
        self.season_rules = config["season_rules"]
        self.vessel_map = config["vessel_map"]
        self.default_bucket = config.get("default_bucket", "global")

    def _get_region(self, lat: float, lon: float) -> Optional[str]:
        """Finds the region containing the given point.
        优先使用轻量 bbox 定义（config.regions），否则在可用时使用 GeoPandas 多边形。
        """
        # 1) 简单 bbox 列表：[{name: 'NSR', bbox: [w,s,e,n]}]
        if self.region_defs:
            for r in self.region_defs:
                try:
                    w,s,e,n = r.get("bbox")
                    if (w <= lon <= e) and (s <= lat <= n):
                        return str(r.get("name"))
                except Exception:
                    continue
            return None
        # 2) GeoPandas
        if _HAS_GPD and self.regions_gdf is not None:
            try:
                point = gpd.points_from_xy([lon], [lat])[0]  # type: ignore
                for _, row in self.regions_gdf.iterrows():
                    if point.within(row.geometry):
                        # 若无 name 列，回退索引
                        name = row.get("name") if "name" in self.regions_gdf.columns else None
                        return str(name if name is not None else row.name)
            except Exception:
                return None
        return None

    def _get_season(self, timestamp: pd.Timestamp) -> str:
        """Determines the season from the timestamp."""
        month = timestamp.month
        for season, months in self.season_rules.items():
            if month in months:
                return season
        return "unknown_season"

    def _get_vessel_category(self, vessel_type: str) -> str:
        """Maps a specific vessel type to a broader category."""
        return self.vessel_map.get(vessel_type, "unknown_vessel")

    def infer_bucket(self, lat: float, lon: float, timestamp: pd.Timestamp, vessel_type: str) -> str:
        """
        Infers the domain bucket for a given context with a fallback strategy.

        Fallback hierarchy:
        1. (region, season, vessel_category)
        2. (region, season)
        3. (region)
        4. global

        Returns:
            The most specific bucket ID found.
        """
        region = self._get_region(lat, lon)
        season = self._get_season(timestamp)
        vessel_category = self._get_vessel_category(vessel_type)

        # Fallback logic
        if region:
            # Attempt most specific bucket
            bucket_id = f"{region}_{season}_{vessel_category}"
            # Simplification: for now, we don't check if bucket_id is valid.
            # In a real scenario, we'd check against a list of trained buckets.

            # For now, let's just create a composite key and assume it's valid.
            # A more robust implementation would check against a pre-defined set of buckets.
            if vessel_category != "unknown_vessel":
                return f"{region}_{season}_{vessel_category}"
            return f"{region}_{season}"

        return self.default_bucket


def get_bucketer_from_config():
    """Factory function to create a Bucketer instance from global config."""
    # Placeholder: In a real app, this would load from a central config system.
    # config = get_config()
    # return Bucketer(config['domain_buckets'])
    pass # Returning None for now

