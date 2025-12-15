from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data/cmems_cache")
    ap.add_argument("--bbox", nargs=4, type=float, metavar=("MINLON","MAXLON","MINLAT","MAXLAT"),
                    default=[-180,180,60,90])
    ap.add_argument("--days", type=int, default=2)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    minlon,maxlon,minlat,maxlat = args.bbox

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    # Products (examples)
    seaice_product = "SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024"   # daily 12:00 UTC 
    wave_product   = "ARCTIC_ANALYSIS_FORECAST_WAV_002_014"   # WAM 3km, hourly, daily updates 

    try:
        import copernicusmarine
    except Exception as e:
        raise SystemExit(f"copernicusmarine not installed: {e}")

    # NOTE: dataset_id/variables differ per product; use `copernicusmarine.describe()` first to finalize.
    # Keep this script as a template + TODO markers, then harden after a first successful describe/subset.

    print("TIP: Run `copernicusmarine describe --contains SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 --include-datasets`")
    print("TIP: Run `copernicusmarine describe --contains ARCTIC_ANALYSIS_FORECAST_WAV_002_014 --include-datasets`")

    print("Done (template). Fill dataset_id + variables after describe.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())


