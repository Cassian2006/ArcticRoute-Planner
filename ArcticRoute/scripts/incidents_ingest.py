#!/usr/bin/env python3
"""Ingest raw incident data and normalize schema for processing.

@role: pipeline
"""

"""
ACC-1 incidents integration and standardisation pipeline.

- Recursively read data_raw/incidents/*.csv
- Standardise columns: time_utc, lat, lon, type, severity, source
- Drop invalid records (missing/duplicate/geometry/time issues)
- Write data_processed/incidents_clean.parquet
- Emit a stats summary to stdout
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data_raw" / "incidents"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data_processed" / "incidents_clean.parquet"

TARGET_COLUMNS = ["time_utc", "lat", "lon", "type", "severity", "source"]

TIME_CANDIDATES = (
    "time",
    "timestamp",
    "datetime",
    "date_time",
    "occurrence_date_time",
    "occurrence_date_and_time",
    "occurrence_time",
)
LAT_CANDIDATES = ("lat", "latitude")
LON_CANDIDATES = ("lon", "longitude")
COORD_CANDIDATES = ("coordinates", "coordinate", "position")
TYPE_CANDIDATES = ("type", "event_type", "casualty_event", "casualty_type")
SEVERITY_CANDIDATES = ("severity", "casualty_severity", "incident_severity")
SOURCE_CANDIDATES = ("source", "data_source", "reference", "report_id")


def _normalize_column(name: str) -> str:
    """Lower-case + snake-case helper for fuzzy column matching."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in name)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_")


def _parse_coordinate_component(part: str, direction: str) -> float | None:
    """Convert single latitude/longitude component into signed decimal degrees."""
    if not part:
        return None

    cleaned = (
        part.replace("º", " ")
        .replace("°", " ")
        .replace("˚", " ")
        .replace("’", " ")
        .replace("'", " ")
        .replace('"', " ")
        .replace(",", ".")
        .strip()
    )
    tokens = [tok for tok in cleaned.split() if tok]
    if not tokens:
        return None

    numbers: list[float] = []
    for token in tokens:
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if not numbers:
        return None

    if len(numbers) == 1:
        value = numbers[0]
    else:
        degrees = abs(numbers[0])
        minutes = numbers[1] if len(numbers) >= 2 else 0.0
        seconds = numbers[2] if len(numbers) >= 3 else 0.0
        value = degrees + minutes / 60.0 + seconds / 3600.0
        value = math.copysign(value, numbers[0])

    if direction.upper() in {"S", "W"}:
        value = -abs(value)
    else:
        value = abs(value)
    return value


def _parse_coordinate_pair(value: str) -> tuple[float | None, float | None]:
    """Split raw coordinate string like `12° 31.6' N 45° 12.0' E` into decimal lat/lon."""
    if not isinstance(value, str):
        return (None, None)
    text = value.strip()
    if not text or text.lower() in {"nan", "na", "n/a", "unknown", "-"}:
        return (None, None)
    text = (
        text.replace(",", " ")
        .replace(";", " ")
        .replace("/", " ")
        .replace("\n", " ")
        .replace("\t", " ")
    )
    text = " ".join(text.split())

    lat_split = None
    for marker in ("N", "S", "n", "s"):
        idx = text.find(marker)
        if idx > 0:
            lat_split = (text[:idx].strip(), marker.upper(), text[idx + 1 :].strip())
            break
    if not lat_split:
        return (None, None)

    lat_part, lat_dir, remainder = lat_split
    lon_split = None
    for marker in ("E", "W", "e", "w"):
        idx = remainder.find(marker)
        if idx > 0:
            lon_split = (remainder[:idx].strip(), marker.upper())
            break
    if not lon_split:
        return (None, None)

    lon_part, lon_dir = lon_split
    lat = _parse_coordinate_component(lat_part, lat_dir)
    lon = _parse_coordinate_component(lon_part, lon_dir)
    return (lat, lon)


def _clean_string(series: pd.Series, default: str | None = None) -> pd.Series:
    """Trim whitespace and coerce empty strings to NA."""
    result = series.astype("string")
    result = result.str.strip()
    result = result.replace({"": pd.NA})
    if default is not None:
        result = result.fillna(default)
    return result


def _pick_column(columns: dict[str, str], candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return columns[candidate]
    return None


def _load_single_file(path: Path) -> tuple[pd.DataFrame, int, int]:
    """Read and standardise one CSV file."""
    try:
        raw = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except Exception as err:  # pragma: no cover - robustness guard
        print(f"[WARN] Failed to read {path}: {err}")
        return pd.DataFrame(columns=TARGET_COLUMNS), 0, 0

    total_rows = len(raw)
    if total_rows == 0:
        return pd.DataFrame(columns=TARGET_COLUMNS), 0, 0

    lookup = {_normalize_column(col): col for col in raw.columns}

    time_col = _pick_column(lookup, TIME_CANDIDATES)
    lat_col = _pick_column(lookup, LAT_CANDIDATES)
    lon_col = _pick_column(lookup, LON_CANDIDATES)
    coord_col = _pick_column(lookup, COORD_CANDIDATES)
    type_col = _pick_column(lookup, TYPE_CANDIDATES)
    severity_col = _pick_column(lookup, SEVERITY_CANDIDATES)
    source_col = _pick_column(lookup, SOURCE_CANDIDATES)

    out = pd.DataFrame(index=raw.index)

    if time_col:
        time_series = _clean_string(raw[time_col])
        time_series = time_series.str.replace(r"(?i)\s*utc$", "", regex=True)
        time_utc = pd.to_datetime(time_series, utc=True, errors="coerce")
    else:
        time_utc = pd.Series(pd.NaT, index=raw.index)

    lat = pd.Series(float("nan"), index=raw.index, dtype="float64")
    lon = pd.Series(float("nan"), index=raw.index, dtype="float64")

    if lat_col and lon_col:
        lat = pd.to_numeric(raw[lat_col], errors="coerce")
        lon = pd.to_numeric(raw[lon_col], errors="coerce")
    elif coord_col:
        coords = raw[coord_col].apply(_parse_coordinate_pair)
        lat = pd.Series([pair[0] for pair in coords], dtype="float64", index=raw.index)
        lon = pd.Series([pair[1] for pair in coords], dtype="float64", index=raw.index)

    type_series = (
        _clean_string(raw[type_col]) if type_col else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    severity_series = (
        _clean_string(raw[severity_col])
        if severity_col
        else pd.Series(pd.NA, index=raw.index, dtype="string")
    )
    default_source = path.stem
    source_series = (
        _clean_string(raw[source_col], default=default_source)
        if source_col
        else pd.Series(default_source, index=raw.index, dtype="string")
    )

    out["time_utc"] = time_utc
    out["lat"] = lat
    out["lon"] = lon
    out["type"] = type_series
    out["severity"] = severity_series
    out["source"] = source_series

    valid_mask = (
        out["time_utc"].notna()
        & out["lat"].notna()
        & out["lon"].notna()
        & out["lat"].between(-90.0, 90.0)
        & out["lon"].between(-180.0, 180.0)
    )
    cleaned = out.loc[valid_mask, TARGET_COLUMNS]

    return cleaned, total_rows, len(cleaned)


def _ingest_directory(input_dir: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frames: list[pd.DataFrame] = []
    counters = {"rows_total": 0, "rows_valid": 0}

    csv_files = sorted(input_dir.rglob("*.csv"))
    if not csv_files:
        print(f"[WARN] No CSV files found under {input_dir}")
        return pd.DataFrame(columns=TARGET_COLUMNS), counters

    for csv_path in csv_files:
        frame, rows_total, rows_valid = _load_single_file(csv_path)
        counters["rows_total"] += rows_total
        counters["rows_valid"] += rows_valid
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=TARGET_COLUMNS), counters

    combined = pd.concat(frames, ignore_index=True)
    return combined, counters


def _print_summary(df: pd.DataFrame, counters: dict[str, int]) -> None:
    total_rows = counters.get("rows_total", 0)
    valid_rows = counters.get("rows_valid", 0)
    dedup_rows = len(df)
    dropped_missing = total_rows - valid_rows
    dropped_duplicates = valid_rows - dedup_rows

    print(f"[STAT] Total rows read: {total_rows}")
    print(f"[STAT] Rows after cleaning: {valid_rows} (dropped {dropped_missing})")
    print(f"[STAT] Rows after deduplication: {dedup_rows} (removed {dropped_duplicates})")

    if df.empty:
        print("[STAT] Clean dataset is empty; skipping range statistics.")
        return

    time_min = df["time_utc"].min()
    time_max = df["time_utc"].max()
    lat_min = df["lat"].min()
    lat_max = df["lat"].max()
    lon_min = df["lon"].min()
    lon_max = df["lon"].max()

    print(f"[STAT] UTC time range: {time_min.isoformat()} to {time_max.isoformat()}")
    print(f"[STAT] Latitude range: {lat_min:.4f} to {lat_max:.4f}")
    print(f"[STAT] Longitude range: {lon_min:.4f} to {lon_max:.4f}")

    missing_rates = {
        column: df[column].isna().mean() if column in df else float("nan")
        for column in ("type", "severity", "source")
    }
    print(
        "[STAT] Missing optional fields: "
        + ", ".join(f"{col}={rate:.1%}" for col, rate in missing_rates.items())
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ACC-1 incidents ingestion")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing raw incident CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to write the cleaned incidents parquet",
    )
    args = parser.parse_args(argv)

    input_dir = args.input_dir
    output_path = args.output

    combined, counters = _ingest_directory(input_dir)
    if combined.empty:
        _print_summary(combined, counters)
        print("[INFO] No valid incident records to write.")
        return 0

    combined = combined.drop_duplicates(subset=["time_utc", "lat", "lon"], keep="first").reset_index(drop=True)
    _print_summary(combined, counters)

    # Preserve datetime for stats, then store ISO8601 strings for output.
    time_iso = combined["time_utc"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    combined = combined.assign(time_utc=time_iso)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)
    print(f"[OK] Wrote cleaned incidents to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
