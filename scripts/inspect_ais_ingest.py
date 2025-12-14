#!/usr/bin/env python3
"""
快速检查 AIS 摄取是否命中真实 JSON，而不是 sample CSV。

运行：python -m scripts.inspect_ais_ingest
"""

from __future__ import annotations

from pathlib import Path

from arcticroute.core.ais_ingest import load_ais_from_raw_dir

MAX_RECORDS_PER_FILE = 50_000  # 防止一次性吃爆内存，仍然远超 5 万


def print_json_head(root: Path, max_lines: int = 5) -> None:
    json_files = sorted(root.glob("*.json"))
    print(f"[DEBUG] json files under {root} = {len(json_files)}")
    for p in json_files:
        print(f"- {p.name}")
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f):
                    print("    " + line.strip())
                    if i + 1 >= max_lines:
                        break
        except Exception as e:
            print(f"    [ERROR] failed to preview: {e}")


def main():
    root = Path("data_real/ais/raw")
    print(f"[AIS] inspecting dir: {root.resolve()}")
    if not root.exists():
        print("[AIS] raw dir not found.")
        return

    print_json_head(root)

    df = load_ais_from_raw_dir(
        root,
        prefer_json=True,
        max_records_per_file=MAX_RECORDS_PER_FILE,
    )
    print(f"[AIS] rows={len(df)}")
    if df.empty:
        return
    print("[AIS] head:")
    print(df.head())
    print("[AIS] lat range:", df["lat"].min(), df["lat"].max())
    print("[AIS] lon range:", df["lon"].min(), df["lon"].max())
    print("[AIS] time range:", df["timestamp"].min(), df["timestamp"].max())
    print("[AIS] unique MMSI:", df["mmsi"].nunique())


if __name__ == "__main__":
    main()
