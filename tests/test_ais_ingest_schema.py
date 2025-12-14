"""
Step 1 测试：AIS schema 探测与快速 QA
"""

import pandas as pd
import pytest
from datetime import datetime
from pathlib import Path

from arcticroute.core.ais_ingest import inspect_ais_csv, load_ais_from_raw_dir


def get_test_ais_csv_path() -> str:
    """获得测试 AIS CSV 路径。"""
    return str(Path(__file__).parent / "data" / "ais_sample.csv")


def get_test_ais_json_path() -> str:
    return str(Path(__file__).parent / "data" / "ais_sample.json")


def test_inspect_ais_csv_basic():
    """测试 inspect_ais_csv 能否正确读取基本信息。"""
    csv_path = get_test_ais_csv_path()
    summary = inspect_ais_csv(csv_path, sample_n=100)

    assert summary.path == csv_path
    assert summary.num_rows == 9  # 测试文件有 9 行数据
    assert "mmsi" in summary.columns
    assert "lat" in summary.columns
    assert "lon" in summary.columns
    assert "timestamp" in summary.columns


def test_inspect_ais_csv_has_required_columns():
    """测试必需列的检查。"""
    csv_path = get_test_ais_csv_path()
    summary = inspect_ais_csv(csv_path)

    assert summary.has_mmsi is True
    assert summary.has_lat is True
    assert summary.has_lon is True
    assert summary.has_timestamp is True


def test_inspect_ais_csv_ranges():
    """测试范围信息的提取。"""
    csv_path = get_test_ais_csv_path()
    summary = inspect_ais_csv(csv_path)

    assert summary.lat_min is not None
    assert summary.lat_max is not None
    assert summary.lat_min >= 75.0  # 测试数据在 75-76N
    assert summary.lat_max <= 76.5

    assert summary.lon_min is not None
    assert summary.lon_max is not None
    assert summary.lon_min >= 20.0  # 测试数据在 20-22E
    assert summary.lon_max <= 22.0

    assert summary.time_min is not None
    assert summary.time_max is not None


def test_inspect_ais_csv_nonexistent_file():
    """测试处理不存在的文件。"""
    summary = inspect_ais_csv("/nonexistent/path/ais.csv")

    assert summary.num_rows == 0
    assert summary.has_mmsi is False
    assert summary.has_lat is False
    assert summary.has_lon is False
    assert summary.has_timestamp is False


def test_inspect_ais_csv_sample_limit():
    """测试 sample_n 参数的效果。"""
    csv_path = get_test_ais_csv_path()

    summary_all = inspect_ais_csv(csv_path, sample_n=1000)
    summary_5 = inspect_ais_csv(csv_path, sample_n=5)

    assert summary_all.num_rows == 9
    assert summary_5.num_rows == 5


def test_load_ais_from_raw_dir_multi_file(tmp_path):
    src = Path(__file__).parent / "data" / "ais_sample.csv"
    dst1 = tmp_path / "part1.csv"
    dst2 = tmp_path / "part2.csv"
    dst1.write_text(src.read_text(), encoding="utf-8")
    dst2.write_text(src.read_text(), encoding="utf-8")

    df = load_ais_from_raw_dir(tmp_path, prefer_json=False)

    assert len(df) == 18
    assert set(df.columns) == {"mmsi", "timestamp", "lat", "lon", "sog", "cog", "nav_status"}
    assert df["lat"].between(60.0, 90.0).all()


def test_load_ais_from_raw_dir_time_filter(tmp_path):
    src = Path(__file__).parent / "data" / "ais_sample.csv"
    (tmp_path / "part1.csv").write_text(src.read_text(), encoding="utf-8")
    (tmp_path / "part2.csv").write_text(src.read_text(), encoding="utf-8")

    t_min = datetime(2024, 1, 15, 10, 50)
    t_max = datetime(2024, 1, 15, 11, 0)
    df = load_ais_from_raw_dir(tmp_path, time_min=t_min, time_max=t_max, prefer_json=False)

    assert len(df) == 6  # 3 rows per file within the window
    assert df["timestamp"].min() >= pd.to_datetime(t_min, utc=True)
    assert df["timestamp"].max() <= pd.to_datetime(t_max, utc=True)


def test_load_ais_from_json(tmp_path):
    src_json = Path(get_test_ais_json_path())
    dst_json = tmp_path / "ais_sample.json"
    dst_json.write_text(src_json.read_text(), encoding="utf-8")

    # 混入一个 csv，验证 prefer_json 时只吃 JSON
    src_csv = Path(get_test_ais_csv_path())
    (tmp_path / "ais_sample.csv").write_text(src_csv.read_text(), encoding="utf-8")

    df = load_ais_from_raw_dir(tmp_path, prefer_json=True)

    assert len(df) == 7  # 8 条里一条越界被丢弃
    assert {"mmsi", "lat", "lon", "timestamp"}.issubset(set(df.columns))
    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()
    assert df["timestamp"].min() >= pd.to_datetime("2024-01-01", utc=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])





