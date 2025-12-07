import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

pytestmark = pytest.mark.xfail(reason="Enable in P1/P2", strict=False)


def test_plan_cli_creates_linestring(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    tag = "pytest"

    cmd = [
        sys.executable,
        "-m",
        "api.cli",
        "plan",
        "--cfg",
        "config/runtime.yaml",
        "--tag",
        tag,
        "--beta",
        "3",
        "--gamma",
        "0",
        "--time-step-nodes",
        "0",
        "--bbox",
        "75,9,72,11",
        "--coarsen",
        "3",
        "--start",
        "75,10",
        "--goal",
        "75,10",
        "--output-dir",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"CLI 鎵ц澶辫触:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    geojson_path = tmp_path / f"route_{tag}.geojson"
    assert geojson_path.exists(), "鏈敓鎴?GeoJSON"
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    assert data.get("type") == "FeatureCollection"
    features = data.get("features", [])
    assert features, "缂哄皯 Feature"
    geometry = features[0].get("geometry", {})
    assert geometry.get("type") == "LineString"
    props = features[0].get("properties", {})
    assert isinstance(props.get("time_switch_nodes"), list)
    assert "turning_count" in props
    assert "max_gradient" in props
    assert "fuel_proxy" in props


def test_plan_cli_with_accident_field(tmp_path):
    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / "data_processed" / "env_clean.nc"

    env_ds = xr.open_dataset(env_path)
    try:
        lat_vals = env_ds["latitude"].values
        lon_vals = env_ds["longitude"].values
        time_vals = env_ds["time"].values[:1]
        accident_data = np.ones((1, len(lat_vals), len(lon_vals)), dtype="float32")
        accident_da = xr.DataArray(
            accident_data,
            dims=("time", "latitude", "longitude"),
            coords={
                "time": time_vals,
                "latitude": lat_vals,
                "longitude": lon_vals,
            },
            name="acc_density_time",
            attrs={"source_incidents": "tests/data/incidents_mock.parquet"},
        )
        accident_ds = xr.Dataset({"acc_density_time": accident_da}, attrs={"source_incidents": "tests/data/incidents_mock.parquet"})
    finally:
        env_ds.close()

    accident_path = tmp_path / "accident_density.nc"
    accident_ds.to_netcdf(accident_path)

    tag = "accident"
    cmd = [
        sys.executable,
        "-m",
        "api.cli",
        "plan",
        "--cfg",
        "config/runtime.yaml",
        "--tag",
        tag,
        "--beta",
        "3",
        "--gamma",
        "0",
        "--beta-a",
        "0.5",
        "--time-step-nodes",
        "0",
        "--accident-density",
        str(accident_path),
        "--acc-mode",
        "time",
        "--bbox",
        "75,9,72,11",
        "--coarsen",
        "3",
        "--start",
        "75,10",
        "--goal",
        "75,10",
        "--output-dir",
        str(tmp_path),
    ]

    result = subprocess.run(cmd, cwd=project_root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    assert result.returncode == 0, f"CLI 运行失败:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    geojson_path = tmp_path / f"route_{tag}.geojson"
    assert geojson_path.exists(), "未生成 GeoJSON 输出"
    data = json.loads(geojson_path.read_text(encoding="utf-8"))
    props = data["features"][0]["properties"]
    assert props.get("beta_a") == 0.5
    assert props.get("acc_density_stats") is not None, "缺少事故密度统计"
    assert props.get("acc_mode") == "time"
    assert isinstance(props.get("time_switch_nodes"), list)
    assert "turning_count" in props
    assert "max_gradient" in props
    assert "fuel_proxy" in props
