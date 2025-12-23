from __future__ import annotations

from pathlib import Path

from arcticroute.ui import data_discovery


def test_discovery_filters_sit_drift_correctly(tmp_path, monkeypatch):
    data_dir = tmp_path / "inputs"
    data_dir.mkdir()

    # 噪声文件（不应被 SIT/Drift 命中）
    (data_dir / "ais_density_static.nc").write_text("")
    (data_dir / "random.nc").write_text("")

    # 目标文件
    sit_path = data_dir / "sit_202401.nc"
    drift_path = data_dir / "ice_drift_202401.nc"
    sit_path.write_text("")
    drift_path.write_text("")

    # 通过额外路径暴露给扫描器
    monkeypatch.setenv("ARCTICROUTE_EXTRA_DATA_PATHS", str(tmp_path))

    snapshot = data_discovery.scan_all()
    summary = data_discovery.summarize_discovery(snapshot)

    assert summary["sit"]["found"]
    assert summary["drift"]["found"]
    assert summary["sit"]["selected_path"].endswith("sit_202401.nc")
    assert summary["drift"]["selected_path"].endswith("ice_drift_202401.nc")

    # 确认噪声文件未被误选
    assert all("ais_density_static" not in p for p in summary["sit"]["found_paths"])

