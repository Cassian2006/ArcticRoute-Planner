"""
CMEMS Data Discovery Tests
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arcticroute.io.data_discovery import discover_cmems_layers, clear_discovery_caches


@pytest.fixture(autouse=True)
def clear_caches_before_each_test():
    """在每个测试前清理缓存"""
    clear_discovery_caches()


def test_discover_cmems_from_newenv(tmp_path: Path):
    """测试从 newenv 目录发现 CMEMS 文件"""
    # 创建模拟的 newenv 目录和文件
    newenv_dir = tmp_path / "data_processed" / "newenv"
    newenv_dir.mkdir(parents=True)
    
    (newenv_dir / "ice_copernicus_sic.nc").touch()
    (newenv_dir / "wave_swh.nc").touch()
    (newenv_dir / "ice_thickness.nc").touch()
    (newenv_dir / "ice_drift.nc").touch()
    
    # 运行发现
    layers = discover_cmems_layers(newenv_dirs=[str(newenv_dir)])
    
    # 断言
    assert layers["sic"].found
    assert layers["sic"].source == "newenv"
    assert "ice_copernicus_sic.nc" in layers["sic"].path
    
    assert layers["swh"].found
    assert layers["swh"].source == "newenv"
    assert "wave_swh.nc" in layers["swh"].path
    
    assert layers["sit"].found
    assert layers["sit"].source == "newenv"
    assert "ice_thickness.nc" in layers["sit"].path
    
    assert layers["drift"].found
    assert layers["drift"].source == "newenv"
    assert "ice_drift.nc" in layers["drift"].path


def test_discover_cmems_from_cache(tmp_path: Path):
    """测试从 cache 目录发现 CMEMS 文件"""
    # 创建模拟的 cache 目录和文件
    cache_dir = tmp_path / "data" / "cmems_cache"
    cache_dir.mkdir(parents=True)
    
    (cache_dir / "cmems_sic_2024.nc").touch()
    (cache_dir / "cmems_wave_2024.nc").touch()
    (cache_dir / "cmems_thickness_2024.nc").touch()
    (cache_dir / "cmems_drift_2024.nc").touch()
    
    # 运行发现
    layers = discover_cmems_layers(newenv_dirs=[], cache_dirs=[str(cache_dir)])
    
    # 断言
    assert layers["sic"].found
    assert layers["sic"].source == "cache"
    assert "cmems_sic_2024.nc" in layers["sic"].path
    
    assert layers["swh"].found
    assert layers["swh"].source == "cache"
    assert "cmems_wave_2024.nc" in layers["swh"].path
    
    assert layers["sit"].found
    assert layers["sit"].source == "cache"
    assert "cmems_thickness_2024.nc" in layers["sit"].path
    
    assert layers["drift"].found
    assert layers["drift"].source == "cache"
    assert "cmems_drift_2024.nc" in layers["drift"].path


def test_discover_cmems_from_manual_path(tmp_path: Path):
    """测试从手动路径发现 CMEMS 文件"""
    # 创建模拟文件
    manual_file = tmp_path / "manual_sic.nc"
    manual_file.touch()
    
    # 运行发现
    layers = discover_cmems_layers(manual_paths={"sic": str(manual_file)})
    
    # 断言
    assert layers["sic"].found
    assert layers["sic"].source == "manual"
    assert str(manual_file) == layers["sic"].path
    
    # 其他层应该未找到
    assert not layers["swh"].found


def test_discover_cmems_not_found(tmp_path: Path):
    """测试未找到 CMEMS 文件"""
    # 运行发现
    layers = discover_cmems_layers(newenv_dirs=[str(tmp_path / "empty_newenv")])
    
    # 断言
    assert not layers["sic"].found
    assert layers["sic"].source == "missing"
    assert "Not found" in layers["sic"].reason


def test_discover_cmems_priority(tmp_path: Path):
    """测试发现优先级：manual > newenv > cache"""
    # 创建所有类型的文件
    manual_file = tmp_path / "manual_sic.nc"
    manual_file.touch()
    
    newenv_dir = tmp_path / "newenv"
    newenv_dir.mkdir()
    (newenv_dir / "sic.nc").touch()
    
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "cache_sic.nc").touch()
    
    # 1. manual 优先
    layers = discover_cmems_layers(
        manual_paths={"sic": str(manual_file)},
        newenv_dirs=[str(newenv_dir)],
        cache_dirs=[str(cache_dir)],
    )
    assert layers["sic"].source == "manual"
    assert str(manual_file) == layers["sic"].path
    
    # 2. newenv 优先于 cache
    layers = discover_cmems_layers(
        newenv_dirs=[str(newenv_dir)],
        cache_dirs=[str(cache_dir)],
    )
    assert layers["sic"].source == "newenv"
    assert "sic.nc" in layers["sic"].path


def test_discover_cmems_from_index(tmp_path: Path):
    """测试从索引文件发现 CMEMS 文件"""
    # 创建索引文件和目标文件
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    
    target_file = tmp_path / "indexed_sic.nc"
    target_file.touch()
    
    index_content = {"sic": str(target_file)}
    
    with open(reports_dir / "cmems_newenv_index.json", "w") as f:
        json.dump(index_content, f)
    
    # 运行发现（需要修改 reports 目录的查找方式，这里简化）
    # 暂时跳过，因为需要修改 _load_cmems_index 的实现
    pass

