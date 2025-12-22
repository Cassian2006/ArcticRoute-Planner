"""
AIS Data Discovery Tests
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from arcticroute.io.data_discovery import discover_ais_density_nc, clear_discovery_caches


@pytest.fixture(autouse=True)
def clear_caches_before_each_test():
    """在每个测试前清理缓存"""
    clear_discovery_caches()


def test_discover_ais_basic(tmp_path: Path):
    """测试基本的 AIS 文件发现"""
    # 创建模拟的 AIS 目录和文件
    ais_dir = tmp_path / "data" / "ais_density"
    ais_dir.mkdir(parents=True)
    
    (ais_dir / "ais_density_2024.nc").touch()
    (ais_dir / "ais_traffic_2024.nc").touch()
    
    # 运行发现
    candidates, best = discover_ais_density_nc(search_dirs=[str(ais_dir)])
    
    # 断言
    assert len(candidates) == 2
    assert best is not None
    assert "ais" in best.path.lower()


def test_discover_ais_recursive(tmp_path: Path):
    """测试递归搜索 AIS 文件"""
    # 创建嵌套目录结构
    ais_dir = tmp_path / "data" / "ais"
    sub_dir = ais_dir / "derived" / "2024"
    sub_dir.mkdir(parents=True)
    
    (sub_dir / "ais_density.nc").touch()
    
    # 运行发现
    candidates, best = discover_ais_density_nc(search_dirs=[str(ais_dir)])
    
    # 断言
    assert len(candidates) == 1
    assert "ais_density.nc" in candidates[0].path


def test_discover_ais_mtime_sorting(tmp_path: Path):
    """测试按修改时间排序"""
    # 创建多个文件，不同的修改时间
    ais_dir = tmp_path / "data" / "ais"
    ais_dir.mkdir(parents=True)
    
    old_file = ais_dir / "ais_old.nc"
    old_file.touch()
    time.sleep(0.1)  # 确保时间戳不同
    
    new_file = ais_dir / "ais_new.nc"
    new_file.touch()
    
    # 运行发现
    candidates, best = discover_ais_density_nc(search_dirs=[str(ais_dir)])
    
    # 断言
    assert len(candidates) == 2
    assert best is not None
    # 最新的文件应该是 best
    assert "ais_new.nc" in best.path
    # 候选列表应该按时间排序
    assert "ais_new.nc" in candidates[0].path
    assert "ais_old.nc" in candidates[1].path


def test_discover_ais_keyword_matching(tmp_path: Path):
    """测试关键词匹配"""
    # 创建包含和不包含关键词的文件
    ais_dir = tmp_path / "data"
    ais_dir.mkdir(parents=True)
    
    (ais_dir / "ais_density.nc").touch()
    (ais_dir / "traffic_data.nc").touch()
    (ais_dir / "corridor_map.nc").touch()
    (ais_dir / "random_data.nc").touch()  # 不包含关键词
    
    # 运行发现
    candidates, best = discover_ais_density_nc(search_dirs=[str(ais_dir)])
    
    # 断言：应该找到所有 .nc 文件（包括不含关键词的）
    assert len(candidates) == 4


def test_discover_ais_empty_dir(tmp_path: Path):
    """测试空目录"""
    # 创建空目录
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    
    # 运行发现
    candidates, best = discover_ais_density_nc(search_dirs=[str(empty_dir)])
    
    # 断言
    assert len(candidates) == 0
    assert best is None


def test_discover_ais_additional_dirs(tmp_path: Path):
    """测试额外的搜索目录"""
    # 创建两个目录
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    (dir1 / "ais1.nc").touch()
    
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    (dir2 / "ais2.nc").touch()
    
    # 运行发现
    candidates, best = discover_ais_density_nc(
        search_dirs=[str(dir1)],
        additional_dirs=[str(dir2)],
    )
    
    # 断言：应该找到两个目录中的文件
    assert len(candidates) == 2


def test_discover_ais_nonexistent_dir(tmp_path: Path):
    """测试不存在的目录"""
    # 运行发现
    candidates, best = discover_ais_density_nc(
        search_dirs=[str(tmp_path / "nonexistent")]
    )
    
    # 断言：不应该抛出异常，返回空列表
    assert len(candidates) == 0
    assert best is None

