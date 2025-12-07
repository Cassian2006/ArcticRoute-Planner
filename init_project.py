#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
初始化《ArcticRoute》项目骨架，生成目录与关键文件：
- .gitignore
- requirements.txt
- data_download/download_era5.py
- docs/ 与 data_download/ 的 README
- notebooks/check_era5_data.ipynb (空壳提示)
运行：python init_project.py
"""
from pathlib import Path
import json

ROOT = Path.cwd() / "ArcticRoute"
dirs = [
    "data_download",
    "data_raw/era5",
    "data_raw/ais",
    "data_processed",
    "core",
    "notebooks",
    "scripts",
    "docs",
]
gitignore = """__pycache__/
*.pyc
*.pyo
*.pyd
*.DS_Store
.env
venv/
.mypy_cache/
.ipynb_checkpoints/
data_raw/
data_processed/
*.nc
*.parquet
"""
requirements = """cdsapi
xarray
netCDF4
pandas
matplotlib
"""

download_era5_py = r"""import cdsapi
from pathlib import Path

# 说明：
# 1) 先在 ~/.cdsapirc 写入你的 CDS API key
# 2) 可修改 year/month/day/time/area
# 3) 输出到 data_raw/era5/

def main():
    out_dir = Path(__file__).resolve().parents[1] / "data_raw" / "era5"
    out_dir.mkdir(parents=True, exist_ok=True)

    c = cdsapi.Client()
    target = out_dir / "era5_env_2023_q1.nc"
    print(f"Downloading to {target} ...")
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "significant_height_of_combined_wind_waves_and_swell",
                "sea_ice_cover",
            ],
            "year": ["2023"],
            "month": ["01", "02", "03"],
            "day": ["01", "05", "10", "15", "20", "25"],
            "time": ["00:00", "06:00", "12:00", "18:00"],
            "area": [85, -180, 60, 180],  # N, W, S, E
            "format": "netcdf",
        },
        str(target),
    )
    print("Done.")

if __name__ == "__main__":
    main()
"""

docs_readme = """# ArcticRoute / 北极航线（阶段一骨架）
本仓库用于搭建最小可行系统（MVP）：从 Copernicus 下载 ERA5 风/浪/冰数据 → 验证 → 后续与 AIS 对齐。
"""

dl_readme = """# data_download 使用说明
- 来源: Copernicus Climate Data Store (CDS)
- 产品: reanalysis-era5-single-levels
- 变量: 10m_u, 10m_v, significant_wave_height, sea_ice_cover
- 时间: 2023年1–3月（样例，可扩展为全年）
- 区域: 北纬60°–85°, 全经度
- 输出: data_raw/era5/era5_env_2023_q1.nc
"""

data_sources_md = """# 数据来源与下载说明
- Copernicus CDS: ERA5 单层再分析（风、浪、海冰）
- 下载脚本：`data_download/download_era5.py`
- API Key: 放置于 `~/.cdsapirc`
"""

# 轻量空壳 notebook（仅提示用）
nb = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# check_era5_data.ipynb\n",
                "运行：\n",
                "```python\n",
                "import xarray as xr\n",
                "ds = xr.open_dataset('../data_raw/era5/era5_env_2023_q1.nc')\n",
                "ds\n",
                "```\n",
            ],
        }
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}


def write(path: Path, content: str, binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    with open(path, mode, encoding=None if binary else "utf-8") as f:
        if binary:
            f.write(content)
        else:
            f.write(content)


def main():
    print(f"Creating project at: {ROOT}")
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)

    write(ROOT / ".gitignore", gitignore)
    write(ROOT / "requirements.txt", requirements)
    write(ROOT / "data_download" / "download_era5.py", download_era5_py)
    write(ROOT / "docs" / "README.md", docs_readme)
    write(ROOT / "data_download" / "README.md", dl_readme)
    write(ROOT / "docs" / "data_sources.md", data_sources_md)
    write(ROOT / "core" / "__init__.py", "")
    write(ROOT / "notebooks" / "check_era5_data.ipynb", json.dumps(nb), binary=False)

    print("\n✅ 初始化完成。下一步：")
    print("1) cd ArcticRoute")
    print("2) python -m venv venv  &&  (Linux/Mac) source venv/bin/activate  |  (Win) venv\\Scripts\\activate")
    print("3) pip install -r requirements.txt")
    print("4) python data_download/download_era5.py  # 开始下载 ERA5")
    print("\n完成后验证：")
    print("python - <<'EOF'")
    print("import xarray as xr")
    print("ds = xr.open_dataset('data_raw/era5/era5_env_2023_q1.nc')")
    print("print(ds)")
    print("EOF")


if __name__ == "__main__":
    main()
