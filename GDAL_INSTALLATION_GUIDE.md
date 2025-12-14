# GDAL 安装指南

## 当前状态
✓ 已成功安装：
- fiona 1.10.1
- rasterio 1.4.4
- geopandas 1.1.1
- pyproj 3.7.2
- shapely 2.1.2

❌ 失败原因：
GDAL 需要 Microsoft Visual C++ 14.0 或更高版本来编译，但你的系统没有安装。

## 解决方案

### 方案 1：安装 Microsoft C++ Build Tools（推荐）

1. 访问：https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. 下载 "Visual Studio Build Tools"
3. 运行安装程序，选择 "Desktop development with C++"
4. 安装完成后，重新运行：
   ```bash
   pip install gdal
   ```

### 方案 2：使用 Conda（更简单）

如果你不想安装 Visual C++ Build Tools，可以使用 Conda：

1. 下载 Miniconda：https://docs.conda.io/projects/miniconda/en/latest/
2. 安装 Miniconda
3. 创建新环境：
   ```bash
   conda create -n ar-final python=3.11
   conda activate ar-final
   ```
4. 用 conda-forge 安装 GDAL：
   ```bash
   conda install -c conda-forge gdal fiona rasterio geopandas
   ```

### 方案 3：使用 GDAL 的替代品（最简单）

如果你的代码只需要读写地理数据，可以用已安装的库替代：
- **Fiona** 替代 GDAL 的向量数据功能
- **Rasterio** 替代 GDAL 的栅格数据功能
- **GeoPandas** 用于地理数据分析

这些库已经安装，可以直接使用：

```python
# 读取 Shapefile
import geopandas as gpd
gdf = gpd.read_file('data.shp')

# 读取栅格数据
import rasterio
with rasterio.open('data.tif') as src:
    data = src.read()

# 读取 GeoJSON
gdf = gpd.read_file('data.geojson')
```

## 推荐步骤

1. **立即尝试**：用 Fiona/Rasterio/GeoPandas 完成你的任务
2. **如果需要 GDAL**：安装 Visual C++ Build Tools（方案 1）
3. **最终方案**：如果方案 1 失败，使用 Conda（方案 2）

## 验证安装

```bash
# 验证已安装的库
python -c "import fiona, rasterio, geopandas; print('All installed!')"

# 如果安装了 GDAL
python -c "from osgeo import gdal; print('GDAL installed!')"
```

