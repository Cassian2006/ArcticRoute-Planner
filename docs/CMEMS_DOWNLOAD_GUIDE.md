# CMEMS 近实时数据下载完整指南

本指南介绍如何使用本项目的 CMEMS 数据下载闭环，自动获取北极海冰和波浪的近实时数据。

## 概述

该闭环包含三个主要步骤：

1. **元数据查询** (`cmems_resolve.py`)：自动从 Copernicus Marine 目录中查询并提取 dataset-id 和变量名
2. **配置生成** (`cmems_resolved.json`)：保存已解析的数据集配置
3. **数据下载** (`cmems_download.py`)：使用 `copernicusmarine subset` 命令下载实际数据

## 前置条件

### 1. 安装 Copernicus Marine Toolbox

```bash
pip install copernicusmarine
```

### 2. 配置认证（可选但推荐）

```bash
copernicusmarine login
```

## 快速开始

### 第一步：获取元数据

执行以下命令获取海冰和波浪数据的元数据：

```powershell
cd C:\Users\sgddsf\Desktop\AR_final

# 获取海冰数据元数据
copernicusmarine describe --contains "SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024" --return-fields all | Out-File -Encoding UTF8 reports/cmems_sic_describe.json

# 获取波浪数据元数据
copernicusmarine describe --contains "ARCTIC_ANALYSIS_FORECAST_WAV_002_014" --return-fields all | Out-File -Encoding UTF8 reports/cmems_wav_describe.json
```

### 第二步：解析配置

运行解析脚本自动提取 dataset-id 和变量名：

```bash
python scripts/cmems_resolve.py
```

**输出示例：**

```json
{
  "sic": {
    "dataset_id": "cmems_obs-si_arc_phy_my_l4_P1D",
    "variables": ["sic", "uncertainty_sic"]
  },
  "wav": {
    "dataset_id": "dataset-wam-arctic-1hr3km-be",
    "variables": [
      "sea_surface_wave_significant_height",
      ...
    ]
  }
}
```

### 第三步：下载数据

执行下载脚本：

```bash
python scripts/cmems_download.py
```

或使用 PowerShell 脚本（支持循环模式）：

```powershell
# 仅执行一次
.\scripts\cmems_download.ps1

# 循环模式：每 60 分钟执行一次
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 60
```

## 数据产品说明

### 海冰浓度 (SIC)

- **产品 ID**: `SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024`
- **Dataset ID**: `cmems_obs-si_arc_phy_my_l4_P1D`
- **更新频率**: 每日 12:00 UTC
- **格式**: NetCDF-4
- **主要变量**: `sic`（海冰浓度，0-100%）
- **空间分辨率**: 25 km
- **覆盖范围**: 北极

### 北极波浪预报 (WAV)

- **产品 ID**: `ARCTIC_ANALYSIS_FORECAST_WAV_002_014`
- **Dataset ID**: `dataset-wam-arctic-1hr3km-be`
- **更新频率**: 每日两次（WAM 模型运行）
- **格式**: NetCDF
- **主要变量**: `sea_surface_wave_significant_height`（有效波高）
- **时间分辨率**: 小时级
- **空间分辨率**: 3 km
- **覆盖范围**: 北极

## 配置参数

### 时间范围

默认下载近 2 天的数据。可在 `cmems_download.py` 中修改：

```python
end_date = datetime.utcnow().date()
start_date = end_date - timedelta(days=2)  # 修改这里
```

### 地理范围 (Bounding Box)

默认北极范围：

```python
bbox = {
    "min_lon": -40,   # 西边界
    "max_lon": 60,    # 东边界
    "min_lat": 65,    # 南边界
    "max_lat": 85,    # 北边界
}
```

### 输出目录

数据默认保存到 `data/cmems_cache/`：

- `sic_latest.nc` - 海冰浓度数据
- `swh_latest.nc` - 波浪数据

## 常见问题

### Q: 如何验证下载的数据？

```bash
# 使用 ncdump 查看 NetCDF 文件结构
ncdump -h data/cmems_cache/sic_latest.nc

# 或使用 Python
import xarray as xr
ds = xr.open_dataset("data/cmems_cache/sic_latest.nc")
print(ds)
```

### Q: 如何处理下载失败？

检查以下几点：

1. **网络连接**: 确保能访问 Copernicus Marine 服务
2. **认证**: 运行 `copernicusmarine login` 进行认证
3. **时间范围**: 确保请求的数据已发布（海冰数据延迟 1-2 天）
4. **变量名**: 检查 `cmems_resolved.json` 中的变量名是否正确

### Q: 如何自动化定期下载？

#### 方案 1: Windows 任务计划程序

```powershell
# 创建任务计划程序条目
$TaskName = "CMEMS_Download"
$TaskPath = "C:\Users\sgddsf\Desktop\AR_final\scripts\cmems_download.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 13:00  # 每天 13:00 UTC 执行
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File $TaskPath"
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Force
```

#### 方案 2: 使用 PowerShell 循环模式

```powershell
# 每 60 分钟执行一次
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 60
```

#### 方案 3: 使用 cron (Linux/macOS)

```bash
# 每天 13:00 UTC 执行
0 13 * * * cd /path/to/AR_final && python scripts/cmems_download.py
```

## 脚本详解

### cmems_resolve.py

**功能**: 从 describe JSON 中自动提取 dataset-id 和变量名

**关键函数**:
- `pick()`: 启发式搜索 JSON 结构，提取 dataset-id 和变量
- `main()`: 加载两个 describe JSON，调用 `pick()` 处理，输出结果

**输出**: `reports/cmems_resolved.json`

### cmems_download.py

**功能**: 使用 copernicusmarine subset 下载数据

**关键函数**:
- `load_resolved_config()`: 加载 `cmems_resolved.json`
- `run_subset()`: 构造并执行 `copernicusmarine subset` 命令
- `main()`: 协调下载流程

**输出**: `data/cmems_cache/sic_latest.nc`, `data/cmems_cache/swh_latest.nc`

### cmems_download.ps1

**功能**: PowerShell 包装脚本，支持循环模式

**参数**:
- `-IntervalMinutes`: 循环间隔（分钟）
- `-Loop`: 启用循环模式

## 数据使用示例

### 使用 xarray 读取数据

```python
import xarray as xr
import numpy as np

# 读取海冰数据
sic_ds = xr.open_dataset("data/cmems_cache/sic_latest.nc")
sic = sic_ds["sic"]  # 海冰浓度

# 读取波浪数据
wav_ds = xr.open_dataset("data/cmems_cache/swh_latest.nc")
swh = wav_ds["sea_surface_wave_significant_height"]  # 有效波高

# 基本统计
print(f"SIC 范围: {sic.min().values:.2f} - {sic.max().values:.2f}")
print(f"SWH 范围: {swh.min().values:.2f} - {swh.max().values:.2f}")
```

### 使用 NetCDF4 读取数据

```python
from netCDF4 import Dataset

# 读取海冰数据
sic_file = Dataset("data/cmems_cache/sic_latest.nc")
sic = sic_file.variables["sic"][:]
sic_file.close()

print(f"SIC 形状: {sic.shape}")
print(f"SIC 数据类型: {sic.dtype}")
```

## 故障排除

### 错误: "Unexpected UTF-8 BOM"

**原因**: PowerShell 的 `Out-File` 添加了 BOM

**解决**: 脚本已配置使用 `utf-8-sig` 编码处理

### 错误: "No such option: --include-datasets"

**原因**: copernicusmarine 版本不同，选项名称可能不同

**解决**: 运行 `copernicusmarine describe -h` 查看正确的选项

### 错误: "Dataset not found"

**原因**: 
- dataset-id 不正确
- 数据尚未发布（海冰数据通常延迟 1-2 天）

**解决**: 
- 检查 `cmems_resolved.json` 中的 dataset-id
- 尝试扩大时间范围

## 参考资源

- [Copernicus Marine Toolbox 文档](https://help.marine.copernicus.eu/)
- [SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 产品页面](https://data.marine.copernicus.eu/products/SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024)
- [ARCTIC_ANALYSIS_FORECAST_WAV_002_014 产品页面](https://data.marine.copernicus.eu/products/ARCTIC_ANALYSIS_FORECAST_WAV_002_014)

## 许可证

本脚本遵循项目许可证。数据使用需遵守 Copernicus Marine 的许可条款。

