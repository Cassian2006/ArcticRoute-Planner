# CMEMS 近实时数据下载闭环工作流

## 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    CMEMS 数据下载闭环                            │
└─────────────────────────────────────────────────────────────────┘

第一步: 元数据查询 (Metadata Discovery)
├─ 输入: 产品 ID
│  ├─ SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 (海冰)
│  └─ ARCTIC_ANALYSIS_FORECAST_WAV_002_014 (波浪)
├─ 命令: copernicusmarine describe --contains <product_id>
└─ 输出: reports/cmems_sic_describe.json, reports/cmems_wav_describe.json

第二步: 配置解析 (Config Resolution)
├─ 输入: describe JSON 文件
├─ 脚本: scripts/cmems_resolve.py
│  ├─ 启发式搜索 dataset-id
│  └─ 提取相关变量名
└─ 输出: reports/cmems_resolved.json
   └─ {
       "sic": {
         "dataset_id": "cmems_obs-si_arc_phy_my_l4_P1D",
         "variables": ["sic", "uncertainty_sic"]
       },
       "wav": {
         "dataset_id": "dataset-wam-arctic-1hr3km-be",
         "variables": ["sea_surface_wave_significant_height", ...]
       }
     }

第三步: 数据下载 (Data Download)
├─ 输入: cmems_resolved.json
├─ 脚本: scripts/cmems_download.py
│  ├─ 加载配置
│  ├─ 定义时间范围 (近 2 天)
│  ├─ 定义地理范围 (北极 bbox)
│  └─ 执行 copernicusmarine subset
└─ 输出: data/cmems_cache/
   ├─ sic_latest.nc (海冰浓度)
   └─ swh_latest.nc (有效波高)

可选: 自动化循环 (Automation Loop)
├─ PowerShell: scripts/cmems_download.ps1 -Loop -IntervalMinutes 60
├─ 任务计划程序: 每日定时执行
└─ Cron: 定期运行 (Linux/macOS)
```

## 执行流程详解

### 步骤 1: 元数据查询

**目的**: 找到正确的 dataset-id 和变量名

**为什么需要?**
- Copernicus Marine 的产品 ID 和 dataset ID 不同
- 同一产品可能有多个 dataset
- 变量名需要精确匹配

**执行命令**:

```powershell
cd C:\Users\sgddsf\Desktop\AR_final

# 海冰数据
copernicusmarine describe --contains "SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024" --return-fields all | Out-File -Encoding UTF8 reports/cmems_sic_describe.json

# 波浪数据
copernicusmarine describe --contains "ARCTIC_ANALYSIS_FORECAST_WAV_002_014" --return-fields all | Out-File -Encoding UTF8 reports/cmems_wav_describe.json
```

**输出大小**: 每个 JSON 文件约 100-200 KB

**执行时间**: 约 1-2 分钟（取决于网络）

### 步骤 2: 配置解析

**目的**: 自动从 JSON 中提取关键信息

**脚本逻辑**:

```python
def pick(obj, product_id, prefer_keywords, prefer_var_keywords):
    # 1. 验证 product_id 在 JSON 中
    # 2. 递归搜索 dataset_id 字段
    # 3. 按关键词优先级选择最匹配的 dataset
    # 4. 搜索变量名字段
    # 5. 按关键词优先级选择相关变量
    # 返回: {"dataset_id": ..., "variables": [...]}
```

**执行命令**:

```bash
python scripts/cmems_resolve.py
```

**输出**:

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
      "sea_surface_wind_wave_significant_height",
      ...
    ]
  }
}
```

**执行时间**: < 1 秒

### 步骤 3: 数据下载

**目的**: 使用 subset 命令下载实际数据

**关键参数**:

| 参数 | 值 | 说明 |
|------|-----|------|
| `--dataset-id` | 从 cmems_resolved.json 读取 | 数据集 ID |
| `--variable` | 从 cmems_resolved.json 读取 | 变量名 |
| `--start-datetime` | 2025-12-13 | 开始日期 |
| `--end-datetime` | 2025-12-15 | 结束日期 |
| `--minimum-longitude` | -40 | 西边界 |
| `--maximum-longitude` | 60 | 东边界 |
| `--minimum-latitude` | 65 | 南边界 |
| `--maximum-latitude` | 85 | 北边界 |
| `--output-directory` | data/cmems_cache | 输出目录 |
| `--output-filename` | sic_latest.nc | 输出文件名 |

**执行命令**:

```bash
python scripts/cmems_download.py
```

**内部执行的 copernicusmarine 命令**:

```bash
# 海冰数据
copernicusmarine subset \
  --dataset-id cmems_obs-si_arc_phy_my_l4_P1D \
  --variable sic \
  --start-datetime 2025-12-13 \
  --end-datetime 2025-12-15 \
  --minimum-longitude -40 \
  --maximum-longitude 60 \
  --minimum-latitude 65 \
  --maximum-latitude 85 \
  --output-directory data/cmems_cache \
  --output-filename sic_latest.nc

# 波浪数据
copernicusmarine subset \
  --dataset-id dataset-wam-arctic-1hr3km-be \
  --variable sea_surface_wave_significant_height \
  --start-datetime 2025-12-13 \
  --end-datetime 2025-12-15 \
  --minimum-longitude -40 \
  --maximum-longitude 60 \
  --minimum-latitude 65 \
  --maximum-latitude 85 \
  --output-directory data/cmems_cache \
  --output-filename swh_latest.nc
```

**输出文件**:

```
data/cmems_cache/
├── sic_latest.nc      # 海冰浓度 (NetCDF-4)
└── swh_latest.nc      # 有效波高 (NetCDF)
```

**执行时间**: 5-15 分钟（取决于数据量和网络）

## 关键设计决策

### 1. 为什么分离为三个步骤?

| 步骤 | 频率 | 原因 |
|------|------|------|
| 元数据查询 | 一次性或定期 | 产品结构变化不频繁 |
| 配置解析 | 一次性或定期 | 变量名可能更新 |
| 数据下载 | 频繁 | 数据每日更新 |

### 2. 为什么使用启发式搜索?

- **灵活性**: 应对 JSON 结构变化
- **容错性**: 即使字段名不同也能找到数据
- **自动化**: 无需手动指定 dataset-id

### 3. 为什么选择这两个产品?

| 产品 | 优点 | 用途 |
|------|------|------|
| SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 | 官方产品，高质量 | 海冰浓度分析 |
| ARCTIC_ANALYSIS_FORECAST_WAV_002_014 | 高分辨率 (3km)，近实时 | 波浪预报 |

## 自动化方案

### 方案 A: Windows 任务计划程序

```powershell
# 创建任务
$TaskName = "CMEMS_Download"
$TaskPath = "C:\Users\sgddsf\Desktop\AR_final\scripts\cmems_download.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 13:00
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File $TaskPath"
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Force

# 查看任务
Get-ScheduledTask -TaskName $TaskName

# 删除任务
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
```

### 方案 B: PowerShell 循环

```powershell
# 每 60 分钟执行一次
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 60
```

### 方案 C: Cron (Linux/macOS)

```bash
# 编辑 crontab
crontab -e

# 每天 13:00 UTC 执行
0 13 * * * cd /path/to/AR_final && python scripts/cmems_download.py >> logs/cmems_download.log 2>&1
```

## 故障恢复

### 网络中断

```bash
# 脚本会自动重试（copernicusmarine 内置重试机制）
# 或手动重新运行
python scripts/cmems_download.py
```

### 数据不可用

```bash
# 检查数据发布状态
copernicusmarine describe --dataset-id cmems_obs-si_arc_phy_my_l4_P1D

# 扩大时间范围重试
# 修改 cmems_download.py 中的 timedelta(days=2) -> timedelta(days=5)
```

### 磁盘空间不足

```bash
# 清理旧数据
rm data/cmems_cache/*.nc

# 或修改输出目录到更大的磁盘
# 修改 cmems_download.py 中的 output_dir
```

## 性能优化

### 1. 减少下载数据量

```python
# 缩小地理范围
bbox = {
    "min_lon": 0,
    "max_lon": 40,
    "min_lat": 70,
    "max_lat": 85,
}

# 缩短时间范围
start_date = end_date - timedelta(days=1)
```

### 2. 并行下载

```python
# 使用 ThreadPoolExecutor 并行下载多个变量
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as executor:
    executor.submit(run_subset, sic_config, ...)
    executor.submit(run_subset, wav_config, ...)
```

### 3. 增量更新

```python
# 只下载新数据（检查文件修改时间）
import os
from datetime import datetime, timedelta

last_modified = os.path.getmtime("data/cmems_cache/sic_latest.nc")
if datetime.now() - datetime.fromtimestamp(last_modified) > timedelta(hours=24):
    # 下载新数据
```

## 监控和日志

### 添加日志记录

```python
import logging

logging.basicConfig(
    filename="logs/cmems_download.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)
logger.info("开始下载...")
```

### 监控脚本

```bash
# 查看最新日志
tail -f logs/cmems_download.log

# 统计下载次数
grep "下载完成" logs/cmems_download.log | wc -l
```

## 总结

这个闭环提供了一个**完整、自动化、可维护**的方案来获取北极近实时数据：

✅ **自动化**: 无需手动指定 dataset-id 和变量名  
✅ **容错性**: 启发式搜索应对 API 变化  
✅ **可扩展**: 易于添加新产品或修改参数  
✅ **可监控**: 支持日志和自动化执行  
✅ **生产就绪**: 支持定期自动更新  

下一步可以考虑：
- 集成数据质量检查
- 添加数据可视化
- 实现增量更新机制
- 构建数据管道

