# Phase 8: CMEMS 数据摄入与集成

**目标**: 将下载的 CMEMS 数据（海冰浓度 SIC 和有效波高 SWH）集成到 RealEnvLayers，实现"下载→落盘→自动加载→参与规划"的完整闭环。

**完成日期**: 2025-12-15  
**状态**: ✅ 完成

---

## 📋 实现内容

### 1. 新增 I/O 模块 (`arcticroute/io/`)

#### `arcticroute/io/__init__.py`
- 导出公共接口

#### `arcticroute/io/cmems_loader.py` (290 行)
**功能**: CMEMS NetCDF 数据加载和对齐

**核心函数**:
- `find_latest_nc(outdir, pattern)` - 在目录中查找最新的 NetCDF 文件
- `load_sic_from_nc(path)` - 加载海冰浓度数据，返回 (sic_2d, metadata)
- `load_swh_from_nc(path)` - 加载有效波高数据，返回 (swh_2d, metadata)
- `align_to_grid(data_2d, source_coords, target_grid, method)` - 将数据重采样到目标网格

**特性**:
- ✅ 自动检测变量名（支持多种命名约定）
- ✅ 处理 3D 时间维度数据（自动取最后一个时间步）
- ✅ 自动规范化数据范围（0-100 → 0-1）
- ✅ 提取和返回完整的元数据（坐标、时间、统计信息）
- ✅ 使用 xarray 进行高效的网格对齐

### 2. 修改 RealEnvLayers (`arcticroute/core/env_real.py`)

#### 新增 `from_cmems` 类方法
```python
@classmethod
def from_cmems(
    cls,
    grid: Grid2D,
    land_mask: Optional[np.ndarray] = None,
    sic_nc: Optional[Path | str] = None,
    swh_nc: Optional[Path | str] = None,
    allow_partial: bool = True,
) -> "RealEnvLayers"
```

**功能**:
- 从 CMEMS NetCDF 文件创建 RealEnvLayers 实例
- 自动加载和对齐 SIC 和 SWH 数据
- 支持部分数据缺失（allow_partial=True 时不抛出异常）
- 返回完整的 RealEnvLayers 对象，可直接用于规划

**使用示例**:
```python
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import Grid2D

# 创建环境层
env = RealEnvLayers.from_cmems(
    grid=grid,
    land_mask=land_mask,
    sic_nc="data/cmems_cache/sic_latest.nc",
    swh_nc="data/cmems_cache/swh_latest.nc",
    allow_partial=True,
)

# 现在可以用于规划
# env.sic, env.wave_swh 已准备好
```

### 3. 新增刷新脚本 (`scripts/cmems_refresh_and_export.py`) (200 行)

**功能**: 自动下载最新数据并生成元数据记录

**工作流**:
1. 读取 `reports/cmems_resolved.json` 获取 dataset-id 和变量名
2. 自动运行 `copernicusmarine subset` 下载最近 N 天的数据
3. 生成带时间戳的输出文件:
   - `sic_YYYYMMDD.nc` - 海冰数据
   - `swh_YYYYMMDDHH.nc` - 波浪数据
4. 生成元数据记录 `reports/cmems_refresh_last.json`

**使用示例**:
```bash
# 下载最近 2 天的数据（默认）
python scripts/cmems_refresh_and_export.py

# 自定义参数
python scripts/cmems_refresh_and_export.py \
  --days 5 \
  --output-dir data/cmems_cache \
  --bbox-min-lon -40 \
  --bbox-max-lon 60 \
  --bbox-min-lat 65 \
  --bbox-max-lat 85
```

**输出元数据示例**:
```json
{
  "timestamp": "2025-12-15T03:18:44.231Z",
  "start_date": "2025-12-13",
  "end_date": "2025-12-15",
  "bbox": {...},
  "downloads": {
    "sic": {
      "dataset_id": "cmems_obs-si_arc_phy_my_l4_P1D",
      "variable": "sic",
      "filename": "sic_20251215.nc",
      "path": "data/cmems_cache/sic_20251215.nc",
      "timestamp": "2025-12-15T03:18:44.231Z",
      "success": true
    },
    "swh": {
      "dataset_id": "dataset-wam-arctic-1hr3km-be",
      "variable": "sea_surface_wave_significant_height",
      "filename": "swh_202512150300.nc",
      "path": "data/cmems_cache/swh_202512150300.nc",
      "timestamp": "2025-12-15T03:18:44.231Z",
      "success": true
    }
  }
}
```

### 4. 新增测试 (`tests/test_cmems_loader.py`) (300 行)

**测试覆盖**:
- ✅ `test_load_sic_from_nc` - 加载 SIC 数据
- ✅ `test_load_swh_from_nc` - 加载 SWH 数据
- ✅ `test_find_latest_nc` - 查找最新文件
- ✅ `test_load_sic_with_time_dimension` - 处理时间维度
- ✅ `test_real_env_layers_from_cmems` - 完整集成测试
- ✅ `test_real_env_layers_from_cmems_partial` - 部分数据加载

**测试结果**: 6/6 通过 ✅

---

## 🔄 完整工作流

```
Phase 7: CMEMS 下载
    ↓
[cmems_resolve.py] → reports/cmems_resolved.json
    ↓
[cmems_download.py] → data/cmems_cache/sic_latest.nc, swh_latest.nc
    ↓
Phase 8: CMEMS 摄入 ← 你在这里
    ↓
[cmems_refresh_and_export.py] → 带时间戳的文件 + 元数据
    ↓
[RealEnvLayers.from_cmems()] → 加载到内存
    ↓
[规划器] → 使用 env.sic, env.wave_swh 参与规划
    ↓
[可视化/导出] → 结果和解释
```

---

## 📊 关键特性

### 自动化
- ✅ 自动检测变量名（支持多种命名约定）
- ✅ 自动处理时间维度
- ✅ 自动规范化数据范围
- ✅ 自动网格对齐

### 容错性
- ✅ 部分数据缺失时不抛出异常（allow_partial=True）
- ✅ 缺失数据时打印警告但继续运行
- ✅ 支持回退到 demo 数据

### 可维护性
- ✅ 清晰的模块结构（io/cmems_loader.py）
- ✅ 完整的文档字符串
- ✅ 全面的测试覆盖
- ✅ 易于扩展（支持添加新变量）

---

## 🚀 使用指南

### 快速开始

```python
from pathlib import Path
from arcticroute.core.env_real import RealEnvLayers
from arcticroute.core.grid import Grid2D

# 1. 准备网格
grid = Grid2D(...)  # 你的网格

# 2. 加载 CMEMS 数据
env = RealEnvLayers.from_cmems(
    grid=grid,
    sic_nc=Path("data/cmems_cache/sic_latest.nc"),
    swh_nc=Path("data/cmems_cache/swh_latest.nc"),
)

# 3. 检查数据
if env.sic is not None:
    print(f"SIC 形状: {env.sic.shape}, 范围: [{env.sic.min():.3f}, {env.sic.max():.3f}]")

if env.wave_swh is not None:
    print(f"SWH 形状: {env.wave_swh.shape}, 范围: [{env.wave_swh.min():.3f}, {env.wave_swh.max():.3f}]")

# 4. 用于规划
# 现在 env 可以传给规划器使用
```

### 自动化更新

```bash
# 每天定时运行
0 13 * * * cd /path/to/AR_final && python scripts/cmems_refresh_and_export.py

# 或使用 PowerShell 循环
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 1440  # 每 24 小时
```

---

## 📈 项目统计

| 项目 | 数量 |
|------|------|
| 新增文件 | 3 个 |
| 修改文件 | 1 个 |
| 新增代码行数 | ~790 |
| 测试用例 | 6 个 |
| 测试通过率 | 100% |

---

## ✅ 验证清单

- [x] 新增 `arcticroute/io/cmems_loader.py`
- [x] 新增 `arcticroute/io/__init__.py`
- [x] 修改 `arcticroute/core/env_real.py` 添加 `from_cmems` 方法
- [x] 新增 `scripts/cmems_refresh_and_export.py`
- [x] 新增 `tests/test_cmems_loader.py`
- [x] 所有测试通过 (6/6)
- [x] 代码文档完整
- [x] 支持部分数据加载
- [x] 支持网格对齐
- [x] 支持自动化刷新

---

## 🔗 相关文件

- **Phase 7 输出**: `reports/cmems_resolved.json`, `data/cmems_cache/*.nc`
- **Phase 8 输入**: 上述文件
- **Phase 8 输出**: `RealEnvLayers` 对象，可直接用于规划
- **后续使用**: 在规划器中调用 `RealEnvLayers.from_cmems()`

---

## 📝 Git 提交

```bash
git checkout feat/polar-rules
git pull
git checkout -b feat/cmems-ingestion

# 添加所有文件
git add -A

# 提交
git commit -m "feat: ingest Copernicus Marine SIC/SWH NetCDF and wire into RealEnvLayers with alignment+tests"

# 推送
git push -u origin feat/cmems-ingestion
```

---

## 🎯 下一步 (Phase 9)

1. **集成到规划器**: 在 `planner_service.py` 中调用 `RealEnvLayers.from_cmems()`
2. **UI 集成**: 在 Streamlit UI 中添加 CMEMS 数据选择和加载
3. **性能优化**: 缓存加载的数据，避免重复读取
4. **可视化**: 在地图上显示 SIC 和 SWH 数据
5. **质量检查**: 添加数据完整性和有效性检查

---

**完成状态**: ✅ Phase 8 完成，已准备好进行 Phase 9 的规划器集成。

