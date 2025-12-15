# Phase 8 快速参考

## 核心改动

### 1. 新增 I/O 模块
```
arcticroute/io/
├── __init__.py
└── cmems_loader.py (290 行)
    ├── find_latest_nc()
    ├── load_sic_from_nc()
    ├── load_swh_from_nc()
    └── align_to_grid()
```

### 2. 修改 RealEnvLayers
```python
# 新增类方法
env = RealEnvLayers.from_cmems(
    grid=grid,
    sic_nc="data/cmems_cache/sic_latest.nc",
    swh_nc="data/cmems_cache/swh_latest.nc",
    allow_partial=True,
)
```

### 3. 新增刷新脚本
```bash
python scripts/cmems_refresh_and_export.py [--days N] [--output-dir DIR]
```

### 4. 新增测试
```bash
pytest tests/test_cmems_loader.py -v
# 6/6 通过 ✅
```

---

## 使用流程

### 步骤 1: 下载数据（Phase 7）
```bash
python scripts/cmems_download.py
# 输出: data/cmems_cache/sic_latest.nc, swh_latest.nc
```

### 步骤 2: 加载到 RealEnvLayers（Phase 8）
```python
from arcticroute.core.env_real import RealEnvLayers
from pathlib import Path

env = RealEnvLayers.from_cmems(
    grid=your_grid,
    sic_nc=Path("data/cmems_cache/sic_latest.nc"),
    swh_nc=Path("data/cmems_cache/swh_latest.nc"),
)

# 现在可以使用
print(env.sic)        # 海冰浓度数组
print(env.wave_swh)   # 有效波高数组
```

### 步骤 3: 用于规划
```python
# 在规划器中使用
result = planner.plan(
    start=(lat, lon),
    end=(lat, lon),
    env=env,  # 直接传入 RealEnvLayers
)
```

---

## 关键函数

### `load_sic_from_nc(path)`
```python
from arcticroute.io.cmems_loader import load_sic_from_nc

sic_2d, metadata = load_sic_from_nc("data/cmems_cache/sic_latest.nc")
# sic_2d: (ny, nx) 数组，范围 0-1
# metadata: {"variable": "sic", "shape": (ny, nx), "min": ..., "max": ...}
```

### `load_swh_from_nc(path)`
```python
from arcticroute.io.cmems_loader import load_swh_from_nc

swh_2d, metadata = load_swh_from_nc("data/cmems_cache/swh_latest.nc")
# swh_2d: (ny, nx) 数组，单位米
# metadata: {"variable": "...", "shape": (ny, nx), "unit": "m", ...}
```

### `find_latest_nc(outdir, pattern)`
```python
from arcticroute.io.cmems_loader import find_latest_nc

latest = find_latest_nc("data/cmems_cache", "sic_*.nc")
# 返回最新的 sic_*.nc 文件路径
```

### `RealEnvLayers.from_cmems()`
```python
env = RealEnvLayers.from_cmems(
    grid=grid,                          # 必需
    land_mask=land_mask,                # 可选
    sic_nc="data/cmems_cache/sic_latest.nc",  # 可选
    swh_nc="data/cmems_cache/swh_latest.nc",  # 可选
    allow_partial=True,                 # 允许部分数据缺失
)

# 返回 RealEnvLayers 对象
# env.sic: 海冰浓度（或 None）
# env.wave_swh: 有效波高（或 None）
# env.grid: 网格
# env.land_mask: 陆地掩码
```

---

## 自动化刷新

### 方案 A: 手动运行
```bash
python scripts/cmems_refresh_and_export.py
```

### 方案 B: 定时任务（Cron）
```bash
0 13 * * * cd /path/to/AR_final && python scripts/cmems_refresh_and_export.py
```

### 方案 C: PowerShell 循环
```powershell
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 1440
```

---

## 输出文件

### 数据文件
```
data/cmems_cache/
├── sic_20251215.nc         # 海冰数据（日期戳）
├── swh_202512150300.nc     # 波浪数据（日期+小时戳）
└── sic_latest.nc           # 最新海冰数据（软链接或副本）
```

### 元数据
```
reports/cmems_refresh_last.json
{
  "timestamp": "2025-12-15T03:18:44.231Z",
  "start_date": "2025-12-13",
  "end_date": "2025-12-15",
  "downloads": {
    "sic": {...},
    "swh": {...}
  }
}
```

---

## 错误处理

### 数据缺失
```python
# allow_partial=True 时，缺失数据不会抛出异常
env = RealEnvLayers.from_cmems(
    grid=grid,
    sic_nc="nonexistent.nc",  # 不存在
    allow_partial=True,        # 继续运行
)
# env.sic 会是 None，但 env 对象仍然有效
```

### 网格不匹配
```python
# 自动进行网格对齐
env = RealEnvLayers.from_cmems(
    grid=grid,  # 目标网格
    sic_nc="data/cmems_cache/sic_latest.nc",  # 可能是不同的网格
)
# 数据会自动重采样到目标网格
```

---

## 测试

### 运行所有测试
```bash
pytest tests/test_cmems_loader.py -v
```

### 运行特定测试
```bash
pytest tests/test_cmems_loader.py::TestCMEMSLoader::test_load_sic_from_nc -v
```

### 测试覆盖
- ✅ 加载 SIC 数据
- ✅ 加载 SWH 数据
- ✅ 查找最新文件
- ✅ 处理时间维度
- ✅ 完整集成测试
- ✅ 部分数据加载

---

## 常见问题

### Q: 如何检查数据是否成功加载？
```python
if env.sic is not None:
    print(f"SIC 已加载: {env.sic.shape}")
else:
    print("SIC 加载失败")

if env.wave_swh is not None:
    print(f"SWH 已加载: {env.wave_swh.shape}")
else:
    print("SWH 加载失败")
```

### Q: 如何自定义下载范围？
```bash
python scripts/cmems_refresh_and_export.py \
  --bbox-min-lon 0 \
  --bbox-max-lon 40 \
  --bbox-min-lat 70 \
  --bbox-max-lat 85
```

### Q: 如何处理网格不匹配？
```python
# 自动处理，无需手动干预
env = RealEnvLayers.from_cmems(
    grid=your_grid,  # 目标网格
    sic_nc="data/cmems_cache/sic_latest.nc",
)
# 数据会自动重采样到 your_grid 的形状
```

### Q: 如何回退到 demo 数据？
```python
# 如果 CMEMS 数据加载失败，可以回退
try:
    env = RealEnvLayers.from_cmems(grid=grid, sic_nc=sic_path)
except Exception:
    # 回退到 demo
    env = load_demo_env(grid)
```

---

## 下一步（Phase 9）

1. **规划器集成**: 在 `planner_service.py` 中调用 `from_cmems()`
2. **UI 集成**: 在 Streamlit 中添加数据选择
3. **缓存优化**: 避免重复加载
4. **可视化**: 显示 SIC/SWH 地图
5. **质量检查**: 数据有效性验证

---

**状态**: ✅ Phase 8 完成  
**下一步**: Phase 9 规划器集成

