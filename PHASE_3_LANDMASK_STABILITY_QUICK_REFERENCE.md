# Phase 3 Landmask 稳定化 - 快速参考

## 核心 API

### 1. 扫描候选文件
```python
from arcticroute.core.landmask_select import scan_landmask_candidates

candidates = scan_landmask_candidates(search_dirs=["data_real/landmask", "data_real/env"])
# 返回: List[LandmaskCandidate]
#   - path: 文件路径
#   - grid_signature: 网格签名（若有）
#   - shape: 数据形状
#   - varname: 变量名
#   - note: 读取状态或错误信息
```

### 2. 选择最佳候选
```python
from arcticroute.core.landmask_select import select_best_candidate

best = select_best_candidate(
    candidates,
    target_signature="40x80_65.0000_80.0000_0.0000_160.0000",
    prefer_path="data_real/landmask/my_landmask.nc"
)
# 优先级: prefer_path > signature > filename > shape
```

### 3. 加载并对齐
```python
from arcticroute.core.landmask_select import load_and_align_landmask
from arcticroute.core.grid import Grid2D

landmask, meta = load_and_align_landmask(best_candidate, grid, method="nearest")
# 返回: (np.ndarray[bool], dict)
#   - landmask: shape 与 grid 相同的 bool 数组 (True=land)
#   - meta: 包含 source_path, original_shape, resampled, land_fraction 等
```

### 4. 统一加载入口
```python
from arcticroute.core.landmask import load_landmask_for_grid

landmask, meta = load_landmask_for_grid(
    grid,
    prefer_real=True,
    explicit_path="data_real/landmask/land_mask.nc",
    search_dirs=["data_real/landmask", "data_real/env"]
)
# 自动处理候选扫描、选择、加载、对齐、回退
```

### 5. 网格+Landmask 一体加载
```python
from arcticroute.core.grid import load_grid_with_landmask

grid, land_mask, meta = load_grid_with_landmask(
    prefer_real=True,
    explicit_landmask_path="data_real/landmask/land_mask.nc",
    landmask_search_dirs=["data_real/landmask", "data_real/env"]
)
# meta 包含: source, data_root, landmask_path, landmask_resampled, landmask_land_fraction
```

---

## 诊断脚本

```bash
python -m scripts.check_grid_and_landmask
```

**输出内容**：
- [0] 数据根目录配置
- [1] 候选列表（signature/shape/varname/note）
- [2] 加载结果
- [3] 网格信息
- [4] 陆地掩码统计
- [5] Landmask 加载详情
- [6] 网格范围
- [7] 修复指引（若需要）

---

## 元数据字段

### load_and_align_landmask 返回的 meta

| 字段 | 类型 | 说明 |
|-----|------|------|
| `source_path` | str | 加载的文件路径或 "demo" |
| `original_shape` | tuple | 原始文件的 shape |
| `target_shape` | tuple | 目标网格的 shape |
| `resampled` | bool | 是否进行了重采样 |
| `cache_hit` | bool | 是否命中缓存 |
| `method` | str | 使用的插值方法 |
| `varname` | str | 使用的变量名 |
| `land_fraction` | float | 陆地比例 [0, 1] |
| `nan_count` | int | NaN 值个数 |
| `error` | str | 错误信息（若有） |
| `warning` | str | 警告信息（若有） |

### load_landmask_for_grid 返回的 meta

| 字段 | 类型 | 说明 |
|-----|------|------|
| `source_path` | str | 加载的文件路径或 "demo" |
| `original_shape` | tuple | 原始文件的 shape |
| `target_shape` | tuple | 目标网格的 shape |
| `resampled` | bool | 是否进行了重采样 |
| `varname` | str | 使用的变量名 |
| `land_fraction` | float | 陆地比例 [0, 1] |
| `fallback_demo` | bool | 是否回退到 demo |
| `reason` | str | 回退原因 |
| `warning` | str | 警告信息 |

### load_grid_with_landmask 返回的 meta

| 字段 | 类型 | 说明 |
|-----|------|------|
| `source` | str | 网格来源 ("real" / "demo") |
| `data_root` | str | 数据根目录 |
| `landmask_path` | str | landmask 文件路径 |
| `landmask_resampled` | bool | 是否重采样 |
| `landmask_land_fraction` | float | 陆地比例 |
| `landmask_note` | str | 诊断信息 |

---

## 语义归一化

支持的 landmask 编码方式：

| 编码方式 | 说明 | 处理方式 |
|---------|------|---------|
| 0/1 | 0=ocean, 1=land | 自动检测（基于陆地比例） |
| 反转 0/1 | 0=land, 1=ocean | 自动检测（基于陆地比例） |
| bool | True=land, False=ocean | 直接使用 |
| float | >0.5=land, ≤0.5=ocean | 阈值判断 |
| NaN | NaN 当 ocean | NaN 转换为 False |

**陆地比例启发式**：
- 若 1 的比例在 5%-50% 之间，认为 1 是 land
- 否则认为 0 是 land

---

## 文件格式要求

### NetCDF 文件结构

```
variables:
  - land_mask / landmask / mask / lsm / land / is_land (优先级顺序)
    shape: (ny, nx)
    dtype: bool, int, float
    
attributes (可选):
  - grid_signature: "40x80_65.0000_80.0000_0.0000_160.0000"
```

### 文件命名建议

- `land_mask.nc` (推荐)
- `land_mask_gebco.nc`
- `landmask.nc`
- `landmask_gebco.nc`

### 搜索目录

默认搜索目录（按优先级）：
1. `data_real/landmask/`
2. `data_real/env/`
3. `data_real/`

---

## UI 集成

### 诊断区显示

在 Streamlit UI 的 "诊断与依赖状态" 展开器中：

```
陆地掩码诊断
  📍 来源: {landmask_path}
  🔄 已进行重采样
  🏔️ 陆地比例: 42.61%
  📝 备注: successfully loaded real landmask
  ⚠️ 已回退到演示 landmask: 未找到任何 landmask 候选文件
```

### 参数输入

- **Landmask 文件（可选）**: 文本框，输入显式指定的 landmask 路径
  - 示例: `data_real/landmask/land_mask.nc`
  - 若为空，则自动扫描候选

---

## 测试

### 运行所有 landmask 测试

```bash
pytest tests/test_landmask_selection.py -v
```

### 运行特定测试

```bash
pytest tests/test_landmask_selection.py::TestLandmaskSelection::test_load_and_align_landmask_with_resampling -v
```

### 测试覆盖

- ✅ 候选扫描和识别
- ✅ 签名匹配和优先级选择
- ✅ 形状匹配和重采样
- ✅ 语义归一化（0/1、反转、float、NaN）
- ✅ 陆地比例合理性检查
- ✅ 异常情况处理

---

## 常见问题

### Q: 如何指定特定的 landmask 文件？

```python
landmask, meta = load_landmask_for_grid(
    grid,
    explicit_path="path/to/my_landmask.nc"
)
```

### Q: 如何自定义搜索目录？

```python
landmask, meta = load_landmask_for_grid(
    grid,
    search_dirs=["custom/dir1", "custom/dir2"]
)
```

### Q: 如何强制使用 demo landmask？

```python
landmask, meta = load_landmask_for_grid(
    grid,
    prefer_real=False
)
```

### Q: 如何检查是否回退到 demo？

```python
if meta.get("fallback_demo"):
    reason = meta.get("reason")
    print(f"Fallback reason: {reason}")
```

### Q: 陆地比例异常时如何处理？

```python
if meta.get("warning"):
    print(f"Warning: {meta['warning']}")
    # 可能需要检查 landmask 文件的语义编码
```

---

## 性能指标

- **扫描**: ~100ms（扫描 10 个 .nc 文件）
- **加载**: ~50ms（读取单个 landmask 文件）
- **重采样**: ~100ms（从 100x100 重采样到 1000x1000）
- **缓存命中**: <1ms

---

## 版本信息

- **Phase**: 3
- **Branch**: `feat/landmask-stability`
- **Commit**: 480e81e
- **Date**: 2025-12-14
- **Status**: ✅ Complete

---

## 相关文档

- [ADR-0001: LayerGraph + Catalog + Plugins Architecture](docs/adr/ADR-0001-layergraph.md)
- [AIS Density Selection Implementation](arcticroute/core/ais_density_select.py)
- [Grid Loader Implementation](arcticroute/core/grid.py)

