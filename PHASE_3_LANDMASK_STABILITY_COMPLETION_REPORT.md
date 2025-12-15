# Phase 3: 真实 Landmask 稳定化加载机制 - 完成报告

**完成日期**: 2025-12-14  
**分支**: `feat/landmask-stability`  
**提交**: `480e81e`

---

## 执行总结

Phase 3 目标是将真实 landmask 加载机制稳定化，对标 AIS density 的成熟设计。通过实现候选扫描、显式选择、自动匹配、最近邻重采样、缓存和清晰诊断，确保 `load_grid_with_landmask(prefer_real=True)` 在真实数据存在时几乎不会回退到 demo。

**所有 7 个执行步骤已完成，测试通过率 100%**。

---

## 核心实现

### 1. 新增核心模块：`arcticroute/core/landmask_select.py`

**功能**：
- `LandmaskCandidate`: 候选文件信息数据类
- `scan_landmask_candidates()`: 递归扫描 .nc 文件，识别变量和 shape
- `select_best_candidate()`: 优先级选择（prefer_path > signature > filename > shape）
- `load_and_align_landmask()`: 加载并对齐到目标网格，支持最近邻/线性插值
- `compute_grid_signature()`: 生成网格签名用于匹配和缓存
- `_normalize_landmask_semantics()`: 语义归一化（支持 0/1、bool、float、NaN）

**缓存策略**：
- 文件读取缓存：key = (path, mtime)
- 重采样缓存：可选，key = (path, mtime, target_signature, method)

**API 对齐 AIS Density**：
- 统一的扫描/选择/加载流程
- 清晰的元数据输出
- 自动回退机制

---

### 2. 接入到 Landmask 核心：`arcticroute/core/landmask.py`

**新增函数**：
```python
def load_landmask_for_grid(
    grid: Grid2D,
    prefer_real: bool = True,
    explicit_path: Optional[str] = None,
    search_dirs: Optional[List[str]] = None,
) -> Tuple[np.ndarray, dict]
```

**行为**：
- 优先加载真实 landmask（若 prefer_real=True）
- 候选扫描 → 最佳选择 → 加载对齐
- 失败时回退到 demo，并在 meta 中标注原因
- 返回 (landmask_bool_2d, meta)

**元数据包含**：
- `source_path`: 加载的文件路径或 "demo"
- `original_shape`: 原始文件 shape
- `target_shape`: 目标网格 shape
- `resampled`: 是否进行了重采样
- `varname`: 使用的变量名
- `land_fraction`: 陆地比例
- `fallback_demo`: 是否回退
- `reason`: 回退原因
- `warning`: 异常陆地比例警告

---

### 3. 强化网格+Landmask 一体加载：`arcticroute/core/grid.py`

**改造函数**：
```python
def load_grid_with_landmask(
    prefer_real: bool = True,
    explicit_landmask_path: Optional[str] = None,
    landmask_search_dirs: Optional[list[str]] = None,
) -> tuple[Grid2D, np.ndarray, dict]
```

**新增参数**：
- `explicit_landmask_path`: 显式指定的 landmask 文件路径
- `landmask_search_dirs`: 搜索目录列表

**返回 meta 包含**：
- `source`: 网格来源（"real" / "demo"）
- `data_root`: 数据根目录
- `landmask_path`: 加载的 landmask 路径
- `landmask_resampled`: 是否重采样
- `landmask_land_fraction`: 陆地比例
- `landmask_note`: 诊断信息

**行为要求**：
- real grid 成功加载时，优先用 real landmask
- 只有"确实找不到/不可读/解析失败"才回退 demo
- 回退时给出清晰 reason

---

### 4. 升级 CLI 自检脚本：`scripts/check_grid_and_landmask.py`

**增强输出**：
- [0] 数据根目录配置
- [1] 候选列表（含 signature/shape/varname/note）
- [2] 加载网格与 landmask
- [3] 网格信息（shape、坐标范围）
- [4] 陆地掩码统计（land_fraction、ocean_fraction）
- [5] Landmask 加载详情（path、resampled、note）
- [6] 网格范围（四角坐标）
- [7] 修复指引（当使用 demo 时）

**修复指引内容**：
- 当前 ARCTICROUTE_DATA_ROOT
- 预期候选搜索目录
- 若无候选，提示放置位置
- 文件名和格式要求
- 变量名候选列表

---

### 5. 新增防回归测试：`tests/test_landmask_selection.py`

**测试覆盖**（13 个测试，全部通过）：

| 测试名称 | 覆盖内容 |
|---------|---------|
| `test_scan_landmask_candidates_finds_nc_files` | 扫描能找到 .nc 文件 |
| `test_select_best_candidate_prefers_explicit_path` | 优先路径选择 |
| `test_select_best_candidate_matches_signature` | 签名精确匹配 |
| `test_load_and_align_landmask_shape_match` | 形状已匹配直接返回 |
| `test_load_and_align_landmask_with_resampling` | 最近邻重采样 |
| `test_normalize_landmask_semantics_0_1_encoding` | 0/1 编码处理 |
| `test_normalize_landmask_semantics_inverted_encoding` | 反转编码处理 |
| `test_normalize_landmask_semantics_float_encoding` | float 编码处理 |
| `test_normalize_landmask_semantics_nan_handling` | NaN 处理 |
| `test_load_and_align_landmask_land_fraction_sanity` | 陆地比例合理性 |
| `test_load_and_align_landmask_warning_on_extreme_fraction` | 异常比例警告 |
| `test_compute_grid_signature` | 网格签名计算 |
| `test_load_and_align_landmask_file_not_found` | 文件不存在处理 |

**特点**：
- 不依赖真实数据，使用临时 NetCDF 文件
- 覆盖所有语义翻转场景
- 包含陆地比例 sanity check
- 异常情况下产生 warning

---

### 6. UI 最小展示：`arcticroute/ui/planner_minimal.py`

**诊断区展示**（在 "诊断与依赖状态" 展开器中）：

```
陆地掩码诊断
  📍 来源: {landmask_path}
  🔄 已进行重采样 (if resampled)
  🏔️ 陆地比例: {land_fraction:.2%}
  📝 备注: {landmask_note}
  ⚠️ 已回退到演示 landmask: {reason} (if fallback_demo)
```

**新增参数**：
- 文本框：Landmask 文件（可选）
  - 用户可显式指定 landmask 路径
  - 传入 `load_grid_with_landmask(explicit_landmask_path=...)`

**集成点**：
- Pipeline 第 2 个节点：加载网格与 landmask
- 诊断区自动显示加载结果
- 回退时显示清晰警告

---

## 验收口径

### ✅ 测试通过

```
pytest -q
66 passed, 2 skipped (landmask 和 grid 相关测试全部通过)
```

### ✅ 诊断脚本输出

```
python -m scripts.check_grid_and_landmask
```

输出包含：
- ✅ 候选列表（含 signature/shape/varname）
- ✅ 最终采用的 landmask 路径
- ✅ land_fraction 统计
- ✅ 是否重采样标记
- ✅ 缺失时的修复指引

### ✅ UI 诊断区

- ✅ 显示 landmask 来源与回退原因
- ✅ 显示陆地比例
- ✅ 显示是否重采样
- ✅ 支持显式指定 landmask 路径

---

## 关键改进

### 1. 稳定性提升
- **候选扫描**：自动发现所有可用 landmask 文件
- **智能选择**：多级优先级确保最佳匹配
- **自动对齐**：支持任意形状的 landmask 重采样到目标网格
- **缓存机制**：避免重复读取和计算

### 2. 诊断能力
- **清晰的元数据**：每个加载步骤都记录详细信息
- **修复指引**：当加载失败时提供具体的修复建议
- **异常检测**：陆地比例异常时产生 warning

### 3. 用户体验
- **最小侵入**：保持现有 API 兼容
- **UI 集成**：在诊断区显示所有关键信息
- **灵活配置**：支持显式指定路径和搜索目录

### 4. 对标 AIS Density
- **统一的 API 设计**：scan → select → load 流程
- **一致的元数据格式**：便于 UI 和脚本集成
- **相同的缓存策略**：基于 mtime 的智能缓存

---

## 文件变更

### 新增文件
- `arcticroute/core/landmask_select.py` (500+ 行)
- `tests/test_landmask_selection.py` (400+ 行)

### 修改文件
- `arcticroute/core/landmask.py`: 新增 `load_landmask_for_grid()` 函数
- `arcticroute/core/grid.py`: 改造 `load_grid_with_landmask()` 函数
- `scripts/check_grid_and_landmask.py`: 增强诊断输出
- `arcticroute/ui/planner_minimal.py`: 添加 landmask 诊断区和参数

---

## 后续工作建议

1. **真实数据集成**：将真实 landmask 文件放入 `data_real/landmask/` 目录
2. **性能优化**：考虑二级缓存（重采样结果缓存）
3. **可视化增强**：在地图上显示 landmask 覆盖范围
4. **文档完善**：补充 landmask 数据准备指南

---

## 提交信息

```
commit 480e81e
Author: Cascade <cascade@ai>
Date:   2025-12-14

    feat: stabilize real landmask loading with selection/resampling/cache and diagnostics
    
    - New module: arcticroute/core/landmask_select.py
      * scan_landmask_candidates: recursive .nc file discovery
      * select_best_candidate: multi-level priority selection
      * load_and_align_landmask: load and resample to target grid
      * _normalize_landmask_semantics: handle 0/1, bool, float, NaN encodings
    
    - Enhanced: arcticroute/core/landmask.py
      * New unified entry: load_landmask_for_grid()
      * Clear metadata output with fallback reasons
    
    - Enhanced: arcticroute/core/grid.py
      * load_grid_with_landmask() now supports explicit_landmask_path and search_dirs
      * Improved metadata with landmask diagnostics
    
    - Enhanced: scripts/check_grid_and_landmask.py
      * Detailed candidate list with signature/shape/varname
      * Repair guidance when real data not found
    
    - New tests: tests/test_landmask_selection.py
      * 13 comprehensive tests covering all scenarios
      * No external data dependency
    
    - Enhanced: arcticroute/ui/planner_minimal.py
      * Landmask diagnostics panel in expandable section
      * Optional explicit landmask path input
```

---

## 验收签字

- ✅ 所有 7 个执行步骤完成
- ✅ 测试通过率 100% (66 passed, 2 skipped)
- ✅ 诊断脚本输出完整
- ✅ UI 集成成功
- ✅ 代码推送到 `feat/landmask-stability` 分支

**状态**: ✅ **COMPLETE**


