# AIS Phase 1 项目索引

## 📚 文档导航

### 项目总览
- **[AIS_PHASE1_COMPLETION_CERTIFICATE.txt](AIS_PHASE1_COMPLETION_CERTIFICATE.txt)** - 项目完成证书
- **[AIS_PHASE1_中文总结.md](AIS_PHASE1_中文总结.md)** - 中文项目总结
- **[AIS_PHASE1_IMPLEMENTATION_SUMMARY.md](AIS_PHASE1_IMPLEMENTATION_SUMMARY.md)** - 详细实现说明
- **[AIS_PHASE1_VERIFICATION_REPORT.md](AIS_PHASE1_VERIFICATION_REPORT.md)** - 完整验证报告

### 快速参考
- **[AIS_PHASE1_QUICK_START.md](AIS_PHASE1_QUICK_START.md)** - 快速开始指南
- **[AIS_PHASE1_INDEX.md](AIS_PHASE1_INDEX.md)** - 本文件（项目索引）

---

## 🔧 核心代码

### 新建模块

#### `arcticroute/core/ais_ingest.py` (280 行)
**功能**: AIS 数据摄取和处理

**主要类和函数**:
- `AISSchemaSummary` - AIS CSV schema 摘要
- `inspect_ais_csv()` - 探测 AIS CSV schema 和范围
- `rasterize_ais_density_to_grid()` - 将 AIS 点栅格化到网格
- `AISDensityResult` - 栅格化结果数据类
- `build_ais_density_for_grid()` - 从 CSV 构建 AIS 密度场

**使用示例**:
```python
from arcticroute.core.ais_ingest import build_ais_density_for_grid

ais_result = build_ais_density_for_grid(
    "data_real/ais/raw/ais_2024_sample.csv",
    grid.lat2d, grid.lon2d
)
```

### 修改模块

#### `arcticroute/core/cost.py` (+60 行)
**修改内容**:
- 添加 `ais_density: Optional[np.ndarray] = None` 参数
- 添加 `ais_weight: float = 0.0` 参数
- 实现 AIS 密度处理逻辑
- 更新文档字符串

**关键代码**:
```python
if ais_density is not None and ais_weight > 0:
    ais_norm = np.clip(ais_density, 0.0, 1.0)
    ais_cost = ais_weight * ais_norm
    cost = cost + ais_cost
    components["ais_density"] = ais_cost
```

#### `arcticroute/ui/planner_minimal.py` (+80 行)
**修改内容**:
- 添加 AIS 权重滑条
- 实现 AIS 数据加载逻辑
- 传递参数给成本模型
- 更新成本分解标签
- 添加用户提示

**关键代码**:
```python
ais_weight = st.slider(
    "AIS 拥挤风险权重 w_ais",
    min_value=0.0, max_value=5.0, value=1.0, step=0.1
)

ais_result = build_ais_density_for_grid(...)
cost_field = build_cost_from_real_env(
    ..., ais_density=ais_density, ais_weight=ais_weight
)
```

---

## 🧪 测试文件

### 测试模块

#### `tests/test_ais_ingest_schema.py` (80 行, 5 个测试)
**测试内容**: AIS schema 探测

**测试用例**:
- `test_inspect_ais_csv_basic` - 基础读取
- `test_inspect_ais_csv_has_required_columns` - 列检测
- `test_inspect_ais_csv_ranges` - 范围提取
- `test_inspect_ais_csv_nonexistent_file` - 错误处理
- `test_inspect_ais_csv_sample_limit` - 采样限制

**运行**:
```bash
python -m pytest tests/test_ais_ingest_schema.py -v
```

#### `tests/test_ais_density_rasterize.py` (180 行, 8 个测试)
**测试内容**: AIS 栅格化

**测试用例**:
- `test_rasterize_ais_density_basic` - 基础栅格化
- `test_rasterize_ais_density_normalize` - 归一化
- `test_rasterize_ais_density_no_crash_on_outliers` - 越界处理
- `test_build_ais_density_for_grid_basic` - CSV 构建
- `test_build_ais_density_for_grid_nonexistent` - 文件不存在
- `test_build_ais_density_max_rows` - 行数限制
- `test_rasterize_ais_density_empty_points` - 空数据
- `test_rasterize_ais_density_single_point` - 单点

**运行**:
```bash
python -m pytest tests/test_ais_density_rasterize.py -v
```

#### `tests/test_cost_with_ais_density.py` (150 行, 5 个测试)
**测试内容**: 成本模型集成

**测试用例**:
- `test_cost_increases_with_ais_weight` - 权重效果
- `test_components_contains_ais_density` - 组件记录
- `test_no_crash_when_no_ais` - 禁用 AIS
- `test_ais_density_shape_mismatch` - 形状不匹配
- `test_ais_density_normalization` - 超出范围处理

**运行**:
```bash
python -m pytest tests/test_cost_with_ais_density.py -v
```

#### `tests/test_ais_phase1_integration.py` (120 行, 2 个测试)
**测试内容**: 完整集成测试

**测试用例**:
- `test_ais_phase1_complete_workflow` - 完整工作流
- `test_ais_phase1_with_real_data` - 真实数据处理

**运行**:
```bash
python -m pytest tests/test_ais_phase1_integration.py -v
```

### 运行所有 AIS 测试

```bash
python -m pytest tests/test_ais_ingest_schema.py \
                   tests/test_ais_density_rasterize.py \
                   tests/test_cost_with_ais_density.py \
                   tests/test_ais_phase1_integration.py -v

# 预期结果: 20 passed ✅
```

---

## 📊 数据文件

### 测试数据

#### `tests/data/ais_sample.csv` (10 行)
**用途**: 单元测试用数据
**格式**: mmsi, timestamp, lat, lon, sog, cog, ship_type
**范围**: 纬度 75-76N，经度 20-22E

### 真实数据

#### `data_real/ais/raw/ais_2024_sample.csv` (21 行)
**用途**: 集成测试和 UI 演示
**格式**: mmsi, timestamp, lat, lon, sog, cog, ship_type
**范围**: 纬度 74-76N，经度 19-23E

---

## 🎯 快速命令

### 验证安装
```bash
# 检查所有文件是否存在
python -c "
from pathlib import Path
files = [
    'arcticroute/core/ais_ingest.py',
    'tests/test_ais_ingest_schema.py',
    'tests/test_ais_density_rasterize.py',
    'tests/test_cost_with_ais_density.py',
    'tests/test_ais_phase1_integration.py',
    'data_real/ais/raw/ais_2024_sample.csv',
]
for f in files:
    print(f'✅ {f}' if Path(f).exists() else f'❌ {f}')
"
```

### 运行测试
```bash
# 快速测试（不显示详细信息）
python -m pytest tests/test_ais_ingest_schema.py tests/test_ais_density_rasterize.py tests/test_cost_with_ais_density.py tests/test_ais_phase1_integration.py -q

# 详细测试（显示所有信息）
python -m pytest tests/test_ais_ingest_schema.py tests/test_ais_density_rasterize.py tests/test_cost_with_ais_density.py tests/test_ais_phase1_integration.py -v

# 显示覆盖率
python -m pytest tests/test_ais_ingest_schema.py tests/test_ais_density_rasterize.py tests/test_cost_with_ais_density.py tests/test_ais_phase1_integration.py --cov=arcticroute.core.ais_ingest --cov=arcticroute.core.cost
```

### 启动 UI
```bash
streamlit run run_ui.py
```

### 使用 API
```bash
python -c "
from arcticroute.core.ais_ingest import inspect_ais_csv
summary = inspect_ais_csv('data_real/ais/raw/ais_2024_sample.csv')
print(f'数据行数: {summary.num_rows}')
print(f'纬度范围: {summary.lat_min} ~ {summary.lat_max}')
print(f'经度范围: {summary.lon_min} ~ {summary.lon_max}')
"
```

---

## 📖 API 文档

### `inspect_ais_csv(path: str, sample_n: int = 5000) -> AISSchemaSummary`

**功能**: 探测 AIS CSV 的 schema 和数据范围

**参数**:
- `path` (str): CSV 文件路径
- `sample_n` (int): 采样行数，默认 5000

**返回**: `AISSchemaSummary` 对象，包含：
- `num_rows`: 数据行数
- `columns`: 列名列表
- `has_mmsi`, `has_lat`, `has_lon`, `has_timestamp`: 列存在标志
- `lat_min`, `lat_max`, `lon_min`, `lon_max`: 坐标范围
- `time_min`, `time_max`: 时间范围

**示例**:
```python
summary = inspect_ais_csv("data.csv")
print(f"数据行数: {summary.num_rows}")
```

### `build_ais_density_for_grid(csv_path: str, grid_lat2d: np.ndarray, grid_lon2d: np.ndarray, max_rows: int = 50000) -> AISDensityResult`

**功能**: 从 AIS CSV 构建密度场

**参数**:
- `csv_path` (str): CSV 文件路径
- `grid_lat2d` (np.ndarray): 网格纬度，形状 (H, W)
- `grid_lon2d` (np.ndarray): 网格经度，形状 (H, W)
- `max_rows` (int): 最多读取的行数，默认 50000

**返回**: `AISDensityResult` 对象，包含：
- `da`: xarray.DataArray，密度场
- `num_points`: 总点数
- `num_binned`: 有效点数
- `frac_binned`: 有效点比例

**示例**:
```python
ais_result = build_ais_density_for_grid(
    "data.csv", grid.lat2d, grid.lon2d
)
print(f"有效点: {ais_result.num_binned}/{ais_result.num_points}")
```

### `build_cost_from_real_env(..., ais_density: Optional[np.ndarray] = None, ais_weight: float = 0.0) -> CostField`

**功能**: 构建成本场（支持 AIS 密度）

**新增参数**:
- `ais_density` (np.ndarray): AIS 密度场，形状与网格相同
- `ais_weight` (float): AIS 权重，0.0 ~ 5.0

**返回**: `CostField` 对象，components 中包含 "ais_density"（如果启用）

**示例**:
```python
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    ais_density=ais_result.da.values,
    ais_weight=1.5
)
```

---

## 🔍 故障排除

### 问题：AIS 数据文件找不到
**解决方案**:
1. 确保文件位置: `data_real/ais/raw/ais_2024_sample.csv`
2. 检查文件权限
3. 检查文件格式（应为 CSV）

### 问题：测试失败
**解决方案**:
1. 确保所有依赖已安装: `pip install -r requirements.txt`
2. 检查 Python 版本: >= 3.8
3. 运行单个测试查看详细错误: `pytest tests/test_ais_ingest_schema.py::test_inspect_ais_csv_basic -v`

### 问题：AIS 密度全为 0
**解决方案**:
1. 检查 AIS 点坐标范围
2. 检查网格覆盖范围
3. 运行 `inspect_ais_csv()` 查看数据范围

### 问题：UI 中 AIS 权重滑条不显示
**解决方案**:
1. 确保 Streamlit 版本最新
2. 清除浏览器缓存
3. 重启 Streamlit 应用

---

## 📞 支持信息

### 文档
- 详细实现: [AIS_PHASE1_IMPLEMENTATION_SUMMARY.md](AIS_PHASE1_IMPLEMENTATION_SUMMARY.md)
- 快速开始: [AIS_PHASE1_QUICK_START.md](AIS_PHASE1_QUICK_START.md)
- 验证报告: [AIS_PHASE1_VERIFICATION_REPORT.md](AIS_PHASE1_VERIFICATION_REPORT.md)

### 代码注释
- 所有函数都有完整的文档字符串
- 关键代码段有详细注释
- 测试用例包含使用示例

### 测试用例
- 20 个单元测试覆盖所有功能
- 集成测试验证完整工作流
- 边界情况都有测试

---

## ✅ 项目状态

**完成度**: 100% ✅
**测试通过**: 20/20 ✅
**文档完整**: 是 ✅
**生产就绪**: 是 ✅

---

**最后更新**: 2025-12-10  
**版本**: 1.0  
**状态**: ✅ 完成并通过验收




