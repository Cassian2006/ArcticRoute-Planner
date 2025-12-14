# AIS Phase 1 实现总结

## 概述

成功完成了 AIS Phase 1 的完整实现，包括数据探测、栅格化、成本模型集成和 UI 展示。

## 实现步骤

### Step 0: 前置约定 ✅

- **数据路径约定**：`data_real/ais/raw/ais_2024_sample.csv`
- **测试数据**：`tests/data/ais_sample.csv`（9 行示例数据）
- **真实数据**：`data_real/ais/raw/ais_2024_sample.csv`（20 行示例数据）

### Step 1: AIS Schema 探测 ✅

**新建模块**：`arcticroute/core/ais_ingest.py`

实现功能：
- `AISSchemaSummary` 数据类：记录 CSV 的基础信息
- `inspect_ais_csv()` 函数：
  - 读取前 N 行 AIS CSV
  - 推断 schema（列名、数据类型）
  - 提取范围信息（时间、纬度、经度）
  - 优雅处理缺失列和错误

**测试**：`tests/test_ais_ingest_schema.py`（5 个测试）
- ✅ 基础读取和列检测
- ✅ 范围信息提取
- ✅ 处理不存在的文件
- ✅ 采样行数限制

### Step 2: AIS 栅格化为密度场 ✅

**新增函数**：`arcticroute/core/ais_ingest.py`

实现功能：
- `rasterize_ais_density_to_grid()` 函数：
  - 将 AIS 经纬度点栅格化到给定网格
  - 使用最近邻算法找到最近栅格
  - 支持归一化到 [0, 1]
  - 返回 xarray.DataArray

- `AISDensityResult` 数据类：记录栅格化结果
- `build_ais_density_for_grid()` 函数：
  - 从 CSV 读取数据
  - 过滤缺失值
  - 调用栅格化函数
  - 返回完整结果（包含统计信息）

**测试**：`tests/test_ais_density_rasterize.py`（8 个测试）
- ✅ 基础栅格化
- ✅ 归一化功能
- ✅ 越界坐标处理
- ✅ 从 CSV 构建密度场
- ✅ 处理不存在的文件
- ✅ max_rows 参数
- ✅ 空点集处理
- ✅ 单点栅格化

### Step 3: 成本模型集成 ✅

**修改**：`arcticroute/core/cost.py`

新增参数：
- `ais_density: Optional[np.ndarray] = None`：AIS 密度场
- `ais_weight: float = 0.0`：AIS 权重

实现逻辑：
```python
if ais_density is not None and ais_weight > 0:
    # Safe 归一化到 [0, 1]
    ais_norm = np.clip(ais_density, 0.0, 1.0)
    
    # 计算 AIS 成本
    ais_cost = ais_weight * ais_norm
    
    # 累加进总成本
    cost = cost + ais_cost
    
    # 记录到 components
    components["ais_density"] = ais_cost
```

**测试**：`tests/test_cost_with_ais_density.py`（5 个测试）
- ✅ 权重增加时成本单调上升
- ✅ components 包含 ais_density
- ✅ 没有 AIS 时行为正常
- ✅ 形状不匹配处理
- ✅ 超出范围的密度归一化

### Step 4: UI 集成 ✅

**修改**：`arcticroute/ui/planner_minimal.py`

新增功能：

1. **Sidebar 滑条**：
   ```python
   ais_weight = st.slider(
       "AIS 拥挤风险权重 w_ais",
       min_value=0.0,
       max_value=5.0,
       value=1.0,
       step=0.1,
   )
   ```

2. **AIS 数据加载**：
   - 检查 `data_real/ais/raw/ais_2024_sample.csv` 是否存在
   - 调用 `build_ais_density_for_grid()` 构建密度场
   - 显示加载统计信息

3. **成本模型传参**：
   - 将 `ais_density` 和 `ais_weight` 传递给 `build_cost_from_real_env()`

4. **成本分解展示**：
   - 在 `COMPONENT_LABELS` 中添加 `"ais_density": "AIS 拥挤风险 🚢"`
   - 在成本分解表格中显示 AIS 密度成本
   - 添加提示信息（当 AIS 已启用但未产生分量时）

## 测试覆盖

**总计 20 个测试，全部通过** ✅

| 测试文件 | 测试数 | 状态 |
|---------|--------|------|
| test_ais_ingest_schema.py | 5 | ✅ |
| test_ais_density_rasterize.py | 8 | ✅ |
| test_cost_with_ais_density.py | 5 | ✅ |
| test_ais_phase1_integration.py | 2 | ✅ |

## 关键特性

### 1. 鲁棒性
- 优雅处理缺失数据和错误
- 形状不匹配检测和处理
- 超出范围坐标的自动 clip

### 2. 灵活性
- AIS 权重可调（0.0 ~ 5.0）
- 支持任意网格大小
- 自动归一化密度场

### 3. 集成性
- 无缝集成到现有成本模型
- 与 EDL、冰级约束等兼容
- UI 友好的参数控制

### 4. 可观测性
- 详细的日志输出
- 成本分解中的 AIS 组件
- 加载统计信息展示

## 数据流

```
AIS CSV
  ↓
inspect_ais_csv() → AISSchemaSummary
  ↓
build_ais_density_for_grid() → AISDensityResult (xarray.DataArray)
  ↓
build_cost_from_real_env(..., ais_density=..., ais_weight=...) → CostField
  ↓
UI 展示（成本分解表格中的 AIS 拥挤风险）
```

## 使用示例

### Python API

```python
from arcticroute.core.ais_ingest import build_ais_density_for_grid
from arcticroute.core.cost import build_cost_from_real_env

# 构建 AIS 密度场
ais_result = build_ais_density_for_grid(
    "data_real/ais/raw/ais_2024_sample.csv",
    grid.lat2d,
    grid.lon2d,
    max_rows=50000,
)

# 集成到成本模型
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    ais_density=ais_result.da.values,
    ais_weight=1.5,  # 调整权重
)
```

### UI 使用

1. 在 Sidebar 中调整 "AIS 拥挤风险权重 w_ais" 滑条
2. 点击"规划三条方案"
3. 在成本分解表格中查看 "AIS 拥挤风险 🚢" 行
4. 观察路线如何避开高 AIS 密度区域

## 后续扩展

### 可能的改进

1. **数据源扩展**
   - 支持多个 AIS 数据源
   - 时间序列 AIS 数据
   - 实时 AIS 流接入

2. **算法优化**
   - 使用 KD-tree 加速最近邻搜索
   - 支持自定义栅格化方法（如高斯核）
   - 时间衰减权重

3. **UI 增强**
   - AIS 密度热力图可视化
   - 历史 AIS 轨迹展示
   - AIS 数据质量指标

4. **模型融合**
   - AIS 密度与船舶类型关联
   - 季节性 AIS 模式
   - 基于 AIS 的风险预测

## 注意事项

1. **数据准备**：
   - 确保 AIS CSV 包含 `mmsi`, `lat`, `lon`, `timestamp` 列
   - 坐标范围应在 [-180, 180] 和 [-90, 90] 内

2. **性能**：
   - 大规模 AIS 数据（>100k 点）可能需要优化
   - 建议使用 max_rows 参数限制数据量

3. **精度**：
   - AIS 密度基于最近邻栅格化，精度取决于网格分辨率
   - 建议网格分辨率 ≤ 0.1 度

## 完成状态

✅ **AIS Phase 1 完全实现并通过所有测试**

所有 5 个 Step 均已完成：
- Step 0: 前置约定 ✅
- Step 1: Schema 探测 ✅
- Step 2: 栅格化 ✅
- Step 3: 成本集成 ✅
- Step 4: UI 集成 ✅

系统已准备好进入 AIS Phase 2（如有需要）。






