# Phase 7 总结：真实 SIC 成本模式实现

## 概述

Phase 7 成功实现了"真实海冰 SIC 成本模式"，在不破坏现有 demo 功能的前提下，添加了从 NetCDF 文件加载真实海冰浓度（SIC）数据并构建成本场的能力。

## 实现的文件

### 1. 新建：`arcticroute/core/env_real.py`

**功能**：真实环境数据加载模块

**主要类和函数**：
- `RealEnvLayers` dataclass：存储真实环境层数据（目前支持 sic）
- `load_real_sic_for_grid(grid, nc_path, var_candidates, time_index)` 函数：
  - 从 NetCDF 文件中加载与网格对齐的海冰浓度数据
  - 支持 1D 和 2D 坐标、有/无时间维度
  - 自动处理 0..100 到 0..1 的数据缩放
  - 失败时优雅返回 None，不抛异常

**特点**：
- 默认尝试加载 `get_newenv_path() / "ice_copernicus_sic.nc"`
- 支持多个变量名候选（sic, SIC, ice_concentration）
- 完整的错误处理和日志输出

### 2. 修改：`arcticroute/core/cost.py`

**新增函数**：`build_cost_from_sic(grid, land_mask, env, ice_penalty)`

**功能**：
- 使用真实 SIC 数据构建成本场
- 成本规则：
  - 海洋基础成本：1.0
  - 冰风险成本：`ice_penalty * sic^1.5`（非线性放大）
  - 陆地成本：np.inf（不可通行）
- 返回 CostField 对象，包含 components 分解（base_distance、ice_risk）

**向后兼容性**：
- 原有的 `build_demo_cost()` 函数完全不变
- 新函数仅在显式调用时使用

### 3. 修改：`arcticroute/ui/planner_minimal.py`

**新增功能**：
1. **成本模式选择框**：
   - 在 Sidebar 中新增 "成本模式" 选择框
   - 选项：
     - "演示冰带成本"（demo_icebelt）
     - "真实 SIC 成本（若可用）"（real_sic_if_available）

2. **修改 `plan_three_routes()` 函数**：
   - 新增 `cost_mode` 参数
   - 返回值从 `(routes_info, cost_fields)` 改为 `(routes_info, cost_fields, meta)`
   - meta 包含：
     - `cost_mode`：当前使用的成本模式
     - `real_sic_available`：真实 SIC 是否可用
     - `fallback_reason`：如果回退到 demo，原因是什么

3. **自动回退机制**：
   - 当选择 "real_sic_if_available" 但真实 SIC 不可用时
   - 自动回退到 "demo_icebelt" 模式
   - UI 中显示警告信息

4. **摘要信息更新**：
   - 在方案摘要的 caption 中显示当前成本模式

### 4. 新建：`tests/test_real_env_cost.py`

**测试覆盖**：11 个新测试，分为 3 个测试类

#### TestBuildCostFromSic（4 个测试）
- `test_build_cost_from_sic_shapes_and_monotonic`：验证形状和单调性
- `test_build_cost_from_sic_land_mask_respected`：验证陆地掩码被正确应用
- `test_build_cost_from_sic_with_none_sic`：验证 sic 为 None 时的行为
- `test_build_cost_from_sic_ice_penalty_scaling`：验证 ice_penalty 的缩放效果

#### TestLoadRealSicForGrid（5 个测试）
- `test_load_real_sic_from_tiny_nc`：从小型 NetCDF 加载 SIC
- `test_load_real_sic_missing_file_returns_none`：缺失文件返回 None
- `test_load_real_sic_shape_mismatch_returns_none`：形状不匹配返回 None
- `test_load_real_sic_with_time_dimension`：处理有时间维度的数据
- `test_load_real_sic_auto_scale_0_100`：自动缩放 0..100 的数据

#### TestRealSicCostBreakdown（2 个测试）
- `test_real_sic_cost_breakdown_components`：验证成本分解包含预期组件
- `test_real_sic_vs_demo_cost_difference`：验证真实 SIC 与 demo 成本的差异

## 测试结果

✅ **所有测试通过**：58 个测试（原 47 个 + 新增 11 个）

```
======================== 58 passed, 1 warning in 2.27s ========================
```

## 使用示例

### 在代码中使用真实 SIC 成本

```python
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.env_real import load_real_sic_for_grid
from arcticroute.core.cost import build_cost_from_sic

grid, land_mask = make_demo_grid()

# 尝试加载真实 SIC
env = load_real_sic_for_grid(grid)

if env is not None and env.sic is not None:
    # 使用真实 SIC 构建成本场
    cost_field = build_cost_from_sic(grid, land_mask, env, ice_penalty=4.0)
else:
    # 回退到 demo 成本
    from arcticroute.core.cost import build_demo_cost
    cost_field = build_demo_cost(grid, land_mask)
```

### 在 UI 中使用

1. 启动 UI：`streamlit run run_ui.py`
2. 在左侧 Sidebar 中选择"成本模式"
3. 选择"真实 SIC 成本（若可用）"
4. 如果真实 SIC 数据可用，将使用真实数据；否则自动回退到演示冰带成本

## 关键设计决策

### 1. 优雅的失败机制
- 所有加载函数返回 None 而不是抛异常
- UI 自动检测失败并回退到 demo 模式
- 用户不会遇到崩溃或错误

### 2. 非线性冰风险成本
- 使用 `ice_penalty * sic^1.5` 而不是线性关系
- 这反映了海冰对航行的非线性影响
- 高浓度冰区的成本增长更快

### 3. 完全向后兼容
- 默认行为完全不变（demo 模式）
- 真实 SIC 是可选的增强功能
- 所有现有代码继续工作

### 4. 灵活的数据加载
- 支持多个变量名（sic, SIC, ice_concentration）
- 支持不同的数据维度（2D 或 3D with time）
- 自动数据缩放和范围检查

## 后续扩展建议

1. **支持其他环境变量**：
   - 波浪高度（wave height）
   - 风速（wind speed）
   - 洋流（ocean current）

2. **改进成本函数**：
   - 考虑多个环境因素的组合效应
   - 基于船型的自适应成本

3. **数据管理**：
   - 支持多个时间步长的数据
   - 时间插值和预报

4. **性能优化**：
   - 缓存加载的数据
   - 支持大规模网格的高效处理

## 文件修改清单

| 文件 | 操作 | 行数 |
|------|------|------|
| `arcticroute/core/env_real.py` | 新建 | 150+ |
| `arcticroute/core/cost.py` | 修改 | +70 |
| `arcticroute/ui/planner_minimal.py` | 修改 | +30 |
| `tests/test_real_env_cost.py` | 新建 | 300+ |

## 验证清单

- [x] 所有新代码都有完整的 docstring
- [x] 所有新测试都通过
- [x] 向后兼容性验证通过
- [x] 错误处理完整
- [x] 日志输出清晰
- [x] 代码风格一致

## 总结

Phase 7 成功实现了真实 SIC 成本模式，提供了一个灵活、可靠的框架来集成真实环境数据。系统设计确保了：

1. **可靠性**：优雅的失败机制和完整的错误处理
2. **易用性**：简单的 UI 集成和自动回退
3. **可维护性**：清晰的代码结构和完整的测试覆盖
4. **可扩展性**：易于添加新的环境变量和成本模型

整个实现遵循了"有则用之，无则优雅退回"的原则，确保了系统的健壮性。













