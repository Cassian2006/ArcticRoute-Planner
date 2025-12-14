# Phase 7 快速开始指南

## 新增模块概览

### 1. `arcticroute/core/env_real.py`

**主要 API**：

```python
from arcticroute.core.env_real import RealEnvLayers, load_real_sic_for_grid

# 加载真实 SIC 数据
env = load_real_sic_for_grid(grid)
if env is not None:
    print(f"SIC shape: {env.sic.shape}")
    print(f"SIC range: [{env.sic.min():.3f}, {env.sic.max():.3f}]")
```

### 2. `arcticroute/core/cost.py` - 新增函数

**主要 API**：

```python
from arcticroute.core.cost import build_cost_from_sic

# 使用真实 SIC 构建成本场
cost_field = build_cost_from_sic(
    grid=grid,
    land_mask=land_mask,
    env=env,
    ice_penalty=4.0  # 可调参数
)

# 访问成本分解
print(cost_field.components)  # {'base_distance': ..., 'ice_risk': ...}
```

### 3. `arcticroute/ui/planner_minimal.py` - 修改

**新增参数**：

```python
from arcticroute.ui.planner_minimal import plan_three_routes

routes_info, cost_fields, meta = plan_three_routes(
    grid, land_mask,
    start_lat=66.0, start_lon=5.0,
    end_lat=78.0, end_lon=150.0,
    allow_diag=True,
    cost_mode="real_sic_if_available"  # 新参数
)

# 检查是否使用了真实 SIC
if meta['real_sic_available']:
    print("Using real SIC data")
else:
    print(f"Fallback reason: {meta['fallback_reason']}")
```

## 使用场景

### 场景 1：Demo 模式（默认）

```python
# 使用演示冰带成本（原有行为）
routes_info, cost_fields, meta = plan_three_routes(
    grid, land_mask,
    start_lat=66.0, start_lon=5.0,
    end_lat=78.0, end_lon=150.0,
    cost_mode="demo_icebelt"  # 显式指定
)
```

### 场景 2：真实 SIC 模式（有数据时）

```python
# 尝试使用真实 SIC，无数据时自动回退
routes_info, cost_fields, meta = plan_three_routes(
    grid, land_mask,
    start_lat=66.0, start_lon=5.0,
    end_lat=78.0, end_lon=150.0,
    cost_mode="real_sic_if_available"  # 新模式
)

# 检查实际使用的模式
print(f"Cost mode: {meta['cost_mode']}")
print(f"Real SIC available: {meta['real_sic_available']}")
```

### 场景 3：直接使用真实 SIC 成本

```python
from arcticroute.core.env_real import load_real_sic_for_grid
from arcticroute.core.cost import build_cost_from_sic

# 加载真实 SIC
env = load_real_sic_for_grid(grid)

if env is not None and env.sic is not None:
    # 使用真实 SIC
    cost_field = build_cost_from_sic(grid, land_mask, env, ice_penalty=4.0)
else:
    # 回退到 demo
    from arcticroute.core.cost import build_demo_cost
    cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)

# 规划路线
from arcticroute.core.astar import plan_route_latlon
path = plan_route_latlon(cost_field, start_lat, start_lon, end_lat, end_lon)
```

## UI 使用步骤

1. **启动 UI**：
   ```bash
   streamlit run run_ui.py
   ```

2. **选择成本模式**：
   - 在左侧 Sidebar 找到"成本模式"选择框
   - 选择"演示冰带成本"或"真实 SIC 成本（若可用）"

3. **查看结果**：
   - 如果选择"真实 SIC 成本"但数据不可用，会显示警告
   - 自动回退到演示冰带成本
   - 方案摘要会显示当前使用的成本模式

## 数据文件约定

真实 SIC 数据的默认路径：
```
ArcticRoute_data_backup/data_processed/newenv/ice_copernicus_sic.nc
```

**文件要求**：
- 格式：NetCDF 4
- 变量名：`sic`、`SIC` 或 `ice_concentration`（自动检测）
- 坐标：`lat` 和 `lon`（支持 1D 或 2D）
- 可选：`time` 维度（默认取第 0 个时间步）
- 值域：0..1 或 0..100（自动检测和缩放）

## 测试验证

运行所有测试：
```bash
pytest -xvs
```

运行仅 Phase 7 相关的测试：
```bash
pytest -xvs tests/test_real_env_cost.py
```

## 常见问题

### Q1：如果真实 SIC 文件不存在怎么办？
A：系统会自动回退到演示冰带成本，UI 会显示警告信息。

### Q2：如何自定义 SIC 文件路径？
A：在代码中显式传入 `nc_path` 参数：
```python
env = load_real_sic_for_grid(grid, nc_path="/path/to/custom/sic.nc")
```

### Q3：如何调整冰风险的权重？
A：使用 `ice_penalty` 参数：
```python
cost_field = build_cost_from_sic(grid, land_mask, env, ice_penalty=8.0)
```

### Q4：成本分解中的 ice_risk 是如何计算的？
A：使用公式 `ice_risk = ice_penalty * sic^1.5`，其中 sic 的值域为 0..1。

### Q5：如何在自己的代码中使用真实 SIC？
A：参考"场景 3"中的代码示例。

## 性能提示

1. **首次加载**：加载 NetCDF 文件可能需要几秒钟
2. **缓存**：考虑在应用中缓存加载的 SIC 数据
3. **大文件**：如果 SIC 文件很大，可能需要优化内存使用

## 后续开发

- [ ] 支持多个时间步长
- [ ] 添加其他环境变量（波浪、风速等）
- [ ] 实现数据缓存机制
- [ ] 支持自定义成本函数

## 相关文件

- `arcticroute/core/env_real.py` - 环境数据加载
- `arcticroute/core/cost.py` - 成本构建
- `arcticroute/ui/planner_minimal.py` - UI 集成
- `tests/test_real_env_cost.py` - 单元测试
- `PHASE_7_SUMMARY.md` - 详细总结













