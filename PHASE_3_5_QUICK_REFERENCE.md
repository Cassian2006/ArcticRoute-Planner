# Phase 3.5 快速参考指南

## 快速开始

### 1. 运行所有测试
```bash
cd C:\Users\sgddsf\Desktop\AR_final
python -m pytest tests/ -v
```

### 2. 启动 UI
```bash
cd C:\Users\sgddsf\Desktop\AR_final
streamlit run run_ui.py
```

## 核心 API

### 导入
```python
from arcticroute.core.landmask import (
    evaluate_route_against_landmask,
    RouteLandmaskStats,
)
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_demo_cost
from arcticroute.core.astar import plan_route_latlon
```

### 基本使用

```python
# 1. 创建网格和陆地掩码
grid, land_mask = make_demo_grid()

# 2. 构建成本场
cost_field = build_demo_cost(grid, land_mask, ice_penalty=4.0)

# 3. 规划路线
route = plan_route_latlon(
    cost_field=cost_field,
    start_lat=66.0,
    start_lon=5.0,
    end_lat=78.0,
    end_lon=150.0,
    neighbor8=True,
)

# 4. 检查路线是否踩陆
stats = evaluate_route_against_landmask(grid, land_mask, route)

# 5. 查看统计信息
print(f"总步数: {stats.total_steps}")
print(f"踩陆步数: {stats.on_land_steps}")
print(f"海上步数: {stats.on_ocean_steps}")
print(f"第一次踩陆: {stats.first_land_latlon}")
```

## RouteLandmaskStats 数据结构

```python
@dataclass
class RouteLandmaskStats:
    total_steps: int                           # 路线总步数
    on_land_steps: int                         # 踩陆步数
    on_ocean_steps: int                        # 海上步数
    first_land_index: int | None               # 第一次踩陆的索引
    first_land_latlon: Tuple[float, float] | None  # 第一次踩陆的坐标
```

## 常见场景

### 场景 1: 验证路线不踩陆
```python
stats = evaluate_route_against_landmask(grid, land_mask, route)
assert stats.on_land_steps == 0, "路线不应该踩陆"
```

### 场景 2: 找到第一个踩陆点
```python
stats = evaluate_route_against_landmask(grid, land_mask, route)
if stats.first_land_latlon:
    print(f"第一个踩陆点: {stats.first_land_latlon}")
    print(f"位置索引: {stats.first_land_index}")
```

### 场景 3: 统计踩陆比例
```python
stats = evaluate_route_against_landmask(grid, land_mask, route)
land_ratio = stats.on_land_steps / stats.total_steps if stats.total_steps > 0 else 0
print(f"踩陆比例: {land_ratio * 100:.2f}%")
```

## UI 功能

### 表格列说明

| 列名 | 说明 |
|------|------|
| 方案 | 路线方案名称（efficient/balanced/safe） |
| 可达 | 是否成功规划（✓/✗） |
| 路径点数 | 路线包含的点数 |
| 粗略距离_km | 路线的大圆距离 |
| 冰带权重 | 冰带成本权重 |
| 允许对角线 | 是否允许 8 邻接 |
| **on_land_steps** | **新增：踩陆步数** |
| **on_ocean_steps** | **新增：海上步数** |

### 提示条说明

- 🟢 **绿色成功提示**: "根据当前 landmask，三条路线均未踩陆（demo 世界下行为正常）"
  - 表示所有路线都不踩陆，符合预期

- 🔴 **红色错误提示**: "警告：根据当前 landmask，有路线踩到了陆地，请检查成本场或掩码数据"
  - 表示至少有一条路线踩到了陆地，需要调查原因

## 测试用例

### 新增测试 (3 个)

1. **test_demo_routes_do_not_cross_land**
   - 验证三条 demo 路线都不踩陆

2. **test_empty_route**
   - 验证空路线的处理

3. **test_route_with_single_point**
   - 验证单点路线的分类

### 运行特定测试
```bash
# 运行新增的陆地一致性测试
python -m pytest tests/test_route_landmask_consistency.py -v

# 运行所有测试
python -m pytest tests/ -v

# 运行单个测试
python -m pytest tests/test_route_landmask_consistency.py::test_demo_routes_do_not_cross_land -v
```

## 故障排除

### 问题 1: 路线踩陆但不应该踩陆

**可能原因**:
- 成本场设置不正确（陆地成本应为 inf）
- 陆地掩码数据有误
- 起止点映射到了陆地

**解决方案**:
```python
# 检查成本场
print(f"陆地成本: {cost_field.cost[land_mask]}")  # 应该全是 inf

# 检查起止点映射
from arcticroute.core.astar import _nearest_ocean_cell
start_ij = _nearest_ocean_cell(grid, land_mask, 66.0, 5.0)
print(f"起点映射到: {start_ij}")
```

### 问题 2: 路线为空

**可能原因**:
- 起止点无法映射到海洋格点
- 起止点之间不可达

**解决方案**:
```python
# 检查起止点是否能映射到海洋
start_ij = _nearest_ocean_cell(grid, land_mask, 66.0, 5.0)
end_ij = _nearest_ocean_cell(grid, land_mask, 78.0, 150.0)
if start_ij is None or end_ij is None:
    print("起止点无法映射到海洋格点")
```

## 性能提示

- `evaluate_route_against_landmask()` 的时间复杂度为 O(n)，其中 n 是路线点数
- 对于大规模路线（>10000 点），建议使用向量化操作优化
- 当前实现使用最近邻映射，精度取决于网格分辨率

## 扩展建议

1. **可视化踩陆点**: 在地图上标记踩陆的路线段
2. **详细报告**: 导出踩陆统计报告
3. **自动修复**: 当检测到踩陆时自动调整路线
4. **多掩码支持**: 支持多个陆地掩码层
5. **性能优化**: 使用 KD-Tree 加速坐标映射

## 相关文件

- 核心实现: `arcticroute/core/landmask.py`
- UI 集成: `arcticroute/ui/planner_minimal.py`
- 测试: `tests/test_route_landmask_consistency.py`
- 完整文档: `PHASE_3_5_IMPLEMENTATION.md`













