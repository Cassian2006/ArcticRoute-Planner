# Phase 8 技术细节文档

## 架构设计

### 数据流

```
┌─────────────────────────────────────────────────────────────┐
│                    用户输入 (UI)                              │
│  grid_mode, cost_mode, wave_penalty, ice_penalty            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              load_real_env_for_grid()                        │
│  ├─ 尝试加载 sic (ice_copernicus_sic.nc)                    │
│  ├─ 尝试加载 wave_swh (wave_swh.nc)                         │
│  └─ 返回 RealEnvLayers(sic=..., wave_swh=...)              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│            build_cost_from_real_env()                        │
│  ├─ base_distance = 1.0 (ocean) / inf (land)               │
│  ├─ ice_risk = ice_penalty × sic^1.5 (if sic available)    │
│  ├─ wave_risk = wave_penalty × (wave_norm^1.5)             │
│  │              (if wave available and wave_penalty > 0)    │
│  └─ cost = base_distance + ice_risk + wave_risk            │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              plan_route_latlon()                             │
│  A* 搜索，找到最低成本路径                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│         compute_route_cost_breakdown()                       │
│  沿路径计算各分量的贡献                                      │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              UI 显示结果                                      │
│  ├─ 地图上显示路线                                           │
│  ├─ 摘要表格                                                 │
│  └─ 成本分解表 (包括 wave_risk)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 核心数据结构

### RealEnvLayers

```python
@dataclass
class RealEnvLayers:
    """真实环境层集合"""
    sic: Optional[np.ndarray]        # shape=(ny,nx), 值域[0,1]
    wave_swh: Optional[np.ndarray] = None  # shape=(ny,nx), 值域[0,10]
```

**设计考虑**:
- 两个字段都是可选的，支持灵活组合
- wave_swh 默认为 None，保持向后兼容
- 数据类型统一为 np.ndarray，便于数值计算

### CostField

```python
@dataclass
class CostField:
    """成本场"""
    grid: Grid2D
    cost: np.ndarray              # shape=(ny,nx), 总成本
    land_mask: np.ndarray         # shape=(ny,nx), bool
    components: Dict[str, np.ndarray]  # 成本分量分解
```

**components 示例**:
```python
{
    "base_distance": np.array(...),  # 必有
    "ice_risk": np.array(...),       # 必有
    "wave_risk": np.array(...),      # 可选，仅当有 wave 数据且 wave_penalty > 0
}
```

---

## 算法细节

### 1. 数据加载 (load_real_env_for_grid)

#### SIC 加载流程

```python
def load_real_sic(nc_path, grid):
    ds = xr.open_dataset(nc_path)
    
    # 尝试找变量
    for var_name in ["sic", "SIC", "ice_concentration"]:
        if var_name in ds:
            sic_da = ds[var_name]
            break
    
    # 处理维度
    sic = sic_da.values
    if sic.ndim == 3:  # (time, y, x)
        sic = sic[time_index, :, :]
    
    # 形状验证
    if sic.shape != (ny, nx):
        return None
    
    # 数据处理
    sic = np.asarray(sic, dtype=float)
    if np.nanmax(sic) > 1.5:  # 假设 0..100 范围
        sic = sic / 100.0
    sic = np.clip(sic, 0.0, 1.0)
    
    return sic
```

#### Wave 加载流程

```python
def load_wave(nc_path, grid):
    # 类似 SIC，但目标范围是 [0, 10]
    wave = ...
    wave = np.clip(wave, 0.0, 10.0)
    return wave
```

**关键特性**:
- ✅ 自动检测 0..100 vs 0..1 范围
- ✅ 自动处理 (time, y, x) 维度
- ✅ 形状不匹配时返回 None（不报错）
- ✅ 失败时打印日志，不中断程序

### 2. 成本计算 (build_cost_from_real_env)

#### 成本函数

```python
def build_cost_from_real_env(grid, landmask, env, ice_penalty, wave_penalty):
    ny, nx = grid.shape()
    
    # 基础距离成本
    base_distance = np.ones((ny, nx))
    base_distance = np.where(landmask, np.inf, base_distance)
    
    # 冰风险成本
    ice_risk = np.zeros((ny, nx))
    if env.sic is not None:
        sic = np.clip(env.sic, 0.0, 1.0)
        ice_risk = ice_penalty * np.power(sic, 1.5)
    
    # 波浪风险成本
    wave_risk = np.zeros((ny, nx))
    if env.wave_swh is not None and wave_penalty > 0:
        wave = np.clip(env.wave_swh, 0.0, 10.0)
        wave_norm = wave / 6.0  # 归一化
        wave_risk = wave_penalty * np.power(wave_norm, 1.5)
    
    # 总成本
    cost = base_distance + ice_risk + wave_risk
    cost = np.where(landmask, np.inf, cost)
    
    return CostField(..., components={...})
```

#### 非线性放大

**为什么使用 ^1.5 的幂次？**

```
sic^1.5 和 wave_norm^1.5 的效果:
- 低值 (0..0.5): 增长缓慢，低风险区域差异小
- 中值 (0.5..0.8): 增长加速，中等风险区域差异明显
- 高值 (0.8..1.0): 增长快速，高风险区域强烈避免

示例:
  sic = 0.3 → ice_risk = 4.0 × 0.3^1.5 = 0.65
  sic = 0.5 → ice_risk = 4.0 × 0.5^1.5 = 1.41
  sic = 0.8 → ice_risk = 4.0 × 0.8^1.5 = 2.86
  sic = 1.0 → ice_risk = 4.0 × 1.0^1.5 = 4.00
```

**波浪归一化**:
```
wave_swh 范围: [0, 10] 米
max_wave = 6.0 米（参考值）
wave_norm = min(wave_swh / 6.0, 1.0)

原因: 波浪有效波高 6 米已经是较大的浪，
      超过 6 米的情况相对罕见，
      归一化到 [0, 1] 便于与 sic 的权重平衡
```

### 3. 路由规划 (plan_route_latlon)

```python
def plan_route_latlon(cost_field, start_lat, start_lon, end_lat, end_lon):
    # 转换为网格坐标
    start_idx = grid.latlon_to_idx(start_lat, start_lon)
    end_idx = grid.latlon_to_idx(end_lat, end_lon)
    
    # A* 搜索
    path_idx = astar(cost_field.cost, start_idx, end_idx)
    
    # 转换回经纬度
    path_latlon = [grid.idx_to_latlon(idx) for idx in path_idx]
    
    return path_latlon
```

**A* 启发式函数**:
```python
def heuristic(current, goal):
    # 使用 Haversine 距离作为启发式
    return haversine_distance(current, goal)
```

---

## 向后兼容性设计

### 1. RealEnvLayers 扩展

```python
# Phase 7 代码仍然有效
env = RealEnvLayers(sic=sic_data)
# wave_swh 默认为 None，不影响现有代码

# Phase 8 新代码
env = RealEnvLayers(sic=sic_data, wave_swh=wave_data)
```

### 2. build_cost_from_sic 包装

```python
# Phase 7 代码仍然有效
cost = build_cost_from_sic(grid, landmask, env, ice_penalty=4.0)

# 内部实现
def build_cost_from_sic(...):
    return build_cost_from_real_env(
        ..., ice_penalty=ice_penalty, wave_penalty=0.0
    )
```

### 3. plan_three_routes 扩展

```python
# Phase 7 代码仍然有效
routes, fields, meta = plan_three_routes(
    grid, landmask, start_lat, start_lon, end_lat, end_lon
)

# Phase 8 新参数
routes, fields, meta = plan_three_routes(
    grid, landmask, start_lat, start_lon, end_lat, end_lon,
    wave_penalty=2.0  # 新参数，默认为 0.0
)
```

---

## 测试策略

### 单元测试覆盖

#### 1. 数据加载测试

```python
# 测试 load_real_env_for_grid
✓ 同时加载 sic 和 wave
✓ 只加载 sic
✓ 只加载 wave
✓ 两者都缺失返回 None
✓ 形状不匹配处理
✓ 时间维度处理
```

#### 2. 成本计算测试

```python
# 测试 build_cost_from_real_env
✓ wave_risk 正确添加
✓ wave_penalty=0 时不添加 wave_risk
✓ wave_swh=None 时不添加 wave_risk
✓ wave_penalty 线性影响
✓ 陆地掩码尊重
✓ 成本单调性
```

#### 3. 向后兼容性测试

```python
# 测试 Phase 7 功能
✓ build_cost_from_sic 行为不变
✓ load_real_sic_for_grid 行为不变
✓ plan_three_routes 默认行为不变
✓ 所有 Phase 7 测试通过
```

### 集成测试

```python
# 端到端测试
✓ UI 导入成功
✓ plan_three_routes 接受 wave_penalty
✓ 成本分解表显示 wave_risk
✓ 路线规划结果合理
```

---

## 性能分析

### 时间复杂度

| 操作 | 复杂度 | 说明 |
|------|--------|------|
| load_real_env_for_grid | O(ny × nx) | 数据 I/O 和处理 |
| build_cost_from_real_env | O(ny × nx) | 矩阵运算 |
| plan_route_latlon | O(ny × nx × log(ny×nx)) | A* 搜索 |
| compute_route_cost_breakdown | O(路径长度) | 沿路径求和 |

### 空间复杂度

```
Grid 100×150:
  base_distance: 100 × 150 × 8 bytes = 120 KB
  ice_risk: 100 × 150 × 8 bytes = 120 KB
  wave_risk: 100 × 150 × 8 bytes = 120 KB
  ─────────────────────────────────────
  总计: ~360 KB
```

### 优化机会

1. **数据缓存**: 避免重复加载相同的 nc 文件
2. **增量更新**: 仅更新变化的网格点
3. **GPU 加速**: 使用 CuPy 加速矩阵运算
4. **并行规划**: 三个方案并行计算

---

## 错误处理

### 数据加载失败

```python
# 情况 1: 文件不存在
result = load_real_env_for_grid(grid)
# → 打印 "[ENV] real SIC not available: file not found at ..."
# → 返回 None

# 情况 2: 形状不匹配
result = load_real_env_for_grid(grid)
# → 打印 "[ENV] real SIC not available: sic shape (50, 100) != grid shape (100, 150)"
# → 返回 None

# 情况 3: 变量不存在
result = load_real_env_for_grid(grid)
# → 打印 "[ENV] real SIC not available: no variable found in ..."
# → 返回 None
```

### 成本计算异常

```python
# 情况 1: sic 形状不匹配
cost = build_cost_from_real_env(grid, landmask, env, ...)
# → 打印 "[COST] warning: sic shape ... != grid shape ..."
# → ice_risk = 0（不中断）

# 情况 2: wave 形状不匹配
cost = build_cost_from_real_env(grid, landmask, env, ...)
# → 打印 "[COST] warning: wave shape ... != grid shape ..."
# → wave_risk = 0（不中断）
```

### UI 降级

```python
# 情况: 真实环境数据不可用
cost_mode = "real_sic_if_available"
real_env = load_real_env_for_grid(grid)  # 返回 None

# UI 自动降级
if real_env is None:
    st.warning("真实环境数据不可用，自动回退为演示冰带成本。")
    cost_field = build_demo_cost(...)  # 使用 demo 成本
```

---

## 扩展点

### 1. 添加新的环保指标

```python
# 示例: 添加风速风险
@dataclass
class RealEnvLayers:
    sic: Optional[np.ndarray]
    wave_swh: Optional[np.ndarray] = None
    wind_speed: Optional[np.ndarray] = None  # 新增

# 在 build_cost_from_real_env 中添加
if env.wind_speed is not None and wind_penalty > 0:
    wind_norm = wind_speed / max_wind
    wind_risk = wind_penalty * np.power(wind_norm, 1.5)
    components["wind_risk"] = wind_risk
```

### 2. 时间序列规划

```python
# 支持多个时间步
def plan_route_with_time_series(
    grid, landmask, env_list, start, end, time_steps
):
    # env_list[t] = RealEnvLayers at time t
    # 规划考虑时间演变
    pass
```

### 3. 动态权重调整

```python
# 根据路线特性动态调整权重
def adaptive_wave_penalty(route, env):
    # 如果路线经过高浪区，增加 wave_penalty
    # 如果路线避开高浪区，减少 wave_penalty
    pass
```

---

## 调试技巧

### 1. 打印成本场

```python
import matplotlib.pyplot as plt

cost_field = build_cost_from_real_env(...)

# 绘制总成本
plt.imshow(cost_field.cost, cmap='viridis')
plt.colorbar(label='Cost')
plt.title('Total Cost Field')
plt.show()

# 绘制各分量
for comp_name, comp_data in cost_field.components.items():
    plt.imshow(comp_data, cmap='viridis')
    plt.colorbar(label=comp_name)
    plt.title(f'{comp_name} Component')
    plt.show()
```

### 2. 检查数据加载

```python
env = load_real_env_for_grid(grid)

print(f"SIC available: {env.sic is not None}")
if env.sic is not None:
    print(f"  Shape: {env.sic.shape}")
    print(f"  Range: [{np.nanmin(env.sic):.3f}, {np.nanmax(env.sic):.3f}]")

print(f"Wave available: {env.wave_swh is not None}")
if env.wave_swh is not None:
    print(f"  Shape: {env.wave_swh.shape}")
    print(f"  Range: [{np.nanmin(env.wave_swh):.3f}, {np.nanmax(env.wave_swh):.3f}]")
```

### 3. 验证成本分解

```python
breakdown = compute_route_cost_breakdown(grid, cost_field, route)

print(f"Total cost: {breakdown.total_cost:.2f}")
for comp_name, comp_total in breakdown.component_totals.items():
    fraction = breakdown.component_fractions[comp_name]
    print(f"  {comp_name}: {comp_total:.2f} ({fraction:.1%})")
```

---

## 参考资源

- **论文**: 北极航线成本模型研究
- **数据源**: Copernicus 海冰浓度数据
- **标准**: WMO 波浪有效波高定义
- **工具**: xarray, numpy, scipy

---

## 版本历史

| 版本 | 日期 | 主要改进 |
|------|------|---------|
| Phase 7 | 2025-12-08 | 真实 SIC 成本模式 |
| Phase 8 | 2025-12-08 | 添加波浪风险支持 |
| Phase 9 | 计划中 | 多时间步规划 |
| Phase 10 | 计划中 | 实时天气集成 |

















