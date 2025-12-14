# Phase 8 快速开始指南

## 新增功能概览

Phase 8 引入了**波浪风险（wave_swh）**支持，允许在路由规划中考虑波浪有效波高。

### 核心改进

| 功能 | Phase 7 | Phase 8 |
|------|--------|--------|
| 成本分量 | base_distance, ice_risk | base_distance, ice_risk, **wave_risk** |
| 环境数据 | sic 只读 | sic + wave_swh |
| 用户控制 | ice_penalty | ice_penalty + **wave_penalty** |
| 成本函数 | build_cost_from_sic() | build_cost_from_real_env() |

---

## 使用方式

### 1. 启动 UI

```bash
streamlit run run_ui.py
```

### 2. 在 Sidebar 中配置

#### 新增滑条：波浪权重
```
风险权重
├─ 波浪权重 (wave_penalty)
   ├─ 范围: 0.0 ~ 10.0
   ├─ 默认: 2.0
   ├─ 步长: 0.5
   └─ 说明: 仅在真实环境数据模式下有效
```

### 3. 选择成本模式

#### 模式 A: Demo 冰带（推荐用于测试）
```
成本模式 = "demo_icebelt"
wave_penalty = 任意值（被忽略）
→ 行为与 Phase 7 完全相同
```

#### 模式 B: 真实 SIC（需要 sic 数据文件）
```
成本模式 = "real_sic_if_available"
wave_penalty = 0.0
→ 只考虑冰风险，不考虑波浪
```

#### 模式 C: 真实 SIC + 波浪（需要 sic 和 wave 数据文件）
```
成本模式 = "real_sic_if_available"
wave_penalty = 2.0 ~ 5.0
→ 同时考虑冰风险和波浪风险
```

---

## 数据准备

### 文件位置

```
$DATA_ROOT/newenv/
├─ ice_copernicus_sic.nc      # SIC 数据（可选）
└─ wave_swh.nc                # 波浪数据（可选）
```

### 数据格式要求

#### SIC 文件
```
变量名候选: "sic", "SIC", "ice_concentration"
维度: (y, x) 或 (time, y, x)
值域: 0..1 或 0..100（自动检测）
```

#### Wave 文件
```
变量名候选: "wave_swh", "swh", "SWH"
维度: (y, x) 或 (time, y, x)
值域: 0..10 米（自动 clip）
```

### 创建示例数据

```python
import numpy as np
import xarray as xr

# 创建 SIC 数据
ny, nx = 100, 150
sic_data = np.random.uniform(0, 1, (ny, nx))
ds_sic = xr.Dataset({
    "sic": (["y", "x"], sic_data),
    "lat": (["y"], np.linspace(60, 85, ny)),
    "lon": (["x"], np.linspace(-30, 60, nx)),
})
ds_sic.to_netcdf("ice_copernicus_sic.nc")

# 创建 Wave 数据
wave_data = np.random.uniform(0, 6, (ny, nx))
ds_wave = xr.Dataset({
    "wave_swh": (["y", "x"], wave_data),
    "lat": (["y"], np.linspace(60, 85, ny)),
    "lon": (["x"], np.linspace(-30, 60, nx)),
})
ds_wave.to_netcdf("wave_swh.nc")
```

---

## 成本分解解读

### 成本分量说明

#### base_distance
- **含义**: 基础距离成本
- **值**: 海洋 1.0，陆地 ∞
- **用途**: 确保路线不穿陆

#### ice_risk
- **含义**: 冰风险成本
- **计算**: ice_penalty × sic^1.5
- **范围**: 0 ~ ice_penalty
- **调节**: ice_penalty 滑条（demo 模式）

#### wave_risk（新增）
- **含义**: 波浪风险成本
- **计算**: wave_penalty × (wave_norm^1.5)
  - wave_norm = wave_swh / 6.0
- **范围**: 0 ~ wave_penalty
- **调节**: wave_penalty 滑条（真实环境模式）

### 成本分解表示例

```
方案: balanced

component          total_contribution    fraction
─────────────────────────────────────────────────
base_distance      150.5                 60.2%
ice_risk           80.3                  32.1%
wave_risk          19.2                  7.7%
─────────────────────────────────────────────────
总成本             250.0                 100%
```

---

## 参数调优建议

### wave_penalty 取值

| 值 | 效果 | 使用场景 |
|----|------|---------|
| 0.0 | 忽略波浪 | 冰风险为主 |
| 1.0 | 轻微考虑 | 波浪辅助因素 |
| 2.0 | 中等考虑 | 平衡考虑（推荐） |
| 5.0 | 重点考虑 | 波浪风险为主 |
| 10.0 | 极端考虑 | 极端天气 |

### ice_penalty 与 wave_penalty 配合

```
低冰险 + 低波浪险:
  ice_penalty = 1.0, wave_penalty = 1.0
  → 快速路由，风险承受度高

平衡方案:
  ice_penalty = 4.0, wave_penalty = 2.0
  → 综合考虑，风险适中

高安全性:
  ice_penalty = 8.0, wave_penalty = 5.0
  → 保守路由，风险承受度低
```

---

## 常见问题

### Q1: 如何只使用 wave 数据，不使用 sic？

**A**: 将 sic 数据文件移除或改名，保留 wave 文件。系统会自动加载可用的数据。

```python
# 代码示例
env = load_real_env_for_grid(grid)
# env.sic = None, env.wave_swh = <数据>
```

### Q2: wave_penalty = 0 时会发生什么？

**A**: wave_risk 分量不被计算，成本分解表中不显示 wave_risk。行为与 Phase 7 完全相同。

### Q3: 如何验证 wave 数据是否被正确加载？

**A**: 查看 UI 的成本分解表。如果有 wave_risk 分量且数值非零，说明加载成功。

```
✓ wave_risk 在 components 中
✓ 数值范围合理（0 ~ wave_penalty）
```

### Q4: 波浪数据缺失时会怎样？

**A**: 系统自动降级，wave_risk = 0，不影响其他分量。

```python
# 自动处理
if env.wave_swh is None:
    wave_risk = 0  # 自动跳过
```

### Q5: 能否同时调节 ice_penalty 和 wave_penalty？

**A**: 可以。UI 中有两个独立的滑条，可以分别调节。

---

## 编程接口

### 加载环境数据

```python
from arcticroute.core.env_real import load_real_env_for_grid
from arcticroute.core.grid import make_demo_grid

grid, _ = make_demo_grid()
env = load_real_env_for_grid(grid)

# 检查数据可用性
if env is None:
    print("数据不可用")
elif env.sic is not None and env.wave_swh is not None:
    print("sic 和 wave 都可用")
elif env.sic is not None:
    print("只有 sic 可用")
elif env.wave_swh is not None:
    print("只有 wave 可用")
```

### 构建成本场

```python
from arcticroute.core.cost import build_cost_from_real_env

# 考虑冰和波浪
cost_field = build_cost_from_real_env(
    grid=grid,
    landmask=landmask,
    env=env,
    ice_penalty=4.0,
    wave_penalty=2.0,
)

# 查看成本分量
print(cost_field.components.keys())
# dict_keys(['base_distance', 'ice_risk', 'wave_risk'])
```

### 规划路线

```python
from arcticroute.ui.planner_minimal import plan_three_routes

routes, fields, meta = plan_three_routes(
    grid=grid,
    land_mask=landmask,
    start_lat=66.0,
    start_lon=5.0,
    end_lat=78.0,
    end_lon=150.0,
    cost_mode="real_sic_if_available",
    wave_penalty=2.0,  # 新参数
)

# 检查元数据
print(f"Real env available: {meta['real_env_available']}")
print(f"Wave penalty: {meta['wave_penalty']}")
```

---

## 测试验证

### 运行单元测试

```bash
# 运行所有测试
pytest

# 运行 wave 相关测试
pytest tests/test_real_env_cost.py::TestBuildCostFromRealEnvWithWave -v

# 运行 load_real_env 测试
pytest tests/test_real_env_cost.py::TestLoadRealEnvForGrid -v
```

### 预期结果

```
66 passed, 1 warning in 2.35s

其中包括:
- 4 个 build_cost_from_real_env wave 测试
- 4 个 load_real_env_for_grid 测试
- 11 个 Phase 7 向后兼容性测试
```

---

## 性能考虑

### 计算复杂度

| 操作 | 复杂度 | 备注 |
|------|--------|------|
| load_real_env_for_grid | O(ny × nx) | 数据加载 |
| build_cost_from_real_env | O(ny × nx) | 成本计算 |
| plan_route | O(ny × nx × log(ny×nx)) | A* 搜索 |

### 内存使用

```
Grid (100×150):
  base_distance: 60 KB
  ice_risk: 60 KB
  wave_risk: 60 KB
  总计: ~180 KB
```

### 优化建议

1. **数据缓存**: 重复使用同一时间步的数据
2. **增量更新**: 仅更新变化的网格点
3. **并行计算**: 多个方案并行规划

---

## 下一步

### 立即可做

- ✅ 准备 wave_swh 数据文件
- ✅ 调试 wave_penalty 参数
- ✅ 验证成本分解结果

### 后续计划

- 🔄 集成更多环保指标
- 🔄 实现时间序列规划
- 🔄 添加天气预报集成

---

## 支持和反馈

如有问题或建议，请参考：
- 完整报告: `PHASE_8_COMPLETION_REPORT.md`
- 代码注释: 各源文件中的详细说明
- 测试用例: `tests/test_real_env_cost.py`











