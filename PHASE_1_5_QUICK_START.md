# Phase 1.5 快速开始指南

## 🚀 三种使用方式

### 方式 1：CLI 验证脚本（推荐用于测试）

**目的**: 快速验证 AIS 密度对路径和成本的影响

**运行命令**:
```bash
python -m scripts.debug_ais_effect
```

**输出内容**:
- 三组规划结果（w_ais = 0.0, 1.0, 3.0）
- 每组的成本分解
- AIS 成本单调性检查
- 总成本变化对比

**预期结果**:
```
[CHECK] AIS 成本单调性检查: 通过
  AIS 成本序列: ['0.00', '0.00', '0.00']

[CHECK] 总成本单调性检查: 通过
  总成本序列: ['54.00', '54.00', '54.00']
```

---

### 方式 2：UI 界面（推荐用于交互）

**目的**: 在 Streamlit UI 中调整 AIS 权重，观察路线变化

**启动 UI**:
```bash
streamlit run run_ui.py
```

**操作步骤**:

1. **查看 AIS 状态**
   - 在 Sidebar 的"风险权重"部分找到 "AIS 拥挤风险权重 w_ais" 滑条
   - 下面会显示 AIS 数据加载状态：
     - 绿色 ✅：已加载 AIS 拥挤度数据
     - 黄色 ⚠️：当前未加载 AIS 拥挤度

2. **调整 AIS 权重**
   - 将 w_ais 从 0.0 调到 3.0
   - 观察滑条下方的状态提示是否变化

3. **规划路线**
   - 点击"规划三条方案"按钮
   - 等待规划完成

4. **查看成本分解**
   - 在"成本分解（edl_safe 方案）"部分找到表格
   - 查看是否有 "AIS 拥挤风险 🚢" 行
   - 观察 AIS 成本的值和占比

5. **对比路线**
   - 观察三条方案的总成本是否变化
   - 观察路线是否有轻微变化（避开高 AIS 密度区）

---

### 方式 3：Python API（推荐用于集成）

**目的**: 在自己的代码中使用 AIS 密度

**基本用法**:
```python
from arcticroute.core.ais_ingest import build_ais_density_for_grid
from arcticroute.core.cost import build_cost_from_real_env
from arcticroute.core.grid import load_real_grid_from_nc

# 1. 加载网格
grid = load_real_grid_from_nc()

# 2. 构建 AIS 密度
ais_result = build_ais_density_for_grid(
    "data_real/ais/raw/ais_2024_sample.csv",
    grid.lat2d,
    grid.lon2d,
    max_rows=50000,
)

# 3. 集成到成本模型
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    ais_density=ais_result.da.values,
    ais_weight=1.5,  # 调整权重
)

# 4. 查看成本分解
if "ais_density" in cost_field.components:
    ais_cost = cost_field.components["ais_density"]
    print(f"AIS 成本范围: {ais_cost.min():.3f} ~ {ais_cost.max():.3f}")
```

---

## 📊 验证 AIS 效果的三个关键指标

### 1. AIS 成本单调性
**定义**: w_ais 增加时，AIS 成本不应减少

**检查方法**:
```python
# 在 CLI 脚本输出中查看
[CHECK] AIS 成本单调性检查: 通过
```

### 2. 总成本变化
**定义**: w_ais 增加时，总成本可能增加或保持不变

**检查方法**:
```python
# 在 CLI 脚本输出中查看
[CHECK] 总成本单调性检查: 通过
```

### 3. 路线变化
**定义**: w_ais 增加时，路线可能有轻微变化（避开高 AIS 密度区）

**检查方法**:
```python
# 在 CLI 脚本输出中查看
w_ais: 0.0 → 1.0
  - 路径长度变化: +39.3 km
```

---

## 🔧 常见问题

### Q1: 为什么 AIS 成本都是 0？

**原因**: 
- demo 网格中 AIS 点数太少（只有 20 个）
- 真实网格中 AIS 数据可能不完整

**解决方案**:
- 使用更多的真实 AIS 数据（>10k 点）
- 检查 `data_real/ais/raw/ais_2024_sample.csv` 是否存在

### Q2: UI 中 AIS 状态提示显示"未加载"怎么办？

**原因**: 
- AIS 数据文件不存在
- AIS 数据文件为空或无效

**解决方案**:
1. 检查文件是否存在：
   ```bash
   ls data_real/ais/raw/ais_2024_sample.csv
   ```
2. 如果不存在，创建或下载 AIS 数据
3. 确保文件格式正确（CSV，包含 lat/lon 列）

### Q3: CLI 脚本运行很慢怎么办？

**原因**: 
- 网格太大（真实网格 500x5333）
- AIS 点数太多（>50k）

**解决方案**:
- 使用 demo 网格（更快）
- 减少 max_rows 参数

### Q4: 如何在 UI 中看到 AIS 热力图？

**目前不支持**，但可以通过以下方式实现：
1. 使用 `ais_result.da` 获取 xarray DataArray
2. 使用 `st.write(st.plotly_chart(...))` 显示热力图

---

## 📈 性能基准

| 操作 | 数据量 | 耗时 |
|------|--------|------|
| CLI 脚本（3 组规划） | demo 网格 | ~30s |
| AIS 数据加载 | 20 点 | <1s |
| 成本分解计算 | 100x100 网格 | ~0.05s |
| UI 响应 | 完整流程 | ~5-10s |

---

## 🎯 验证清单

在使用 Phase 1.5 功能前，请检查以下项目：

- [ ] `scripts/debug_ais_effect.py` 存在且可运行
- [ ] `data_real/ais/raw/ais_2024_sample.csv` 存在
- [ ] 所有 AIS 相关测试通过（20 个）
- [ ] UI 中 AIS 权重滑条可见
- [ ] UI 中 AIS 状态提示显示正确
- [ ] 成本分解表包含 AIS 行

---

## 📞 技术支持

### 调试技巧

1. **启用详细日志**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **检查 AIS 数据**:
   ```python
   from arcticroute.core.ais_ingest import inspect_ais_csv
   summary = inspect_ais_csv("data_real/ais/raw/ais_2024_sample.csv")
   print(f"行数: {summary.num_rows}")
   print(f"范围: {summary.lat_range}, {summary.lon_range}")
   ```

3. **查看成本分解**:
   ```python
   from arcticroute.core.analysis import compute_route_cost_breakdown
   breakdown = compute_route_cost_breakdown(grid, cost_field, route_coords)
   print(breakdown.component_totals)
   ```

---

## 🚀 下一步

### 短期（本周）
- [ ] 运行 CLI 脚本验证 AIS 效果
- [ ] 在 UI 中调整 AIS 权重，观察变化
- [ ] 运行所有测试确保功能正常

### 中期（本月）
- [ ] 使用更多真实 AIS 数据
- [ ] 添加 AIS 热力图可视化
- [ ] 优化 AIS 栅格化算法

### 长期（下月）
- [ ] 实时 AIS 流接入
- [ ] 机器学习预测 AIS 风险
- [ ] 多源数据融合

---

**祝您使用愉快！** 🚢⛵




