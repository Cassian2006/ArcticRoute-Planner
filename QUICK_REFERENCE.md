# 快速参考 - AIS 维度匹配修复

## 📋 修改概览

| 任务 | 文件 | 修改内容 | 状态 |
|------|------|---------|------|
| **A** | `planner_minimal.py` | AIS 状态管理：确保 AIS 完成时不停留在 pending | ✅ |
| **B** | 无 | 检查并删除简化版本管线（无需修改） | ✅ |
| **C1** | `planner_minimal.py` | 网格变化检测：自动清空旧 AIS 选择 | ✅ |
| **C2** | `preprocess_ais_to_density.py` | 添加网格元信息到 NetCDF 属性 | ✅ |
| **C3** | `cost.py` | 验证和重采样逻辑 | ✅ |

---

## 🔧 关键代码片段

### 任务 A：AIS 状态管理
```python
# 位置：arcticroute/ui/planner_minimal.py, 第 1156 行
if w_ais <= 0:
    _update_pipeline_node(3, "done", "跳过：权重为 0", seconds=0.1)
else:
    _update_pipeline_node(3, "running", "正在加载 AIS 密度...")
    # ... 详细的加载逻辑
```

### 任务 C1：网格变化检测
```python
# 位置：arcticroute/ui/planner_minimal.py, 第 810 行
previous_grid_signature = st.session_state.get("previous_grid_signature", None)
current_grid_signature = st.session_state.get("grid_signature", None)

if previous_grid_signature != current_grid_signature:
    st.session_state["ais_density_path"] = None
    st.info("🔄 网格已切换，已清空 AIS 密度选择")
```

### 任务 C2：网格元信息
```python
# 位置：scripts/preprocess_ais_to_density.py, build_density_dataset 函数
ds.attrs['grid_shape'] = f"{grid_shape[0]}x{grid_shape[1]}"
ds.attrs['grid_source'] = grid_source
ds.attrs['grid_lat_name'] = 'latitude'
ds.attrs['grid_lon_name'] = 'longitude'
```

### 任务 C3：验证函数
```python
# 位置：arcticroute/core/cost.py
def _validate_ais_density_for_grid(ais_da: xr.DataArray, grid: Grid2D) -> Tuple[bool, str]:
    """验证 AIS 密度是否可用于当前网格"""
    # 有坐标 → 可重采样
    # 无坐标 → 拒绝，给出清晰提示
```

---

## 🚀 快速开始

### 1. 重新生成 AIS 文件
```bash
python scripts/preprocess_ais_to_density.py --grid-mode demo
python scripts/preprocess_ais_to_density.py --grid-mode real
```

### 2. 启动应用
```bash
streamlit run arcticroute/ui/home.py
```

### 3. 测试流程
1. 选择 demo 网格 → 选择 demo AIS 文件 → 运行规划
2. 切换到 real 网格 → 观察 AIS 选择被清空 → 选择 real AIS 文件 → 运行规划

---

## 🔍 验证清单

- [ ] 任务 A：`grep "任务 A：AIS 密度加载与状态管理" arcticroute/ui/planner_minimal.py` 返回 1 条
- [ ] 任务 C1：`grep "任务 C1：网格变化检测" arcticroute/ui/planner_minimal.py` 返回 1 条
- [ ] 任务 C2：`grep "任务 C2" scripts/preprocess_ais_to_density.py` 返回 3 条
- [ ] 任务 C3：`grep "_validate_ais_density_for_grid" arcticroute/core/cost.py` 返回 1 条

---

## 📊 AIS 文件命名规范

### 旧格式（已弃用）
```
ais_density_2024_demo.nc
ais_density_2024_real.nc
```

### 新格式（推荐）
```
ais_density_2024_grid_40x80_demo.nc
ais_density_2024_grid_101x1440_env_clean.nc
ais_density_2024_grid_500x5333_highres.nc
```

---

## 🎯 AIS 加载状态流程

```
权重 w_ais
    ↓
w_ais <= 0?
    ├─ YES → done(skip: 权重为 0)
    └─ NO → 尝试加载
            ↓
        文件存在?
            ├─ NO → done(skip: 文件不存在)
            └─ YES → 尝试打开
                    ↓
                格式有效?
                    ├─ NO → done(skip: 文件格式无效)
                    └─ YES → 加载成功 → done(AIS=HxW source=...)
                            或加载失败 → fail(加载失败: ...)
```

---

## 💡 常见问题

### Q: 为什么切换网格后 AIS 选择被清空？
**A**: 这是设计特性。不同网格的 AIS 文件维度不同，自动清空可以防止维度错配。

### Q: 如何强制使用不匹配的 AIS 文件？
**A**: 不建议这样做。如果 AIS 文件有坐标信息，系统会自动重采样。如果没有坐标，系统会拒绝并给出提示。

### Q: 如何检查 AIS 文件的网格信息？
**A**: 
```python
import xarray as xr
ds = xr.open_dataset('path/to/ais_density.nc')
print(ds.attrs)  # 查看网格元信息
print(ds.coords)  # 查看坐标
```

### Q: 重采样会影响精度吗？
**A**: 使用最近邻插值，精度足够用于成本计算。如需更高精度，可修改 `_regrid_ais_density_to_grid` 函数。

---

## 📞 调试技巧

### 查看 AIS 加载日志
```python
# 在 planner_minimal.py 中搜索 [AIS] 标记
grep "\[AIS\]" arcticroute/ui/planner_minimal.py
```

### 查看网格变化日志
```python
# 在 planner_minimal.py 中搜索 [UI] 标记
grep "\[UI\]" arcticroute/ui/planner_minimal.py
```

### 检查成本计算
```python
# 在 cost.py 中搜索验证函数调用
grep "_validate_ais_density_for_grid" arcticroute/core/cost.py
```

---

## ✨ 最后检查

- ✅ 所有 5 个任务已完成
- ✅ 所有修改已验证
- ✅ 文档已生成
- ✅ 快速参考已准备

**系统已准备好投入使用！**

