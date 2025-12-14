# Phase 4 快速开始指南

## 📋 概览
Phase 4 实现了 **Mini-ECO 模块** + **船型指标面板**，支持简化版能耗估算。

---

## 🚀 快速启动

### 1. 运行 UI
```bash
cd C:\Users\sgddsf\Desktop\AR_final
streamlit run run_ui.py
```

### 2. 在浏览器中操作
- 打开 http://localhost:8501
- 在左侧 Sidebar 选择船型（Handysize / Panamax / Ice-Class）
- 设置起点和终点坐标
- 点击「规划三条方案」

### 3. 查看结果
摘要表格中会显示：
- `distance_km`: 航程距离
- `travel_time_h`: 航行时间
- `fuel_total_t`: 燃油消耗
- `co2_total_t`: CO2 排放

---

## 📦 新增模块

### `arcticroute/core/eco/vessel_profiles.py`
```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles

profiles = get_default_profiles()
# 返回: {"handy": VesselProfile(...), "panamax": ..., "ice_class": ...}
```

### `arcticroute/core/eco/eco_model.py`
```python
from arcticroute.core.eco.eco_model import estimate_route_eco

eco = estimate_route_eco(route_latlon, vessel)
# 返回: EcoRouteEstimate(distance_km, travel_time_h, fuel_total_t, co2_total_t)
```

---

## 🧪 运行测试

### 运行所有测试
```bash
pytest
```

### 仅运行 ECO 测试
```bash
pytest tests/test_eco_demo.py -v
```

### 预期结果
```
26 passed in 1.22s
```

---

## 🎯 关键特性

| 特性 | 说明 |
|-----|------|
| **3 种船型** | Handysize, Panamax, Ice-Class Cargo |
| **ECO 指标** | 距离、时间、燃油、CO2 |
| **动态选择** | UI 中实时切换船型 |
| **完整测试** | 10 个 ECO 功能测试 |
| **向后兼容** | 所有旧测试仍通过 |

---

## 📊 船型参数对比

| 船型 | DWT | 航速 | 油耗 |
|-----|-----|------|------|
| Handysize | 30k | 13 kn | 0.035 t/km |
| Panamax | 80k | 14 kn | 0.050 t/km |
| Ice-Class | 50k | 12 kn | 0.060 t/km |

---

## 💡 使用示例

### Python 脚本中使用
```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles
from arcticroute.core.eco.eco_model import estimate_route_eco

# 获取船型
profiles = get_default_profiles()
vessel = profiles["panamax"]

# 定义路线
route = [(70.0, 10.0), (70.5, 15.0), (71.0, 20.0)]

# 估算 ECO
eco = estimate_route_eco(route, vessel)
print(f"距离: {eco.distance_km:.1f} km")
print(f"时间: {eco.travel_time_h:.1f} h")
print(f"燃油: {eco.fuel_total_t:.2f} t")
print(f"CO2: {eco.co2_total_t:.2f} t")
```

### 自定义 CO2 系数
```python
eco = estimate_route_eco(route, vessel, co2_per_ton_fuel=3.5)
```

---

## ⚠️ 注意事项

1. **Demo 数据**：当前使用 demo 网格和 landmask，非真实海陆分布
2. **简化模型**：ECO 估算为简化版，不考虑海况、风向等因素
3. **绝对值**：表格中的数值仅供参考，不应过度解读
4. **扩展性**：模块设计易于扩展，后续可集成更复杂的模型

---

## 📝 修改文件清单

```
✏️  arcticroute/core/eco/vessel_profiles.py
✏️  arcticroute/core/eco/eco_model.py
✏️  arcticroute/ui/planner_minimal.py
✨ tests/test_eco_demo.py (新增)
```

---

## 🔗 相关文档

- 完整报告: `PHASE_4_COMPLETION_REPORT.md`
- 项目 README: `README.md`
- Phase 3 总结: `PHASE_3_5_FINAL_REPORT.md`

---

## ✅ 验证清单

- [x] 3 种船型配置正确加载
- [x] ECO 估算逻辑正确
- [x] UI 船型选择正常
- [x] 摘要表格显示 ECO 指标
- [x] 所有 26 个测试通过
- [x] 无破坏性修改

---

**状态**: ✅ Phase 4 完成  
**最后更新**: 2025-12-08













