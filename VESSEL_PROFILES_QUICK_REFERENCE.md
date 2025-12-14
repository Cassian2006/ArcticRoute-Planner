# 船舶参数配置 - 快速参考

## 🚀 5 分钟快速开始

### 1. 获取预定义配置

```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles

profiles = get_default_profiles()
panamax = profiles["panamax"]

print(f"船舶: {panamax.name}")
print(f"最大冰厚: {panamax.max_ice_thickness_m}m")
print(f"有效冰厚: {panamax.get_effective_max_ice_thickness():.2f}m")
```

### 2. 创建自定义配置

```python
from arcticroute.core.eco.vessel_profiles import (
    create_vessel_profile,
    VesselType,
    IceClass,
)

# Handysize + FSICR 1A
profile = create_vessel_profile(
    VesselType.HANDYSIZE,
    IceClass.FSICR_1A,
)
```

### 3. 在成本构建中使用

```python
from arcticroute.core.cost import build_cost_from_real_env

cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    vessel_profile=profile,
)
```

## 📊 冰厚阈值速查表

| 冰级 | 最大冰厚 | 有效冰厚* | 软约束起点** |
|------|---------|---------|------------|
| No Ice Class | 0.25m | 0.21m | 0.18m |
| FSICR 1C | 0.30m | 0.27m | 0.21m |
| FSICR 1B | 0.50m | 0.45m | 0.35m |
| FSICR 1A | 0.80m | 0.72m | 0.56m |
| FSICR 1A Super | 1.00m | 0.90m | 0.70m |
| **Polar PC7** | **1.20m** | **1.14m** | **0.84m** |
| Polar PC6 | 1.50m | 1.43m | 1.05m |
| Polar PC5 | 2.00m | 1.90m | 1.40m |

*有效冰厚 = 最大冰厚 × 0.95（默认安全裕度）  
**软约束起点 = 最大冰厚 × 0.70

## 🔧 常用代码片段

### 列出所有选项

```python
from arcticroute.core.eco.vessel_profiles import (
    list_available_profiles,
    get_ice_class_options,
    get_vessel_type_options,
)

# 预定义配置
profiles = list_available_profiles()
# {'handy': 'Handysize (No Ice Class)', ...}

# 冰级选项
ice_classes = get_ice_class_options()
# {'no_ice_class': 'No Ice Class', 'fsicr_1c': 'FSICR 1C', ...}

# 业务船型选项
vessel_types = get_vessel_type_options()
# {'feeder': 'Feeder', 'handysize': 'Handysize', ...}
```

### 获取冰级信息

```python
profile = profiles["panamax_pc7"]
info = profile.get_ice_class_info()

print(info["label"])  # "Polar Class PC7"
print(info["description"])  # "IMO Polar Class PC7，可通行厚度 ~1.2m 的一年冰"
print(info["standard"])  # "IMO Polar Code"
```

### 调整安全裕度

```python
profile.ice_margin_factor = 0.85  # 更保守
effective = profile.get_effective_max_ice_thickness()
print(f"有效冰厚: {effective:.2f}m")
```

## 📱 UI 集成示例

### Streamlit

```python
import streamlit as st
from arcticroute.core.eco.vessel_profiles import (
    list_available_profiles,
    get_profile_by_key,
)

profiles = list_available_profiles()
selected_key = st.selectbox(
    "选择船舶",
    options=list(profiles.keys()),
    format_func=lambda k: profiles[k],
)

profile = get_profile_by_key(selected_key)
st.write(f"最大冰厚: {profile.max_ice_thickness_m}m")
st.write(f"有效冰厚: {profile.get_effective_max_ice_thickness():.2f}m")
```

## 🎯 业务船型对照

| 船型 | DWT | 航速 | 油耗 | 用途 |
|------|-----|------|------|------|
| Feeder | 5k-15k | 13 | 0.020 | 支线船 |
| Handysize | 20k-40k | 13 | 0.035 | 灵便散货 |
| Panamax | 65k-85k | 14 | 0.050 | 巴拿马运河 |
| Aframax | 80k-120k | 13.5 | 0.055 | 油轮 |
| Suezmax | 120k-200k | 14 | 0.070 | 苏伊士运河 |
| Capesize | 150k-220k | 13 | 0.080 | 大型散货 |
| Container | 40k-200k | 18 | 0.065 | 集装箱 |
| LNG | 130k-180k | 19 | 0.045 | 液化气 |

## ❓ 常见问题

### Q: 如何选择冰级？

**A**: 根据航线和季节：
- **夏季北冰洋**：PC7 或 FSICR 1A
- **冬季波罗的海**：FSICR 1B 或 1A
- **非冰区**：No Ice Class

### Q: 有效冰厚是什么？

**A**: 考虑安全裕度后的实际最大冰厚：
```
有效冰厚 = 最大冰厚 × 安全裕度系数
```

### Q: 软约束和硬约束的区别？

**A**:
- **硬约束**：超过有效冰厚 → 不可通行（成本 = ∞）
- **软约束**：在软约束区间 → 可通行但成本高

### Q: 参数何时更新？

**A**: 
- **现在**：使用工程估计参数
- **近期**：基于 AIS 轨迹校准
- **长期**：使用 EDL 模型优化

## 📚 参考文档

| 文档 | 说明 |
|------|------|
| `VESSEL_PROFILES_DOCUMENTATION.md` | 完整系统文档 |
| `configs/vessel_profiles.yaml` | YAML 配置文件 |
| `arcticroute/core/eco/vessel_profiles.py` | Python 源代码 |

## 🔗 相关模块

- `arcticroute.core.cost` - 成本构建（使用 VesselProfile）
- `arcticroute.core.eco.eco_model` - 经济模型
- `arcticroute.config.scenarios` - 场景配置

## 📞 支持

问题或建议？参考完整文档：`VESSEL_PROFILES_DOCUMENTATION.md`

---

**版本**: 1.0  
**最后更新**: 2024-12-12  
**状态**: ✅ 生产就绪




