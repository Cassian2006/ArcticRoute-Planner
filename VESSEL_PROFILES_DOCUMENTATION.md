# 船舶参数配置系统 - 完整文档

**版本**: 1.0  
**更新时间**: 2024-12-12  
**状态**: ✅ 生产就绪

## 概述

本文档描述了 ArcticRoute 项目中的船舶参数配置系统，采用**两层结构**设计：

1. **业务船型层**（Vessel Type）：Handysize、Panamax、Capesize 等
2. **冰级标准层**（Ice Class）：No ice class、FSICR 1C/1B/1A/1A Super、Polar Class PC7~PC3

## 系统架构

### 两层结构

```
┌─────────────────────────────────────────────────────────┐
│                    VesselProfile                         │
├─────────────────────────────────────────────────────────┤
│  业务船型（Vessel Type）                                 │
│  ├─ Handysize (20k-40k DWT)                             │
│  ├─ Panamax (65k-85k DWT)                               │
│  ├─ Capesize (150k-220k DWT)                            │
│  └─ ... 其他船型                                         │
├─────────────────────────────────────────────────────────┤
│  冰级标准（Ice Class）                                   │
│  ├─ No Ice Class (0.25m)                                │
│  ├─ FSICR 1C/1B/1A/1A Super (0.3-1.0m)                 │
│  ├─ Polar Class PC7/PC6/PC5 (1.2-2.0m)                 │
│  └─ Polar Class PC4/PC3 (2.5-3.0m)                     │
├─────────────────────────────────────────────────────────┤
│  参数                                                     │
│  ├─ dwt: 载重吨                                          │
│  ├─ design_speed_kn: 设计航速                            │
│  ├─ base_fuel_per_km: 基础油耗                           │
│  ├─ max_ice_thickness_m: 最大冰厚（工程估计）            │
│  ├─ ice_margin_factor: 安全裕度系数                      │
│  └─ ice_class_label: 冰级标签                            │
└─────────────────────────────────────────────────────────┘
```

## 冰厚阈值参考

### FSICR（芬兰-瑞典冰级规则）

| 冰级 | 最大冰厚 | 冰情描述 | 适用场景 |
|------|---------|---------|---------|
| No Ice Class | 0.25m | 薄冰 | 非冰级船，仅可通行薄冰 |
| 1C | 0.30m | 薄冰 | 波罗的海冬季 |
| 1B | 0.50m | 中等冰 | 波罗的海严冬 |
| 1A | 0.80m | 厚冰 | 北冰洋边缘 |
| 1A Super | 1.00m | 很厚冰 | 北冰洋内部 |

### IMO Polar Class（国际海事组织极地规则）

| 冰级 | 最大冰厚 | 冰情描述 | 适用场景 |
|------|---------|---------|---------|
| PC7 | 1.20m | 一年冰 | 北冰洋边缘，夏季 |
| PC6 | 1.50m | 一年冰 | 北冰洋内部，夏季 |
| PC5 | 2.00m | 一年冰 | 北冰洋内部，春秋季 |
| PC4 | 2.50m | 多年冰 | 北冰洋中心，冬季（暂不开放） |
| PC3 | 3.00m | 多年冰 | 北冰洋中心，严冬（暂不开放） |

### 冰情分级体系

```
薄冰（Thin Ice）
  范围：< 0.3m
  特征：新冰、年轻冰
  通行性：大多数船舶可通行

一年冰（First-Year Ice）
  范围：0.3 - 2.0m
  特征：单个冬季形成的冰
  通行性：取决于冰级

多年冰（Multi-Year Ice）
  范围：> 2.0m
  特征：多个冬季积累的冰
  通行性：仅高冰级船舶可通行
```

## 关键参数说明

### max_ice_thickness_m（最大冰厚）

**定义**：船舶的设计可通行最大冰厚（米）

**来源**：
- 基于 Polar Class 和 FSICR 标准的工程估计
- 反映船舶的冰级能力

**用途**：
- 在成本构建中用于硬约束（hard constraint）
- 超过此值时，路由成本设为 ∞（不可通行）

**示例**：
```python
# Panamax + Polar Class PC7
max_ice_thickness_m = 1.20  # 可通行 1.2m 以下的一年冰
```

### ice_margin_factor（安全裕度系数）

**定义**：应用于 max_ice_thickness_m 的安全系数（0..1）

**物理意义**：
- 考虑设计裕度、测量误差、动态因素等
- 有效最大冰厚 = max_ice_thickness_m × ice_margin_factor

**推荐值**：
- 无冰级船：0.85（保守）
- 冰级船：0.90-0.95（适中）

**示例**：
```python
# Panamax + PC7，ice_margin_factor = 0.95
effective_max_ice_thickness = 1.20 × 0.95 = 1.14m
```

### 软约束区间

**定义**：在 [soft_threshold, max_ice_thickness_m] 范围内施加二次惩罚

**计算**：
```
soft_threshold = 0.7 × max_ice_thickness_m
soft_penalty = ice_class_soft_weight × (ratio^2)

其中 ratio = (ice_thickness - soft_threshold) / (max_ice_thickness_m - soft_threshold)
```

**目的**：
- 避免硬约束的陡峭跳跃
- 鼓励船舶选择冰厚较低的路线
- 提供更平滑的成本梯度

**示例**：
```python
# Panamax + PC7
max_ice_thickness_m = 1.20m
soft_threshold = 0.7 × 1.20 = 0.84m

冰厚 0.84m：无额外惩罚
冰厚 1.02m：中等惩罚
冰厚 1.20m：最大惩罚
冰厚 > 1.20m：不可通行（硬约束）
```

## 使用指南

### 1. 获取预定义配置

```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles

profiles = get_default_profiles()

# 获取特定配置
panamax_profile = profiles["panamax"]
print(panamax_profile.name)  # "Panamax + No Ice Class"
print(panamax_profile.max_ice_thickness_m)  # 0.5
```

### 2. 创建自定义配置

```python
from arcticroute.core.eco.vessel_profiles import (
    create_vessel_profile,
    VesselType,
    IceClass,
)

# 创建 Handysize + FSICR 1A 配置
profile = create_vessel_profile(
    vessel_type=VesselType.HANDYSIZE,
    ice_class=IceClass.FSICR_1A,
    ice_margin_factor=0.90,
)

print(profile.name)  # "Handysize + FSICR 1A"
print(profile.max_ice_thickness_m)  # 0.8
print(profile.get_effective_max_ice_thickness())  # 0.72
```

### 3. 在成本构建中使用

```python
from arcticroute.core.cost import build_cost_from_real_env

# 使用船舶配置构建成本场
cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    vessel_profile=profile,  # 传入 VesselProfile 对象
    ice_penalty=4.0,
    wave_penalty=1.0,
)
```

### 4. 获取冰级信息

```python
# 获取冰级详细信息
ice_info = profile.get_ice_class_info()
print(ice_info["label"])  # "FSICR 1A"
print(ice_info["description"])  # "芬兰-瑞典冰级 1A，可通行厚度 ~0.8m 的一年冰"
print(ice_info["standard"])  # "Finnish-Swedish Ice Class Rules"
```

### 5. 列出可用选项

```python
from arcticroute.core.eco.vessel_profiles import (
    list_available_profiles,
    get_ice_class_options,
    get_vessel_type_options,
)

# 列出所有预定义配置
profiles = list_available_profiles()
# {'handy': 'Handysize (No Ice Class)', 'panamax': 'Panamax (No Ice Class)', ...}

# 列出所有冰级选项
ice_classes = get_ice_class_options()
# {'no_ice_class': 'No Ice Class', 'fsicr_1c': 'FSICR 1C', ...}

# 列出所有业务船型选项
vessel_types = get_vessel_type_options()
# {'feeder': 'Feeder', 'handysize': 'Handysize', ...}
```

## 在 UI 中集成

### 1. 船舶选择界面

```python
import streamlit as st
from arcticroute.core.eco.vessel_profiles import list_available_profiles

profiles = list_available_profiles()
selected_profile_key = st.selectbox(
    "选择船舶配置",
    options=list(profiles.keys()),
    format_func=lambda k: profiles[k],
)
```

### 2. 冰级选择界面

```python
from arcticroute.core.eco.vessel_profiles import (
    get_vessel_type_options,
    get_ice_class_options,
    create_vessel_profile,
    VesselType,
    IceClass,
)

vessel_types = get_vessel_type_options()
ice_classes = get_ice_class_options()

col1, col2 = st.columns(2)

with col1:
    selected_vessel_type = st.selectbox(
        "业务船型",
        options=list(vessel_types.keys()),
        format_func=lambda k: vessel_types[k],
    )

with col2:
    selected_ice_class = st.selectbox(
        "冰级标准",
        options=list(ice_classes.keys()),
        format_func=lambda k: ice_classes[k],
    )

# 创建配置
profile = create_vessel_profile(
    vessel_type=VesselType(selected_vessel_type),
    ice_class=IceClass(selected_ice_class),
)
```

### 3. 参数调整界面

```python
# 显示当前参数
st.write(f"**船舶名称**: {profile.name}")
st.write(f"**载重吨**: {profile.dwt:.0f} DWT")
st.write(f"**设计航速**: {profile.design_speed_kn} 节")
st.write(f"**最大冰厚**: {profile.max_ice_thickness_m} m")
st.write(f"**有效最大冰厚**: {profile.get_effective_max_ice_thickness():.2f} m")

# 允许用户调整安全裕度
ice_margin_factor = st.slider(
    "冰厚安全裕度系数",
    min_value=0.7,
    max_value=1.0,
    value=profile.ice_margin_factor,
    step=0.05,
)

# 更新配置
profile.ice_margin_factor = ice_margin_factor
```

## 参数校准

### 当前状态

✅ **已实现**：工程估计参数（基于 Polar Class 和 FSICR 标准）

### 计划中的校准

#### 1. AIS 轨迹分析（短期）

```bash
# 收集 AIS 数据
python scripts/preprocess_ais_to_density.py --ym 202412

# 分析船舶通行能力
python scripts/analyze_vessel_ice_capability.py --ym 202412
```

**目标**：
- 识别不同冰级船舶的实际通行冰厚
- 验证工程估计的合理性
- 识别异常情况

#### 2. EDL 模型训练（中期）

```bash
# 使用 AIS 轨迹训练 EDL 模型
python scripts/calibrate_env_exponents.py \
  --ym 202412 \
  --grid-mode real \
  --sample-n 200000 \
  --bootstrap-n 200
```

**目标**：
- 优化 max_ice_thickness_m 和 ice_margin_factor
- 估计参数的不确定性
- 生成校准报告

#### 3. 多月份对比（长期）

```bash
# 对多个月份进行校准，分析季节性变化
for month in 202401 202402 ... 202412; do
  python scripts/calibrate_env_exponents.py --ym $month
done
```

**目标**：
- 理解季节性变化
- 建立月份特定的参数
- 支持动态参数更新

### 校准工作流

```
初始参数（工程估计）
    ↓
AIS 轨迹分析
    ↓
EDL 模型训练
    ↓
网格搜索优化
    ↓
Bootstrap 置信区间
    ↓
校准报告生成
    ↓
参数更新
    ↓
UI 集成
```

## 文件清单

### Python 模块

- `arcticroute/core/eco/vessel_profiles.py` (400+ 行)
  - VesselProfile 数据类
  - VesselType 和 IceClass 枚举
  - 冰级参数映射表
  - 工厂函数和工具函数

### 配置文件

- `configs/vessel_profiles.yaml`
  - 业务船型定义
  - 冰级标准定义
  - 预定义配置
  - 冰厚约束配置
  - 参数校准配置

### 文档

- `VESSEL_PROFILES_DOCUMENTATION.md`（本文件）
  - 完整系统文档
  - 使用指南
  - 参考标准

## 常见问题

### Q1: 为什么使用两层结构？

**A**: 两层结构提供了灵活性和可扩展性：
- **业务层**：支持多种船型（Handysize、Panamax 等）
- **冰级层**：支持多种标准（FSICR、Polar Class 等）
- **组合**：可以创建任意的业务船型 + 冰级组合

### Q2: 冰厚阈值如何确定？

**A**: 冰厚阈值基于以下标准：
1. **Polar Class（IMO）**：国际海事组织的极地规则
2. **FSICR**：芬兰-瑞典冰级规则
3. **工程经验**：实际船舶的通行能力

这些是初始估计，后续将通过 AIS 轨迹和 EDL 模型进行校准。

### Q3: 如何更新参数？

**A**: 有三种方式：

1. **直接修改代码**（开发阶段）
   ```python
   profile.ice_margin_factor = 0.95
   ```

2. **从 YAML 配置读取**（部署阶段）
   ```yaml
   default_profiles:
     panamax_pc7:
       ice_margin_factor: 0.95
   ```

3. **通过 UI 调整**（用户交互）
   ```python
   ice_margin_factor = st.slider("安全裕度", 0.7, 1.0, 0.95)
   ```

### Q4: 软约束和硬约束的区别？

**A**:
- **硬约束**：冰厚超过有效最大冰厚时，成本设为 ∞（不可通行）
- **软约束**：在软约束区间内，施加二次惩罚（可通行但成本高）

硬约束确保安全，软约束鼓励选择更安全的路线。

### Q5: 为什么 PC4 和 PC3 暂不开放？

**A**: 因为：
1. 北冰洋中心的多年冰区域通常不在商业航运范围内
2. 能够通行 PC4/PC3 的船舶数量极少
3. 后续可根据需求开放

## 参考标准

### IMO Polar Code

- **官方网站**：https://www.imo.org/en/OurWork/Environment/PolarCode/Pages/default.aspx
- **定义**：国际海事组织的极地规则
- **冰级**：PC1（最弱）到 PC3（最强）

### Finnish-Swedish Ice Class Rules

- **定义**：芬兰和瑞典共同制定的冰级规则
- **冰级**：1C、1B、1A、1A Super
- **应用**：波罗的海和北冰洋边缘

### 冰情分级体系

- **薄冰**：< 0.3m（新冰、年轻冰）
- **一年冰**：0.3-2.0m（单个冬季形成）
- **多年冰**：> 2.0m（多个冬季积累）

## 后续工作

### 短期（立即）

- [x] 实现两层结构系统
- [x] 定义冰厚阈值
- [ ] 在 UI 中集成船舶选择
- [ ] 收集用户反馈

### 中期（1-2 周）

- [ ] 收集 AIS 轨迹数据
- [ ] 进行参数校准
- [ ] 更新默认参数
- [ ] 生成校准报告

### 长期（1-3 月）

- [ ] 使用 EDL 模型进行深度学习校准
- [ ] 建立自动参数更新机制
- [ ] 支持多月份、多船型的参数定制
- [ ] 集成到生产系统

## 总结

本系统提供了一个完整的、可扩展的船舶参数配置框架，具有以下特点：

✅ **两层结构**：业务船型 × 冰级标准  
✅ **标准化参数**：基于 Polar Class 和 FSICR  
✅ **灵活配置**：支持预定义和自定义配置  
✅ **可校准**：支持后续的 AIS 和 EDL 校准  
✅ **易于集成**：提供 Python API 和 YAML 配置  

---

**维护者**: AI Assistant  
**最后更新**: 2024-12-12  
**版本**: 1.0


