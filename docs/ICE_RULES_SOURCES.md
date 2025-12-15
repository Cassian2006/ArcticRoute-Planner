# Ice Rules Sources (Traceable Parameters)

## 原则

任何进入"禁行/可达性/惩罚"的阈值，都必须对应可追溯来源（标准/论文/权威仓库）。

不在代码中硬编码数值；所有参数来自配置文件 `arcticroute/config/polar_rules.yaml`。

缺失值时，系统按 `missing_value_policy` 处理（默认：警告并禁用该规则）。

---

## 参数填充清单

| Rule | Parameter | Unit | Intended Source | Status | Notes |
|---|---|---:|---|---|---|
| wave | swh_max_m | m | Copernicus ERA5 / IMO Polar Code guidance | TODO | Significant wave height threshold; varies by vessel type & ice class |
| sic | sic_max | 0..1 | IMO Polar Code / peer-reviewed literature | TODO | Sea ice concentration; typically 0.8-0.95 for different classes |
| ice_thickness | thickness_max_m | m | IMO Polar Code / class guidance / literature | TODO | Maximum ice thickness; varies by ice class (e.g., 1A: 4.0m, 1B: 2.5m) |
| speed_penalty | model_params | - | TBD (future) | PENDING | Optional speed/fuel penalty model; not yet implemented |

---

## 填充指南

### 1. Wave (SWH) Threshold

**来源候选：**
- IMO Polar Code (2014/2015 amendments)
- Copernicus Climate Data Store (ERA5 wave data guidance)
- 学术文献：Arctic navigation wave statistics

**预期范围：**
- PC6 (最强冰级): 3.5-4.5 m
- PC7 (中等冰级): 4.0-5.0 m
- 无冰级限制: 5.0+ m

**状态：** 待填充权威数值

---

### 2. Sea Ice Concentration (SIC) Threshold

**来源候选：**
- IMO Polar Code (ice navigation guidelines)
- NSIDC (National Snow & Ice Data Center) recommendations
- 学术文献：Arctic ice navigation safety

**预期范围：**
- 1A (最强): 0.80-0.85
- 1B: 0.85-0.90
- 1C: 0.90-0.95
- 无限制: 1.0 (open water)

**状态：** 待填充权威数值

---

### 3. Ice Thickness Threshold

**来源候选：**
- IMO Polar Code (ice class definitions)
- 船级社规范 (DNV, ABS, ClassNK)
- 学术文献：ice-hull interaction

**预期范围：**
- 1A: 3.0-4.0 m
- 1B: 2.0-2.5 m
- 1C: 1.0-1.5 m

**状态：** 待填充权威数值

---

## 使用流程

1. **配置阶段**：在 `arcticroute/config/polar_rules.yaml` 中填入具体数值
2. **来源追溯**：在本文档中记录每个参数的来源与引用
3. **测试验证**：运行 `pytest tests/test_polar_rules.py` 验证边界条件
4. **UI 展示**：在诊断区显示启用的规则、命中率、阈值来源

---

## 当前状态

- ✅ 配置文件骨架搭建
- ✅ 来源文档框架
- ⏳ 权威数值填充（待 Phase 6 后续迭代）
- ⏳ 学术文献引用补充

---

## 相关文件

- `arcticroute/config/polar_rules.yaml` - 配置文件
- `arcticroute/core/constraints/polar_rules.py` - 规则引擎
- `tests/test_polar_rules.py` - 单元测试


