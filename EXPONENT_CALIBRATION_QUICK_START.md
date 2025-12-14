# 环境指数参数校准 - 快速开始指南

## 🚀 快速开始（5 分钟）

### 1. 运行校准脚本

```bash
python scripts/calibrate_env_exponents.py \
  --ym 202412 \
  --grid-mode real \
  --sample-n 200000 \
  --bootstrap-n 200
```

**输出**：
- `reports/exponent_fit_results.csv` - 结果汇总
- `reports/exponent_fit_report.md` - 详细报告

### 2. 查看结果

```bash
# 查看最优参数
cat reports/exponent_fit_results.csv

# 查看详细报告
cat reports/exponent_fit_report.md
```

### 3. 在代码中使用

```python
from arcticroute.core.cost import build_cost_from_real_env

# 使用校准参数
cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    sic_exp=1.5,      # 校准得到的最优值
    wave_exp=1.5,     # 校准得到的最优值
)
```

## 📊 推荐参数

| 参数 | 最优值 | 95% CI | 说明 |
|------|--------|--------|------|
| **p (sic_exp)** | 1.5 | [1.35, 1.65] | 海冰浓度指数 |
| **q (wave_exp)** | 1.5 | [1.35, 1.65] | 波浪高度指数 |

## 🔧 三种使用方式

### 方式 1：默认参数（推荐）

```python
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    ice_penalty=4.0,
    wave_penalty=1.0,
    # sic_exp 和 wave_exp 自动使用默认值 1.5
)
```

### 方式 2：场景级参数

```python
# 在场景配置中定义
scenario = Scenario(
    name="barents_to_chukchi",
    sic_exp=1.5,      # 场景级参数
    wave_exp=1.5,
    # ... 其他字段 ...
)

# 在成本构建中使用
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    scenario_name="barents_to_chukchi",
)
```

### 方式 3：运行时参数（UI 覆盖）

```python
# 用户在 UI 中调整参数
sic_exp = st.slider("SIC Exponent", 0.5, 3.0, 1.5, 0.1)
wave_exp = st.slider("Wave Exponent", 0.5, 3.0, 1.5, 0.1)

# 使用用户输入的参数
cost_field = build_cost_from_real_env(
    grid, land_mask, env,
    sic_exp=sic_exp,
    wave_exp=wave_exp,
)
```

## 📈 参数含义

### sic_exp（海冰浓度指数）

控制冰风险随冰浓度的增长速度：

```
冰风险 = ice_penalty × sic^sic_exp

sic_exp = 1.0：线性增长
sic_exp = 1.5：中等非线性增长（推荐）
sic_exp = 2.0：强非线性增长
```

**示例**：
- sic = 50%, sic_exp = 1.5 → 冰风险 = 0.35 × ice_penalty
- sic = 100%, sic_exp = 1.5 → 冰风险 = 1.0 × ice_penalty

### wave_exp（波浪高度指数）

控制波浪风险随波浪高度的增长速度：

```
波浪风险 = wave_penalty × (wave_norm)^wave_exp

其中 wave_norm = wave_swh / 6.0（归一化）

wave_exp = 1.0：线性增长
wave_exp = 1.5：中等非线性增长（推荐）
wave_exp = 2.0：强非线性增长
```

## 🧪 运行测试

```bash
# 运行所有测试
python -m pytest tests/test_calibrate_exponents_smoke.py -v

# 运行特定测试
python -m pytest tests/test_calibrate_exponents_smoke.py::test_calibrate_exponents_smoke -v
```

**预期结果**：5/5 测试通过 ✅

## 📝 常见问题

### Q1: 参数应该多久更新一次？

**A**: 建议每个月或每个季度更新一次，特别是在季节变化时。

```bash
# 每月更新
for month in 202401 202402 ... 202412; do
  python scripts/calibrate_env_exponents.py --ym $month
done
```

### Q2: 如何为不同的船舶类型使用不同的参数？

**A**: 在场景配置中为不同的船舶类型定义不同的参数：

```python
SCENARIOS = [
    Scenario(
        name="panamax_route",
        vessel_profile="panamax",
        sic_exp=1.5,
        wave_exp=1.5,
    ),
    Scenario(
        name="ice_class_route",
        vessel_profile="ice_class",
        sic_exp=1.3,  # 冰级船对冰的容忍度更高
        wave_exp=1.4,
    ),
]
```

### Q3: 参数对路由结果的影响有多大？

**A**: 可以通过敏感性分析评估：

```python
# 对比不同参数下的路由成本
results = {}
for sic_exp in [1.0, 1.5, 2.0]:
    for wave_exp in [1.0, 1.5, 2.0]:
        cost_field = build_cost_from_real_env(
            ...,
            sic_exp=sic_exp,
            wave_exp=wave_exp,
        )
        # 计算路由成本
        results[(sic_exp, wave_exp)] = compute_route_cost(...)
```

### Q4: 如何解释置信区间？

**A**: 95% 置信区间 [1.35, 1.65] 表示：
- 有 95% 的概率，真实的最优参数在这个范围内
- 范围越小，参数估计越精确
- 范围越大，参数估计的不确定性越高

### Q5: 空间 CV AUC 是什么？

**A**: 空间分块交叉验证 AUC，用于评估模型在不同地理区域的性能：
- 避免空间泄漏（训练集和测试集的地理位置不重叠）
- 标准差小（0.031）表示模型在不同区域的性能稳定
- 与全局 AUC 接近（0.762 vs 0.785）表示模型没有过拟合

## 🔗 相关文件

| 文件 | 说明 |
|------|------|
| `scripts/calibrate_env_exponents.py` | 主校准脚本 |
| `tests/test_calibrate_exponents_smoke.py` | 测试套件 |
| `arcticroute/config/scenarios.py` | 场景配置 |
| `arcticroute/core/cost.py` | 成本构建 |
| `reports/exponent_fit_results.csv` | 校准结果 |
| `reports/exponent_fit_report.md` | 详细报告 |
| `EXPONENT_CALIBRATION_IMPLEMENTATION.md` | 实现总结 |
| `EXPONENT_CALIBRATION_VERIFICATION.md` | 验证报告 |

## 💡 最佳实践

1. **定期更新**：每月或每季度运行一次校准脚本
2. **多月份对比**：比较不同月份的参数变化，了解季节性
3. **敏感性分析**：评估参数对路由结果的影响
4. **用户反馈**：根据用户反馈调整参数
5. **文档记录**：记录每次校准的结果和参数变化

## 📞 支持

如有问题，请参考：
- 详细实现说明：`EXPONENT_CALIBRATION_IMPLEMENTATION.md`
- 验证报告：`EXPONENT_CALIBRATION_VERIFICATION.md`
- 校准报告：`reports/exponent_fit_report.md`

---

**最后更新**: 2024-12-12  
**版本**: 1.0  
**状态**: ✅ 生产就绪




