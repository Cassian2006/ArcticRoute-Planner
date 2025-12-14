# Phase 3 EDL 行为体检 - 快速开始指南

## 一句话总结

在 AR_final 项目中实现了 EDL 灵敏度分析框架，通过对比 3 种模式（baseline、EDL-safe、EDL-robust）在 4 个标准场景上的表现，量化 EDL 的成本影响和不确定性分布。

---

## 快速开始

### 1. 运行灵敏度分析（最简单）

```bash
cd C:\Users\sgddsf\Desktop\AR_final
python -m scripts.run_edl_sensitivity_study
```

**输出**:
- `reports/edl_sensitivity_results.csv` - 分析结果表
- `reports/edl_sensitivity_*.png` - 对比图表（4 张）
- 控制台摘要表

### 2. 查看结果

```bash
# 查看 CSV 文件
cat reports/edl_sensitivity_results.csv

# 或用 Excel/Python 打开
import pandas as pd
df = pd.read_csv("reports/edl_sensitivity_results.csv")
print(df)
```

### 3. 运行测试

```bash
pytest tests/test_edl_sensitivity_script.py -v
```

**预期**: 19 个测试全部通过 ✅

---

## 核心概念

### 三种规划模式

| 模式 | 说明 | 何时使用 |
|-----|------|--------|
| **efficient** | 基准方案，无 EDL | 对比基础 |
| **edl_safe** | 考虑 EDL 风险 | 评估 EDL 贡献 |
| **edl_robust** | 风险 + 不确定性 | 最保守方案 |

### 关键指标

| 指标 | 含义 | 关注点 |
|-----|------|--------|
| `distance_km` | 路线距离 | 路线长度变化 |
| `total_cost` | 总成本 | 三种模式的成本差异 |
| `edl_risk_cost` | EDL 风险成本 | EDL 的实际贡献 |
| `mean_uncertainty` | 平均不确定性 | EDL 模型的信心度 |

---

## 常用命令

### 干运行（验证脚本，不实际计算）
```bash
python -m scripts.run_edl_sensitivity_study --dry-run
```

### 使用真实数据
```bash
python -m scripts.run_edl_sensitivity_study --use-real-data
```

### 自定义输出路径
```bash
python -m scripts.run_edl_sensitivity_study \
  --output-csv my_results.csv \
  --output-dir my_charts
```

### 在 Python 中调用
```python
from scripts.run_edl_sensitivity_study import run_all_scenarios, print_summary

results = run_all_scenarios()
print_summary(results)
```

---

## 文件位置

```
scripts/
├── edl_scenarios.py                    # 场景库
└── run_edl_sensitivity_study.py        # 主脚本

tests/
└── test_edl_sensitivity_script.py      # 测试

docs/
└── EDL_BEHAVIOR_CHECK.md               # 详细文档

reports/
├── edl_sensitivity_results.csv         # 结果表
└── edl_sensitivity_*.png               # 图表
```

---

## 数据分析示例

### 按场景统计
```python
import pandas as pd

df = pd.read_csv("reports/edl_sensitivity_results.csv")

# 按场景分组
by_scenario = df.groupby("scenario").agg({
    "total_cost": ["min", "max", "mean"],
    "edl_risk_cost": "mean",
    "mean_uncertainty": "mean",
})
print(by_scenario)
```

### 计算 EDL 贡献度
```python
# 计算 EDL 风险占比
df["edl_fraction"] = df["edl_risk_cost"] / df["total_cost"]

# 找出 EDL 贡献最大的场景
top_edl = df.nlargest(5, "edl_fraction")[["scenario", "mode", "edl_fraction"]]
print(top_edl)
```

### 对比三种模式
```python
# 每个场景的三种模式对比
for scenario in df["scenario"].unique():
    subset = df[df["scenario"] == scenario]
    print(f"\n{scenario}:")
    print(subset[["mode", "distance_km", "total_cost", "edl_risk_cost"]])
```

---

## 关键发现

### 当前状态（Demo 网格）

在 demo 网格上的测试结果：
- ✅ 所有 4 个场景都可达
- ✅ 三种模式的路线相同（因为 demo 模式下不启用 EDL）
- ✅ 成本分解正确（base_distance + ice_risk）

### 预期在真实数据上

使用 `--use-real-data` 时：
- EDL 风险应该在高冰区有显著贡献（5%~20%）
- 不确定性应该在复杂区域较高（0.3~0.7）
- 三种模式应该产生不同的路线

---

## 参数调优

### w_edl（EDL 风险权重）

**当前**: 1.0

**调优指南**:
- 若 EDL 风险占比 < 2% → 增加到 1.5~2.0
- 若 EDL 风险占比 > 30% → 减少到 0.5~0.7
- 若 EDL 风险占比 5%~15% → 保持当前值

### edl_uncertainty_weight（不确定性权重）

**当前**: 1.0

**调优指南**:
- 若不确定性成本占比 < 1% → 增加到 2.0~3.0
- 若不确定性成本占比 > 20% → 减少到 0.3~0.5
- 若不确定性成本占比 5%~10% → 保持当前值

---

## 常见问题

**Q: 为什么 EDL 风险成本都是 0？**  
A: 在 demo 网格上，EDL 不启用。使用 `--use-real-data` 选项来启用 EDL。

**Q: 三种模式的路线为什么相同？**  
A: 在 demo 网格上，EDL 不启用，所以三种模式的成本场相同。

**Q: 如何添加新的场景？**  
A: 编辑 `scripts/edl_scenarios.py`，在 `SCENARIOS` 列表中添加新的 `Scenario` 对象。

**Q: 如何修改权重参数？**  
A: 编辑 `scripts/run_edl_sensitivity_study.py` 中的 `MODES` 字典。

---

## 下一步

1. **数据分析**: 用 Python/Excel 分析 CSV 结果
2. **参数调优**: 根据结果调整 w_edl 和 edl_uncertainty_weight
3. **真实数据**: 使用 `--use-real-data` 在真实数据上运行
4. **自定义场景**: 在 `edl_scenarios.py` 中添加新场景
5. **模型改进**: 根据不确定性分布改进 EDL 模型

---

## 技术细节

### 脚本架构

```
run_edl_sensitivity_study.py
├── MODES: 三种规划模式的配置
├── SensitivityResult: 结果数据类
├── run_single_scenario_mode(): 单个场景+模式
├── run_all_scenarios(): 批量运行
├── write_results_to_csv(): 输出 CSV
├── print_summary(): 打印摘要
└── generate_charts(): 生成图表
```

### 数据流

```
edl_scenarios.SCENARIOS
    ↓
run_all_scenarios()
    ├─ run_single_scenario_mode() × 12
    │   ├─ 加载网格和陆地掩码
    │   ├─ 构建成本场
    │   ├─ 规划路线
    │   └─ 计算成本分解
    ↓
write_results_to_csv()
    └─ reports/edl_sensitivity_results.csv

generate_charts()
    └─ reports/edl_sensitivity_*.png
```

---

## 文件大小

- `scripts/edl_scenarios.py`: ~100 行
- `scripts/run_edl_sensitivity_study.py`: ~600 行
- `tests/test_edl_sensitivity_script.py`: ~400 行
- `docs/EDL_BEHAVIOR_CHECK.md`: ~800 行
- **总计**: ~1900 行代码 + 文档

---

## 性能

- **干运行**: < 1 秒
- **实际运行（demo 网格）**: ~5 秒
- **实际运行（真实数据）**: ~30 秒（取决于网格大小和数据可用性）

---

## 支持

- **文档**: `docs/EDL_BEHAVIOR_CHECK.md`
- **测试**: `tests/test_edl_sensitivity_script.py`
- **示例**: 本文件中的代码示例

---

**版本**: 1.0  
**日期**: 2024-12-08  
**状态**: ✅ 完成
















