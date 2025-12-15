# Phase EVAL-1 快速开始指南

## 5 分钟上手

### 1️⃣ 确保有场景结果数据

```bash
# 如果还没有运行过场景套件
python -m scripts.run_scenario_suite
```

这会生成 `reports/scenario_suite_results.csv`

### 2️⃣ 运行评估脚本

```bash
python -m scripts.eval_scenario_results
```

### 3️⃣ 查看结果

**终端会打印**：
- 各场景的对比表（Δdist、Δcost、risk_reduction）
- 全局统计摘要（平均风险下降、绕航增加等）

**生成的文件**：
- `reports/eval_mode_comparison.csv` - 详细对比数据

---

## 常见用法

### 自定义输入/输出路径

```bash
python -m scripts.eval_scenario_results \
    --input my_results.csv \
    --output my_eval.csv
```

### 仅生成 CSV，不打印终端表格

```bash
python -m scripts.eval_scenario_results --pretty-print False
```

### 查看帮助

```bash
python -m scripts.eval_scenario_results --help
```

---

## 理解输出

### 终端表格示例

```
[barents_to_chukchi]
Mode            Δdist(km)   Δdist(%)      Δcost   Δcost(%)  risk_red(%)
--------------------------------------------------------------------------------
edl_safe           123.50       2.85       1.23       2.27        61.88
edl_robust         253.80       5.87       2.69       4.97        79.88
```

**列说明**：
- `Δdist(km)` - 距离增加多少公里
- `Δdist(%)` - 距离增加百分比
- `Δcost` - 成本增加多少
- `Δcost(%)` - 成本增加百分比
- `risk_red(%)` - 风险下降百分比（**越高越好**）

### 全局统计示例

```
EDL_SAFE:
  Avg risk reduction:             59.53%
  Avg distance increase:           3.12%
  Scenarios with better risk:         4
  Better risk + small detour:         4
```

**含义**：
- edl_safe 平均降低风险 59.53%
- 平均增加绕航 3.12%
- 4 个场景都有风险改善
- 4 个场景既有风险改善又绕航 ≤5%（最优）

---

## 用于论文/汇报

### 直接复制的数据

从终端摘要复制关键数字：

```
"我们的 EDL-Safe 方案在 4 个测试场景中平均降低风险 59.53%，
同时仅增加 3.12% 的航程。"
```

### 导入到 Excel

1. 打开 `reports/eval_mode_comparison.csv`
2. 在 Excel 中打开
3. 制作图表（推荐：柱状图或散点图）

### 导入到 LaTeX

```latex
\begin{table}
\input{reports/eval_mode_comparison.csv}
\end{table}
```

---

## 运行测试

```bash
# 运行所有测试
pytest tests/test_eval_scenario_results.py -v

# 运行特定测试
pytest tests/test_eval_scenario_results.py::test_evaluate_delta_calculations -v
```

✅ 所有 9 个测试应该通过

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| `FileNotFoundError: reports/scenario_suite_results.csv` | 先运行 `python -m scripts.run_scenario_suite` |
| 所有 `risk_red(%)` 都是 NaN | 检查输入数据的 `edl_risk_cost` 列是否有非零值 |
| 某个场景被跳过 | 检查该场景是否有 `efficient` 模式的数据 |
| 输出为空 | 检查输入 CSV 是否有 `reachable=True` 的行 |

---

## 下一步

- 📊 查看 `reports/eval_mode_comparison.csv` 的详细数据
- 📈 制作可视化图表
- 📝 在论文中引用结果
- 🔧 根据需要调整参数重新运行

---

**更多信息**：见 `PHASE_EVAL_1_IMPLEMENTATION_REPORT.md`









