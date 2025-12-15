# Phase 4 Pareto 前沿 - 快速参考

## 📋 交付物检查清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `arcticroute/core/pareto.py` | ✅ | 核心模块：ParetoSolution, dominates, pareto_front 等 |
| `scripts/run_pareto_suite.py` | ✅ | CLI 工具：一键生成候选与前沿 |
| `arcticroute/ui/pareto_panel.py` | ✅ | UI 面板：交互式前沿展示 |
| `arcticroute/ui/planner_minimal.py` | ✅ | 已集成 Pareto 面板 |
| `tests/test_pareto_front.py` | ✅ | 基础功能测试 |
| `tests/test_pareto_demo_smoke.py` | ✅ | 演示烟雾测试 |
| `reports/pareto_solutions.csv` | ✅ | 候选解集合（23 行） |
| `reports/pareto_front.csv` | ✅ | Pareto 前沿（3 行） |

## 🚀 快速开始

### 1. 生成 Pareto 前沿
```bash
python -m scripts.run_pareto_suite --n 20
```
**输出**:
- `reports/pareto_solutions.csv` - 所有候选解
- `reports/pareto_front.csv` - Pareto 前沿解

### 2. 运行测试
```bash
python -m pytest tests/test_pareto_front.py tests/test_pareto_demo_smoke.py -v
```
**预期**: 2 passed

### 3. 在 UI 中使用
启动 Streamlit：
```bash
streamlit run run_ui.py
```
然后在规划界面中找到 "[object Object]目标前沿（实验）" expander，点击 "🚀 生成 Pareto 前沿"。

## 📊 核心 API

### ParetoSolution 数据类
```python
from arcticroute.core.pareto import ParetoSolution

sol = ParetoSolution(
    key="efficient",                          # 候选 ID
    objectives={                              # 目标向量
        "distance_km": 5076.6,
        "total_cost": 105.3,
        "edl_risk": 0.0,
        "edl_uncertainty": 0.0
    },
    route=[(66.0, 5.0), (78.0, 150.0)],      # 路线坐标
    component_totals={                        # 成本分量
        "ice_risk": 25.6,
        "wave_risk": 10.7
    },
    meta={"ice_penalty": 2.0}                 # 配置元数据
)
```

### 计算 Pareto 前沿
```python
from arcticroute.core.pareto import pareto_front

front = pareto_front(
    cands=[sol1, sol2, sol3, ...],           # 候选解列表
    fields=["distance_km", "total_cost"]     # 目标维度
)
```

### 提取目标向量
```python
from arcticroute.core.pareto import extract_objectives_from_breakdown

objectives = extract_objectives_from_breakdown(breakdown)
# 返回: {"distance_km": ..., "total_cost": ..., "edl_risk": ..., ...}
```

### 转换为 DataFrame
```python
from arcticroute.core.pareto import solutions_to_dataframe

df = solutions_to_dataframe(solutions)
# 包含列: key, distance_km, total_cost, edl_risk, edl_uncertainty, ...
```

## 🎯 使用场景

### 场景 1：比较多个规划方案
```python
from scripts.run_pareto_suite import run_pareto_suite

# 生成候选解
solutions, front = run_pareto_suite(n_random=50)

# 查看 Pareto 前沿
for sol in front:
    print(f"{sol.key}: distance={sol.objectives['distance_km']:.1f}km, "
          f"cost={sol.objectives['total_cost']:.1f}")
```

### 场景 2：在 Streamlit 中展示
```python
import streamlit as st
from arcticroute.ui.pareto_panel import render_pareto_panel
from scripts.run_pareto_suite import run_pareto_suite

solutions, _ = run_pareto_suite(n_random=20)
render_pareto_panel(solutions)
```

### 场景 3：自定义目标维度
```python
from arcticroute.core.pareto import pareto_front

# 只考虑距离和成本
front = pareto_front(solutions, fields=["distance_km", "total_cost"])

# 考虑所有 4 个维度
front = pareto_front(
    solutions, 
    fields=["distance_km", "total_cost", "edl_risk", "edl_uncertainty"]
)
```

## 📈 Pareto 前沿分析结果

### 当前演示结果（--n 20）
```
总候选数: 23 (3 个预设 + 20 个随机)
前沿大小: 3
支配率: 86.96%

前沿解:
1. efficient: distance=5076.6km, cost=105.3, edl_risk=0.0, edl_unc=0.0
2. rand_001: distance=4835.4km, cost=149.1, edl_risk=30.0, edl_unc=10.9
3. rand_009: distance=5017.3km, cost=164.3, edl_risk=45.9, edl_unc=0.0
```

## 🔧 CLI 参数说明

```bash
python -m scripts.run_pareto_suite [OPTIONS]

OPTIONS:
  --n INT                    随机候选数量（除 3 个预设外）[default: 20]
  --seed INT                 随机种子 [default: 7]
  --outdir PATH              输出目录 [default: reports]
  --pareto-fields STR        目标维度（逗号分隔）
                             [default: distance_km,total_cost,edl_uncertainty]
```

## 📝 输出文件格式

### pareto_solutions.csv
```
key,distance_km,total_cost,edl_risk,edl_uncertainty,ice_risk,wave_risk,base_distance
efficient,5076.601070580531,105.27060014554836,0.0,0.0,25.61416759554096,10.65643255000739,69.0
edl_safe,5832.8399330632255,194.33203478901694,46.74781799316406,0.0,50.79880184762636,24.78541256404064,72.0
...
```

### pareto_front.csv
```
key,distance_km,total_cost,edl_risk,edl_uncertainty,ice_risk,wave_risk,base_distance
efficient,5076.601070580531,105.27060014554836,0.0,0.0,25.61416759554096,10.65643255000739,69.0
rand_001,4835.405123145022,149.12884155598286,30.04204559326172,10.903141083266055,22.23324671427241,17.950405125345718,68.0
rand_009,5017.256746023602,164.2707755205531,45.9073600769043,0.0,29.024322113358906,21.339097681428957,68.0
```

## 🧪 测试验收

### 单元测试
```bash
$ python -m pytest tests/test_pareto_front.py -v
# 测试: 基础支配关系和前沿计算
# 预期: 1 passed
```

### 集成测试
```bash
$ python -m pytest tests/test_pareto_demo_smoke.py -v
# 测试: 完整的规划流程（环境 → 规划 → 前沿 → 导出）
# 预期: 1 passed
```

### 完整测试
```bash
$ python -m pytest -q
# 预期: 所有测试通过，0 failed
```

## 🎨 UI 功能说明

### Pareto 面板功能
1. **目标维度选择**: 多选框选择要最小化的目标
2. **前沿表格**: 展示所有 Pareto 前沿解
3. **散点图**: 可视化前沿解在 2D 空间中的分布
4. **解选择**: 选择一条前沿解查看详细信息
5. **路线预览**: 显示选中解的路线坐标和地图
6. **下载**: 导出前沿解和所有候选解的 CSV 文件

## 💡 关键概念

### 支配关系（Dominance）
在最小化问题中，解 A 支配解 B 当且仅当：
- A 在所有目标上都不劣于 B
- A 在至少一个目标上严格优于 B

### Pareto 前沿（Pareto Front）
不被任何其他候选解支配的解的集合。前沿上的每个解都代表一种不同的目标权衡。

### 支配率（Dominance Rate）
被支配的候选解数 / 总候选数。支配率越高，说明前沿越紧凑。

## 🔗 相关文档

- [Phase 4 验收报告](PHASE_4_PARETO_ACCEPTANCE_REPORT.md)
- [执行总结（中文）](PHASE_4_PARETO_执行总结_中文.md)
- [ADR-0001: LayerGraph 架构](docs/adr/ADR-0001-layergraph.md)

## ❓ 常见问题

### Q: 为什么前沿解这么少？
A: 这取决于候选解的多样性。如果大多数候选解在某个目标上都很相似，前沿会比较小。增加随机候选数量（--n）可以得到更大的前沿。

### Q: 如何自定义目标维度？
A: 使用 `--pareto-fields` 参数指定目标维度，例如：
```bash
python -m scripts.run_pareto_suite --pareto-fields "distance_km,edl_risk"
```

### Q: 如何在自己的代码中使用？
A: 导入相关模块并调用 API，例如：
```python
from arcticroute.core.pareto import pareto_front, solutions_to_dataframe
front = pareto_front(my_solutions, fields=["distance_km", "total_cost"])
df = solutions_to_dataframe(front)
```

### Q: 支持多少个目标维度？
A: 理论上没有限制，但实际上 3-4 个维度是最实用的。超过 4 个维度时，可视化会变得困难。

---

**最后更新**: 2025-12-14  
**版本**: Phase 4 Final  
**状态**: ✅ 生产就绪


