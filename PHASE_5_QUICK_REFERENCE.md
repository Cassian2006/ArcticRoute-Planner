# Phase 5 快速参考：实验导出与 UI 下载

## 快速开始

### CLI 导出（最常用）

```bash
# 基础用法
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode efficient

# 导出 CSV
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode edl_safe \
    --out-csv result.csv

# 导出 JSON
python -m scripts.run_case_export \
    --scenario kara_short \
    --mode edl_robust \
    --out-json result.json

# 同时导出 CSV 和 JSON
python -m scripts.run_case_export \
    --scenario southern_route \
    --mode efficient \
    --out-csv result.csv \
    --out-json result.json

# 使用真实数据
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode edl_safe \
    --use-real-data \
    --out-csv result_real.csv
```

### Python 代码使用

```python
from arcticroute.experiments.runner import run_single_case, run_case_grid

# 单个案例
result = run_single_case("barents_to_chukchi", "efficient", use_real_data=False)
print(f"Distance: {result.distance_km} km")
print(f"Total cost: {result.total_cost}")

# 批量运行
df = run_case_grid(
    scenarios=["barents_to_chukchi", "kara_short"],
    modes=["efficient", "edl_safe"],
    use_real_data=False,
)
df.to_csv("results.csv", index=False)
```

### UI 导出

1. 打开 Streamlit UI
2. 选择场景和规划风格
3. 点击"规划三条方案"
4. 在下方找到"📥 导出当前规划结果"
5. 点击下载按钮

---

## 可用场景

| 场景名称 | 描述 | 起点 | 终点 | 船舶 |
|---------|------|------|------|------|
| `barents_to_chukchi` | 巴伦支海到楚科奇海（高冰区，长距离） | 69.0°N, 33.0°E | 70.5°N, 170.0°E | panamax |
| `kara_short` | 卡拉海短途（中等冰区，冰级船） | 73.0°N, 60.0°E | 76.0°N, 120.0°E | ice_class |
| `southern_route` | 南向北冰洋边缘（低冰区，短距离） | 60.0°N, 30.0°E | 68.0°N, 90.0°E | panamax |
| `west_to_east_demo` | 西向东跨越北冰洋（全程高纬，多冰区） | 72.0°N, 10.0°E | 75.0°N, 150.0°E | panamax |

---

## 规划模式

| 模式 | 描述 | EDL 权重 | 不确定性 | 用途 |
|------|------|---------|---------|------|
| `efficient` | 弱 EDL，偏燃油/距离 | 0.3 | ❌ | 成本敏感 |
| `edl_safe` | 中等 EDL，偏风险规避 | 1.0 | ❌ | 平衡方案 |
| `edl_robust` | 强 EDL，风险 + 不确定性 | 1.0 | ✅ | 风险厌恶 |

---

## 导出数据字段

### 基础字段
- `scenario`: 场景名称
- `mode`: 规划模式
- `reachable`: 是否可达（True/False）
- `distance_km`: 路线距离（km）
- `total_cost`: 总成本

### 成本分量
- `edl_risk_cost`: EDL 风险成本
- `edl_unc_cost`: EDL 不确定性成本
- `ice_cost`: 冰风险成本
- `wave_cost`: 波浪风险成本
- `ice_class_soft_cost`: 冰级软约束成本
- `ice_class_hard_cost`: 冰级硬约束成本

### 元数据字段
- `meta_ym`: 年月（YYYYMM）
- `meta_use_real_data`: 是否使用真实数据
- `meta_cost_mode`: 成本模式
- `meta_vessel_profile`: 船舶类型
- `meta_edl_backend`: EDL 后端
- `meta_grid_shape`: 网格形状
- `meta_w_edl`: EDL 权重
- `meta_ice_penalty`: 冰风险权重

---

## 输出示例

### 终端摘要
```
======================================================================
[SCENARIO] barents_to_chukchi             [MODE] efficient
======================================================================
Reachable: Yes
Distance: 4326.7 km
Total cost: 54.0

Metadata:
  Year-Month: 202412
  Use Real Data: False
  Cost Mode: demo_icebelt
  Vessel: panamax
  EDL Backend: miles
======================================================================
```

### CSV 格式
```csv
scenario,mode,reachable,distance_km,total_cost,edl_risk_cost,...
barents_to_chukchi,efficient,True,4326.7,54.0,,,,...
```

### JSON 格式
```json
{
  "scenario": "barents_to_chukchi",
  "mode": "efficient",
  "reachable": true,
  "distance_km": 4326.7,
  "total_cost": 54.0,
  "meta": {
    "ym": "202412",
    "use_real_data": false,
    ...
  }
}
```

---

## 常见问题

### Q: 如何批量导出多个场景和模式？

A: 使用 `run_case_grid` 函数：

```python
from arcticroute.experiments.runner import run_case_grid

df = run_case_grid(
    scenarios=["barents_to_chukchi", "kara_short", "southern_route"],
    modes=["efficient", "edl_safe", "edl_robust"],
    use_real_data=False,
)
df.to_csv("batch_results.csv", index=False)
```

### Q: 如何使用真实数据？

A: 添加 `--use-real-data` 标志：

```bash
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode edl_safe \
    --use-real-data \
    --out-csv result_real.csv
```

### Q: 如何处理不可达的案例？

A: 结果中 `reachable` 字段为 `False`，距离和成本字段为 `None`。

### Q: CSV 和 JSON 有什么区别？

A: 
- **CSV**: 表格格式，易于 Excel 打开和数据分析
- **JSON**: 结构化格式，包含完整的元数据，易于程序处理

### Q: 如何在 Python 中读取导出的结果？

A:
```python
import pandas as pd
import json

# 读取 CSV
df = pd.read_csv("result.csv")

# 读取 JSON
with open("result.json") as f:
    data = json.load(f)
```

---

## 文件位置

| 文件 | 位置 | 说明 |
|------|------|------|
| 运行器 | `arcticroute/experiments/runner.py` | Core 层统一运行器 |
| CLI 脚本 | `scripts/run_case_export.py` | 命令行导出脚本 |
| UI 导出 | `arcticroute/ui/planner_minimal.py` | Streamlit UI 导出功能 |
| 测试 | `tests/test_experiment_export.py` | 导出功能测试 |

---

## 测试验证

```bash
# 运行所有导出测试
pytest tests/test_experiment_export.py -v

# 运行所有测试（确保无破坏性改动）
pytest tests/ -q
```

**预期结果**：
- 19 个新测试全部通过
- 224 个现有测试全部通过

---

## 版本信息

- **Phase**: 5 - Experiment & Export
- **完成日期**: 2024-12-09
- **测试状态**: ✅ 全部通过
- **文档状态**: ✅ 完整













