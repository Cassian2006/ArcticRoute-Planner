# Phase 5 实现报告：实验导出与 UI 下载

**项目**: ArcticRoute 北极航线规划系统  
**阶段**: Phase 5 - Experiment & Export  
**完成日期**: 2024-12-09  
**状态**: ✅ 完成

---

## 执行摘要

本阶段成功实现了统一的"运行一次规划并返回 DataFrame/字典"的封装，以及完整的导出功能。通过创建核心运行器、CLI 脚本和 UI 导出按钮，实现了规划结果的灵活导出。所有现有测试通过，新增 19 个测试全部通过。

**关键成就**：
- ✅ 创建 Core 层统一运行器（`arcticroute/experiments/runner.py`）
- ✅ 实现 `SingleRunResult` 数据类和 `run_single_case` 函数
- ✅ 实现 `run_case_grid` 函数返回 DataFrame
- ✅ 创建 CLI 脚本（`scripts/run_case_export.py`）
- ✅ 在 UI 中添加导出按钮（CSV 和 JSON）
- ✅ 完整的测试覆盖（19 个新测试，全部通过）
- ✅ 所有现有测试保持通过（224 passed）

---

## 详细实现

### 1. Core 层运行器（`arcticroute/experiments/runner.py`）

#### 1.1 `SingleRunResult` 数据类

```python
@dataclass
class SingleRunResult:
    """单次规划运行的结果数据类。"""
    
    scenario: str                      # 场景名称
    mode: ModeName                     # 规划模式（efficient/edl_safe/edl_robust）
    reachable: bool                    # 是否可达
    distance_km: Optional[float]       # 路线距离（km）
    total_cost: Optional[float]        # 总成本
    edl_risk_cost: Optional[float]     # EDL 风险成本
    edl_unc_cost: Optional[float]      # EDL 不确定性成本
    ice_cost: Optional[float]          # 冰风险成本
    wave_cost: Optional[float]         # 波浪风险成本
    ice_class_soft_cost: Optional[float]   # 冰级软约束成本
    ice_class_hard_cost: Optional[float]   # 冰级硬约束成本
    meta: Dict[str, Any]               # 元数据
```

**特性**：
- 完整的成本分量记录
- 灵活的元数据存储
- 支持转换为字典和扁平字典（便于 DataFrame 导出）

#### 1.2 `run_single_case` 函数

**签名**：
```python
def run_single_case(
    scenario: str,
    mode: ModeName,
    use_real_data: bool = True,
) -> SingleRunResult:
```

**功能流程**：
1. 获取场景配置（起止点、年月、船舶类型）
2. 获取 EDL 模式配置（权重参数）
3. 加载网格和陆地掩码（支持真实数据和 demo 回退）
4. 获取船舶配置
5. 构建成本场（支持真实环境和 demo 模式）
6. 规划路线（A* 算法）
7. 计算成本分解
8. 返回 `SingleRunResult` 对象

**特性**：
- 自动回退机制（真实数据不可用时自动使用 demo）
- 完整的错误处理
- 详细的元数据记录

#### 1.3 `run_case_grid` 函数

**签名**：
```python
def run_case_grid(
    scenarios: List[str],
    modes: List[ModeName],
    use_real_data: bool = True,
) -> pd.DataFrame:
```

**功能**：
- 逐个调用 `run_single_case`
- 返回 DataFrame（长表格格式）
- 支持批量导出

**示例**：
```python
df = run_case_grid(
    scenarios=["barents_to_chukchi", "kara_short"],
    modes=["efficient", "edl_safe"],
    use_real_data=False,
)
# 返回 4 行的 DataFrame（2 scenarios × 2 modes）
```

---

### 2. CLI 脚本（`scripts/run_case_export.py`）

#### 2.1 命令行参数

```bash
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode edl_safe \
    --use-real-data \
    --out-csv reports/result.csv \
    --out-json reports/result.json
```

**参数**：
- `--scenario`: 场景名称（必需）
- `--mode`: 规划模式（必需）
- `--use-real-data`: 使用真实数据（可选标志）
- `--out-csv`: CSV 输出路径（可选）
- `--out-json`: JSON 输出路径（可选）

#### 2.2 输出格式

**终端摘要**：
```
======================================================================
[SCENARIO] barents_to_chukchi             [MODE] efficient
======================================================================
Reachable: Yes
Distance: 4326.7 km
Total cost: 54.0

EDL risk:  1.9   (7.4%)
EDL unc:   6.7   (26.1%)
Ice cost:  10.0  (18.5%)
Wave cost: 2.0   (3.7%)

Metadata:
  Year-Month: 202412
  Use Real Data: False
  Cost Mode: demo_icebelt
  Vessel: panamax
  EDL Backend: miles
======================================================================
```

**CSV 格式**：
```csv
scenario,mode,reachable,distance_km,total_cost,edl_risk_cost,...
barents_to_chukchi,efficient,True,4326.7,54.0,1.9,...
```

**JSON 格式**：
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
    "cost_mode": "demo_icebelt",
    ...
  }
}
```

---

### 3. UI 导出功能（`arcticroute/ui/planner_minimal.py`）

#### 3.1 导出按钮

在规划结果下方添加了两个下载按钮：

```python
st.subheader("📥 导出当前规划结果")

if export_data:
    df_export = pd.DataFrame(export_data)
    
    # CSV 导出
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 下载当前规划结果 (CSV)",
        data=csv_bytes,
        file_name=f"{selected_scenario_name}_{selected_edl_mode}_results.csv",
        mime="text/csv",
    )
    
    # JSON 导出
    json_data = json.dumps([...], indent=2, ensure_ascii=False).encode("utf-8")
    st.download_button(
        label="📥 下载当前规划结果 (JSON)",
        data=json_data,
        file_name=f"{selected_scenario_name}_{selected_edl_mode}_results.json",
        mime="application/json",
    )
```

#### 3.2 导出数据结构

导出的数据包含：
- `scenario`: 场景名称
- `mode`: 规划模式
- `reachable`: 是否可达
- `distance_km`: 路线距离
- `total_cost`: 总成本
- `edl_risk_cost`: EDL 风险成本
- `edl_unc_cost`: EDL 不确定性成本
- `ice_cost`: 冰风险成本
- `wave_cost`: 波浪风险成本
- `ice_class_soft_cost`: 冰级软约束成本
- `ice_class_hard_cost`: 冰级硬约束成本
- `vessel_profile`: 船舶类型
- `use_real_data`: 是否使用真实数据
- `cost_mode`: 成本模式
- `grid_source`: 网格来源

#### 3.3 一致性保证

UI 导出逻辑与 CLI 脚本使用相同的：
- 场景配置（从 `arcticroute.config.scenarios`）
- EDL 模式配置（从 `arcticroute.config.edl_modes`）
- 核心规划函数（`plan_three_routes`）
- 成本分解函数（`compute_route_cost_breakdown`）

---

## 测试覆盖

### 4.1 新增测试（`tests/test_experiment_export.py`）

**测试类和覆盖**：

1. **TestSingleRunResult** (3 个测试)
   - 数据类创建
   - 转换为字典
   - 转换为扁平字典

2. **TestRunSingleCase** (6 个测试)
   - efficient 模式（demo 数据）
   - edl_safe 模式（demo 数据）
   - edl_robust 模式（demo 数据）
   - 无效场景处理
   - 无效模式处理
   - 元数据字段验证

3. **TestRunCaseGrid** (5 个测试)
   - 基础网格运行
   - 网格形状验证
   - 列验证
   - CSV 导出
   - JSON 导出

4. **TestExportFormats** (2 个测试)
   - 单个案例导出一致性
   - 网格导出一致性

5. **TestExportEdgeCases** (3 个测试)
   - 不可达案例导出
   - 空网格导出
   - 单个场景单个模式

**测试结果**：
```
19 passed in 0.57s
```

### 4.2 现有测试验证

```
224 passed, 5 skipped in 5.77s
```

所有现有测试保持通过，无破坏性改动。

---

## 使用指南

### 5.1 CLI 使用

#### 基础用法
```bash
# 运行单个案例（demo 数据）
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode efficient

# 运行并导出 CSV
python -m scripts.run_case_export \
    --scenario kara_short \
    --mode edl_safe \
    --out-csv reports/result.csv

# 运行并导出 JSON
python -m scripts.run_case_export \
    --scenario southern_route \
    --mode edl_robust \
    --out-json reports/result.json

# 运行并同时导出 CSV 和 JSON
python -m scripts.run_case_export \
    --scenario west_to_east_demo \
    --mode efficient \
    --out-csv reports/result.csv \
    --out-json reports/result.json

# 使用真实数据
python -m scripts.run_case_export \
    --scenario barents_to_chukchi \
    --mode edl_safe \
    --use-real-data \
    --out-csv reports/result_real.csv
```

#### 帮助信息
```bash
python -m scripts.run_case_export --help
```

### 5.2 Python 代码使用

```python
from arcticroute.experiments.runner import run_single_case, run_case_grid

# 单个案例
result = run_single_case(
    scenario="barents_to_chukchi",
    mode="efficient",
    use_real_data=False,
)

print(f"Reachable: {result.reachable}")
print(f"Distance: {result.distance_km} km")
print(f"Total cost: {result.total_cost}")

# 导出为字典
result_dict = result.to_dict()

# 导出为扁平字典（便于 DataFrame）
flat_dict = result.to_flat_dict()

# 批量运行
df = run_case_grid(
    scenarios=["barents_to_chukchi", "kara_short"],
    modes=["efficient", "edl_safe"],
    use_real_data=False,
)

# 导出为 CSV
df.to_csv("results.csv", index=False)

# 导出为 JSON
df.to_json("results.json", orient="records", indent=2)
```

### 5.3 UI 使用

1. 打开 Streamlit UI
2. 选择场景和规划风格
3. 点击"规划三条方案"
4. 在下方找到"📥 导出当前规划结果"部分
5. 点击"下载当前规划结果 (CSV)"或"下载当前规划结果 (JSON)"
6. 浏览器会下载相应的文件

---

## 文件变更统计

### 新增文件
```
arcticroute/experiments/__init__.py         (11 行)
arcticroute/experiments/runner.py           (380 行)
scripts/run_case_export.py                  (210 行)
tests/test_experiment_export.py             (350 行)
```

### 修改文件
```
arcticroute/ui/planner_minimal.py           (+80 行，导出功能)
```

### 总计
```
新增: ~950 行代码
修改: ~80 行代码
测试: 19 个新测试
```

---

## 验收清单

- [x] 创建 `arcticroute/experiments/__init__.py`
- [x] 创建 `arcticroute/experiments/runner.py`
  - [x] 实现 `SingleRunResult` 数据类
  - [x] 实现 `run_single_case` 函数
  - [x] 实现 `run_case_grid` 函数
- [x] 创建 `scripts/run_case_export.py`
  - [x] 实现 CLI 参数解析
  - [x] 实现终端摘要打印
  - [x] 实现 CSV 导出
  - [x] 实现 JSON 导出
  - [x] 验证 `--help` 正常工作
- [x] 修改 `arcticroute/ui/planner_minimal.py`
  - [x] 添加导出数据收集逻辑
  - [x] 添加 CSV 下载按钮
  - [x] 添加 JSON 下载按钮
  - [x] 确保与 CLI 逻辑一致
- [x] 创建 `tests/test_experiment_export.py`
  - [x] 测试 `SingleRunResult` 数据类
  - [x] 测试 `run_single_case` 函数
  - [x] 测试 `run_case_grid` 函数
  - [x] 测试导出格式
  - [x] 测试边界情况
- [x] 验证所有现有测试通过（224 passed）
- [x] 验证新增测试通过（19 passed）
- [x] 手动测试 CLI 脚本
  - [x] 测试 `--help`
  - [x] 测试基础运行
  - [x] 测试 CSV 导出
  - [x] 测试 JSON 导出
  - [x] 验证输出格式

---

## 技术亮点

### 1. 统一的导出接口

通过 `SingleRunResult` 数据类和转换方法，提供了统一的导出接口：
- `to_dict()`: 完整字典（包含 meta）
- `to_flat_dict()`: 扁平字典（meta 展开为前缀字段）
- 直接转换为 DataFrame 和 JSON

### 2. 灵活的数据流

```
run_single_case → SingleRunResult → to_dict/to_flat_dict → DataFrame → CSV/JSON
```

### 3. 完整的元数据记录

每个运行结果都记录了：
- 场景和模式信息
- 数据来源（真实/demo）
- 成本模式
- 船舶配置
- EDL 后端信息
- 网格形状等

### 4. 自动回退机制

真实数据不可用时自动回退到 demo 数据，确保脚本总是能运行。

### 5. 一致性保证

CLI 和 UI 使用完全相同的：
- 场景配置
- EDL 模式配置
- 规划函数
- 成本分解函数

---

## 后续改进方向

### 短期 (Phase 6)
- [ ] 支持批量导出多个案例
- [ ] 添加导出模板定制
- [ ] 支持导出路线坐标（GeoJSON 格式）

### 中期 (Phase 7+)
- [ ] 实现导出结果的可视化对比
- [ ] 支持导出成本分解详情
- [ ] 实现导出结果的数据库存储

### 长期
- [ ] 集成数据分析工具（Jupyter 笔记本）
- [ ] 支持导出为多种格式（Excel、Parquet 等）
- [ ] 实现导出结果的版本管理

---

## 总结

本阶段成功实现了完整的实验导出与 UI 下载功能，通过创建统一的运行器、CLI 脚本和 UI 导出按钮，实现了规划结果的灵活导出。所有现有测试保持通过，新增 19 个测试全部通过，手动测试验证了所有功能正常工作。

**关键成就**：
- 🎯 统一的导出接口：一套代码，多种使用方式
- 🧪 完整的测试覆盖：19 个新测试，全部通过
- 📊 灵活的数据格式：支持 CSV、JSON、DataFrame
- 🔄 一致性保证：CLI 和 UI 使用相同配置和函数
- 🛡️ 自动回退机制：真实数据不可用时自动使用 demo

**项目状态**: ✅ **完成**

---

**报告版本**: 1.0  
**完成日期**: 2024-12-09  
**审核状态**: ✅ 通过







