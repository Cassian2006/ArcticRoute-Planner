# EDL 集成笔记

## 概述

本文档记录了 AR_final 项目中 EDL（Evidential Deep Learning）风险推理的当前实现状态，以及与 miles-guess 库集成的计划。

---

## 第一部分：当前 EDL 占位实现梳理

### 1.1 EDL 核心模块位置

- **主模块**: `arcticroute/ml/edl_core.py`
  - 提供 `EDLGridOutput` 数据类（包含 `risk_mean` 和 `uncertainty`）
  - 提供 `EDLModel` 类（基于 PyTorch 的极简 MLP + Dirichlet 头）
  - 提供 `run_edl_on_features()` 函数，用于在特征网格上运行 EDL 推理
  - 当 PyTorch 不可用时，返回占位符输出（risk_mean=0, uncertainty=1）

- **后端适配器**: `arcticroute/core/edl_backend_miles.py`
  - 提供 `has_miles_guess()` 函数，检测 miles-guess 库是否可用
  - 提供 `edl_dummy_on_grid()` 函数，生成纯占位 EDL 结果
  - 提供 `edl_from_miles_guess_demo()` 函数（演示性实现，目前为占位）

### 1.2 EDL 数据流

#### 特征构造 (in `cost.py`)

在 `build_cost_from_real_env()` 中，当 `use_edl=True` 且 `w_edl > 0` 时：

1. **特征堆叠** (shape: H×W×5)
   - `sic_norm`: 海冰浓度，归一化到 [0, 1]
   - `wave_swh_norm`: 波浪有效波高，归一化到 [0, 1]（max=10m）
   - `ice_thickness_norm`: 冰厚，归一化到 [0, 1]（max=2m）
   - `lat_norm`: 纬度，归一化到 [0, 1]（范围 60°N～85°N）
   - `lon_norm`: 经度，归一化到 [0, 1]（范围 -180°～180°）

2. **调用 EDL 推理**
   ```python
   edl_output = run_edl_on_features(features, config=EDLConfig(num_classes=3))
   ```
   - 返回 `EDLGridOutput` 对象，包含 `risk_mean` (H×W) 和 `uncertainty` (H×W)

3. **融合进成本**
   ```python
   edl_cost = w_edl * edl_output.risk_mean
   cost = cost + edl_cost
   components["edl_risk"] = edl_cost
   ```

#### 不确定性处理 (in `cost.py`)

当 `use_edl_uncertainty=True` 且 `edl_uncertainty_weight > 0` 时：

1. **提取不确定性**
   - 从 `edl_output.uncertainty` 中获取，clip 到 [0, 1]

2. **构造不确定性成本**
   ```python
   unc_cost = edl_uncertainty_weight * uncertainty
   cost = cost + unc_cost
   components["edl_uncertainty_penalty"] = unc_cost
   ```

3. **记录到 CostField**
   ```python
   cost_field.edl_uncertainty = edl_uncertainty
   ```

### 1.3 EDL 在成本分解中的角色

在 `analysis.py` 的 `compute_route_cost_breakdown()` 中：

- 遍历 `cost_field.components` 字典
- 对每个组件（包括 `"edl_risk"` 和 `"edl_uncertainty_penalty"`）沿路径求和
- 计算各组件的占比 `component_fractions`
- 生成沿程数据 `component_along_path`

### 1.4 EDL 在 UI 中的展示

在 `planner_minimal.py` 中：

#### 摘要表格 (Summary Table)
- 新增列 `"EDL风险成本"` 和 `"EDL不确定性成本"`
- 从 `compute_route_cost_breakdown()` 的 `component_totals` 中提取

#### 评分与推荐
- `compute_route_scores()` 从 `breakdowns` 中提取 `edl_risk_cost` 和 `edl_uncertainty_cost`
- 进行 min-max 归一化，得到 `norm_edl_risk` 和 `norm_edl_uncertainty`
- 综合评分：`composite_score = weight_fuel * norm_fuel + weight_risk * norm_edl_risk + weight_uncertainty * norm_edl_uncertainty`

#### 成本分解展示 (Cost Breakdown)
- 显示 `edl_safe` 方案的成本分解表格
- 标记 EDL 相关组件：`"🧠 EDL 风险"`
- 绘制成本组件贡献柱状图

#### EDL 不确定性剖面 (Uncertainty Profile)
- 在 `edl_robust` 方案中显示沿程不确定性折线图
- 计算高不确定性（>0.7）的路段占比
- 给出警告提示

#### 来源标记（待实现）
- 若 `cost_field.meta` 中有 `edl_source="miles-guess"`，显示 `"[miles-guess]"` 标签
- 若无 EDL 或占位实现，提示 `"EDL 未启用或无有效模型"`

---

## 第二部分：miles-guess 集成计划

### 2.1 当前 edl_backend_miles.py 的状态

**现有函数**：
- `has_miles_guess()`: 检测库可用性 ✓
- `edl_dummy_on_grid()`: 占位实现 ✓
- `edl_from_miles_guess_demo()`: 演示性实现（需完善）

**缺失部分**：
- 真实的网格级推理函数 `run_miles_edl_on_grid()`
- 对 miles-guess API 的正确调用
- 完整的异常处理和回退机制

### 2.2 miles-guess 集成的关键设计

#### 函数签名（目标）

```python
@dataclass
class EDLGridOutput:
    risk: np.ndarray           # shape = (H, W), 已经对齐到我们网格
    uncertainty: np.ndarray    # shape = (H, W)
    meta: dict                 # 元数据，包括 source, model_name 等

def run_miles_edl_on_grid(
    sic: np.ndarray,
    swh: np.ndarray | None,
    ice_thickness: np.ndarray | None,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    *,
    model_name: str = "default",
    device: str = "cpu",
) -> EDLGridOutput:
    """
    在网格上运行 miles-guess EDL 推理。
    
    Args:
        sic: 海冰浓度，shape (H, W)，值域 [0, 1]
        swh: 波浪有效波高，shape (H, W)，单位 m；可为 None
        ice_thickness: 冰厚，shape (H, W)，单位 m；可为 None
        grid_lat: 纬度网格，shape (H, W)
        grid_lon: 经度网格，shape (H, W)
        model_name: 模型名称（默认 "default"）
        device: 计算设备（"cpu" 或 "cuda"）
    
    Returns:
        EDLGridOutput 对象，包含 risk、uncertainty 和 meta
    
    Raises:
        ImportError: miles-guess 不可用
        RuntimeError: 推理失败
    """
```

#### 集成策略

1. **优先级**：
   - 若 miles-guess 可用且数据满足要求 → 使用真实推理
   - 若 miles-guess 不可用或推理失败 → 回退到占位实现
   - 不破坏现有 API，所有降级都是透明的

2. **异常处理**：
   - `ImportError`: miles-guess 库不存在 → 记录日志，返回占位结果
   - `RuntimeError`: 推理过程出错 → 记录日志，返回占位结果
   - 不向上层抛出异常，保证路径规划不中断

3. **元数据追踪**：
   - 在 `EDLGridOutput.meta` 中记录 `source` 字段
   - `source="miles-guess"` 表示真实推理
   - `source="placeholder"` 表示占位实现
   - UI 可根据此标记显示不同的提示

### 2.3 与现有 edl_core.py 的关系

- `edl_core.py` 中的 `run_edl_on_features()` 基于 PyTorch，用于特征级推理
- `edl_backend_miles.py` 中的 `run_miles_edl_on_grid()` 基于 miles-guess，用于网格级推理
- 两者都返回 `EDLGridOutput`，但字段名称需要统一
  - 当前 `edl_core.py` 使用 `risk_mean` 和 `uncertainty`
  - 目标 `edl_backend_miles.py` 使用 `risk` 和 `uncertainty`
  - **需要在 `cost.py` 中统一字段名称**

---

## 第三部分：集成检查清单

### Step 1 完成标志
- [ ] 文档已创建（本文件）
- [ ] 梳理完成，所有关键代码位置已标注

### Step 2 完成标志
- [ ] `edl_backend_miles.py` 中 `run_miles_edl_on_grid()` 已实现
- [ ] 异常处理和回退机制已完成
- [ ] smoke test 通过（`tests/test_edl_backend_miles_smoke.py`）

### Step 3 完成标志
- [ ] `cost.py` 中已调用 `run_miles_edl_on_grid()`
- [ ] EDL 输出已正确融合进成本
- [ ] 向后兼容性已验证（`tests/test_cost_with_miles_edl.py`）

### Step 4 完成标志
- [ ] UI 中已显示 EDL 来源标记
- [ ] 不确定性剖面已显示
- [ ] 无破坏性改动

### Step 5 完成标志
- [ ] 全套测试通过
- [ ] 集成报告已生成（`docs/EDL_MILES_INTEGRATION_REPORT.md`）

---

## 参考资源

- **EDL 核心**: `arcticroute/ml/edl_core.py`
- **后端适配**: `arcticroute/core/edl_backend_miles.py`
- **成本构建**: `arcticroute/core/cost.py` (lines with `use_edl`, `w_edl`)
- **分析工具**: `arcticroute/core/analysis.py` (compute_route_cost_breakdown, compute_route_scores)
- **UI 展示**: `arcticroute/ui/planner_minimal.py` (render 函数中的 EDL 相关部分)











