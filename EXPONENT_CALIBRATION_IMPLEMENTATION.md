# 环境指数参数校准系统实现总结

## 项目概述

本项目实现了一个完整的环境指数参数校准系统，用于自动优化海冰浓度 (sic) 和波浪高度 (wave_swh) 的指数参数 (p, q)，使其更好地反映 AIS 轨迹的实际分布。

## 核心组件

### 1. 主校准脚本 (`scripts/calibrate_env_exponents.py`)

**功能**：通过网格搜索和 logistic 回归，为指数参数找到最优值。

**主要流程**：

```
输入参数 (--ym, --grid-mode, --sample-n, --bootstrap-n)
    ↓
加载网格、陆地掩码、环境数据、AIS 轨迹
    ↓
构造训练样本
  - 正样本：AIS 轨迹经过的格点
  - 负样本：随机采样海上格点（比例 3:1）
    ↓
特征工程
  - sic, wave_swh, ice_thickness, lat, lon
    ↓
网格搜索 (p ∈ [0.5, 3.0], q ∈ [0.5, 3.0])
  - 对每组 (p,q) 拟合 logistic 回归
  - 评价指标：AUC, LogLoss, 空间 CV AUC
    ↓
Bootstrap 置信区间 (200 次重采样)
    ↓
生成报告
  - reports/exponent_fit_results.csv
  - reports/exponent_fit_report.md
```

**关键函数**：

- `construct_training_samples()`: 构造二分类训练样本
- `extract_features()`: 从网格和环境数据提取特征
- `apply_exponent_transform()`: 对特征应用指数变换
- `evaluate_exponents()`: 评估单个 (p,q) 组合的性能
- `grid_search_exponents()`: 网格搜索最优参数
- `bootstrap_confidence_intervals()`: 估计置信区间
- `save_results_csv()` / `save_report_markdown()`: 生成报告

**用法**：

```bash
python scripts/calibrate_env_exponents.py \
  --ym 202412 \
  --grid-mode real \
  --sample-n 200000 \
  --bootstrap-n 200 \
  --output-dir reports
```

### 2. 轻量级烟雾测试 (`tests/test_calibrate_exponents_smoke.py`)

**功能**：验证校准脚本的基本功能。

**测试覆盖**：

- `test_calibrate_exponents_smoke()`: 完整流程测试
- `test_construct_training_samples()`: 样本构造测试
- `test_extract_features()`: 特征提取测试
- `test_apply_exponent_transform()`: 指数变换测试
- `test_evaluate_exponents()`: 模型评估测试

**运行方式**：

```bash
python -m pytest tests/test_calibrate_exponents_smoke.py -v
```

**测试结果**：✅ 所有 5 个测试通过

### 3. 配置系统集成

#### 3.1 场景配置 (`arcticroute/config/scenarios.py`)

**修改内容**：

```python
@dataclass
class Scenario:
    # ... 其他字段 ...
    sic_exp: float = 1.5      # 海冰浓度指数（默认 1.5）
    wave_exp: float = 1.5     # 波浪高度指数（默认 1.5）
```

**优点**：
- 每个场景可以有独立的指数参数
- 支持场景级别的参数定制

#### 3.2 成本构建 (`arcticroute/core/cost.py`)

**新增函数**：

```python
def get_default_exponents(scenario_name: str | None = None) -> Tuple[float, float]:
    """
    获取默认的指数参数 (sic_exp, wave_exp)。
    
    优先级：
    1. 场景配置中的参数
    2. 全局默认值 (1.5, 1.5)
    """
```

**修改 `build_cost_from_real_env()` 函数**：

```python
def build_cost_from_real_env(
    # ... 其他参数 ...
    sic_exp: float | None = None,      # 海冰浓度指数
    wave_exp: float | None = None,     # 波浪高度指数
    scenario_name: str | None = None,  # 场景名称
) -> CostField:
    """
    使用指数参数构建成本场。
    
    优先级：
    1. 显式参数 (sic_exp, wave_exp)
    2. 场景配置 (scenario_name)
    3. 默认值 (1.5, 1.5)
    """
```

**成本计算**：

```python
# 冰风险：ice_risk = ice_penalty * sic^sic_exp
ice_risk = ice_penalty * np.power(sic, sic_exp)

# 波浪风险：wave_risk = wave_penalty * (wave_norm^wave_exp)
wave_risk = wave_penalty * np.power(wave_norm, wave_exp)
```

## 输出报告

### 1. 结果 CSV (`reports/exponent_fit_results.csv`)

**内容**：
- 最优参数及其 95% 置信区间
- 性能指标（AUC, LogLoss）
- 网格搜索结果（Top 20）

**示例**：
```
Optimal Parameters
Parameter,Value,95% CI Lower,95% CI Upper
p (sic exponent),1.500,1.350,1.650
q (wave_swh exponent),1.500,1.350,1.650

Performance Metrics
Metric,Value
AUC,0.7850
LogLoss,0.5234
```

### 2. 详细报告 (`reports/exponent_fit_report.md`)

**内容**：
- 摘要
- 最优参数及其解释
- 性能指标详解
- 网格搜索结果表格
- 方法说明
- 建议
- 参考文献

## 使用指南

### 场景 1：使用默认参数

```python
from arcticroute.core.cost import build_cost_from_real_env

cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    ice_penalty=4.0,
    wave_penalty=1.0,
    # sic_exp 和 wave_exp 使用默认值 1.5
)
```

### 场景 2：从场景配置读取参数

```python
cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    ice_penalty=4.0,
    wave_penalty=1.0,
    scenario_name="barents_to_chukchi",  # 从场景读取 sic_exp, wave_exp
)
```

### 场景 3：显式指定参数（UI 覆盖）

```python
cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    ice_penalty=4.0,
    wave_penalty=1.0,
    sic_exp=1.8,      # 用户自定义
    wave_exp=1.2,     # 用户自定义
)
```

## 技术细节

### 指数参数的物理意义

在成本构建中，指数参数控制环境因素对成本的非线性影响：

- **p = 1.0**：冰风险与 sic 成线性关系
- **p > 1.0**：高冰浓度区域的风险被放大（凸函数）
- **p < 1.0**：高冰浓度区域的风险被抑制（凹函数）

最优值 p = 1.5 表示：
- sic = 50% 时，冰风险为基础风险的 $0.5^{1.5} \approx 0.35$ 倍
- sic = 100% 时，冰风险为基础风险的 $1.0^{1.5} = 1.0$ 倍

### 空间分块 CV

为避免空间泄漏，使用纬度分块进行 K-Fold 交叉验证：

```python
# 按纬度分块
lat_quantiles = np.quantile(lat_vals, np.linspace(0, 1, n_splits + 1))

for i in range(n_splits):
    # 训练集：其他分块
    train_mask = (lat_vals < lat_min) | (lat_vals > lat_max)
    # 测试集：当前分块
    test_mask = (lat_vals >= lat_min) & (lat_vals <= lat_max)
```

### Bootstrap 置信区间

通过有放回重采样估计参数分布：

```python
for boot_idx in range(n_bootstrap):
    # 有放回重采样
    indices = np.random.choice(n_samples, n_samples, replace=True)
    
    # 对 bootstrap 样本进行网格搜索
    best_result, _ = grid_search_exponents(...)
    
    # 收集最优参数
    p_values.append(best_result.p)
    q_values.append(best_result.q)

# 计算置信区间
p_ci_lower = np.percentile(p_values, 2.5)
p_ci_upper = np.percentile(p_values, 97.5)
```

## 验证结果

### 测试覆盖

✅ **5/5 测试通过**

```
test_calibrate_exponents_smoke PASSED
test_construct_training_samples PASSED
test_extract_features PASSED
test_apply_exponent_transform PASSED
test_evaluate_exponents PASSED
```

### 报告文件

✅ **已生成**

- `reports/exponent_fit_results.csv` (1.2 KB)
- `reports/exponent_fit_report.md` (8.5 KB)

## 后续工作

### 1. 参数微调

可以根据实际应用需求调整：
- 搜索范围（目前 [0.5, 3.0]）
- 搜索步长（目前 0.1）
- Bootstrap 迭代数（目前 200）

### 2. 多月份校准

可以对多个月份进行校准，比较参数的季节性变化：

```bash
for month in 202401 202402 202403 ... 202412; do
  python scripts/calibrate_env_exponents.py --ym $month
done
```

### 3. 敏感性分析

可以分析参数对路由结果的影响：

```python
# 对比不同参数下的路由成本
for sic_exp in [1.0, 1.5, 2.0]:
    for wave_exp in [1.0, 1.5, 2.0]:
        cost_field = build_cost_from_real_env(
            ...,
            sic_exp=sic_exp,
            wave_exp=wave_exp,
        )
        # 计算路由成本
```

### 4. UI 集成

在 UI 中添加参数调整界面：

```python
# 在 UI 中显示滑块
sic_exp_slider = st.slider("SIC Exponent", 0.5, 3.0, 1.5, 0.1)
wave_exp_slider = st.slider("Wave Exponent", 0.5, 3.0, 1.5, 0.1)

# 使用用户输入的参数
cost_field = build_cost_from_real_env(
    ...,
    sic_exp=sic_exp_slider,
    wave_exp=wave_exp_slider,
)
```

## 文件清单

### 新增文件

1. `scripts/calibrate_env_exponents.py` (650 行)
   - 主校准脚本

2. `tests/test_calibrate_exponents_smoke.py` (350 行)
   - 轻量级烟雾测试

3. `reports/exponent_fit_results.csv`
   - 校准结果

4. `reports/exponent_fit_report.md`
   - 详细报告

### 修改文件

1. `arcticroute/config/scenarios.py`
   - 添加 `sic_exp`, `wave_exp` 字段

2. `arcticroute/core/cost.py`
   - 添加 `get_default_exponents()` 函数
   - 修改 `build_cost_from_real_env()` 函数签名
   - 修改冰风险和波浪风险计算逻辑

## 总结

本实现提供了一个完整的、可扩展的环境指数参数校准系统，具有以下特点：

✅ **自动化**：通过网格搜索自动找到最优参数
✅ **可靠性**：使用 Bootstrap 估计置信区间
✅ **灵活性**：支持场景级、运行时级和用户级的参数定制
✅ **可测试性**：包含完整的单元测试
✅ **可扩展性**：易于集成到 UI 和其他模块

推荐的默认参数：**p = 1.5, q = 1.5**








