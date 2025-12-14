# EDL-miles-guess 集成快速参考

## 概览

miles-guess 库已成功集成到 AR_final 项目中，作为真实的 EDL 风险推理后端。系统会自动优先使用 miles-guess，若不可用则回退到 PyTorch 实现。

## 快速开始

### 1. 检查 miles-guess 可用性

```python
from arcticroute.core.edl_backend_miles import has_miles_guess

if has_miles_guess():
    print("✅ miles-guess 可用")
else:
    print("⚠️ miles-guess 不可用，将使用 PyTorch")
```

### 2. 启用 EDL 风险推理

```python
from arcticroute.core.cost import build_cost_from_real_env

cost_field = build_cost_from_real_env(
    grid=grid,
    land_mask=land_mask,
    env=env,
    use_edl=True,           # 启用 EDL
    w_edl=2.0,              # EDL 权重
    use_edl_uncertainty=True,  # 启用不确定性
    edl_uncertainty_weight=1.0,
)

# 检查 EDL 来源
print(f"EDL 来源: {cost_field.meta['edl_source']}")
```

### 3. 访问 EDL 输出

```python
# 访问 EDL 风险成本
edl_risk = cost_field.components.get("edl_risk")
if edl_risk is not None:
    print(f"EDL 风险范围: [{edl_risk.min():.2f}, {edl_risk.max():.2f}]")

# 访问 EDL 不确定性
if cost_field.edl_uncertainty is not None:
    print(f"不确定性范围: [{cost_field.edl_uncertainty.min():.2f}, {cost_field.edl_uncertainty.max():.2f}]")
```

## 关键 API

### EDLGridOutput

```python
@dataclass
class EDLGridOutput:
    risk: np.ndarray           # 风险分数，shape (H, W)，值域 [0, 1]
    uncertainty: np.ndarray    # 不确定性，shape (H, W)
    meta: dict                 # 元数据
```

**meta 字段**:
- `source`: "miles-guess" | "pytorch" | "placeholder"
- `model_name`: 使用的模型名称
- `device`: "cpu" | "cuda"

### run_miles_edl_on_grid()

```python
from arcticroute.core.edl_backend_miles import run_miles_edl_on_grid

edl_output = run_miles_edl_on_grid(
    sic=sic_array,              # 海冰浓度，shape (H, W)
    swh=wave_array,             # 波浪有效波高，shape (H, W)
    ice_thickness=thickness,    # 冰厚，shape (H, W)
    grid_lat=lat_array,         # 纬度，shape (H, W)
    grid_lon=lon_array,         # 经度，shape (H, W)
)

# 检查来源
if edl_output.meta["source"] == "miles-guess":
    print("✅ 使用 miles-guess 推理")
else:
    print("⚠️ 使用占位实现")
```

## 工作流程

```
用户启用 EDL (use_edl=True)
    ↓
build_cost_from_real_env()
    ↓
尝试 miles-guess 后端
    ├─ 成功 → 使用真实推理 (meta["source"]="miles-guess")
    └─ 失败 → 尝试 PyTorch 实现
        ├─ 成功 → 使用 PyTorch (meta["source"]="pytorch")
        └─ 失败 → 无 EDL (meta["source"]=None)
    ↓
融合进成本场
    ├─ components["edl_risk"] = w_edl * risk
    └─ edl_uncertainty = uncertainty
    ↓
UI 显示
    ├─ 成本分解表格（带来源标记）
    ├─ 不确定性剖面
    └─ 综合评分
```

## 常见问题

### Q: 如何强制使用 PyTorch 而不是 miles-guess？

A: 目前不支持强制选择。系统会自动优先使用 miles-guess，若不可用则回退。如果需要禁用 miles-guess，可以卸载该库。

### Q: miles-guess 推理失败时会发生什么？

A: 系统会自动回退到 PyTorch 实现，并记录警告日志。路径规划不会中断。

### Q: 如何检查当前使用的是哪个后端？

A: 检查 `cost_field.meta["edl_source"]` 字段：
- `"miles-guess"`: 使用 miles-guess
- `"pytorch"`: 使用 PyTorch
- `None`: 未启用 EDL

### Q: 不确定性的含义是什么？

A: 不确定性表示模型对风险预测的置信度。值越高，模型越不确定。可用于识别高风险区域。

### Q: 如何调整 EDL 权重？

A: 使用 `w_edl` 参数控制 EDL 风险在总成本中的影响：
```python
cost_field = build_cost_from_real_env(
    ...,
    w_edl=2.0,  # 增加此值以增加 EDL 权重
    edl_uncertainty_weight=1.0,  # 不确定性权重
)
```

## 测试

### 运行 smoke test

```bash
pytest tests/test_edl_backend_miles_smoke.py -v
```

### 运行集成测试

```bash
pytest tests/test_cost_with_miles_edl.py -v
```

### 运行全套测试

```bash
pytest -q
```

## 文件位置

| 文件 | 说明 |
|------|------|
| `arcticroute/core/edl_backend_miles.py` | miles-guess 后端适配器 |
| `arcticroute/core/cost.py` | 成本构建（已修改以支持 miles-guess） |
| `arcticroute/ui/planner_minimal.py` | UI（已修改以显示 EDL 来源） |
| `tests/test_edl_backend_miles_smoke.py` | smoke test |
| `tests/test_cost_with_miles_edl.py` | 集成测试 |
| `docs/EDL_INTEGRATION_NOTES.md` | 详细梳理文档 |
| `docs/EDL_MILES_INTEGRATION_REPORT.md` | 完整集成报告 |

## 下一步

1. 在实际环境中测试 miles-guess 推理性能
2. 根据实际数据调整特征归一化参数
3. 考虑支持多个 miles-guess 模型的选择
4. 收集用户反馈，优化 UI 显示

## 联系方式

如有问题或建议，请参考完整的集成报告：`docs/EDL_MILES_INTEGRATION_REPORT.md`

















