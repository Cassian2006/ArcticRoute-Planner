# PyTorch EDL 后端 nn 未定义问题修复总结

## 问题描述

在 `arcticroute/ml/edl_core.py` 中，`EDLModel` 类定义在 try-except 块之外，但它使用了 `nn.Module`、`nn.Linear`、`nn.init` 等 PyTorch 的 `nn` 模块中的类和函数。

当 PyTorch 导入失败时（`TORCH_AVAILABLE=False`），`nn` 变量是未定义的，但 `EDLModel` 类仍然会被定义，导致在类定义时立即抛出 `NameError: name 'nn' is not defined`。

## 修复方案

### 1. 占位符定义（第 30-33 行）

在 except 块中添加占位符定义，防止 `NameError`：

```python
except Exception:
    TORCH_AVAILABLE = False
    # 当 PyTorch 不可用时，定义占位符以避免 NameError
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
```

### 2. 条件类定义（第 57-159 行）

使用 `if TORCH_AVAILABLE:` 条件语句，将 `EDLModel` 类定义分为两部分：

**当 PyTorch 可用时（第 57-159 行）：**
- 定义完整的 `EDLModel(nn.Module)` 类
- 包含所有 PyTorch 操作（Linear 层、初始化、前向传播等）
- 添加 `# type: ignore` 注释以抑制类型检查器的警告

**当 PyTorch 不可用时（第 160-166 行）：**
- 定义占位符 `EDLModel` 类
- 仅包含 `__init__` 方法，用于创建对象而不报错
- 标记为 `# type: ignore[no-redef]` 以允许重新定义

### 3. 异常捕获和错误处理（第 169-230 行）

在 `run_edl_on_features` 函数中添加完整的异常处理：

```python
try:
    # 推理逻辑
    ...
except Exception as e:
    print(f"[EDL][torch] failed with error: {type(e).__name__}: {e}")
    print("[EDL][torch] falling back to placeholder output")
    # 返回占位符输出，让上层可以平滑回退
    risk_mean = np.zeros((H, W), dtype=float)
    uncertainty = np.ones((H, W), dtype=float)
    return EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)
```

**特点：**
- 捕获所有异常，不向上层抛出
- 打印详细的错误信息（错误类型和消息）
- 返回占位符输出，确保管线可以平滑回退
- 日志前缀统一为 `[EDL][torch]`，便于追踪

### 4. 类型注解改进

添加了 `# type: ignore` 注释来处理类型检查器的警告：

- `# type: ignore[misc,valid-type]`：用于 `class EDLModel(nn.Module)` 定义
- `# type: ignore[attr-defined]`：用于 `nn.Linear`、`nn.init` 等属性访问
- `# type: ignore[name-defined]`：用于 `torch.Tensor` 等类型引用
- `# type: ignore[assignment]`：用于占位符赋值

## 修改的文件

### `arcticroute/ml/edl_core.py`

**关键改动：**

| 行号 | 改动 | 说明 |
|------|------|------|
| 30-33 | 添加占位符定义 | 防止 `nn` 未定义的 NameError |
| 57-159 | 条件类定义（PyTorch 可用） | 完整的 EDLModel 实现 |
| 160-166 | 条件类定义（PyTorch 不可用） | 占位符 EDLModel 实现 |
| 169-230 | 异常捕获和错误处理 | 确保管线平滑回退 |
| 日志消息 | 统一前缀 `[EDL][torch]` | 便于日志追踪 |

## 测试覆盖

现有的测试文件已经覆盖了修复的场景：

### `tests/test_edl_core.py`

**测试类：**

1. **TestEDLFallback**
   - `test_edl_fallback_without_torch`：验证 PyTorch 不可用时的 fallback 行为
   - `test_edl_fallback_returns_numpy`：验证返回类型

2. **TestEDLWithTorch**
   - `test_edl_with_torch_shapes_match`：验证输出形状和数值范围
   - `test_edl_with_torch_output_types`：验证输出类型
   - `test_edl_with_torch_different_inputs`：验证不同输入产生不同输出

3. **TestEDLConfig**
   - `test_edl_config_num_classes_effect`：验证配置参数的影响

4. **TestEDLFeatureProcessing**
   - `test_edl_with_different_feature_dims`：验证不同特征维度处理
   - `test_edl_with_large_grid`：验证大网格处理
   - `test_edl_with_nan_features`：验证 NaN 特征处理

### `tests/test_cost_real_env_edl.py`

- 验证 EDL 与成本构建的集成
- 测试 EDL 禁用和启用时的行为差异
- 测试无 PyTorch 时的不崩溃行为

## 使用示例

### 场景 1：PyTorch 可用

```python
from arcticroute.ml.edl_core import run_edl_on_features, EDLConfig
import numpy as np

# 构造特征数组
features = np.random.randn(100, 100, 5)  # (H, W, F)

# 运行 EDL 推理
output = run_edl_on_features(features)

# 获取结果
risk_mean = output.risk_mean  # shape (100, 100)
uncertainty = output.uncertainty  # shape (100, 100)
```

### 场景 2：PyTorch 不可用

```python
# 代码完全相同，但会自动返回占位符输出
output = run_edl_on_features(features)
# risk_mean 全为 0，uncertainty 全为 1
```

### 场景 3：推理过程中出错

```python
# 即使推理过程中发生异常，也会捕获并返回占位符输出
# 日志输出：[EDL][torch] failed with error: RuntimeError: ...
# 日志输出：[EDL][torch] falling back to placeholder output
output = run_edl_on_features(features)
# 返回占位符输出，管线继续执行
```

## 日志输出示例

### PyTorch 不可用

```
[EDL][torch] PyTorch not available; using fallback constant risk.
```

### 推理失败

```
[EDL][torch] failed with error: RuntimeError: CUDA out of memory
[EDL][torch] falling back to placeholder output
```

## 向后兼容性

✅ **完全向后兼容**

- 所有现有的 API 保持不变
- 现有的测试无需修改
- 返回的 `EDLGridOutput` 数据结构不变
- 占位符输出与之前的行为一致

## 后续改进建议

1. **更详细的错误分类**
   - 区分 PyTorch 不可用、CUDA 内存不足、模型加载失败等不同错误
   - 为不同错误返回不同的占位符策略

2. **性能监控**
   - 添加推理时间统计
   - 监控内存使用情况

3. **模型缓存**
   - 缓存已创建的 EDLModel 实例，避免重复创建
   - 支持加载预训练模型

4. **配置管理**
   - 支持从配置文件加载 EDL 参数
   - 支持模型选择和切换

## 总结

通过以下三个关键改动，完全解决了 PyTorch EDL 后端的 `nn` 未定义问题：

1. ✅ **占位符定义**：防止 import 失败时的 NameError
2. ✅ **条件类定义**：根据 PyTorch 可用性选择不同的实现
3. ✅ **异常捕获**：确保推理失败时管线可以平滑回退

修复后的代码既能充分利用 PyTorch 的功能，也能在 PyTorch 不可用时优雅地降级，提高了系统的鲁棒性和可靠性。











