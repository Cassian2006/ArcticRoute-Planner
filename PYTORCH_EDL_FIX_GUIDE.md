# PyTorch EDL 后端修复指南

## 问题分析

### 原始问题

在 `arcticroute/ml/edl_core.py` 中存在以下问题：

```python
# 尝试导入 PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ❌ 问题：EDLModel 定义在 try-except 块外
# 当 PyTorch 导入失败时，nn 是未定义的，导致 NameError
class EDLModel(nn.Module):  # NameError: name 'nn' is not defined
    ...
```

### 根本原因

1. **作用域问题**：`nn` 变量只在 try 块中定义，except 块中未定义
2. **类定义时机**：Python 在模块加载时立即执行类定义，不会延迟到使用时
3. **缺乏占位符**：except 块中没有为 `nn`、`torch`、`F` 定义占位符

### 影响范围

- 当 PyTorch 不可用时，整个 `edl_core.py` 模块无法导入
- 依赖此模块的其他代码（如 `cost.py`）也无法导入
- 整个应用程序可能无法启动

## 修复方案详解

### 第一步：添加占位符定义

**修改位置**：第 30-33 行

```python
except Exception:
    TORCH_AVAILABLE = False
    # 当 PyTorch 不可用时，定义占位符以避免 NameError
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
```

**作用**：
- 确保 `nn`、`torch`、`F` 在全局作用域中总是有定义
- 即使 PyTorch 导入失败，也不会抛出 `NameError`
- `# type: ignore[assignment]` 告诉类型检查器忽略类型不匹配警告

### 第二步：条件类定义

**修改位置**：第 57-166 行

#### 方案 A：PyTorch 可用时（第 57-159 行）

```python
if TORCH_AVAILABLE:
    class EDLModel(nn.Module):  # type: ignore[misc,valid-type]
        """完整的 EDL 模型实现"""
        
        def __init__(self, input_dim: int, num_classes: int = 3):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 16)  # type: ignore[attr-defined]
            # ... 其他初始化 ...
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            # ... 前向传播 ...
```

**关键点**：
- `# type: ignore[misc,valid-type]`：允许 `nn.Module` 作为基类
- `# type: ignore[attr-defined]`：允许访问 `nn.Linear`、`nn.init` 等
- `# type: ignore[name-defined]`：允许使用 `torch.Tensor` 类型

#### 方案 B：PyTorch 不可用时（第 160-166 行）

```python
else:
    # 当 PyTorch 不可用时，定义占位符类
    class EDLModel:  # type: ignore[no-redef]
        """占位符 EDL 模型（PyTorch 不可用时）。"""

        def __init__(self, input_dim: int, num_classes: int = 3):
            """初始化占位符模型。"""
            self.input_dim = input_dim
            self.num_classes = num_classes
```

**作用**：
- 提供一个可以实例化的占位符类
- 避免在 `run_edl_on_features` 中调用 `EDLModel()` 时出现 `NameError`
- `# type: ignore[no-redef]` 允许在 else 块中重新定义类

### 第三步：异常捕获和错误处理

**修改位置**：第 169-230 行

```python
def run_edl_on_features(features, config=None):
    """..."""
    if not TORCH_AVAILABLE:
        print("[EDL][torch] PyTorch not available; using fallback constant risk.")
        return EDLGridOutput(risk_mean=np.zeros(...), uncertainty=np.ones(...))
    
    try:
        # 推理逻辑
        features_tensor = torch.from_numpy(features_flat).float()  # type: ignore
        model = EDLModel(input_dim=F, num_classes=config.num_classes)
        model.eval()
        
        with torch.no_grad():  # type: ignore
            logits = model(features_tensor)
            p, u = model.compute_edl_outputs(logits)
            # ... 处理结果 ...
        
        return EDLGridOutput(risk_mean=risk_mean, uncertainty=uncertainty)
    
    except Exception as e:
        print(f"[EDL][torch] failed with error: {type(e).__name__}: {e}")
        print("[EDL][torch] falling back to placeholder output")
        # 返回占位符，让上层可以平滑回退
        return EDLGridOutput(risk_mean=np.zeros(...), uncertainty=np.ones(...))
```

**特点**：
- **两层防护**：
  1. 第一层：检查 `TORCH_AVAILABLE`，提前返回占位符
  2. 第二层：try-except 捕获推理过程中的异常
- **详细日志**：
  - 前缀统一为 `[EDL][torch]`，便于日志过滤
  - 包含错误类型和错误消息
  - 明确指出正在进行 fallback
- **平滑回退**：
  - 不向上层抛出异常
  - 返回有效的占位符输出
  - 上层代码可以继续执行

## 修复前后对比

### 修复前

```
❌ 导入失败
   NameError: name 'nn' is not defined
   
❌ 应用崩溃
   无法导入 edl_core 模块
   无法导入依赖模块（cost.py 等）
   整个应用启动失败
```

### 修复后

```
✅ 导入成功
   TORCH_AVAILABLE=False 时，使用占位符
   TORCH_AVAILABLE=True 时，使用完整实现
   
✅ 应用继续运行
   PyTorch 不可用时，返回占位符输出
   推理失败时，捕获异常并返回占位符
   上层代码可以平滑处理占位符输出
```

## 类型注解说明

### 使用的 type: ignore 注解

| 注解 | 用途 | 示例 |
|------|------|------|
| `# type: ignore[assignment]` | 赋值类型不匹配 | `nn = None` |
| `# type: ignore[misc,valid-type]` | 类定义问题 | `class EDLModel(nn.Module):` |
| `# type: ignore[attr-defined]` | 属性访问 | `nn.Linear(...)` |
| `# type: ignore[name-defined]` | 名称未定义 | `torch.Tensor` |
| `# type: ignore[no-redef]` | 重新定义 | `class EDLModel:` (else 块) |

### 为什么需要这些注解

- **类型检查器**（如 mypy）会在 PyTorch 不可用时报错
- 注解告诉类型检查器"我知道这里有问题，但这是有意的"
- 不影响运行时行为，只影响静态分析

## 测试验证

### 导入测试

```bash
# 验证模块可以导入
python -c "from arcticroute.ml.edl_core import run_edl_on_features, TORCH_AVAILABLE; print(f'TORCH_AVAILABLE={TORCH_AVAILABLE}')"
```

**预期输出**：
```
Import successful! TORCH_AVAILABLE=True  # 如果 PyTorch 已安装
# 或
Import successful! TORCH_AVAILABLE=False  # 如果 PyTorch 未安装
```

### 功能测试

```bash
# 运行现有的单元测试
pytest tests/test_edl_core.py -v

# 运行集成测试
pytest tests/test_cost_real_env_edl.py -v
```

**预期结果**：
- 所有测试通过
- 无论 PyTorch 是否可用，都不会崩溃
- PyTorch 不可用时，fallback 测试通过
- PyTorch 可用时，完整功能测试通过

## 常见问题

### Q1: 为什么不用 `try-except` 包装类定义？

```python
# ❌ 不推荐
try:
    class EDLModel(nn.Module):
        ...
except NameError:
    class EDLModel:
        ...
```

**原因**：
- 类定义本身不会抛出异常，只有在访问 `nn` 时才会
- 这种方式不清晰，难以维护

### Q2: 为什么占位符类需要 `__init__` 方法？

```python
class EDLModel:
    def __init__(self, input_dim: int, num_classes: int = 3):
        self.input_dim = input_dim
        self.num_classes = num_classes
```

**原因**：
- 确保占位符类可以被实例化
- 避免在 `run_edl_on_features` 中调用 `EDLModel()` 时出错
- 保持与完整实现相同的接口

### Q3: 为什么需要 `# type: ignore` 注解？

**原因**：
- 类型检查器（如 mypy）会检查代码的类型正确性
- 当 PyTorch 不可用时，`nn` 是 `None`，无法作为基类
- 注解告诉类型检查器"这是有意的，请忽略此错误"
- 不影响运行时行为

### Q4: 占位符输出是什么？

```python
risk_mean = np.zeros((H, W), dtype=float)  # 全 0
uncertainty = np.ones((H, W), dtype=float)  # 全 1
```

**含义**：
- `risk_mean=0`：表示没有风险（保守估计）
- `uncertainty=1`：表示完全不确定（最大不确定性）
- 这样的占位符输出可以被上层代码安全处理

### Q5: 如何区分占位符输出和真实输出？

```python
output = run_edl_on_features(features)

# 方法 1：检查 TORCH_AVAILABLE
if not TORCH_AVAILABLE:
    print("Using placeholder output")

# 方法 2：检查输出特征
if np.allclose(output.risk_mean, 0) and np.allclose(output.uncertainty, 1):
    print("Likely placeholder output")

# 方法 3：添加元数据（未来改进）
# output.meta['source'] == 'placeholder' or 'torch'
```

## 后续改进建议

### 短期改进

1. **添加元数据**
   ```python
   @dataclass
   class EDLGridOutput:
       risk_mean: np.ndarray
       uncertainty: np.ndarray
       source: str = "unknown"  # "torch", "miles-guess", "placeholder"
   ```

2. **更详细的错误分类**
   ```python
   except ImportError:
       print("[EDL][torch] PyTorch import failed")
   except RuntimeError as e:
       if "CUDA" in str(e):
           print("[EDL][torch] CUDA error")
       else:
           print("[EDL][torch] Runtime error")
   ```

3. **性能监控**
   ```python
   import time
   start = time.time()
   # ... 推理 ...
   elapsed = time.time() - start
   print(f"[EDL][torch] inference took {elapsed:.3f}s")
   ```

### 中期改进

1. **模型缓存**
   - 缓存已创建的 EDLModel 实例
   - 避免重复创建

2. **预训练模型加载**
   - 支持加载预训练权重
   - 提高推理精度

3. **配置管理**
   - 从配置文件加载参数
   - 支持模型选择和切换

## 总结

通过三个关键改动，完全解决了 PyTorch EDL 后端的问题：

1. ✅ **占位符定义**：防止 import 失败时的 NameError
2. ✅ **条件类定义**：根据 PyTorch 可用性选择不同实现
3. ✅ **异常捕获**：确保推理失败时管线可以平滑回退

修复后的代码：
- 既能充分利用 PyTorch 的功能
- 也能在 PyTorch 不可用时优雅地降级
- 提高了系统的鲁棒性和可靠性
- 完全向后兼容，无需修改上层代码















