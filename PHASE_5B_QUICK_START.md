# Phase 5B 快速开始指南

**版本**：Phase 5B（PolarRoute Pipeline 端到端自动化）  
**更新日期**：2025-12-14

---

## [object Object] 分钟快速开始

### 1. 检查 Pipeline CLI

```bash
# 基本检查
python -m scripts.polarroute_pipeline_doctor

# 带诊断的检查
python -m scripts.polarroute_pipeline_doctor --pipeline-dir "D:\polarroute-pipeline"
```

**预期输出**：
```
✓ pipeline CLI 已找到: /path/to/pipeline
✓ pipeline --help 成功
✓ pipeline status --help 成功
✓ pipeline status --short 成功
```

### 2. 在 UI 中使用 Pipeline 模式

1. 启动 Streamlit UI：
   ```bash
   streamlit run run_ui.py
   ```

2. 在左侧面板中找到"规划内核"部分

3. 从下拉框中选择 **"PolarRoute (pipeline dir)"**

4. 输入 Pipeline 目录路径：
   ```
   D:\polarroute-pipeline
   ```

5. 点击 **"Status"** 按钮检查 pipeline 状态

6. 点击 **"Execute"** 按钮运行 pipeline（首次需要，可能需要几分钟）

7. 系统会自动显示最新的 vessel_mesh.json 路径

8. 设置起终点坐标，点击"规划路线"

### 3. 编程方式使用

```python
from arcticroute.core.planners.polarroute_backend import PolarRouteBackend

# 初始化 Pipeline 模式
backend = PolarRouteBackend(
    pipeline_dir="/path/to/polarroute-pipeline"
)

# 规划路线
path = backend.plan(
    start_latlon=(75.0, 30.0),
    end_latlon=(70.0, 50.0)
)

print(f"规划成功！路径包含 {len(path)} 个点")
```

---

## 📋 常见任务

### 任务 1：查找最新的 vessel_mesh.json

```python
from arcticroute.integrations.polarroute_artifacts import find_latest_vessel_mesh

mesh_path = find_latest_vessel_mesh("/path/to/pipeline")
if mesh_path:
    print(f"最新 mesh: {mesh_path}")
else:
    print("未找到 vessel_mesh.json，请先执行 pipeline execute")
```

### 任务 2：获取 Pipeline 状态

```python
from arcticroute.integrations.polarroute_pipeline import pipeline_status

success, output = pipeline_status(
    "/path/to/pipeline",
    short=True
)

if success:
    print("Pipeline 状态：")
    print(output)
else:
    print("获取状态失败")
```

### 任务 3：执行 Pipeline

```python
from arcticroute.integrations.polarroute_pipeline import pipeline_execute

success, output = pipeline_execute("/path/to/pipeline")

if success:
    print("Pipeline 执行成功")
else:
    print("Pipeline 执行失败")
    print(output)
```

### 任务 4：重置 Pipeline

```python
from arcticroute.integrations.polarroute_pipeline import pipeline_reset

success, output = pipeline_reset("/path/to/pipeline")

if success:
    print("Pipeline 已重置")
else:
    print("重置失败")
```

---

## 🔧 故障排除

### 问题 1：pipeline 命令未找到

**症状**：
```
❌ pipeline CLI 不可用。请先安装：
`pip install polar-route`
```

**解决方案**：
```bash
pip install polar-route
```

### 问题 2：未找到 vessel_mesh.json

**症状**：
```
⚠️ 未找到 vessel_mesh.json。请先执行 pipeline execute
```

**解决方案**：
1. 点击 UI 中的 "Execute" 按钮
2. 或运行：
   ```python
   from arcticroute.integrations.polarroute_pipeline import pipeline_execute
   pipeline_execute("/path/to/pipeline")
   ```

### 问题 3：未找到 route_config.json

**症状**：
```
未找到 route_config.json 在 Pipeline 目录中
```

**解决方案**：
确保 `route_config.json` 在以下位置之一：
- `<pipeline>/route_config.json`
- `<pipeline>/config/route_config.json`
- `<pipeline>/configs/route_config.json`

### 问题 4：Pipeline 执行失败

**症状**：
```
✗ Pipeline Execute 失败
```

**解决方案**：
1. 查看日志文件：
   ```
   reports/polarroute_pipeline_last_out.log
   reports/polarroute_pipeline_last_err.log
   ```

2. 检查 pipeline 目录结构

3. 运行医生脚本诊断：
   ```bash
   python -m scripts.polarroute_pipeline_doctor --pipeline-dir "D:\polarroute-pipeline"
   ```

---

## 📚 API 参考

### PolarRouteBackend

```python
from arcticroute.core.planners.polarroute_backend import PolarRouteBackend

# Phase 5A：外部文件模式
backend = PolarRouteBackend(
    vessel_mesh_path="/path/to/vessel_mesh.json",
    route_config_path="/path/to/route_config.json"
)

# Phase 5B：Pipeline 模式
backend = PolarRouteBackend(
    pipeline_dir="/path/to/pipeline"
)

# 规划路线
path = backend.plan(
    start_latlon=(lat, lon),
    end_latlon=(lat, lon)
)
```

### Pipeline 命令

```python
from arcticroute.integrations.polarroute_pipeline import (
    pipeline_build,
    pipeline_status,
    pipeline_execute,
    pipeline_reset,
    pipeline_halt,
)

# 构建 pipeline
success, output = pipeline_build(pipeline_dir)

# 获取状态
success, output = pipeline_status(pipeline_dir, short=True)

# 执行 pipeline
success, output = pipeline_execute(pipeline_dir)

# 重置 pipeline
success, output = pipeline_reset(pipeline_dir)

# 停止 pipeline
success, output = pipeline_halt(pipeline_dir)
```

### 工件查找

```python
from arcticroute.integrations.polarroute_artifacts import (
    find_latest_vessel_mesh,
    find_latest_route_json,
    find_latest_route_config,
)

# 查找最新的 vessel_mesh.json
mesh_path = find_latest_vessel_mesh(pipeline_dir)

# 查找最新的 route.json
route_path = find_latest_route_json(pipeline_dir)

# 查找最新的 route_config.json
config_path = find_latest_route_config(pipeline_dir)
```

---

## 🎯 最佳实践

### 1. 定期检查 Pipeline 状态

```python
from arcticroute.integrations.polarroute_pipeline import pipeline_status

# 在规划前检查状态
success, output = pipeline_status(pipeline_dir, short=True)
if not success:
    print("Pipeline 异常，请检查")
```

### 2. 处理错误

```python
from arcticroute.core.planners.base import PlannerBackendError

try:
    backend = PolarRouteBackend(pipeline_dir=pipeline_dir)
    path = backend.plan(start, end)
except PlannerBackendError as e:
    print(f"规划失败: {e}")
    # 回退到 A* 或其他方案
```

### 3. 查看日志

```bash
# 查看最后一次 pipeline 命令的输出
cat reports/polarroute_pipeline_last_out.log

# 查看最后一次 pipeline 命令的错误
cat reports/polarroute_pipeline_last_err.log
```

### 4. 缓存 vessel_mesh 路径

```python
from arcticroute.integrations.polarroute_artifacts import find_latest_vessel_mesh

# 首次查找
mesh_path = find_latest_vessel_mesh(pipeline_dir)

# 后续使用缓存的路径
backend = PolarRouteBackend(
    vessel_mesh_path=mesh_path,
    route_config_path=config_path
)
```

---

## 🧪 测试

### 运行所有测试

```bash
python -m pytest tests/ -q
```

### 运行 Pipeline 可选测试

```bash
# 需要 pipeline CLI 和 AR_POLAR_PIPELINE_DIR 环境变量
export AR_POLAR_PIPELINE_DIR="/path/to/pipeline"
python -m pytest tests/test_polarroute_pipeline_optional.py -v
```

### 运行医生脚本

```bash
python -m scripts.polarroute_pipeline_doctor --pipeline-dir "D:\polarroute-pipeline"
```

---

## 📖 更多信息

- 详细总结：[PHASE_5B_POLARROUTE_PIPELINE_INTEGRATION_SUMMARY.md](PHASE_5B_POLARROUTE_PIPELINE_INTEGRATION_SUMMARY.md)
- 执行总结：[PHASE_5B_EXECUTION_SUMMARY.md](PHASE_5B_EXECUTION_SUMMARY.md)
- 官方文档：[PolarRoute-pipeline 官方文档](https://bas-amop.github.io)

---

## 💡 提示

- **首次使用**：运行医生脚本检查环境
- **调试**：查看 `reports/` 目录中的日志文件
- **性能**：Pipeline 执行可能需要几分钟，请耐心等待
- **兼容性**：Phase 5A 的外部文件模式仍然可用

---

**需要帮助？**

1. 查看故障排除部分
2. 运行医生脚本诊断
3. 查看日志文件
4. 查阅官方文档

祝你使用愉快！[object Object]
