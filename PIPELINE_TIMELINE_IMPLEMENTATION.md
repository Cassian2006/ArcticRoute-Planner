# Pipeline Timeline 实现文档

## 概述

本实现为 ArcticRoute UI 的规划流程添加了一个实时的"计算流程管线（Timeline）"组件，用于展示规划过程中各个阶段的执行状态、耗时和额外信息。

## 文件结构

### 新增文件

1. **arcticroute/ui/components/pipeline_timeline.py**
   - `PipelineStage` dataclass：表示管线中的单个阶段
   - `Pipeline` 类：管理所有阶段的状态和时间
   - `render_pipeline()` 函数：将管线渲染为 Streamlit UI
   - `init_pipeline_in_session()` 和 `get_pipeline()` 函数：session 状态管理

2. **arcticroute/ui/components/__init__.py**
   - 导出 Pipeline 组件的公共 API

### 修改的文件

1. **arcticroute/ui/planner_minimal.py**
   - 添加 Pipeline 导入
   - 在规划按钮之后初始化 Pipeline 和 stages
   - 在规划流程的各个关键点添加 `start()`、`done()` 和 `fail()` 调用
   - 在每个 stage 完成时调用 `render_pipeline()` 更新显示
   - 在规划完成后自动折叠 pipeline 并保存结果到 session_state

## 核心功能

### 1. PipelineStage 数据类

```python
@dataclass
class PipelineStage:
    key: str                    # 唯一标识符
    label: str                  # 显示标签
    status: str = "pending"     # pending / running / done / fail
    dt_s: float = 0.0          # 耗时（秒）
    extra_info: str = ""        # 额外信息
    fail_reason: str = ""       # 失败原因
```

### 2. Pipeline 类

主要方法：
- `add_stage(key, label)`：添加新阶段
- `start(key)`：标记阶段开始执行
- `done(key, extra_info="")`：标记阶段完成，自动计算耗时
- `fail(key, fail_reason="")`：标记阶段失败
- `get_stages_list()`：获取所有阶段列表

### 3. render_pipeline 函数

使用 Streamlit 的 `st.columns()` 横向渲染管线：
- 节点状态图标：⚪待执行 → 🟡执行中 → 🟢完成（失败🔴）
- 节点下方显示耗时（秒）或"运行中..."
- 节点间用 → 箭头连接
- 显示额外信息和失败原因

## Pipeline Stages 定义

规划流程中定义的 7 个阶段：

1. **grid_env**（加载网格）
   - 加载网格和 landmask
   - 额外信息：`grid=500×5333`

2. **ais**（加载 AIS）
   - 加载 AIS 密度数据
   - 额外信息：`candidates=4`

3. **cost_build**（构建成本场）
   - 为三个方案构建成本场

4. **snap**（起止点吸附）
   - 将起止点吸附到最近的海洋单元

5. **astar**（A* 路由）
   - 执行三次 A* 路由
   - 额外信息：`routes reachable=3/3`

6. **analysis**（成本分析）
   - 计算成本分解和剖面

7. **render**（数据准备）
   - 组织数据以供地图和表格渲染

## Session State 控制

- `pipeline_expanded`：控制 expander 的展开/折叠状态
  - 初始值：`True`（默认展开）
  - 点击规划按钮时：强制 `True`（展开）
  - 规划完成后：设置为 `False`（自动折叠）并调用 `st.rerun()`

- `last_plan_result`：保存规划结果
  - 规划完成后保存到 session_state
  - 在 rerun 后仍可用，确保结果不丢失

## 实时更新机制

1. 在 expander 外部创建 `pipeline_placeholder = st.empty()`
2. 每个 stage 完成时调用 `render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)`
3. `st.empty()` 容器会被新的内容替换，实现实时更新

## 错误处理

- 如果某个 stage 失败，调用 `pipeline.fail(key, fail_reason="...")`
- 失败节点显示 🔴 图标和失败原因
- 规划流程继续进行（不中断）

## 使用示例

```python
# 初始化
pipeline = init_pipeline_in_session()
pipeline.add_stage("grid_env", "加载网格")

# 执行阶段
pipeline.start("grid_env")
# ... 执行加载网格的代码 ...
pipeline.done("grid_env", extra_info="grid=500×5333")
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)

# 或者失败
try:
    # ... 执行代码 ...
except Exception as e:
    pipeline.fail("grid_env", fail_reason=str(e))
    render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)
```

## 测试

运行以下命令进行集成测试：

```bash
python test_pipeline_integration.py
```

测试项目：
- ✅ Pipeline 组件导入
- ✅ Pipeline 类功能（start/done/fail）
- ✅ planner_minimal.py 语法
- ✅ Pipeline 在 planner_minimal.py 中的集成

## 运行 UI

```bash
streamlit run run_ui.py
```

然后：
1. 在左侧设置起止点和规划参数
2. 点击"规划三条方案"按钮
3. 观察"⏱️ 计算流程管线"中的进度
4. 每个 stage 完成时，节点会变色并显示耗时
5. 规划完成后，管线自动折叠

## 注意事项

1. **Streamlit 的 expander 行为**
   - `expanded=...` 参数只在下一次 rerun 时生效
   - 因此规划完成后需要调用 `st.rerun()` 来应用折叠

2. **Session State 持久化**
   - 规划结果保存到 `st.session_state['last_plan_result']`
   - 在 rerun 后仍可用，确保用户看到的是最新的结果

3. **Placeholder 作用域**
   - `pipeline_placeholder` 必须在 expander 外部创建
   - 这样才能在 expander 内外都能访问和更新

4. **性能考虑**
   - 每个 stage 完成时都会调用 `render_pipeline()`
   - 这会导致 Streamlit 重新渲染整个 expander
   - 对于快速执行的 stage，这是可以接受的

## 未来改进

1. 添加更详细的进度信息（如百分比）
2. 支持并行 stage 的显示
3. 添加 stage 之间的依赖关系
4. 支持自定义 stage 样式和颜色
5. 添加 stage 执行日志的展示

## 相关文件

- `arcticroute/ui/components/pipeline_timeline.py` - Pipeline 组件实现
- `arcticroute/ui/components/__init__.py` - 组件导出
- `arcticroute/ui/planner_minimal.py` - Pipeline 集成
- `test_pipeline_integration.py` - 集成测试




