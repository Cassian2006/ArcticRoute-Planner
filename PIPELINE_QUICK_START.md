# Pipeline Timeline 快速启动指南

## 🚀 快速开始

### 1. 验证安装

```bash
# 检查 Pipeline 组件是否正确安装
python test_pipeline_integration.py
```

预期输出：`🎉 All tests passed!`

### 2. 运行 UI

```bash
streamlit run run_ui.py
```

### 3. 使用 Pipeline

1. 在左侧边栏设置：
   - 选择场景
   - 设置起点和终点
   - 配置规划参数

2. 点击"规划三条方案"按钮

3. 观察"⏱️ 计算流程管线"：
   - 节点从 ⚪ 变为 🟡（执行中）
   - 再变为 [object Object] 显示耗时（秒）

4. 规划完成后：
   - 管线自动折叠
   - 结果显示在下方

## 📊 Pipeline 节点说明

| 节点 | 说明 | 额外信息 |
|------|------|--------|
| ⏱️ 加载网格 | 加载网格和 landmask | `grid=500×5333` |
| 🔄 加载 AIS | 加载 AIS 密度数据 | `candidates=4` |
| 🏗️ 构建成本场 | 为三个方案构建成本场 | - |
| 📍 起止点吸附 | 吸附到最近海洋单元 | - |
| [object Object]* 路由 | 执行三次路由 | `routes reachable=3/3` |
| 📈 成本分析 | 计算成本分解 | - |
| 🎨 数据准备 | 组织渲染数据 | - |

## 🎯 关键特性

### ✅ 实时进度显示
- 每个 stage 完成时实时更新
- 显示执行耗时

### ✅ 自动折叠
- 规划完成后自动折叠 pipeline
- 结果仍然可见

### ✅ 错误处理
- 失败节点显示 🔴
- 显示失败原因

### ✅ 额外信息
- 显示网格大小、AIS 候选数等
- 帮助用户理解规划过程

## 🔧 开发者指南

### 添加新的 Stage

在 `planner_minimal.py` 中：

```python
# 1. 在初始化时添加
pipeline.add_stage("my_stage", "我的阶段")

# 2. 在执行时调用
pipeline.start("my_stage")
try:
    # ... 执行代码 ...
    pipeline.done("my_stage", extra_info="some_info")
except Exception as e:
    pipeline.fail("my_stage", fail_reason=str(e))

# 3. 更新显示
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)
```

### 自定义 Stage 样式

编辑 `arcticroute/ui/components/pipeline_timeline.py` 中的 `render_pipeline()` 函数：

```python
# 修改状态图标
status_icons = {
    "pending": "⚪",
    "running": "🟡",
    "done": "🟢",
    "fail": "🔴"
}

# 修改样式
st.markdown(f"<div style='...'>内容</div>", unsafe_allow_html=True)
```

## 📝 常见问题

### Q: Pipeline 没有显示？
A: 检查是否点击了"规划三条方案"按钮。Pipeline 只在规划过程中显示。

### Q: 为什么 Pipeline 在规划完成后折叠了？
A: 这是设计的行为。规划完成后自动折叠以节省空间。点击 expander 可以重新展开查看。

### Q: 如何查看完整的规划结果？
A: 规划完成后，结果会显示在 Pipeline 下方的各个部分（路线对比、成本分析等）。

### Q: Pipeline 中的耗时不准确？
A: 耗时是从 `start()` 到 `done()` 的时间差。如果执行非常快，可能显示 0.00s。

## 🐛 故障排除

### 错误：ImportError: cannot import name 'Pipeline'
- 检查 `arcticroute/ui/components/__init__.py` 是否存在
- 检查 `arcticroute/ui/components/pipeline_timeline.py` 是否存在

### 错误：AttributeError: 'NoneType' object has no attribute 'container'
- 确保 `pipeline_placeholder` 在 expander 外部创建
- 检查 `render_pipeline()` 的调用是否正确

### Pipeline 显示但不更新
- 检查是否调用了 `render_pipeline()`
- 检查 `pipeline_placeholder` 是否正确传递

## 📚 相关文件

- `arcticroute/ui/components/pipeline_timeline.py` - 核心实现
- `arcticroute/ui/planner_minimal.py` - 集成代码
- `test_pipeline_integration.py` - 测试脚本
- `PIPELINE_TIMELINE_IMPLEMENTATION.md` - 详细文档

## 🎓 学习资源

- Streamlit 官方文档：https://docs.streamlit.io/
- Python dataclass：https://docs.python.org/3/library/dataclasses.html
- Streamlit session_state：https://docs.streamlit.io/library/api-reference/session-state

## 💡 提示

1. 使用 `st.session_state` 来保存 Pipeline 对象，确保在 rerun 后仍然可用
2. 在 expander 外部创建 placeholder，这样可以在 expander 内外都能更新
3. 使用 `render_pipeline()` 来实时更新显示，而不是重新创建整个 UI

## 📞 支持

如有问题，请检查：
1. 是否运行了 `test_pipeline_integration.py` 并通过了所有测试
2. 是否正确设置了起止点
3. 是否有足够的内存和计算资源
4. 是否使用了最新版本的 Streamlit（建议 1.28+）




