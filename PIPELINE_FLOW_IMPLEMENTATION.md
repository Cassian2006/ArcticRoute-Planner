# 流动管线 UI 实现文档

## 概述

在 `arcticroute/ui/planner_minimal.py` 中实现了一个"流动管线"UI，用于可视化规划流程的各个步骤。该管线显示 8 个节点，节点之间用会流动的管道连接。

## 核心文件

### 1. `arcticroute/ui/components/pipeline_flow.py`

新增组件文件，包含：

#### `PipeNode` 数据类
```python
@dataclass
class PipeNode:
    key: str                          # 节点唯一标识
    label: str                        # 节点显示标签
    status: str                       # 状态：pending/running/done/fail
    seconds: Optional[float] = None   # 耗时（秒）
    detail: Optional[str] = None      # 详情文本
```

#### `render_pipeline()` 函数
```python
def render_pipeline(
    nodes: List[PipeNode],
    title: str = "计算流程管线",
    expanded: bool = True
) -> None:
```

渲染流动管线 UI，支持：
- 节点状态可视化（pending/running/done/fail）
- CSS 动画（管道流动效果）
- 底部统计（完成数/失败数/总耗时）

## 规划流程的 8 个节点

在 `planner_minimal.py` 中，规划按钮点击后会初始化 8 个节点：

| 序号 | 节点 | 说明 |
|------|------|------|
| ① | 解析场景/参数 | 解析用户输入的场景和参数 |
| ② | 加载网格与 landmask | 加载网格数据和陆地掩码 |
| ③ | 加载环境层 | 加载 SIC（海冰浓度）和 Wave（波浪）数据 |
| ④ | 加载 AIS 密度 | 加载 AIS 船舶密度数据 |
| ⑤ | 构建成本场 | 构建 3 种成本场（efficient/edl_safe/edl_robust） |
| ⑥ | A* 规划 | 执行 A* 路由规划算法 |
| ⑦ | 分析与诊断 | 计算成本分解和路线诊断 |
| ⑧ | 渲染与导出 | 渲染地图和准备导出数据 |

## 集成方式

### 1. 导入组件
```python
from arcticroute.ui.components.pipeline_flow import (
    PipeNode,
    render_pipeline as render_pipeline_flow,
)
```

### 2. 初始化流动管线

在规划按钮点击时：
```python
if do_plan:
    st.session_state.pipeline_flow_expanded = True
    st.session_state.pipeline_flow_start_time = datetime.now()
    st.session_state.pipeline_flow_nodes = [
        PipeNode(key="parse", label="① 解析场景/参数", status="pending"),
        # ... 其他 7 个节点
    ]
```

### 3. 更新节点状态

使用辅助函数 `_update_pipeline_node()` 更新节点：
```python
_update_pipeline_node(
    idx=0,                    # 节点索引（0-7）
    status="running",         # 状态
    detail="正在解析...",     # 详情文本
    seconds=None              # 耗时（可选）
)
```

### 4. 完成流程

规划完成后，自动折叠管线并显示"✅ 完成"标记：
```python
with st.session_state.pipeline_flow_placeholder.container():
    render_pipeline_flow(
        st.session_state.pipeline_flow_nodes,
        title="🔄 规划流程管线 ✅ 完成",
        expanded=False,  # 自动折叠
    )
```

## CSS 动画效果

### 管道流动动画
```css
.pipe.active {
  background: linear-gradient(90deg, ...);
  background-size: 200% 100%;
  animation: pipeflow 1.2s linear infinite;
}

@keyframes pipeflow {
  0% { background-position: 0% 50%; }
  100% { background-position: 200% 50%; }
}
```

### 节点状态样式
- **pending**：灰色，透明度 65%
- **running**：蓝色边框，内阴影
- **done**：绿色边框
- **fail**：红色边框，内阴影

## 美观细节

### 1. 节点 detail 显示关键数值

示例：
```python
_update_pipeline_node(0, "done", f"grid={grid_shape[0]}×{grid_shape[1]}", seconds=0.5)
_update_pipeline_node(3, "done", f"AIS={ais_density.shape}", seconds=0.4)
_update_pipeline_node(5, "done", f"可达={num_reachable}/3", seconds=0.8)
```

### 2. 失败节点显示错误原因

```python
_update_pipeline_node(3, "fail", f"加载失败: {str(e)[:30]}")
```

### 3. 总耗时 badge

底部自动显示：
- 已完成节点数
- 失败节点数
- 总耗时（秒）

## 测试方式

### 方式 1：运行演示脚本
```bash
streamlit run test_pipeline_flow.py
```

这会打开一个交互式演示，可以逐步推进各个节点的状态。

### 方式 2：在完整 UI 中测试
```bash
streamlit run run_ui.py
```

点击"规划三条方案"按钮，观察流动管线的实时更新。

## 技术细节

### Session State 管理
```python
st.session_state.pipeline_flow_nodes      # 节点列表
st.session_state.pipeline_flow_placeholder # 容器引用
st.session_state.pipeline_flow_expanded   # 展开状态
st.session_state.pipeline_flow_start_time # 开始时间
```

### 渲染更新机制
每次调用 `_update_pipeline_node()` 时：
1. 更新 session state 中的节点数据
2. 清空 placeholder 容器
3. 重新渲染整个管线

这确保了实时的视觉反馈。

## 兼容性

- ✅ Streamlit 1.0+
- ✅ 深色主题
- ✅ 响应式设计
- ✅ 水平滚动支持

## 未来改进

1. **进度条**：添加整体进度百分比
2. **详细日志**：点击节点展开详细日志
3. **重试机制**：失败节点支持重试
4. **性能指标**：显示各步骤的性能数据
5. **导出报告**：将流程记录导出为 JSON/CSV

## 参考

- 组件文件：`arcticroute/ui/components/pipeline_flow.py`
- 集成文件：`arcticroute/ui/planner_minimal.py`（第 891-2980 行）
- 测试脚本：`test_pipeline_flow.py`




