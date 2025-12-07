# ArcticRoute 2.0 灵感调研与能力候选清单（草案）

更新时间：2025-11-23

本页由扫描 C:\Users\sgddsf\Desktop\Arctic_insp 目录下 9 个参考项目与两份总纲文件整理而成，用于指导 ArcticRoute 2.0 的设计与落地顺序。

---

## 1. 灵感项目总览与优先级

说明：优先级来自 INSPIRATION_PRIORITY.md（以 1–5 表示）。

| 代号 | 类型 | 优先级 | 建议借鉴点（简要） |
|---|---|---:|---|
| WX_ENGINE（weather-engine-maritime-main） | Web UI / 包装 / 工程模板 | 5 | 仪表盘布局、配色与图标风格；Map + TimeSlider + Alerts 组件化模式；README/产品说明书写法；前后端拆分与部署脚本（Docker/一键脚本）。 |
| KEPLER_ST（streamlit_kepler-main） | 高级可视化 / 地图交互 | 4 | 在 Streamlit 中嵌入 kepler.gl；使用 session_state 管理多数据集；将 kepler map state JSON 回显/持久化作为配置；动态增删图层。 |
| KEPLER_SDK（streamlit-keplergl-main） | 地图组件库 / 配置驱动 | 4 | keplergl_static 封装；以 config 预设图层/样式；示例对比不同嵌入方式，选择更稳的封装路线。 |
| AI_ROUTE_DEMO（Maritime-AI-Route-Optimizer-with--main） | Streamlit 布局样板 | 4 | 侧栏参数 → 指标卡/图表/地图的一页式编排；folium + plotly 的轻量组合；单文件 app 的组织方式。 |
| GNN_RL_ROUTE（maritime-route-optimization-main） | 算法/系统图灵感 | 4 | Methodology/系统架构/管线图写法；graph_builder/environment 模块化思路；GNN + RL 叙事可用于技术说明。 |
| MM_SEAICE（MMSeaIce-main） | 风险建模 / 感知（海冰分割） | 3 | U-Net/SwinTransformer 的海冰感知套路；数据加载与训练配置结构；作为“高级风险源”的背景与说明素材。 |
| MOO_LIB（MultiObjectiveOptimization-master） | 多目标优化算法 | 3 | MGDA/最小范数解等多目标权衡方法名录；可用于我们 Pareto/权重自适应的理论参考与轻量实现。 |
| ORTOOLS_VRP（RouteOptimization-master） | 传统 VRP/TSP 对比背景 | 2 | 作为“我们不是 VRP/TSP”的对照素材；文档中强化我们是“栅格+风险场+A*”。 |
| INDUSTRY_REPORT（Sailing-Smarter-...） | 行业文案/价值叙事 | 2 | 背景/痛点/价值陈述的提炼，服务于 README/报告“意义与场景”章节。 |

注：对于每个项目，“不建议直接抄”的点（例如重型云服务、与现栈冲突部分）已在内评中考虑，本文不单列列。

---

## 2. RouteView 对比：我们的差距（按类别归纳）

- 算法与模型
  - 多模态实时感知链路（SAR/光学/气象）与高分辨率前视预报：我们目前仅有多模态融合的骨架与样例，不具备实时/高时空分辨率能力。
  - 战术/战略双层规划：我们有 Pareto/稳健接口，但缺少“尺度切换 + 运行中持续更新”的产品化能力。
  - 不确定性与稳健性：接口具备，但缺少样本维度的真实来源与可复现实验集；缺少标准化展示。
- 产品功能
  - 实时重规划（航行中周期更新）；
  - 多船/多窗口的任务编排与结果管理；
  - 批量场景运行与对比报告体系；
  - 环境要素快速预览与风险告警面板。
- UI/交互
  - 3D/数字孪生风格的地图呈现与动画回放；
  - 一键化操作流（参数→运行→对比→报告）；
  - 路线/图层的多方案并排对比交互体验有待增强（颜色、图例、联动）。
- 工程与部署
  - 自动化数据链路（下载/清洗/入库）缺位；
  - 监控/日志/配置管理的工程化程度不足；
  - 组件化前后端与容器化交付的样板需要完善。

---

## 3. ArcticRoute 2.0 能力候选清单（草案）

分档依据：竞赛展示影响力 × 实现成本 × 与现有 1.0 的耦合度。

### 3.1 Must（2.0 必须实现）

- 高级可视化：多方案对比视图（kepler.gl Tab）
  - 类别：UI
  - 参考来源：KEPLER_ST / KEPLER_SDK / RouteView（对比视图）
  - 说明：在 Planner 增设“高级可视化/对比”Tab，支持多条路线与关键图层的联动、配色区分、动画回放/时间滑块。

- 稳定的“高级风险融合”示范链路（至少 1 条）
  - 类别：算法
  - 参考来源：现有 core/fusion_adv + MM_SEAICE（思路）
  - 说明：固化一条可复现的高级融合（如 UNet 先验 + PoE/Evidential 聚合）到 Planner 的“高级选项”，确保演示可稳定复现。

- 批量场景管理与结果总览
  - 类别：产品功能
  - 参考来源：GNN_RL_ROUTE（管线化叙事）/ 我们 1.0 的 scenarios.yaml 与 reports
  - 说明：提供多场景批量规划入口与结果面板（列表/筛选/一键对比/导出），生成简要对比报告。

- 战略/战术模式切换（轻量版）
  - 类别：产品功能 / UI
  - 参考来源：RouteView
  - 说明：通过不同分辨率/时窗的环境与参数预设，快速切换“规划尺度”，并在 UI 上显式标注，哪怕暂时是离线样例。

### 3.2 Should（强烈推荐，时间充裕时实现）

- 路线告警与走廊合规检查面板
  - 类别：产品功能 / UI
  - 参考来源：WX_ENGINE（AlertsPanel 模式）
  - 说明：对路线计算出“冰险/超速/偏离走廊”等告警条目，并可点击定位，支持导出。

- Pareto 前端交互升级（合并入 Planner 主流程）
  - 类别：UI / 产品功能
  - 参考来源：我们 1.0 的 Pareto Tab + KEPLER_ST（对比）
  - 说明：在 Planner 内以 Tab/抽屉方式并排对比代表解，支持一键替换“当前方案”。

- 环境要素预览层（冰/雾/风等占位）
  - 类别：UI / 产品功能
  - 参考来源：RouteView（要素预览）
  - 说明：提供开关与图例，至少内置示例栅格/摘录数据，统一配色与透明度策略。

- 轻量实时重规划 Stub（离线数据步进）
  - 类别：产品功能 / 工程
  - 参考来源：RouteView（战术更新）
  - 说明：基于预载的时间序列环境，提供“步进播放 + 重新规划”按钮，演示运行中更新的交互流。

### 3.3 Nice-to-have（不影响主线的加分项）

- 3D/数字孪生风格演示页（kepler 3D/extrusion）
  - 类别：UI
  - 参考来源：KEPLER_ST / KEPLER_SDK / WX_ENGINE
  - 说明：用于路演/视频录屏，突出沉浸式展示。

- AI 解释器话术包升级
  - 类别：产品功能
  - 参考来源：我们 1.0 的 AI 解释器 + INDUSTRY_REPORT
  - 说明：补充“价值/风险/合规”模板话术，让自动生成报告更贴近行业表达。

- 服务化打包（FastAPI 轻服务）
  - 类别：工程
  - 参考来源：WX_ENGINE（backend 模板）
  - 说明：把“计算/数据”做成简单 REST 端点，便于后续容器化与外部调用（演示级别即可）。

---

## 4. 建议的 2.0 实施顺序（面向 Codex 的后续任务入口）

以下步骤从 Must + 顶级 Should 中择优排序，兼顾展示效果与改动风险。每步均应通过配置开关/独立模块接入，避免破坏 1.0 主流程。

- Step 1：高级可视化 Tab（kepler.gl 多方案对比）
  - 目标简述：在 Planner 内新增“高级可视化/对比”Tab，加载当前方案与备选方案，支持配色区分与动画回放。
  - 预计改动范围：
    - ArcticRoute/pages/00_Planner.py（新增 Tab 与状态管理）
    - ArcticRoute/apps/components/（新增 kepler 嵌入组件）
    - ArcticRoute/core/utils/（数据结构/GeoJSON 导出辅助）
    - ArcticRoute/config/runtime.yaml（新增开关）
  - 是否影响 1.0 行为：否（默认关闭 Tab 或回退到现有可视化）。

- Step 2：批量场景管理与结果总览
  - 目标简述：提供批量运行入口与结果列表/筛选/导出，实现一键对比与小结报告。
  - 预计改动范围：
    - ArcticRoute/core/planner_service.py（批处理入口）
    - ArcticRoute/core/reporting/（对比报告生成）
    - ArcticRoute/apps/components/（结果面板）
    - ArcticRoute/config/scenarios.yaml（样例与开关）
  - 是否影响 1.0 行为：否（作为独立入口/Tab，默认不触发批量）。

- Step 3：稳定的高级风险融合链路接线
  - 目标简述：固化并文档化 1 条“可复现”的高级融合配置，统一指标与可视化输出。
  - 预计改动范围：
    - ArcticRoute/core/fusion_adv/（清理与固化 pipeline）
    - ArcticRoute/core/risk/ 与 core/prior/（数据接口对齐）
    - ArcticRoute/apps/route_params.py（高级选项映射）
    - ArcticRoute/config/risk_fuse_*.yaml（示例配置）
  - 是否影响 1.0 行为：否（默认走基础融合；高级模式需显式开启）。

- Step 4：路线告警与走廊合规面板
  - 目标简述：为当前/对比路线生成“冰险/超速/偏离走廊”等告警列表，支持定位与导出。
  - 预计改动范围：
    - ArcticRoute/core/feedback/ 与 core/constraints/（检测逻辑）
    - ArcticRoute/apps/components/（AlertsPanel）
    - ArcticRoute/pages/00_Planner.py（集成面板）
  - 是否影响 1.0 行为：否（只读分析，不改变路由结果；可选开关）。

- Step 5：战略/战术模式切换（轻量版）
  - 目标简述：通过不同分辨率/时窗的参数预设，提供一键切换的“尺度模式”。
  - 预计改动范围：
    - ArcticRoute/config/runtime.yaml（模式开关与预设）
    - ArcticRoute/apps/route_params.py（模式选择控件）
    - ArcticRoute/core/planners/ 与 core/prior/（模式对参数的落地）
  - 是否影响 1.0 行为：否（默认保持当前规划模式）。

---

## 附：数据来源清单
- INSPIRATION_PRIORITY.md（优先级）
- ROUTEVIEW_COMPARISON.md（差距归纳）
- 9 个子项目说明 md（NOTES_*.md）

