# ADR: UI 自主设计（Self-Directed）与旧版兼容同步方案

状态: 提案/可实施
日期: 2025-11-16
关联产物: reports/dev/ui_audit.json

1. 背景与问题
- 当前 apps/app_min.py 体量过大，包含导航、页面、渲染、状态、后端任务等多职责，扩展/回退成本高。
- UI 与 CLI 的映射存在偏差：
  - app_min Pipeline 页调用了 cli 命令“pipeline”（cli.py 不存在），Report 页构造了“api.cli report …”也不匹配（应为 ArcticRoute.api.cli 的 report.build）。
- 页面与控件分散在同一文件，复用与替换困难，无法“页面可插拔”。
- config/runtime.yaml 缺少 UI feature flags 与页面开关，无法做到“新开关全关→外部行为与旧版一致”。

2. 目标与非目标
- 目标
  - 将 app_min 压薄为“路由与布局壳”；引入页面目录 apps/pages/* 与组件库 apps/components/*；建立 apps/state.py 与 apps/registry.py。
  - 提供旧名兼容的控件-参数映射；外部 CLI/脚本无感（旧参数名继续可用）。
  - config/runtime.yaml 增加 UI feature flags，所有新功能默认关闭，保证默认行为与旧版一致（尽量字节级一致）。
  - 页面：Compare/Explain/Live/Review/Reports/Health 可用，最小实现优先 REUSE 现有能力。
- 非目标
  - 不重写核心算法与报告生成；不引入新前端框架，仅在 Streamlit 内结构化。

3. 现状要点（摘自 ui_audit.json）
- 入口：ArcticRoute/apps/app_min.py（Streamlit，包含 Tabs：Live/Pipeline/Layers/Route/Summary/Report/Export/Review/Cache）。
- 可复用模块
  - 控件：ArcticRoute/apps/route_params.py（风险来源/聚合/ECO），ArcticRoute/apps/layers_prior.py（先验面板，可选），ArcticRoute/apps/layers_risk.py（风控视图），ArcticRoute/apps/progress_view.py（进度）。
  - CLI：ArcticRoute/api/cli.py 提供 route.scan、report.build、route.explain、route.replan、watch.run、route.review/apply 等。
- 差异/缺口
  - app_min 对“pipeline”与“report”命令映射不正确；需在 UI 适配层修正为现有 CLI 命令。

4. 方案概述
- 架构分层
  - app_min.py → 仅保留：
    - 读取 config/runtime.yaml feature flags
    - 调用 registry 注册与页面装配
    - 渲染主布局（侧边栏+顶层导航），将页面渲染委托给 apps/pages/*
  - apps/pages/*：每个页面一个文件，提供 render(ctx) 最小接口。建议页面：
    - layers.py / route.py / compare.py / explain.py / live.py / review.py / reports.py / health.py
  - apps/components/*：可复用 UI 组件（legend、weights_panel、route_summary、runmeta_badge、diff_overlay、review_toolbar 等）。优先 REUSE 现有 apps/route_params.py、apps/layers_risk.py、apps/progress_view.py；新组件仅在缺口时补最小实现。
  - apps/state.py：轻量状态层，导入/导出 .ui_state.json；每次运行 route/report 同步写 .ui_action.meta.json（含 run_id/git_sha/config_hash/inputs）。
  - apps/registry.py：插件/页面注册中心，从 config/runtime.yaml 读取启用的页面/控件/功能。支持页面可插拔与控件开关，避免在 app_min 硬编码。

- 控件与参数映射（兼容旧名）
  - 统一侧边栏控件：
    - 聚合模式 mode: mean/q/CVaR(α)
    - 先验权重 w_p
    - 绿色航行开关 & w_e；船型只读/选择（由 eco.yaml 提供）
    - 拥挤/交互开关 interact_weight
    - MoE/域桶提示（只读，依据模型/栅格是否存在）
    - Live 开关（路由重规划）
  - 旧→新映射表（UI 内部维护；外部 CLI 不变）：
    - risk_source → route.scan --risk-source
    - risk_agg/alpha → route.scan --risk-agg/--alpha；route.replan 对齐
    - interact_weight → cost/EnvRiskCostProvider.interact_weight 或 risk.fuse 权重
    - eco_enabled + w_e + vessel_class → route.scan --eco on --w_e；eco.preview --class
    - prior_w_p + prior_path → EnvRiskCost prior_penalty_weight；risk.fuse inputs prior_penalty

- 配置与开关（config/runtime.yaml 新增）
  - ui:
    - ui_sync_strict: true|false（严格模式下，关闭新页面/新控件，维持旧布局/默认值）
    - pages: {compare: false, explain: false, live: true, review: true, reports: true, health: true}
    - components: {legend: true, weights_panel: true, review_toolbar: true}
    - defaults: {risk_source: ice, risk_agg: mean, alpha: 0.95, interact_weight: 0.0, eco_enabled: false, w_e: 0.0}
  - 当 pages/* 全为 false 时，只保留原有 Layers/Route/Export/Summary（与旧版一致）。

- 页面接线（最小实现，优先 REUSE）
  - Compare：读取 ArcticRoute/reports/d_stage/phaseG/pareto_front_{ym}_{scenario}.json，展示 safe/balanced/efficient 三路线与指标表，提供下载。REUSE ArcticRoute/reports/phaseG_report.build_pareto_html 的数据生成与格式（若存在）
  - Explain：调用 CLI route.explain，读取 phaseH/route_attr_*.json，绘制分段贡献条（Risk/Dist/Prior/Interact/Congest/Eco）。校验“分段之和≈目标函数积分（误差≤2%）”。REUSE ArcticRoute/core/route/attribution.py 与 tests/test_phaseH_attribution.py 校验逻辑。
  - Live：展示最近更新时间/触发原因/版本差异；“Replan now” → CLI route.replan，REUSE 已存在 live_tab 逻辑与 replan 报告。
  - Review：REUSE 现有 app_min Review Tab 逻辑（route.review/route.apply.feedback/constraints.check/route.approve），抽出为页面与组件（review_toolbar）。
  - Reports：映射到 CLI report.build / report.animate；聚合下载链接。
  - Health：REUSE ArcticRoute/api/health.py → CLI health.check 输出。

- 性能与体验
  - 首屏加载 < 2s：延迟加载大数据（xarray 打开时仅在需要的页签触发；缓存 session_state）。
  - 任何重算提供进度与取消：REUSE app_min 现有 TaskRecord 与后台线程实现，抽为组件（task_panel）。
  - 图层透明度/图例：组件 legend 支持透明度；REUSE layers_risk 配色。
  - 空态占位：统一灰色占位与“去构建”指引（指向 CLI 命令）。

5. 与旧版一致性与回退
- 默认开关全关时：仅保留旧的 Layers/Route/Summary/Export，参数默认与旧版一致；Report/Compare/Explain/Health/Live/Review 按 flags 隐藏。
- 对 CLI 的修正仅体现在 UI 适配层（registry/adapter），不更改核心 CLI；避免历史脚本破坏。
- 回退方案：
  - 通过 config/runtime.yaml 将 ui.ui_sync_strict=true + pages 全部 false，即回退为旧版布局/交互。
  - 若新增页面导致异常，apps/registry.py 根据 flags 不加载该页面。

6. 风险与缓解
- 风险：Explain 的积分近似无法做到 2% 内。
  - 缓解：在 ADR 与页面中显式提示误差来源；落地校验单元（对齐 tests/test_phaseH_attribution.py），当误差>2% 显示黄色提示但不中断。
- 风险：Report 页旧实现与 CLI 名称不匹配。
  - 缓解：UI 适配层修正为 report.build、report.animate；保留旧按钮但映射新命令；在严格模式隐藏不匹配按钮。
- 风险：app_min 压薄过程中影响现有烟雾测试。
  - 缓解：分支实施，先建立 pages 与 registry，保持 app_min 导航与行为不变，再逐步迁移具体页签。

7. 替代路径
- 方案 A（本提案）：Streamlit 单进程内拆分目录 + 轻量注册中心 + flags；最小改动，复用高。
- 方案 B：引入多页 Streamlit（st.experimental_set_page_config + pages 目录），每页独立路由；缺点：跨页状态共享与回退更复杂。
- 方案 C：保留单文件，增加内部类与分区；缺点：仍难以页面可插拔，不推荐。

8. 复用清单（关键 REUSE）
- apps/route_params.py（风险来源/聚合/ECO 控件）
- apps/layers_prior.py（先验面板）
- apps/layers_risk.py（风险图层展示）
- apps/progress_view.py（进度/轮询）
- core/route/astar_cost.py、metrics.py、abtest.py（路径求解与指标）
- core/reporting/*（report.build/animate、calibration、eco 等）
- api/cli.py（route.scan/explain/robust/replan/review/apply/health.check）

9. 交付物与 DoD
- 必选产物：
  - reports/dev/ui_audit.json（已产出）
  - docs/adr/ADR-UI-SYNC.md（本文件）
- 代码改动最小集（后续 MR）：
  - 新增/复用 apps/pages/*、apps/components/*、apps/state.py、apps/registry.py
  - 更新 app_min.py 为薄壳（读取 flags，调用 registry 渲染）
  - config/runtime.yaml 增加 ui feature flags；默认关闭 Compare/Explain/Reports/Health（确保旧版一致）
- DoD 校验：
  - 新 flags 全关，历史用例结果不变（页面/导出与旧版一致）；
  - Compare/Explain/Live/Review/Reports/Health 页面可用；
  - .ui_state.json / .ui_action.meta.json 正确生成；
  - 新交互均有异常处理；Explain 积分误差 ≤ 2%；
  - 提供“冒烟说明”：本地启动、最小操作路径与已知限制。

10. 冒烟说明（最小操作路径）
- 本地启动：
  - python -m ArcticRoute.api.cli serve --ui --port 8501
- 最小路径：
  - Layers 页确认 merged/ 中有 sic/ice nc；
  - Route 页随机起止点→规划；
  - Live 页 Replan now；
  - Review 页导入一段 jsonl 后 Apply&Replan；
  - Reports 页执行 report.build pareto（修正后映射）。
- 已知限制：
  - Explain 页依赖 route.explain 输出；若输入路线与风险层不匹配可能出现 >2% 误差提示。
  - Compare 页依赖 pareto_front_*.json；无文件时显示“去构建”指引。

11. 变更计划（分阶段）
- P1 结构落地：新增 registry/state 以及空页面文件，app_min 读取 flags 决定加载；Report 页命令修正。
- P2 页面迁移：将 app_min 中 Review/Live/Report/Export 迁移至 pages；复用组件抽取 review_toolbar/task_panel。
- P3 控件映射：统一侧边栏控件与旧名映射表，加入 MoE/域桶提示只读位。
- P4 验收：默认关闭下结果不变；Explain 误差校验；生成冒烟说明。






