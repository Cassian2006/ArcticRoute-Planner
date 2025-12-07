# UI 自检与故障排查（UI-Guided-Smoke）

本章说明如何一键执行 UI 多页冒烟（Live / Reports / Compare / Explain / Review / Health），并对常见错误码给出排查路径。

## 一键运行

- 运行命令：

```
python -m ArcticRoute.api.cli ui.smoke --profile default
```

成功后会生成两份报告：
- 机器可读：reports/dev/ui_smoke_result.json
- 人类可读：reports/dev/ui_smoke_result.html（表格视图）

CLI 控制台会打印简表（每页 OK/FAIL + 错误码）。

## 自检项说明

- Live：调用一次 route.replan（偏向 dry-run/最小依赖）。
- Reports：调用 report.build（小样本 include=pareto）。
- Compare：尝试加载 Pareto 前沿 JSON（pareto_front_{ym}_{scenario}.json）。
- Explain：尝试加载 route_attr_{ym}_*.json，或提示通过 route.explain 生成。
- Review：检查反馈 schema 可导入 + 风险层与场景依赖存在。
- Health：调用 health.check 并写出健康报告。

结果条目形如：

```
{
  "page": "Compare",
  "ok": false,
  "error_code": "NO_PARETO",
  "error_msg": "ArcticRoute/reports/d_stage/phaseG/pareto_front_202412_nsr_wbound_smoke.json",
  "hint": "先运行: python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound_smoke --ym 202412"
}
```

## 常见错误码与解决

- NO_RISK_DATA：缺少融合风险层 risk_fused_*.nc。
  - 解决：`python -m ArcticRoute.api.cli risk.fuse --ym 202412`
- REPLAN_FAIL/REPLAN_ERROR：重规划失败（依赖或场景配置异常）。
  - 解决：检查 configs/scenarios.yaml 是否存在且含目标场景；确认 risk_fused_*.nc 存在。
- REPORT_BUILD_FAIL/REPORT_ERROR：报告构建失败。
  - 解决：先保证 Pareto 产物存在；查看 reports/dev/ui_smoke_result.html 与控制台 stderr。
- NO_PARETO：缺少 Pareto 前沿 JSON。
  - 解决：`python -m ArcticRoute.api.cli route.scan --scenario nsr_wbound_smoke --ym 202412`
- NO_ROUTE_SELECTED：Explain 未选择或找不到路线。
  - 解决：先在 Compare 生成代表路线或手动提供 GeoJSON，再运行 route.explain。
- EXPLAIN_DATA_MISSING：Explain 的 route_attr_* 产物缺失或不可读。
  - 解决：`python -m ArcticRoute.api.cli route.explain --route <path> --ym 202412`
- FEEDBACK_INVALID：Review 的 feedback.jsonl 解析失败或 schema 缺失。
  - 解决：使用 core/feedback/schema.py 校验；按示例格式修正后重试。
- HEALTH_FAIL/HEALTH_ERROR：健康检查失败或入口错误。
  - 解决：查看 reports/health/ 目录内 JSON/HTML 详情；根据条目修复缺失数据或路径。

## UI 页面中查看和复现

每个页面顶部均提供“使用说明 / Usage”折叠区，包含：
- 操作流程（步骤、前提、产物）。
- 1-2 条对应 CLI 示例，便于在命令行复现。

如果页面缺少必要数据或参数不完整，UI 会展示统一样式的错误卡片（error_card），包括：
- 错误码（如 NO_PARETO、NO_RISK_DATA、NO_ROUTE_SELECTED 等）。
- 简短错误描述。
- 建议操作（包含 CLI 示例或跳转指引）。

