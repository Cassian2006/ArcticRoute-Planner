# 审计/体检套件最小冒烟指南

本指南介绍如何运行完整体检（code + data + UI + 汇总）。本次体检仅做扫描与统计，不修改核心逻辑。

## 运行命令

- 全量体检（推荐）：

```
python -m ArcticRoute.api.cli audit.full --ym 202412
```

- 仅代码体检：
```
python -m ArcticRoute.api.cli audit.code
```

- 仅产物体检：
```
python -m ArcticRoute.api.cli audit.data --ym 202412
```

## 最小冒烟示例（与本轮 Audit-Fix 相关）

- risk.nowcast（若组件缺失将回退到 fused）：
```
python -m ArcticRoute.api.cli risk.nowcast --ym 202412 --conf 0.7
```

- route.replan（基于 fused 路面，默认场景见 configs/scenarios.yaml）：
```
python -m ArcticRoute.api.cli route.replan --scenario nsr_wbound_smoke --ym 202412
```

- watch.run（仅跑一轮以验证 CLI）：
```
python -m ArcticRoute.api.cli watch.run --scenario nsr_wbound_smoke --once
```

- risk.ice.build（安全占位，当前禁用）：
```
python -m ArcticRoute.api.cli risk.ice.build --ym 202412 --dry-run
# 期望输出：{"ok": true, "disabled": true, "reason": "...currently disabled..."}
```

## 结果查看

- 代码体检报告：
  - JSON: reports/audit/code_audit.json
  - HTML: reports/audit/code_audit.html

- 产物体检报告：
  - JSON: reports/audit/data_audit.json
  - HTML: reports/audit/data_audit.html

- 汇总 Markdown：
  - docs/audit/ARCTICROUTE-FULL-AUDIT.md

## 报告包含的检查

- 代码体检（code_audit）：
  - 搜索 TODO/FIXME/XXX/pass/NotImplementedError/未完成/占位/stub
  - 按模块分组（core/io/apps/api/config）
  - 解析 api/cli.py：子命令是否在 main 中分发；import 路径是否存在
  - 扫描 core/interfaces.py 及 *interfaces*.py 是否存在未实现入口（非抽象方法）

- 产物体检（data_audit）：
  - AIS/特征：tracks/segment_index 行数、经纬度异常、features/密度的 NaN 比例/方差
  - 先验：prior_penalty_*.nc、prior_centerlines_*.geojson、prior_corridor_selected_*.nc；P_prior∈[0,1]
  - 风险与融合：R_ice/R_wave/R_acc/risk_fused 的范围、均值、方差、NaN 比例
  - 绿色航行/路由：ECO 产物存在性、代表路线 GeoJSON 可解析性
  - 报告：复用 reports/audit/realdata_report.json（若存在）
  - UI：复用 ui.smoke 的结果

## 若发现 broken，怎么做

- 优先在 CLI 层加守卫或友好提示，避免误用（例如缺文件时直接退出并指向上游步骤）。
- 在报告中记录为 broken 或 suspect，不要在体检任务中直接修改核心实现。

## 备注

- 文件头部标注有“体检任务引入，可回退”，可安全回滚。
- 若仓库已有 audit/health 工具，本体检已优先复用（如 ArcticRoute.core.audit、api.health、ui_smoke 等）。






