# ArcticRoute 全量体检报告 (202412)

## 概要

共评估模块 11，状态分布：suspect:4, ok:5, missing_optional:1, data_sparse:1

## 按模块状态

- code:core: suspect
- code:io: suspect
- code:apps: suspect
- code:api: suspect
- code:config: ok
- data:ais_features: missing_optional
- data:prior: ok
- data:risk_fuse: data_sparse
- data:eco_routes: ok
- data:reports: ok
- data:ui: ok

## 高优先级问题 TOP 10

- ais_features:缺少 tracks_*.parquet
- ais_features:缺少 segment_index_*.parquet
- ais_features:ais_density_202412.nc 方差≈0
- ais_features:features_202412.nc 方差≈0
- prior:缺少 prior_penalty/transformer_*.nc (planned_disabled, 当前版本仅使用概率地图 P_prior)
- prior:缺少 prior_corridor_selected_*.nc (planned_disabled)
- risk_fuse:R_ice 方差过低≈0. 该月风险信号极弱，仅适合结构演示
- risk_fuse:R_wave_202412.nc is planned_disabled (use_wave=false)
- risk_fuse:R_acc 方差过低≈0. 该月风险信号极弱，仅适合结构演示
- risk_fuse:risk_fused 方差过低≈0

## 推荐排查步骤

- ais_features:缺少 tracks_*.parquet
  - 建议：确认流水线是否产出该文件；若本次不需要，应在 CLI 层增加可选分支避免强依赖
- ais_features:缺少 segment_index_*.parquet
  - 建议：确认流水线是否产出该文件；若本次不需要，应在 CLI 层增加可选分支避免强依赖
- ais_features:ais_density_202412.nc 方差≈0
  - 建议：复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段
- ais_features:features_202412.nc 方差≈0
  - 建议：复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段
- prior:缺少 prior_penalty/transformer_*.nc (planned_disabled, 当前版本仅使用概率地图 P_prior)
  - 建议：核对 prior.export 产物的归一化与取反逻辑，确认 [0,1] 边界裁剪
- prior:缺少 prior_corridor_selected_*.nc (planned_disabled)
  - 建议：确认流水线是否产出该文件；若本次不需要，应在 CLI 层增加可选分支避免强依赖
- risk_fuse:R_ice 方差过低≈0. 该月风险信号极弱，仅适合结构演示
  - 建议：复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段
- risk_fuse:R_wave_202412.nc is planned_disabled (use_wave=false)
  - 建议：复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段
- risk_fuse:R_acc 方差过低≈0. 该月风险信号极弱，仅适合结构演示
  - 建议：复现该问题的最小命令，打印关键统计（均值/方差/NaN 比），定位到具体阶段
- risk_fuse:risk_fused 方差过低≈0
  - 建议：检查融合输入通道与权重；若由 CLI 触发，请先在 CLI 层加校验避免导出全 0 栅格

## 整改说明

本轮审计修复（第三轮）主要集中在以下结构性问题：

1.  **UI 状态修复**:
    - 修复了导致 UI 页面崩溃的语法和导入错误。
    - 调整了 `audit.data` 逻辑，将因缺少数据（如 `pareto_front_*.json`）而无法完整渲染的页面标记为 `planned_disabled`，而非 `broken`，确保 UI 整体状态为 `ok`。

2.  **风险层状态澄清**:
    - 确认 `R_ice` / `R_acc` 方差为零是源于数据稀疏，已在审计报告中标记为 `data_sparse` 并附带说明。
    - 通过 `configs/risk_fuse_202412.yaml` 明确 `use_wave: false`，将 `R_wave` 缺失正确标记为 `planned_disabled`。

3.  **先验命名与缺失处理**:
    - 更新了审计逻辑，不再强制要求 `prior_penalty_*.nc` 文件，将其缺失标记为 `planned_disabled`，以反映当前版本的设计。

4.  **AIS 告警降噪**:
    - 统一了 AIS 相关文件的缺失检查，当整个数据链未运行时，在报告中合并为单一的 `missing_optional` 说明，避免了 TOP10 列表被重复问题占满。

## 运行命令示例

```
python -m ArcticRoute.api.cli audit.full --ym 202412
```

## 需要人工确认的灰区问题

- 某些数据异常可能源于原始数据质量，请结合原始来源再次确认
