# ADR-AUDIT-FIX: 审计修复第一轮（2025-11-17）

目的：在不大改架构的前提下，修复与澄清审计中最关键、最明确的问题，使 audit.full 更接近真实状态并让现有流水线更稳。

变更摘要
- CLI 分发修复：将以下子命令挂入 main() 分发，并在 --help 中可见：
  - risk.nowcast（已实现）
  - route.replan（已实现）
  - watch.run（已实现）
  - risk.ice.build（安全占位：planned_disabled）
- 安全占位策略：risk.ice.build 使用 stub，执行时输出“experimental and currently disabled”，退出码 0，避免误用导致失败。
- AIS/特征链条审计放宽：若 tracks_YYYYMM.parquet 或 segment_index_YYYYMM.parquet 缺失，但 features_YYYYMM.nc 或 ais_density_YYYYMM.nc 已存在，则在 audit.data 中降级为 missing_optional，并在文档中提示需手动运行 prior.ds.prepare / ais.* 相关命令。
- 先验链条命名对齐：audit.data 接受 prior_transformer_YYYYMM.nc（变量含 PriorPenalty 或 P_prior）作为 prior_penalty 等价产物；中心线接受 reports/phaseE/center 与 data_processed/prior 两种路径。
- audit.full 的帮助文案补充状态定义：ok / suspect / broken / missing_optional / planned_disabled / disabled。

CLI 可用性矩阵
- 已启用（Enabled）
  - risk.nowcast
  - route.replan
  - watch.run
  - audit.code / audit.data / audit.full
  - route.scan / report.build / report.animate / route.robust / eco.preview 等（原有）
- 计划中但当前禁用（Planned, Disabled）
  - risk.ice.build：实验性，依赖上游冰图构建链条（Phase K）与参数打磨；当前不对外启用，执行将提示 disabled。

为什么禁用 risk.ice.build
- 当前仓库中缺少稳定的冰风险构建全链路（数据源 / 融合 / 单元验证），贸然开放会导致审计误判为“真 BUG”。
- 短期目标是让融合与路由链更稳，因此将其标注为 planned_disabled，保留 CLI 名称以兼容脚本，同时避免误触。
- 未来计划：在 Phase K 整理冰层来源与权重策略后启用，并在 audit 中加入数值一致性检查。

AIS/特征链条说明
- tracks_YYYYMM.parquet / segment_index_YYYYMM.parquet：由 prior.ds.prepare 及 AIS ingest/clean/segment 产生；并非所有运行配置都需要，若 features/ais_density 已可用，则视为 missing_optional。
- 最小再现命令（仅体检）
  - 数据准备（干跑）：python -m ArcticRoute.api.cli prior.ds.prepare --ym 202412 --dry-run
  - 生成密度：参见 ArcticRoute/features/ais_density.py 的 build_ais_density（按项目配置调用）
  - 合成特征：参见 ArcticRoute/features/build_features.py 的 build_feature_dataset（按项目配置调用）

先验链条说明
- prior_penalty_YYYYMM.nc 与 prior_transformer_YYYYMM.nc：两者在语义上等价（PriorPenalty = 1 - P_prior）。audit.data 已兼容从 prior_transformer 中读取统计，不再简单标缺。
- prior_centerlines_YYYYMM.geojson：若 reports/phaseE/center/ 或 data_processed/prior/centerlines/ 中存在其一，则 audit 视为可用；若尚未实现导出，则将其视为对解释性报告的影响，不阻断主链。

状态定义（用于审计报告）
- ok：检查通过
- suspect：可疑项，需进一步确认
- broken：明确错误（如无法读取、维度错误、数值越界）
- missing_optional：可选产物缺失（在当前配置下非必需）
- planned_disabled：功能处于计划但主动禁用（CLI 仍占位）
- disabled：由 runtime.yaml 或配置显式关闭

对运行流水线的影响
- 改动均向后兼容；新增的安全占位避免误触导致失败；审计脚本更贴近真实状态，减少“误报”。

回滚策略
- 移除 docs/adr/ADR-AUDIT-FIX.md，并还原 ArcticRoute/api/cli.py 与 ArcticRoute/audit/data_audit.py 本次差异，即可回退。

# ADR-AUDIT-FIX: 审计修复第二轮（2025-11-17）

## 1. 确认代表月份 (Canonical Month)

根据 `config/paper_profiles.yaml` 的 `quick` 和 `ablation` 配置，以及现有文档中的示例，**`202412`** 被确定为项目的“代表月份” (`canonical_ym`)。

所有后续的审计修复工作将围绕此月份展开，以确保其数据链路的健康与可信。`audit.full` 命令的默认月份已更新为此值。
