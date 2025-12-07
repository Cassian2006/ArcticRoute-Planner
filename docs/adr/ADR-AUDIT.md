# ADR-AUDIT: Real-Data & Method Audit

Status: Accepted
Date: 2025-11-12

Context
- 需要一套“真实数据与方法符合性”审计工具，确保产物不含 mock/demo 等样例，方法满足预期（如融合为 unetformer），体量与统计指标合理。
- 审计为非破坏性、可回退；未开启守门时不影响既有流程。

Decision
- 新增审计模块（只读）：ArcticRoute/core/audit/{provenance.py,data_sanity.py,report.py}
- 新增 CLI：audit.real / audit.assert
- 新增配置：configs/audit.yaml（可被用户覆盖）
- 新增报告：reports/audit/realdata_report.{json,html}
- 新增冒烟脚本：scripts/smoke/audit_realdata.(ps1|sh)
- 暂不在 paper.* / serve.api 等入口启用强制守门；后续在获得确认后接入 --require-real。

Checks
- 命名否定：文件名/标签含 mock/demo/sample/toy/synthetic → 不合格
- 方法来源：优先读取 .meta.json 的 method/attrs.method；与 require_method 不一致 → 不合格
- 元数据必填：run_id/git_sha/config_hash/inputs 缺失 → 警告（可通过 --fail-on-warn 升级为失败）
- 体量检查：按 min_size_bytes 校验不同类型（parquet_tracks/risk_nc/prior_nc）
- 风险统计：risk_nonzero_pct ∈ [lo,hi]，std > prior_penalty_std_min
- 先验统计：prior std > prior_penalty_std_min
- 交互相关：pearson(ais_density, R_interact) ≥ interact_corr_min；缺层 → 跳过并提示

Rationale
- 采用“复用优先”的策略，尽量读取现有 .meta.json 与目录约定，不更动原有产物生成链路。
- 所有检查均为只读操作。

Consequences
- 报告生成于 reports/audit/；存在失败将以非零退出码从 CLI 传出。
- 依赖可选：若缺 xarray/numpy 则部分统计与相关性检查会被跳过并记录。

Alternatives Considered
- 直接嵌入到现有 CLI 流程 → 放弃（先以独立审计工具试运行，降低风险）。

Follow-ups
- 若审核通过并稳定：
  - 在 paper.* / serve.api 等入口增加 --require-real 守门
  - 将报告项扩展到 A/B 合并与误报白名单机制
  - 增补单测覆盖审计核心函数








