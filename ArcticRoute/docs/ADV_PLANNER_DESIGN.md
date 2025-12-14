# 高级规划器设计（补充：Evidential-Robust 集成）

本补充说明记录 Phase 2 中“Evidential + Robust”风险聚合在高级规划器中的落地细节。

## 1. 数据源与变量

- 文件：ArcticRoute/data_processed/risk/risk_fused_{ym}.nc（或 risk_fused_{ym}_{fusion_mode}.nc）
- 变量：
  - Risk：均值风险（0..1）
  - RiskVar：风险方差（≥0）
  - 可选 time 维，系统默认取 time=0 切片。

## 2. 鲁棒风险表面构造

在 ArcticRoute/core/advanced_risk.py 中新增：

- build_evidential_robust_surface(ym, fusion_mode, risk_agg_mode, risk_agg_alpha=0.9) → xr.DataArray | None
  - 若 fusion_mode 不含“evidential”：返回 None。
  - 若仅有 Risk 或 risk_agg_mode 为 mean：返回 Risk（二维化）。
  - 若存在 Risk + RiskVar 且请求 cvar/robust：构造 R_robust = Risk + λ*sqrt(RiskVar)，其中
    - α≥0.95 → λ=2.0；α≥0.9 → λ=1.5；否则 λ=1.0。
  - 返回二维 DataArray，name="risk_evidential_robust"，attrs 标注：
    - agg: "mean" 或 "cvar"
    - agg_alpha: 传入的 α
    - source: "evidential_risk_plus_var"

异常情况下返回 None，由上层回退到原逻辑。

## 3. 环境加载中的优先级

在 ArcticRoute/core/planner_service.py 的 load_environment：

1) 若 fusion_mode 含“evidential”，优先调用 build_evidential_robust_surface()；
   - 若返回非 None，则直接作为 fused_da 使用；并设置
     - fusion_mode_effective = fusion_mode
     - risk_agg_mode_effective = fused_da.attrs["agg"]（若缺失则按请求 mode 推断）
2) 仅当上述未命中/返回 None，再回退到 adv_load_fused_risk + aggregate_risk_da（用于其它融合模式或缺文件时）。

成本叠加保持原逻辑：将 fused_da 分位标准化后乘 15 叠加到最终成本。

## 4. 有效模式与 UI 验证

- cost_da.attrs：
  - fusion_mode_effective：成功命中 evidential 时为 "evidential"
  - risk_agg_mode_effective：
    - evidential robust 命中时为 "cvar"
    - 无 RiskVar 或未命中 evidential 时，根据样本维聚合或回退为 "mean"
  - risk_agg_alpha：透传

- 运行 quick demo（python -m ArcticRoute.scripts.run_evidential_robust_demo）时：
  - 若 evidential robust 生效，不应再出现“mode=cvar not applicable; fallback to mean”降级日志。

## 5. 兜底改进：避免同一点起止

为适配部分前端多策略入口在特定分支下可能传入相同起止点（导致退化路线）的情况，后端在 run_planning_pipeline_evidential_robust 内增加了兜底：

- 当检测到 start_ij == end_ij 时，会在局部邻域内搜索最近的可通行海洋格点，自动将 end_ij 轻推到该位置，确保路线可达。
- 该兜底不改变既有参数契约，仅在异常输入场景触发，对正常流程无影响。

