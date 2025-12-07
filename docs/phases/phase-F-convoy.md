# Phase F｜护航/编队与交互风险（Convoy/Escort & Interaction）

## 0) 背景与目标
- 背景：现有 `risk.fuse` 线性加权，未显式考虑护航降冰险与近距交互增碰险。
- 目标：新增 `R_interact` 与 `R_ice_eff`（由 `m_escort` 折减），接入融合与路由；保持 CLI/UI/产物契约不变。

## 1) 范围 / 非目标
- 范围：离线月度识别编队、生成两类新栅格层、最小改动接入 A* 成本。
- 非目标：实时编队追踪、全新路由器替换。

## 2) 产物与契约
- `data_processed/risk/R_interact_<ym>.nc`    # 维度 (y, x) 或 (time, y, x)，值∈[0,1]，var: `risk`
- `data_processed/risk/R_ice_eff_<ym>.nc`     # 由 `m_escort` 对 R_ice 乘性折减后的冰险，var: `risk`
- 融合输出仍为 `data_processed/risk/risk_fused_<ym>.nc`（软链/同名）
- UI 不变：风险图层复用，新增“显示交互风险叠加”勾选（非破坏性）
- CLI：补充 `risk.interact.build` / `risk.ice.apply-escort` / `risk.fuse --method stacking|...`

## 3) 数据与接口对齐
- 输入 AIS：`data_processed/ais/tracks_<ym>.parquet`, `segment_index_<ym>.parquet`
- 现有风险层：`core/risk/fusion.py` 按 (time, y, x)；优先对齐到先验网格（EPSG:3413）
- 路由器：`core/planners/astar_grid_time.py` + `core/route/astar_cost.py`
- 成本提供者：`core/cost/env_risk_cost.py`（新增可选参数：congestion/interact/prior_penalty）

## 4) 任务清单（按顺序）
- [ ] F-01 编队识别：基于 `segment_index_<ym>.parquet` 做时空聚类与 leader/follower 判定；输出 episodes parquet
- [ ] F-02 P_escort_corridor KDE：把 episodes 投影成概率走廊；生成 `P_escort_corridor_<ym>.nc`
- [ ] F-03 m_escort 与 R_ice_eff：`m = 1-η·P_escort_corridor`（门控 η≤0.3）→ 写 `R_ice_eff_<ym>.nc`
- [ ] F-04 相遇抽取与 r_enc：计算 DCPA/TCPA/Δψ/动态船域 → 聚合为 `R_interact_<ym>.nc`
- [ ] F-05 融合接线：修改 `core/risk/fusion.py` 支持外部通道（R_ice_eff 优先于 R_ice；可选添加 R_interact）
- [ ] F-06 CLI：在 `api/cli.py` 新增 `risk.interact.build`、`risk.ice.apply-escort`、`risk.fuse --method stacking`
- [ ] F-07 成本接线：`env_risk_cost.py` 接收 interact/corridor，可通过 `runtime.yaml` 开关权重
- [ ] F-08 UI：`apps/layers_risk.py` 添加 R_interact 叠加开关；`route_params.py` 暴露开关（默认关闭）
- [ ] F-09 报告：新增 “校准曲线/ECE、护航门控热力、路径分段贡献（R_ice vs R_interact）”
- [ ] F-10 基线评测：AUC/Brier/ECE + 路径级总风险/绕行率/主走廊命中；达不到阈值→回退策略

## 5) 验收标准（Definition of Done）
- [ ] 运行 `risk.ice.apply-escort` 后，`R_ice_eff_<ym>.nc` 生成，attrs 写明 `source='escort'`
- [ ] 运行 `risk.interact.build` 后，`R_interact_<ym>.nc` 生成，取值∈[0,1]，NaN 比例 ≤ 原始 R_ice
- [ ] `risk.fuse --method stacking` 输出的 `risk_fused_<ym>.nc` 相比线性：ECE ↓10%+（或绝对值 < 0.08）
- [ ] 路由在开启 `interact_weight>0` 时，拥挤水道的绕行率↑且总距离增加不超过 5%
- [ ] UI 能显示/隐藏交互层，导出报告含上述指标与差异对比

## 6) 风险与回退
- 质控未达：`η → 0`（m_escort 退化为 1），或不纳入 R_interact，保持原融合
- 文档化：记录到 `reports/d_stage/phaseF_metrics_<ym>.json` 与 `.html`

## 7) TL;DR for Codex（执行指令）
> 建议 Codex 每次工作先打开 `docs/_index.md`，再完整阅读本文件，然后 **按以下顺序执行**：
1. **创建新模块与函数（非破坏）**
   - 新建 `ArcticRoute/core/congest/encounter.py`：实现 DCPA/TCPA/动态船域与 `build_interact_layer(ym)->xarray.DataArray`
   - 新建 `ArcticRoute/core/risk/escort.py`：实现 `apply_escort(ym, eta)->xarray.DataArray`（读 R_ice 与 P_escort_corridor）
   - 新建 `ArcticRoute/io/ais_pairs.py`：基于 `segment_index_<ym>.parquet` 识别 convoy episodes
2. **扩展融合与 CLI**
   - 修改 `ArcticRoute/core/risk/fusion.py`：优先读取 `R_ice_eff`；可合并 `R_interact`（权重来自 `config/runtime.yaml`）
   - 修改 `ArcticRoute/api/cli.py`：新增
     - `risk.interact.build --ym 202412 --method dcpa-tcpa --save`
     - `risk.ice.apply-escort --ym 202412 --eta 0.3 --save`
     - `risk.fuse --ym 202412 --method stacking --config config/risk_fuse_202412.yaml`
3. **成本与 UI**
   - 修改 `ArcticRoute/core/cost/env_risk_cost.py`：`compute(...)` 支持 `interact_weight` 与 `prior_penalty_weight`
   - 修改 `ArcticRoute/apps/layers_risk.py` 与 `route_params.py`：新增交互层开关 & 权重滑条（默认 0）
4. **依赖**
   - 在 `requirements.txt`（若不存在请新建）中加入：`scikit-learn`, `hdbscan`, `einops`（用于后续 ST-UNetFormer）
5. **最小可运行用例**
   - 运行：`python -m ArcticRoute.api.cli risk.ice.apply-escort --ym 202412 --eta 0.3`
   - 运行：`python -m ArcticRoute.api.cli risk.interact.build --ym 202412`
   - 运行：`python -m ArcticRoute.api.cli risk.fuse --ym 202412 --method stacking`

## 8) 参考实现提示
- DCPA/TCPA 计算以等时间步插值两船位置，动态船域可用 SOG/船型近似椭圆域；r_enc 用分段映射 ∈[0,1]
- KDE 走廊用 `sklearn.neighbors.KernelDensity`，输出正则化成 `P_escort_corridor`
