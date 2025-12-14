# Prior Transformer 全链条指南（E-T18）

本指南说明如何以最小代价、按步骤重现“先验走廊”从数据到采纳再到路由/前端可视的全链路。适配小样本（如 202412）与 RTX 3070 Ti（bf16/SDPA/梯度累积/激活检查点）。

目录
- 0. 环境与依赖
- 1. 配置与路径
- 2. 数据就绪化（E-T01）
- 3. 切分与对比采样（E-T03）
- 4. 特征与时间位置编码（E-T04）
- 5. 模型定义与训练（E-T05/E-T07/07.5）
- 6. 嵌入导出（E-T09）
- 7. 聚类（HDBSCAN 网格扫描）（E-T10）
- 8. 中心线与带宽（E-T11）
- 9. 栅格化 P_prior 与 PriorPenalty（E-T12）
- 10. 评测：覆盖率/偏差/稳健性（E-T13）
- 11. 采纳判定与回退（E-T14）
- 12. 路由对接与前端（E-T15/E-T16）
- 附录：常见问题与参数建议

---

## 0. 环境与依赖

确保安装 requirements.txt 中的依赖：
- torch>=2.2（SDPA 内核自动选择 flash/mem-efficient）
- pyproj、xarray、netCDF4、polars/pandas、hdbscan、scikit-learn、matplotlib、streamlit 等

建议显卡：RTX 3070 Ti；推荐 bf16（若硬件/驱动支持，否则 AMP fp16）。

## 1. 配置与路径

集中配置：`configs/prior_transformer.yaml`
- 数据路径/投影 CRS/采样/模型/训练/聚类/评测/采纳/路由/前端全部集中管理
- CLI 参数可覆盖 YAML 中的默认值

## 2. 数据就绪化（E-T01）

按月从原始 JSON 复用清洗/分段逻辑，生成 `tracks_<YYYYMM>.parquet` 与 `segment_index_<YYYYMM>.parquet`：

```bash
python -m api.cli prior.ds.prepare --ym 202412
```

日志会输出段数分布与样例；若已存在 tracks_<YM>.parquet 将复用。

## 3. 切分与对比采样（E-T03）

按 MMSI 分层切分 Train/Val；ContrastivePairSampler 提供正样本（同段不同裁剪）与难负采样（同月近时近域不同 MMSI）。

```bash
python -m api.cli prior.ds.split --ym 202412 --train-ratio 0.8 --seed 42 --out reports/prior/split_202412.json
```

断言 Train/Val MMSI 无交集，seed 固定可复现。

## 4. 特征与时间位置编码（E-T04）

输入通道：Δx, Δy, sog, cos(cog), sin(cog), hour_sin, hour_cos, doy_sin, doy_cos；稳健标准化（中位数/MAD）。
- 位置编码：RoPE（默认）或 time2vec。

## 5. 模型定义与训练（E-T05/E-T07/07.5）

Encoder-only（Pre-Norm），SDPA 注意力，激活检查点，支持 AMP/bf16 与梯度累积。

```bash
# 干跑（小步数验证）
python -m api.cli prior.train --ym 202412 --epochs 3 --bf16 --batch 16 --grad-accum 4 --dry-run

# 实跑（默认 20 epoch，参数见 configs）
python -m api.cli prior.train --ym 202412 --epochs 20 --bf16 --batch 16 --grad-accum 4
```

产物：`reports/phaseE/train_logs/prior_<YM>_<ts>/best.ckpt` 与训练日志（CSV/PNG）。

## 6. 嵌入导出（E-T09）

将所有段编码为 256 维嵌入（列扁平为 emb_0..emb_255）：

```bash
python -m api.cli prior.embed --ym 202412 --ckpt reports/phaseE/train_logs/prior_202412_<ts>/best.ckpt
```

产物：`data_processed/ais/embeddings_<YM>.parquet`

## 7. 聚类（HDBSCAN 网格扫描）（E-T10）

对 (min_cluster_size, min_samples) 网格扫描，筛选噪声占比<40%、silhouette/覆盖综合最优：

```bash
python -m api.cli prior.cluster --ym 202412 --min-cluster 30,50,80 --min-samples 5,10,15
```

产物：`cluster_assign_<YM>.parquet` 与扫描报告 `reports/phaseE/cluster/scan_report_<YM>.json`

## 8. 中心线与带宽（E-T11）

DBA 近似中心线 + 带宽分位（默认 p75），每簇一条折线：

```bash
python -m api.cli prior.centerline --ym 202412 --band-quantile 0.75
```

产物：`reports/phaseE/center/prior_centerlines_<YM>.geojson`

> 可扩展：KDE 脊线作为首选，DBA 为保底。

## 9. 栅格化 P_prior 与 PriorPenalty（E-T12）

将中心线/带宽转为 P_prior（max 聚合）与 PriorPenalty=1-P_prior。网格按 P1 对齐（grid_spec.json 或 env_clean.nc 读取）。

```bash
python -m api.cli prior.export --ym 202412 --method transformer
```

产物：`ArcticRoute/data_processed/prior/prior_transformer_<YM>.nc`

## 10. 评测：覆盖率/偏差/稳健性（E-T13）

计算 Val 覆盖率（P_prior≥τ），横向偏差（到中心线最近距离的均值/p95），以及跨日/周稳定性：

```bash
python -m api.cli prior.eval --ym 202412 --method transformer --tau 0.5
```

产物：`reports/phaseE/prior_metrics_<YM>.json` 与 `prior_summary_<YM>.html`

## 11. 采纳判定与回退（E-T14）

规则：若 coverage≥C_min 且 deviation_mean≤D_max → adopt=transformer；否则若存在 prior_density_<YM>.nc 且 coverage≥0.9*C_min → adopt=density；否则 adopt=none。

```bash
python -m api.cli prior.select --ym 202412 --c-min 0.7 --d-max-nm 5 --tau 0.5
```

产物：
- 若 adopt!=none：`ArcticRoute/data_processed/prior/prior_corridor_selected_<YM>.nc`
- 报告：`reports/phaseE/PRIOR_SELECT_<YM>.md`

## 12. 路由对接与前端（E-T15/E-T16）

- CLI 层：可增 `--prior` 与 `--w_p`（>0时启用）；若未指定 prior 则自动发现 `prior_corridor_selected_<YM>.nc` 并提示。
- 前端层：在 Route 页面注入 w_p 滑条与 prior 文件发现提示；w_p>0 时自动叠加 PriorPenalty 到路由代价。

在 Demo 前端（Streamlit）中：
- Layers 页显示 P_prior，提供中心线下载按钮
- Route 页侧栏可调节 w_p（默认 0）

---

## 附录：常见问题与参数建议

- 资源建议（3070 Ti）：
  - bf16 优先（若支持），否则 AMP fp16
  - SDPA 自动启用；梯度累积 4；激活检查点开启（checkpoint=true）
- 密度过稀：
  - 聚类时增大 min_cluster_size，或降低 min_samples；
  - 栅格化时调大带宽分位（p90）；
- 噪声高：
  - 先在 HDBSCAN 网格扫描中挑选噪声占比 <40% 的最佳参数；
- 中心线断裂：
  - 尝试 KDE 脊线方法；或在 DBA 重采样时增大 resample_points；
- 评测指标异常：
  - 检查 grid_spec.json 与 env_clean.nc 的坐标一致性；
  - 检查 τ（tau）阈值是否与需要一致；
- 采纳失败（adopt=none）：
  - 提升训练/采样质量，重跑聚类与中心线；
  - 备选回退：生成 prior_density_<YM>.nc 并复核 coverage。

---

## 报告“Prior”章节插槽（建议）

在总报告（report-html）中新增“Prior”章节：
- 覆盖率-τ 曲线（可扫描 τ∈[0.1..0.9]）
- 偏差箱线图（均值/中位数/p95）
- 示例可视（P_prior 与中心线叠加的局部放大）

实现建议：
- 从 `reports/phaseE/prior_metrics_<YM>.json` 读取基础统计
- 若存在 `prior_summary_<YM>.html`，可内嵌或链接
- 生成图像保存到 `reports/phaseE/prior_figs_<YM>/...png` 并在 HTML 中引用













