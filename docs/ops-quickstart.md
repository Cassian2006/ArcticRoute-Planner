# ArcticRoute 运维快速上手（Ops Quickstart）

本指南帮助你用最少步骤完成：环境配置、容器启动、健康检查、近实况拉取与报告构建。

## 1) 准备环境变量
- 复制项目根的 `env.example.txt` 为 `.env`，按需填写（尤其是 STAC/CDSE 访问令牌/账号）。
- 关键变量：
  - `ARCTICROUTE_ROOT=/app`、`PYTHONPATH=/app`、`ARCTICROUTE_DATA=/data`
  - `CDSE_TOKEN` 或 `CDSE_USERNAME`/`CDSE_PASSWORD`
  - `MPC_STAC_URL`（如使用 MPC）
  - `FORCE_HTTP_MOSAIC`（在不支持 s3:// 环境下可设为 true）

## 2) 构建并启动容器
```bash
# 仓库根
docker build -t arcticroute:latest .
# 启动 UI 与 OPS 容器（后台）
docker compose -f deploy/docker-compose.yml up -d
# 打开 UI: http://localhost:8501
```

## 3) 健康检查（容器内执行）
```bash
docker exec -it arcticroute-ops python -m ArcticRoute.api.cli health.check --out ArcticRoute/reports/health/health_$(date +%Y%m%d).json
```
- 输出：
  - `ArcticRoute/reports/health/summary.json|md`
  - 自定义 `--out` 的 JSON
  - JSON 中 `extras` 包含磁盘信息、数据目录可写性、git sha、最近运行摘要

## 4) 近实况（NRT）拉取（最小对接）
```bash
# dry-run 计划
docker exec -it arcticroute-ops python -m ArcticRoute.api.cli ingest.nrt.pull --ym current --what ice,wave --since -P1D --dry-run
# 实跑（将基于 env_clean.nc 生成占位栅格，并记录 STAC 查询及预览访问）
docker exec -it arcticroute-ops python -m ArcticRoute.api.cli ingest.nrt.pull --ym current --what ice,wave --since -P1D
```
- 过程说明：
  - # REUSE `ArcticRoute.io.stac_ingest.stac_search_sat` 搜索 STAC 项
  - # REUSE `download_asset_preview` 尝试下载首个资产预览字节（验证权限连通性）
  - # REUSE `stub_mosaic_to_grid(env_clean.nc, out.tif)` 基于 `env_clean.nc` 输出占位镶嵌（真实环境可替换为 COG 合成）
  - 在 `ArcticRoute/logs/` 写入 `ingest_nrt_<run_id>.meta.json`（含 inputs/outputs/items），并 #REUSE `register_artifact` 登记索引

## 5) 融合与报告
```bash
docker exec -it arcticroute-ops python -m ArcticRoute.api.cli risk.fuse --ym current --method stacking
# 示例：生成校准报告
docker exec -it arcticroute-ops python -m ArcticRoute.api.cli report.build --ym current --include calibration
```

## 6) 调度（可选）
- Windows：Task Scheduler 导入 `deploy/schedule/pull_nrt_daily.xml`，或直接用 `pull_nrt_daily.ps1`
- Linux：复制 `deploy/systemd/arcticroute-pull.service|timer` 至 `/etc/systemd/system/` 并启用：
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now arcticroute-pull.timer
```

## 7) 故障排查
- 检查 `.env` 是否正确挂载（compose 使用 `env_file: ../.env`）
- `FORCE_HTTP_MOSAIC=true` 可绕过 s3:// 预览限制
- 查看健康检查 `extras.disk` 与 `data_dirs` 的可写性
- 查看 `ArcticRoute/logs/stac_results_*` 与 `ingest_nrt_*.meta.json`

