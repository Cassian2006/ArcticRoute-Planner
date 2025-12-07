# ArcticRoute Engineering Baseline

This repository ships a minimal engineering baseline so the broader ArcticRoute modules can share
the same dependency, configuration, and path conventions.

## Quickstart

1. Copy `.env.example` to `.env` and adjust paths or credentials as needed.
2. Install dependencies with your preferred tool:
   - `uv sync` *(recommended)*, or
   - `pip install -e .[dev]`
3. Run checks:
   - `uv run pytest -q`
   - `uv run ruff check .`

The default settings module (`arcticroute.settings`) reads values from `.env`. The companion
`arcticroute.paths` module ensures that the data, cache, and log directories exist on import.

Verify the settings loader with:

```bash
python -c "from arcticroute.settings import settings; print(settings.ENV)"
```

Secrets and API tokens must only live in your local `.env` file. Keep `.env.example` public and free
of real credentials.

## Smoke Tests & Samples

- Synthetic demo assets live under `data/samples/` (raster, AIS, coastline stubs) for offline tests.
- The CLI automatically degrades to a straight-line GeoJSON when source datasets are absent; the
  output still reports `waypoints`, `eta_hours`, and `cost`.
- Run the fast test harness with `uv run pytest -q -m "e2e_stub or p0"` to exercise the fallback
  pipeline and Otsu guards.

---

# Phase B · AIS Features Pipeline（新成员上手指南）

本节文档帮助你在本机完成 AIS → Features 的最小链路，并理解可配置项与产物目录结构。

## 数据准备

- AIS 源目录（默认）：`ArcticRoute/data_raw/ais`
  - 支持 JSON（数组）与 JSONL（逐行）两种格式。
  - 文件可按月分段，但不是强制要求；系统将根据记录时间 `ts` 自动分桶。
- JSON 字段映射
  - 系统会先运行“架构探测”生成建议映射：`reports/recon/ais_keymap_suggest.json`
  - 归一化目标字段（不区分大小写，示例映射在上面 JSON 中）：
    - 必填：`mmsi, ts, lat, lon`
    - 选填：`sog, cog, heading, vessel_type, loa, beam, nav_status`
  - 时间 `ts` 接受三种格式：ISO8601、epoch_ms、epoch_s；会统一为 UTC 秒（int）。

可选：你也可以手动编辑 `reports/recon/ais_keymap_suggest.json` 来覆盖默认键名映射。

## 一键运行示例

以下命令会将 2024-12 的 AIS 数据处理为分区 Parquet、栅格密度与合成特征，并生成 QA 摘要与 PNG。

1) 构建特征（干跑预览计划）

```bash
python -m ArcticRoute.api.cli features.build \
  --src ArcticRoute/data_raw/ais \
  --months 202412 \
  --time-step 6H \
  --smooth-sigma 0 \
  --dry-run
```

2) 实际执行（落盘并登记工件）

```bash
python -m ArcticRoute.api.cli features.build \
  --src ArcticRoute/data_raw/ais \
  --months 202412 \
  --time-step 6H \
  --smooth-sigma 0
```

3) 生成 QA 与可视化快照

```bash
python -m ArcticRoute.api.cli features.summarize --months 202412
```

提示：如需强制重算（覆盖已存在产物），可加 `--force`；默认会跳过已完成分区（幂等，支持断点续跑）。

## 可选参数与性能

- 时间分桶：`--time-step`（默认读取 `ArcticRoute/config/grid_spec.json` 的 `time.freq`，否则 6H）
- 船型分层：`--classify/--no-classify`（基于 `vessel_type/loa/beam` 生成 `vclass`）
- 段内重采样：`--resample-step`（秒，默认关闭）
- 高斯平滑：`--smooth-sigma`（格点，默认 0 关闭）
- 清洗速度阈值：`--speed-max`（kn，默认 35）
- 并发与内存：`configs/runtime.yaml` 可配置
  
  ```yaml
  features:
    max_workers: 4      # 文件级并行线程数（Windows 友好，避免多进程 spawn）
    chunk_lines: 100000 # pandas 回退路径下的分块行数
  ```

## 产物目录结构（示例）

```
ArcticRoute/
└─ data_processed/
   ├─ ais_parquet/
   │  └─ year=2024/month=12/
   │     ├─ part-202412xx-00001.parquet
   │     ├─ part-...grid.parquet           # 网格索引后（可选）
   │     └─ part-...timebinned.parquet     # 时间分桶后（可选）
   └─ features/
      ├─ ais_density_202412.nc             # AIS 密度栅格
      └─ features_202412.nc                # 合成特征（AIS+可选环境层）

outputs/
├─ features_summary_202412.json            # QA 摘要
└─ features_202412.png                     # 可视化快照

reports/
└─ recon/
   ├─ ais_schema.json
   └─ ais_keymap_suggest.json
```

## UI 预览（可选）

运行最小演示 UI：

```bash
streamlit run ArcticRoute/apps/app_min.py
```

在 Layers 标签页，若检测到 `features_YYYYMM.nc`，可勾选 “AIS Density” 图层进行查看（默认隐藏）。

## 验收（202412 示例）

- 执行：
  1) `features.build --months 202412`
  2) `features.summarize --months 202412`
- 验收文件：
  - `ArcticRoute/data_processed/features/features_202412.nc`
  - `outputs/features_summary_202412.json`
  - `outputs/features_202412.png`
- UI：在 Layers 标签页勾选 “AIS Density” 可见密度图
- 可选汇总：在 `reports/phaseB/ACCEPT_202412.md` 汇总 QA 指标与样例图链接

---

# 开发者附注

- 幂等与断点续跑：默认跳过已存在产物，写盘采用 `.part` 临时名并在完成后原子重命名；`--force` 可覆盖。
- 注册工件：所有产物通过 `ArcticRoute.cache.index_util.register_artifact` 登记，索引位于 `cache/index/cache_index.json`。
- Windows 友好：并行使用线程池（文件级并行），避免多进程 spawn；路径统一 `os.path.join`。
