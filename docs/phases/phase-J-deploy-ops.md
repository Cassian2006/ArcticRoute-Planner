# Phase J | Deployability, Automation & Ops (P5)

## 0) Context & Goal
**Context.** Phases F–I added convoy/interaction, Pareto, explainability, and uncertainty. We now need a **production-like envelope**: containerized app, scheduled NRT data pulls, cache/catalog hygiene, health checks, and one-command startup for demos and competitions.

**Goal.** Deliver:
1) **Containerized stack** (Streamlit UI + CLI runtime) with environment-based config;
2) **NRT data pulls** (ice/wave/incidents/AIS as available) via CLI with scheduling (Windows Task Scheduler or systemd/cron);
3) **Cache & Catalog GC** (TTL, disk watermarks, soft-link compatibility);
4) **Health & audit endpoints** (JSON health, run logs, catalog show);
5) **One-command serve** (docker-compose) and a short operational playbook.

No breaking changes to existing CLI/UI contracts or artifact paths.

---

## 1) Scope / Non-Goals
- **In scope**
  - Dockerfile + docker-compose for local/VM deployment
  - CLI for NRT pulls, cache/catalog GC, health probe
  - Minimal monitoring: health JSON, rotating logs, disk watermarks, alerts-as-files
- **Out of scope**
  - Heavy cloud infra (K8s/Prometheus/Grafana)
  - AuthN/Z and multi-tenant user management

---

## 2) Artifacts & Contracts
- **Inputs** (reused)
  - Configs: `config/env.yaml`, `config/runtime.yaml`, `config/scenarios.yaml`
  - Sources & ingesters: `io/stac_ingest.py`, `io/sources/`, `io/ice_sarima_lstm.py`, `io/loaders.py`
- **Outputs**
  - Docker image: `arcticroute:<tag>`
  - Compose file: `deploy/docker-compose.yml`
  - Schedulers:
    - Windows: `.xml` or `.ps1` for Task Scheduler
    - Linux: `systemd` unit + timer or `cron` entry
  - Ops CLIs:
    - `ingest.nrt.pull`
    - `catalog.gc` & `cache.gc`
    - `health.check`
    - `serve` (optional shortcut)
- **Meta**
  - All generated artifacts keep adjacent `.meta.json` per the contracts in `docs/playbooks/cli-contracts.md`.

---

## 3) Minimal Data & Secrets Checklist
- ✅ You already have monthly AIS, risk, prior artifacts.
- ➕ (Optional NRT) Access to near-real-time ice/wave feeds your current `io/stac_ingest.py` or `io/sources/` supports (keep credentials in `.env`).
- ➕ Incident CSV/Parquet refresh (if available) and AIS deltas (if allowed).
- **Secrets**: store endpoints/keys in `.env` (never commit), then have the CLI read them via `os.environ` → `config/env.yaml` overrides.

---

## 4) Tasks (Execution Order)
- [ ] **J-01 Containerization**
  - Add `Dockerfile` (multi-stage if needed). Goals:
    - Install project + pinned deps (`requirements.txt` or `pyproject.toml`)
    - Create a non-root user; mount `/data` for `data_processed/`, `/reports`, `/outputs`
    - Entrypoint supports both **UI** and **CLI**
  - Add `deploy/docker-compose.yml`:
    - Service `ui`: runs Streamlit (`apps/app_min.py`), port mapping `8501:8501`
    - Bind mounts host `./data_processed`, `./reports`, `./outputs`
    - Optional service `cron`: runs periodic CLI tasks via a lightweight scheduler container
- [ ] **J-02 Ops CLI commands**
  - In `api/cli.py` add:
    - `ingest.nrt.pull --ym <YYYYMM|current> --what ice,wave,incidents --since <ISO8601>`  
      Reuses `io/stac_ingest.py` and `io/sources/*`; writes under `data_processed/env/` or existing domains; logs to catalog meta.
    - `cache.gc --ttl-days 90 --watermark-disk 0.8`  
      Deletes temp caches in `data_processed/*/cv_cache`, prunes old tiles; stops when disk usage < watermark.
    - `catalog.gc --keep-months 6 --dry-run`  
      Lists candidates by `logical_id` time; optionally removes old `_adv_` versions keeping soft-link targets.
    - `health.check --out reports/health/health_<date>.json`  
      Checks file perms, free disk, key paths, python deps, GPU availability, read/writable dirs; emits human-readable hints.
    - (Optional) `serve --ui` → shorthand to run Streamlit with environment.
- [ ] **J-03 Schedulers**
  - **Windows**: add `deploy/schedule/pull_nrt_daily.ps1` and a Task Scheduler export `.xml`
    - Executes: `python -m ArcticRoute.api.cli ingest.nrt.pull --ym current --what ice,wave --since -P1D`
    - Then: `python -m ArcticRoute.api.cli risk.fuse --ym current --method stacking`
    - Then: `python -m ArcticRoute.api.cli report.build --ym current --include calibration`
  - **Linux**: add `deploy/systemd/arcticroute-pull.service` & `arcticroute-pull.timer`
- [ ] **J-04 Health & Logs**
  - Extend `api/health.py`:
    - `health_check()` returns JSON: `ok, warnings[], disk_free_gb, data_dirs, git_sha, last_runs[]`
  - Log rotation: simple size/time rotation under `outputs/logs/`
- [ ] **J-05 Catalog & Soft-link compatibility**
  - Implement `core/catalog.py` helpers:
    - `gc_list(keep_months, tags)` returns candidate artifacts by month/tags
    - `verify(logical_id)` checks path existence and hash (if present)
  - Preserve legacy paths by **symlinks** from `_adv_` to canonical filenames.
- [ ] **J-06 One-command serve**
  - Document: `docker compose up -d` to start UI and a small cron container
  - Provide `.env.example` with placeholders for endpoints/keys
  - 若仓库根无法提交点文件，已提供 `env.example.txt`（与 `.env` 内容等价），请复制为项目根 `.env` 使用
- [ ] **J-07 Tests / Smoke**
  - `docker build` and `docker compose up` run successfully
  - `health.check` JSON contains required fields
  - `ingest.nrt.pull` in `--dry-run` mode prints planned actions and exits 0

---

## 5) CLI (Smoke)
```bash
# 1) Health probe
python -m ArcticRoute.api.cli health.check --out reports/health/health_$(date +%Y%m%d).json

# 2) Dry-run NRT pull (no changes, just plan)
python -m ArcticRoute.api.cli ingest.nrt.pull --ym current --what ice,wave --since -P1D --dry-run

# 3) Cache GC to 80% watermark
python -m ArcticRoute.api.cli cache.gc --ttl-days 90 --watermark-disk 0.8

# 4) Catalog GC plan (keep 6 months)
python -m ArcticRoute.api.cli catalog.gc --keep-months 6 --dry-run

# 5) Local UI via docker-compose
docker compose -f deploy/docker-compose.yml up -d
6) Definition of Done (Acceptance)
 docker build succeeds; docker compose up -d serves Streamlit on :8501

 ingest.nrt.pull (dry-run and real mode) operates without errors; writes meta with inputs and run_id

 cache.gc + catalog.gc reduce disk usage to below watermark, preserve canonical soft-links

 health.check emits JSON with actionable warnings (e.g., low disk, missing dirs)

 A single README snippet allows judges to reproduce: pull → fuse → route → report in one or two commands

7) Risks & Mitigations
Large images → use slim base; cache wheels; separate dev vs runtime stages

GDAL/GEOS heavy deps → avoid unless required; rely on xarray/rioxarray where possible

Secrets leakage → use .env and mount-time env overrides; never commit real keys

Windows vs Linux path quirks → use env vars and pathlib; document both schedulers