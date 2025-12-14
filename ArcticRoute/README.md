# ArcticRoute

## Environment Variables & `.env`

Before using the Moonshot K2 model, configure the API key:

```powershell
setx MOONSHOT_API_KEY "sk-xxxx"
```

If you prefer a `.env` file in the repository root, create it with:

```
MOONSHOT_API_KEY=sk-xxxx
AI_MODEL_NAME=moonshot-k2
AI_MAX_TOKENS=1024
AI_TEMPERATURE=0.7
AI_TIMEOUT_S=30
```

Restart the terminal to propagate environment variables. The loaders automatically read both `.env` and system variables at runtime.

## Running From Any Directory

- CLI commands and helper scripts detect the repository root automatically, regardless of the current working directory.
- From the repository root: `python -m api.cli health --cfg config/runtime.yaml`
- From the parent directory: `python -m ArcticRoute.api.cli health --cfg ArcticRoute/config/runtime.yaml`
- From any other directory: `python C:\path\to\ArcticRoute\scripts\run_cv_test_suite.py --cfg C:\path\to\ArcticRoute\config\runtime.yaml`

## 本地一键启动

1. 创建虚拟环境并激活：
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. 安装依赖：
   ```powershell
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. 启动最小演示界面（Streamlit）：
   ```powershell
   streamlit run ArcticRoute/apps/app_min.py
   ```
4. 浏览器访问 `http://localhost:8501`，通过“Run demo” 发起 CLI 任务，“任务面板” 会展示实时日志与状态。

如需在命令行直接生成示例输出，可运行 `python -m ArcticRoute.api.cli demo --out-dir outputs/demo`.

## 云上 Nginx 反代示例

以下示例通过 Nginx 将公网流量转发至本地的 Streamlit 服务（默认端口 8501）：

```nginx
server {
    listen 80;
    server_name demo.example.com;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_read_timeout 300;
    }
}
```

部署步骤参考：

1. 在云主机创建工作目录，拉取仓库并 `python -m venv venv && venv\Scripts\activate`。
2. 执行 `pip install -r requirements.txt` 并后台运行：
   ```bash
   streamlit run ArcticRoute/apps/app_min.py --server.address 127.0.0.1 --server.port 8501
   ```
3. 写入上述 Nginx 配置至 `/etc/nginx/conf.d/arcticroute.conf`，`nginx -t` 校验后 `systemctl reload nginx`。
4. 若需 HTTPS，可增加 `listen 443 ssl;` 并配置证书路径或使用 certbot。

## CV Readiness Check

- Run `python scripts/prep_cv_readiness.py` to generate assessments.
  - Text report: `logs/cv_readiness_report.txt`
  - JSON report: `logs/cv_readiness_report.json`
- The checklist covers Python/CUDA availability, free disk space (>= 50 GiB considered OK), optional CV dependencies, directory writability, and required satellite credentials.
- Recommended fixes:
  - Missing Python packages -> `pip install rasterio pystac-client stackstac rioxarray shapely scikit-image folium streamlit`
  - Low disk space -> clean the `outputs/` or `data_processed/` directories or extend the drive
  - Directory not writable -> adjust permissions or recreate the folder with elevated privileges
  - Credential missing -> add `CDSE_USERNAME`, `CDSE_PASSWORD`, `MPC_SAS_TOKEN` to `.env` or the system environment

## CV Module

- Select the predictor with `--predictor`:
  - `env_nc` (default): traditional ERA5-driven risk field.
  - `cv_sat`: queries MPC Sentinel-2 tiles, mosaics B02/B03/B04/B08 onto the env grid, derives `ice_prob` (NDWI/Otsu), and exposes both layers in `extra_layers`.
  - `dl_ice`: retains the ML placeholder (outputs `ice_prob` only).
- Blend satellite ice probability into the planner by setting `alpha_ice` in `config/runtime.yaml` or via `--alpha-ice` (choose `0 <= alpha_ice <= 1`; `0` keeps pure `risk_env`, intermediate values mix in `ice_prob`). The resulting run report now records `ice_prob_stats` (mean / max / coverage).
- Example: `python -m api.cli plan --cfg config/runtime.yaml --predictor cv_sat --alpha-ice 0.3 --tag cv_sat_live`

## CV Test Suite

- Run `python scripts/run_cv_test_suite.py` to execute the full CV acceptance workflow:
  1. Environment readiness check (`logs/env_ready_report.json`)
  2. Ice probability cache export (`data_processed/cv_cache/ice_prob_latest.*`)
  3. Planner runs for baseline (`cv_probe`) and blended (`cv_blend`) cases
  4. Route comparison & global histogram (`docs/compare_metrics.md`, `docs/ice_prob_hist.png`)
  5. Alpha sweep over sensitivity values (`outputs/alpha_sweep.csv`, plots in `docs/`)
  6. Consolidated report `logs/cv_test_summary.md` listing outcomes and artifact paths
- Customise inputs with `--cfg`, `--tidx`, `--gamma`, or `--alphas` as needed.
- If the mosaic date needs overriding, refresh the cache explicitly, for example:
  `python scripts/export_ice_cache.py --cfg config/runtime.yaml --tidx 0 --date 2023-07-15 --mission S2`

### CV 套件常见错误与快速修复

- 缺少依赖 (`ModuleNotFoundError`): 使用 `pip install rasterio pystac-client stackstac rioxarray scikit-image` 补齐。
- 找不到 mosaic (`sat_mosaic_*.tif` 缺失): 运行 `python scripts/export_ice_cache.py --cfg config/runtime.yaml --tidx 0 --date YYYY-MM-DD --mission S2 --verbose` 自动重建。
- 日期不匹配: 根据运行的 `tidx` 或日志日期重跑 `export_ice_cache.py`，确保 `--date` 与 env 数据对齐。
- 影像全为 NaN / 缺关键波段: 检查 STAC 日志是否命中正确任务，必要时切换 `--mission` 或调整 bbox；确认 B03/B08 (S2) 或 VV/VH (S1) 可用。
- 阈值失败 (Otsu): 增加输入样本或确认拼接区域覆盖冰水混合区域，再次执行 `export_ice_cache.py --verbose` 查看诊断统计。
