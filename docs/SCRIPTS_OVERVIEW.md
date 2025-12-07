# Scripts Overview

This document summarizes the Python scripts under `ArcticRoute/scripts/` and their intended roles.

See `PROJECT_LAYOUT.md` for the overall project structure.

Last updated: 2025-11-20.

Whenever new scripts are added under `ArcticRoute/scripts/`, please:

1) add a module-level docstring with a `@role` tag, and

2) update this overview accordingly.

提示：本次精简后，根级 aro-ui/、packages/（含 fuel_service）、data_processed/ 已移动至 legacy/，当前主 demo 不依赖这些资源。如遇到与这些目录强耦合的脚本，请先将输入/输出路径改为 ArcticRoute/ 下的数据结构，或仅作离线演示。

## Core

- `ArcticRoute/scripts/__init__.py`  
  简要描述：Namespace initializer for the scripts package.

- `ArcticRoute/scripts/_modpath.py`  
  简要描述：Helper to resolve CLI module path regardless of working directory.

- `ArcticRoute/scripts/check_env_ready.py`  
  简要描述：Check environment readiness for demos (dependencies, data presence, etc.).

- `ArcticRoute/scripts/quick_healthcheck.py`  
  简要描述：Quick health check for core datasets and minimal route computation.

- `ArcticRoute/scripts/route_astar_min.py`  
  简要描述：Minimal A* routing CLI over a single NetCDF variable for debugging.

- `ArcticRoute/scripts/test_planner_route.py`  
  简要描述：Quick smoke test for Planner: runs a single A* route and prints a summary.

## Pipeline

- `ArcticRoute/scripts/accident_density_grid.py`  
  简要描述：Build accident density grid NetCDF from ingested incident data.

- `ArcticRoute/scripts/ais_align_min.py`  
  简要描述：Minimal AIS alignment routine to map tracks onto the model grid.  
  当前状态：产物质量可能尚未稳定，仅保留代码以备后续修复。

- `ArcticRoute/scripts/calc_risk_field.py`  
  简要描述：Compute risk field layers (ice/accident/etc.) and write to NetCDF.

- `ArcticRoute/scripts/cog_mosaic_to_grid.py`  
  简要描述：Mosaic Sentinel COG assets onto the environmental analysis grid (GeoTIFF + sidecar).  
  当前状态：产物质量可能尚未稳定，仅保留代码以备后续修复。

- `ArcticRoute/scripts/convert_ais_json_stream.py`  
  简要描述：Stream-parse AIS JSON/JSONL/GeoJSON into a normalized Parquet table.  
  当前状态：产物质量可能尚未稳定，仅保留代码以备后续修复。

- `ArcticRoute/scripts/convert_prior_centerlines_to_wgs84.py`  
  简要描述：Converts the prior centerlines file to WGS84 coordinate order and writes corrected GeoJSON.

- `ArcticRoute/scripts/corridor_from_ais.py`  
  简要描述：Derive corridor probability grid from aligned AIS tracks (DBSCAN + rasterize).  
  当前状态：产物质量可能尚未稳定，仅保留代码以备后续修复。

- `ArcticRoute/scripts/cv_stub_mosaic.py`  
  简要描述：Stub pipeline to mock CV mosaic outputs during development.

- `ArcticRoute/scripts/export_acc_hotspots.py`  
  简要描述：Export accident hotspot layers derived from analysis to GeoJSON.

- `ArcticRoute/scripts/export_ice_cache.py`  
  简要描述：Export cached computer-vision ice probability layers for downstream use.

- `ArcticRoute/scripts/export_risk_overlay.py`  
  简要描述：Export composite risk overlays (e.g., for map viewers or docs).

- `ArcticRoute/scripts/gen_raw_auto_from_json.py`  
  简要描述：Scan AIS raw JSONs and emit a normalized Parquet with [mmsi, ts, lat, lon, sog, cog].

- `ArcticRoute/scripts/get_cdse_token.py`  
  简要描述：Obtain/refresh auth token for Copernicus/CDSE access.

- `ArcticRoute/scripts/incidents_align_to_grid.py`  
  简要描述：Align incident points to model grid/time index for downstream use.

- `ArcticRoute/scripts/incidents_ingest.py`  
  简要描述：Ingest raw incident data and normalize schema for processing.

- `ArcticRoute/scripts/prep_cv_readiness.py`  
  简要描述：Prepare inputs and sanity checks for CV readiness.

- `ArcticRoute/scripts/preproc_json_fix.py`  
  简要描述：Preprocess non-standard AIS JSON files into valid array JSON under data_raw/ais_fixed.

- `ArcticRoute/scripts/stac_fetch.py`  
  简要描述：Query Sentinel STAC catalogues and optionally validate asset access.  
  当前状态：产物质量可能尚未稳定，仅保留代码以备后续修复。

## Analysis

- `ArcticRoute/scripts/accident_calibration.py`  
  简要描述：Calibration experiments for accident risk modeling (exploratory).

- `ArcticRoute/scripts/accident_overlay.py`  
  简要描述：Visualization of accident overlays vs routes for exploratory analysis.

- `ArcticRoute/scripts/accident_research_plots.py`  
  简要描述：Research plots for accident risk vs background and route distance to hotspots.

- `ArcticRoute/scripts/ai_eval_loop.py`  
  简要描述：Utility script: ai eval loop

- `ArcticRoute/scripts/batch_alpha_sweep.py`  
  简要描述：Batch-run Planner with multiple alpha/weights to compare outcomes.

- `ArcticRoute/scripts/compare_ice_blend.py`  
  简要描述：Compare baseline vs ice-blend runs and generate docs/plots.

- `ArcticRoute/scripts/inspect_prior_centerlines.py`  
  简要描述：Inspects prior centerline GeoJSONs and checks CRS/bounds/high-lat coverage.

- `ArcticRoute/scripts/make_placeholder_corridor.py`  
  简要描述：Generate a placeholder corridor layer for illustrative experiments.

- `ArcticRoute/scripts/plot_scenarios.py`  
  简要描述：Plot scenario results and comparisons from batch runs.

- `ArcticRoute/scripts/probe_ais_json_schema.py`  
  简要描述：Sample and list candidate keys from AIS JSON to infer mmsi/time/lat/lon schema.

- `ArcticRoute/scripts/run_cv_test_suite.py`  
  简要描述：Run a lightweight CV validation suite and summarize artifacts for debugging.

- `ArcticRoute/scripts/run_scenarios.py`  
  简要描述：Batch-run minimal A* scenarios and collect metrics CSV.

- `ArcticRoute/scripts/visualize_env.py`  
  简要描述：Quick visualization helpers for environment/risk grids.

## Legacy

- `ArcticRoute/scripts/maintenance/consistency_check.py`  
  简要描述：High-level repository consistency checks; historic helper.

- `ArcticRoute/scripts/maintenance/fix_env_path.py`  
  简要描述：Maintenance script to normalize environment path variables; kept for reference.

- `ArcticRoute/scripts/maintenance/pre_demo_check.py`  
  简要描述：Pre-demo checklist runner; retained for historical reference.

- `ArcticRoute/scripts/maintenance/repo_tidy.py`  
  简要描述：Repository hygiene utility; historical helper (not part of pipeline).
