# 数据来源与下载说明
- Copernicus CDS: ERA5 单层再分析（风、浪、海冰）
- 下载脚本：`data_download/download_era5.py`
- API Key: 放置于 `~/.cdsapirc`
- 事故样本：`data_raw/incidents.csv`，可通过 `scripts/accident_overlay.py` 生成事故密度栅格供路径规划折扣使用
