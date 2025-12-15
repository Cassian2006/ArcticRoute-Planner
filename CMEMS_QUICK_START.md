# CMEMS 数据下载 - 快速开始

## 一句话总结

通过三个步骤自动下载北极海冰和波浪的近实时数据。

## 快速执行 (5 分钟)

### 1️⃣ 获取元数据 (一次性)

```powershell
cd C:\Users\sgddsf\Desktop\AR_final

# 海冰
copernicusmarine describe --contains "SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024" --return-fields all | Out-File -Encoding UTF8 reports/cmems_sic_describe.json

# 波浪
copernicusmarine describe --contains "ARCTIC_ANALYSIS_FORECAST_WAV_002_014" --return-fields all | Out-File -Encoding UTF8 reports/cmems_wav_describe.json
```

### 2️⃣ 解析配置 (一次性)

```bash
python scripts/cmems_resolve.py
```

**输出**: `reports/cmems_resolved.json` ✅

### 3️⃣ 下载数据 (重复执行)

```bash
python scripts/cmems_download.py
```

**输出**: 
- `data/cmems_cache/sic_latest.nc` (海冰浓度)
- `data/cmems_cache/swh_latest.nc` (有效波高)

✅ **完成！**

---

## 自动化 (可选)

### 方案 A: 每 60 分钟执行一次

```powershell
.\scripts\cmems_download.ps1 -Loop -IntervalMinutes 60
```

### 方案 B: 每天 13:00 UTC 执行

```powershell
$TaskName = "CMEMS_Download"
$TaskPath = "C:\Users\sgddsf\Desktop\AR_final\scripts\cmems_download.ps1"
$Trigger = New-ScheduledTaskTrigger -Daily -At 13:00
$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File $TaskPath"
Register-ScheduledTask -TaskName $TaskName -Trigger $Trigger -Action $Action -Force
```

---

## 数据说明

| 产品 | Dataset ID | 变量 | 更新频率 | 分辨率 |
|------|-----------|------|---------|--------|
| 海冰浓度 | `cmems_obs-si_arc_phy_my_l4_P1D` | `sic` | 每日 12:00 UTC | 25 km |
| 波浪高度 | `dataset-wam-arctic-1hr3km-be` | `sea_surface_wave_significant_height` | 每日两次 | 3 km, 小时级 |

---

## 常见问题

**Q: 数据在哪里?**  
A: `data/cmems_cache/` 目录

**Q: 如何查看数据?**  
```python
import xarray as xr
ds = xr.open_dataset("data/cmems_cache/sic_latest.nc")
print(ds)
```

**Q: 下载失败怎么办?**  
A: 检查网络，重新运行 `python scripts/cmems_download.py`

**Q: 如何修改下载范围?**  
A: 编辑 `scripts/cmems_download.py` 中的 `bbox` 和 `timedelta`

**Q: 如何添加其他产品?**  
A: 修改 `cmems_resolve.py` 和 `cmems_download.py` 中的产品 ID 和关键词

---

## 文件结构

```
AR_final/
├── scripts/
│   ├── cmems_resolve.py      # 解析脚本
│   ├── cmems_download.py     # 下载脚本
│   └── cmems_download.ps1    # PowerShell 包装
├── reports/
│   ├── cmems_sic_describe.json    # 海冰元数据
│   ├── cmems_wav_describe.json    # 波浪元数据
│   └── cmems_resolved.json        # 解析结果
├── data/
│   └── cmems_cache/
│       ├── sic_latest.nc          # 海冰数据
│       └── swh_latest.nc          # 波浪数据
└── docs/
    ├── CMEMS_DOWNLOAD_GUIDE.md    # 完整指南
    └── CMEMS_WORKFLOW.md          # 工作流详解
```

---

## 下一步

- 📖 详细指南: 见 `docs/CMEMS_DOWNLOAD_GUIDE.md`
- 🔧 工作流详解: 见 `docs/CMEMS_WORKFLOW.md`
- 💾 集成到应用: 在 `arcticroute/` 中调用下载脚本
- 📊 数据可视化: 使用 xarray + matplotlib 绘图

---

**最后更新**: 2025-12-15  
**状态**: ✅ 生产就绪

