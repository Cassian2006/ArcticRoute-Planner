# CMEMS 集成 - 快速参考

## 🎯 核心命令

### 1. 生成 Describe JSON
```bash
python scripts/gen_describe_json.py
# 或 PowerShell（Windows）
copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm --return-fields all | Out-File -Encoding UTF8 reports/cmems_sic_describe.json
copernicusmarine describe --contains dataset-wam-arctic-1hr3km-be --return-fields all | Out-File -Encoding UTF8 reports/cmems_swh_describe.json
```

### 2. 解析变量
```bash
python scripts/cmems_resolve.py
# 输出: reports/cmems_resolved.json
```

### 3. 刷新数据
```bash
# 最近 2 天
python scripts/cmems_refresh_and_export.py --days 2

# 仅生成 describe JSON
python scripts/cmems_refresh_and_export.py --describe-only

# 自定义参数
python scripts/cmems_refresh_and_export.py --days 3 --bbox -40,60,65,85
```

### 4. 同步到 Newenv
```bash
python scripts/cmems_newenv_sync.py
# 输出:
# - ArcticRoute/data_processed/newenv/ice_copernicus_sic.nc
# - ArcticRoute/data_processed/newenv/wave_swh.nc
```

### 5. 运行测试
```bash
pytest tests/test_cmems_planner_integration.py -v
```

---

## 📊 数据集信息

### SIC (海冰浓度)
| 项目 | 值 |
|------|-----|
| Dataset ID | `cmems_mod_arc_phy_anfc_nextsim_hm` |
| 变量 | `sic`, `uncertainty_sic` |
| 时间分辨率 | 日 |
| 空间分辨率 | ~12.5 km |

### SWH (有效波高)
| 项目 | 值 |
|------|-----|
| Dataset ID | `dataset-wam-arctic-1hr3km-be` |
| 变量 | `sea_surface_wave_significant_height` |
| 时间分辨率 | 小时 |
| 空间分辨率 | 3 km |

---

## 🔄 工作流程

```
┌─────────────────────────────────────────────────────────┐
│ 1. 生成 Describe JSON                                   │
│    python scripts/gen_describe_json.py                  │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 2. 解析变量                                             │
│    python scripts/cmems_resolve.py                      │
│    → reports/cmems_resolved.json                        │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 3. 刷新数据                                             │
│    python scripts/cmems_refresh_and_export.py --days 2  │
│    → data/cmems_cache/sic_*.nc                          │
│    → data/cmems_cache/swh_*.nc                          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 4. 同步到 Newenv                                        │
│    python scripts/cmems_newenv_sync.py                  │
│    → ArcticRoute/data_processed/newenv/                 │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 5. 启动 UI                                              │
│    streamlit run run_ui.py                              │
│    → 在 sidebar 选择 "CMEMS 近实时数据"                 │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ 常见问题

### Q1: 如何检查 describe JSON 是否生成成功？
```bash
# 检查文件大小
ls -lh reports/cmems_*_describe.json

# 查看内容（前 50 行）
head -50 reports/cmems_sic_describe.json
```

### Q2: 如何验证变量解析是否正确？
```bash
# 查看 cmems_resolved.json
cat reports/cmems_resolved.json

# 应该包含:
# {
#   "sic": {
#     "dataset_id": "...",
#     "variables": ["sic", ...]
#   },
#   "wav": {
#     "dataset_id": "...",
#     "variables": ["sea_surface_wave_significant_height", ...]
#   }
# }
```

### Q3: 如何检查最新下载的数据？
```bash
# 列出缓存目录
ls -lh data/cmems_cache/

# 查看最后刷新记录
cat reports/cmems_refresh_last.json
```

### Q4: 如何手动指定 NC 文件？
在 UI 中：
1. 展开 "☁️ CMEMS 近实时数据" 面板
2. 选择 "手动指定 NC 文件 (manual_nc)"
3. 输入文件路径，例如: `data/cmems_cache/sic_20241215.nc`

### Q5: 如何启用 CMEMS 数据用于规划？
在 UI 中：
1. 展开 "☁️ CMEMS 近实时数据" 面板
2. 选择 "CMEMS 近实时数据 (cmems_latest)"
3. 点击 "🔄 立即刷新 CMEMS 数据"
4. 点击 "规划路线"，系统会自动使用 newenv 数据

---

## 📋 文件清单

### 新增脚本
- `scripts/gen_describe_json.py` - 生成 describe JSON
- `scripts/cmems_utils.py` - 工具函数库
- `scripts/cmems_newenv_sync.py` - Newenv 同步
- `scripts/integrate_cmems_ui.py` - UI 集成（可选）

### 新增 UI 组件
- `arcticroute/ui/cmems_panel.py` - CMEMS 面板

### 新增测试
- `tests/test_cmems_planner_integration.py` - 集成测试

### 修改文件
- `scripts/cmems_refresh_and_export.py` - 完善参数
- `scripts/cmems_resolve.py` - 支持多种格式
- `arcticroute/ui/planner_minimal.py` - 集成面板（待执行）

---

## 🔗 关键函数

### cmems_utils.py
```python
from scripts.cmems_utils import (
    find_latest_nc,           # 查找最新 nc 文件
    load_resolved_config,     # 加载配置
    load_refresh_record,      # 加载刷新记录
    get_sic_variable,         # 获取 SIC 变量
    get_swh_variable,         # 获取 SWH 变量
)
```

### cmems_newenv_sync.py
```python
from scripts.cmems_newenv_sync import (
    find_latest_nc,           # 查找最新 nc 文件
    sync_to_newenv,           # 同步到 newenv
)
```

### cmems_panel.py
```python
from arcticroute.ui.cmems_panel import (
    render_env_source_selector,    # 数据源选择器
    render_cmems_panel,            # 刷新面板
    render_manual_nc_selector,     # 手动选择器
    get_env_source_config,         # 获取配置
)
```

---

## 🚀 Git 工作流

```bash
# 创建分支
git checkout -b feat/cmems-planner-integration

# 提交更改
git add -A
git commit -m "feat: integrate CMEMS near-real-time env into planner pipeline"

# 推送
git push -u origin feat/cmems-planner-integration

# 在 GitHub 创建 PR，合并到 main
```

---

## 📞 支持

如有问题，请检查：
1. CMEMS 认证是否正确
2. 网络连接是否正常
3. 数据集 ID 是否正确
4. 地理范围是否有效

---

**最后更新**: 2024-12-15  
**版本**: 1.0

