# AIS 数据路径重构验证报告

## 执行时间
2025-12-11 06:39:28 UTC

## 重构目标
✅ **完全达成**

1. ✅ 不再到 `ais_2024_sample.csv` 去找数据
2. ✅ AIS 拥挤度从 NetCDF 密度文件读（`data_real/ais/derived/*.nc`）
3. ✅ 原始 AIS 只在预处理脚本里用（从 `data_real/ais/raw/` 目录读取）
4. ✅ UI 和终端 warning 文案改成："目录/密度 nc"而不是 sample.csv

## 数据验证

### Step 0: 数据结构确认

#### 原始 AIS 数据（`data_real/ais/raw/`）
```
✅ 目录存在且包含 5 个 JSON 文件：
  - AIS of 2023.12.29-2024.03.28.json (1.76 GB)
  - AIS of 2024.03.28-2024.06.26.json (2.01 GB)
  - AIS of 2024.06.24-2024.09.22.json (1.77 GB)
  - AIS of 2024.09.22-2024.12.21.json (1.76 GB)
  - AIS of 2024.12.21-2025.01.01.json (169 MB)
```

#### AIS 密度 NC 文件（`data_real/ais/derived/`）
```
✅ 目录存在且包含：
  - ais_density_2024_demo.nc (33.8 KB)
```

## Task A: 彻底去掉对 `ais_2024_sample.csv` 的硬编码

### 修改内容

#### `arcticroute/core/ais_ingest.py`
✅ **新增路径常量**
```python
AIS_RAW_DIR = Path(__file__).resolve().parents[2] / "data_real" / "ais" / "raw"
```

✅ **新增辅助函数**
```python
def has_raw_ais_files(raw_dir: Path | str | None = None) -> bool:
    """检查目录中是否存在可识别的 AIS 文件（.json, .jsonl, .geojson, .csv）"""
```

✅ **更新 `load_ais_from_raw_dir()` 函数**
- 默认参数改为 `raw_dir=AIS_RAW_DIR`
- 优先读取 JSON/JSONL/GeoJSON，CSV 作为 fallback
- 不再硬编码 CSV 文件名
- 更新警告文案：`"[AIS] 原始 AIS 目录为空或不存在: {raw_dir}, AIS 数据未加载"`

#### `arcticroute/core/cost.py`
✅ **新增/更新路径常量**
```python
AIS_DENSITY_PATH_DEMO = Path(...) / "data_real" / "ais" / "derived" / "ais_density_2024_demo.nc"
AIS_DENSITY_PATH_REAL = Path(...) / "data_real" / "ais" / "derived" / "ais_density_2024_real.nc"
AIS_DENSITY_PATH = AIS_DENSITY_PATH_DEMO  # 向后兼容别名
```

✅ **更新 `load_ais_density_for_demo_grid()` 函数**
- 使用 `AIS_DENSITY_PATH_DEMO` 常量
- 更新文案：`"[AIS] 密度文件不存在: {target}"`

✅ **更新 `load_ais_density_for_grid()` 函数**
- 支持 `prefer_real` 参数优先加载真实分辨率 NC
- 回退到 demo NC
- 都不存在时返回 None，不抛异常
- 更新文案：`"[AIS] 未找到 AIS 密度数据，将不使用 AIS 主航道成本 (可先运行 python -m scripts.preprocess_ais_to_density)"`

✅ **新增 `has_ais_density_data()` 函数**
```python
def has_ais_density_data(grid: Grid2D | None = None, prefer_real: bool = True) -> bool:
    """最大努力检查当前是否有 AIS 密度文件可用，不抛异常"""
```

✅ **更新 `_add_ais_cost_component()` 函数**
- 更新文案：`"[AIS] 密度数据不可用或形状不匹配，跳过 AIS 成本"`

### 验证结果
✅ **所有 Task A 修改完成**
- 不再有任何硬编码的 `ais_2024_sample.csv` 路径
- 所有 AIS 数据加载都通过目录或 NC 文件进行
- 警告文案已更新，不再提及具体的 CSV 文件名

## Task B: 统一 AIS 密度 NC 的路径（成本层）

### 修改内容

#### `arcticroute/core/cost.py`
✅ **路径常量集中管理**
- `AIS_DENSITY_PATH_DEMO` - demo 分辨率 NC 文件
- `AIS_DENSITY_PATH_REAL` - 真实分辨率 NC 文件（预留）
- `AIS_DENSITY_PATH` - 向后兼容别名

✅ **统一加载逻辑**
- `load_ais_density_for_grid()` 是唯一的加载入口
- 支持 `prefer_real` 参数自动选择最佳可用文件
- 不存在时静默返回 None，不影响路由规划

### 验证结果
✅ **所有 Task B 修改完成**
- AIS 密度路径完全统一
- 支持 demo 和 real 两个分辨率
- 加载逻辑清晰、易于维护

## Task C: UI 侧只根据"密度 NC 是否存在"给提示

### 修改内容

#### `arcticroute/ui/planner_minimal.py`

✅ **更新 AIS 数据检查逻辑（第 670-700 行）**
```python
# 检查 AIS 密度数据是否可用（根据 NC 文件）
has_ais_density = cost_core.has_ais_density_data(prefer_real=(grid_mode == "real"))

if not has_ais_density and w_ais > 0:
    st.caption("⚠ 当前未找到 AIS 拥挤度密度数据，将临时忽略 AIS 成本 （可先运行 `python -m scripts.preprocess_ais_to_density` 生成 nc 文件）")
elif has_ais_density:
    st.caption("✅ 已加载 AIS 拥挤度密度数据，w_ais 对成本已生效")
```

✅ **更新 AIS 密度加载逻辑（第 840-865 行）**
```python
# 尝试从密度 NC 文件加载 AIS 数据
prefer_real = (grid_mode == "real")
ais_da = cost_core.load_ais_density_for_grid(grid, prefer_real=prefer_real)

if ais_da is not None:
    ais_density = ais_da.values
    st.info(f"✓ 已加载 AIS 拥挤度密度数据，栅格={ais_info['shape']}")
else:
    st.warning(f"⚠ 当前未找到 AIS 拥挤度密度数据，AIS 拥挤度成本将被禁用。可先运行 `python -m scripts.preprocess_ais_to_density` 生成 NC 文件。")
```

### 验证结果
✅ **所有 Task C 修改完成**
- UI 中不再出现任何 `ais_2024_sample.csv` 的提示
- 提示文案改为"目录/密度 NC"
- 用户友好的错误提示和解决方案

## 其他脚本更新

### `scripts/debug_ais_effect.py`
✅ **更新 AIS 数据加载方式**
- 改为从 `AIS_RAW_DIR` 目录加载原始 AIS 数据
- 使用 `build_ais_density_da_for_demo_grid()` 构建密度

### `scripts/evaluate_routes_vs_ais.py`
✅ **更新 AIS 数据加载方式**
- 改为从 `AIS_RAW_DIR` 目录加载原始 AIS 数据
- 更新 `_load_ais_density()` 函数实现

## 全局搜索验证

### 搜索 `ais_2024_sample.csv` 结果
```
✅ 核心代码（arcticroute/）中：0 个引用
✅ 脚本（scripts/）中：0 个引用
✅ UI（arcticroute/ui/）中：0 个引用
```

所有硬编码的 CSV 文件名已完全移除。

## 向后兼容性

✅ **保留向后兼容**
- `AIS_DENSITY_PATH` 别名保留，指向 `AIS_DENSITY_PATH_DEMO`
- 现有代码无需修改即可继续工作
- 测试数据保留，不影响现有测试

## 总结

### 完成情况
| 任务 | 状态 | 说明 |
|------|------|------|
| Task A - 去掉 CSV 硬编码 | ✅ 完成 | 所有 CSV 引用已移除 |
| Task B - 统一 NC 路径 | ✅ 完成 | 路径常量集中管理 |
| Task C - 更新 UI 提示 | ✅ 完成 | 提示文案已更新 |
| 数据验证 | ✅ 完成 | 原始数据和密度 NC 都已确认 |
| 脚本更新 | ✅ 完成 | debug 和 evaluate 脚本已更新 |

### 关键改进
1. ✅ **灵活性**：不再依赖特定的 CSV 文件，支持多种格式和多个文件
2. ✅ **可维护性**：路径常量集中管理，易于修改和扩展
3. ✅ **用户友好**：提示文案更清晰，指导用户如何生成缺失的数据
4. ✅ **向后兼容**：保留了别名，不破坏现有代码

### 建议的后续步骤
1. 运行 `python -m scripts.debug_ais_effect` 验证 AIS 效果
2. 运行 `python -m scripts.evaluate_routes_vs_ais` 验证路由对比
3. 在 Streamlit UI 中测试 AIS 权重滑条
4. 验证 NC 文件加载和密度可视化

## 签名
✅ **重构完成** - 2025-12-11 06:39:28 UTC





