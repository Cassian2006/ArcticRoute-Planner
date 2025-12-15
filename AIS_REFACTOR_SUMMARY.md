# AIS 数据路径重构总结

## 目标
彻底去掉对 `ais_2024_sample.csv` 的硬编码，改为：
1. **原始 AIS 数据**：从 `data_real/ais/raw/` 目录读取（支持 JSON/JSONL/GeoJSON/CSV）
2. **AIS 密度数据**：从 `data_real/ais/derived/*.nc` 读取预处理的密度栅格
3. **UI 和终端提示**：改为提示"目录/密度 NC"而不是具体的 CSV 文件名

## 完成的修改

### Task A: 彻底去掉对 `ais_2024_sample.csv` 的硬编码

#### 1. `arcticroute/core/ais_ingest.py`
- **新增常量**：`AIS_RAW_DIR = Path("data_real/ais/raw")`
- **新增函数**：`has_raw_ais_files(raw_dir)` - 检查目录中是否存在可识别的 AIS 文件
- **更新函数**：`load_ais_from_raw_dir()` 
  - 默认参数改为 `raw_dir=AIS_RAW_DIR`
  - 不再硬编码 CSV 文件名
  - 优先读取 JSON/JSONL/GeoJSON，CSV 作为 fallback
  - 更新警告文案：不再提及 `ais_2024_sample.csv`

#### 2. `arcticroute/core/cost.py`
- **新增常量**：
  - `AIS_DENSITY_PATH_DEMO = Path("data_real/ais/derived/ais_density_2024_demo.nc")`
  - `AIS_DENSITY_PATH_REAL = Path("data_real/ais/derived/ais_density_2024_real.nc")`
  - `AIS_DENSITY_PATH = AIS_DENSITY_PATH_DEMO`（向后兼容别名）
- **更新函数**：`load_ais_density_for_grid(grid, prefer_real=True)`
  - 优先加载真实分辨率 NC（若 `prefer_real=True`）
  - 回退到 demo NC
  - 都不存在时返回 None，不抛异常
  - 更新警告文案：提示运行 `python -m scripts.preprocess_ais_to_density`
- **更新函数**：`has_ais_density_data(grid, prefer_real=True)`
  - 检查是否存在可用的 AIS 密度 NC 文件
  - 不抛异常，返回 bool
- **更新函数**：`_add_ais_cost_component()`
  - 更新文档和警告文案

### Task B: 统一 AIS 密度 NC 的路径（成本层）

#### `arcticroute/core/cost.py`
- 所有 AIS 密度加载都通过 `load_ais_density_for_grid()` 进行
- 支持 demo 和 real 两个分辨率的 NC 文件
- 自动选择可用的文件，无需手动干预

### Task C: UI 侧只根据"密度 NC 是否存在"给提示

#### `arcticroute/ui/planner_minimal.py`
- **更新 AIS 数据检查逻辑**（第 670-700 行）：
  - 改为检查密度 NC 文件是否存在（使用 `cost_core.has_ais_density_data()`）
  - 不再检查 CSV 文件
  - 提示文案改为："已检测到 AIS 拥挤度密度数据（目录/密度 NC）"
  
- **更新 AIS 密度加载逻辑**（第 840-865 行）：
  - 改为从密度 NC 文件加载（使用 `cost_core.load_ais_density_for_grid()`）
  - 不再从 CSV 文件构建
  - 提示文案改为："当前未找到 AIS 拥挤度密度数据，可先运行 `python -m scripts.preprocess_ais_to_density` 生成 NC 文件"

### 其他脚本更新

#### `scripts/debug_ais_effect.py`
- 改为从 `AIS_RAW_DIR` 目录加载原始 AIS 数据
- 使用 `build_ais_density_da_for_demo_grid()` 构建密度

#### `scripts/evaluate_routes_vs_ais.py`
- 改为从 `AIS_RAW_DIR` 目录加载原始 AIS 数据
- 更新 `_load_ais_density()` 函数

## 数据结构验证

### 原始 AIS 数据（`data_real/ais/raw/`）
✅ 存在 5 个 JSON 文件：
- `AIS of 2023.12.29-2024.03.28.json` (1.76 GB)
- `AIS of 2024.03.28-2024.06.26.json` (2.01 GB)
- `AIS of 2024.06.24-2024.09.22.json` (1.77 GB)
- `AIS of 2024.09.22-2024.12.21.json` (1.76 GB)
- `AIS of 2024.12.21-2025.01.01.json` (169 MB)

### AIS 密度 NC 文件（`data_real/ais/derived/`）
✅ 存在：
- `ais_density_2024_demo.nc` (33.8 KB)

## 关键改进

1. **灵活性**：不再依赖特定的 CSV 文件，支持多种格式和多个文件
2. **可维护性**：路径常量集中管理，易于修改
3. **用户友好**：提示文案更清晰，指导用户如何生成缺失的数据
4. **向后兼容**：保留了 `AIS_DENSITY_PATH` 别名，不破坏现有代码

## 测试建议

1. 验证 `has_raw_ais_files()` 能正确检测原始 AIS 文件
2. 验证 `load_ais_from_raw_dir()` 能正确加载 JSON 文件
3. 验证 `load_ais_density_for_grid()` 能正确加载 demo NC 文件
4. 验证 UI 中的 AIS 提示文案正确显示
5. 运行 `python -m scripts.debug_ais_effect` 验证 AIS 效果

## 文件清单

### 修改的核心文件
- `arcticroute/core/ais_ingest.py` - 新增路径常量和辅助函数
- `arcticroute/core/cost.py` - 统一 AIS 密度路径常量和加载逻辑
- `arcticroute/ui/planner_minimal.py` - 更新 UI 中的 AIS 检查和加载逻辑

### 修改的脚本文件
- `scripts/debug_ais_effect.py` - 更新 AIS 数据加载方式
- `scripts/evaluate_routes_vs_ais.py` - 更新 AIS 数据加载方式

### 未修改的文件（仅作为测试数据）
- `tests/test_ais_phase1_integration.py` - 保留原有的 CSV 测试数据
- `verify_ais_phase1.py` - 验证脚本，保留原有的检查逻辑









