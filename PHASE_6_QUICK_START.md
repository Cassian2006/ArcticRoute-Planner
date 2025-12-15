# Phase 6 快速开始指南

## 文件变更总结

### 新增文件
- `arcticroute/core/config_paths.py` - 数据路径配置模块
- `tests/test_real_grid_loader.py` - 12 个新增单元测试

### 修改文件
- `arcticroute/core/grid.py` - 新增 `load_real_grid_from_nc()` 函数
- `arcticroute/core/landmask.py` - 新增 `load_real_landmask_from_nc()` 函数
- `scripts/check_grid_and_landmask.py` - 增加真实网格加载逻辑
- `arcticroute/ui/planner_minimal.py` - 增加网格模式选择开关

## 快速验证

### 1. 运行所有测试
```bash
python -m pytest tests/ -v
# 预期：47 passed
```

### 2. 运行 CLI 脚本
```bash
python -m scripts.check_grid_and_landmask
# 预期：source: demo (如果没有真实数据)
```

### 3. 启动 UI
```bash
streamlit run run_ui.py
# 在左侧栏选择"网格模式"
```

## 核心 API

### 加载真实网格
```python
from arcticroute.core.grid import load_real_grid_from_nc

grid = load_real_grid_from_nc()  # 返回 Grid2D 或 None
if grid is not None:
    print(f"Grid shape: {grid.shape()}")
```

### 加载真实 Landmask
```python
from arcticroute.core.landmask import load_real_landmask_from_nc

landmask = load_real_landmask_from_nc(grid)  # 返回 np.ndarray 或 None
if landmask is not None:
    print(f"Landmask shape: {landmask.shape}")
```

### 获取数据路径
```python
from arcticroute.core.config_paths import get_data_root, get_newenv_path

data_root = get_data_root()  # 数据根目录
newenv = get_newenv_path()   # 处理后的环境数据目录
```

## 环境变量

### ARCTICROUTE_DATA_ROOT
指定数据根目录位置（可选）：
```bash
export ARCTICROUTE_DATA_ROOT=/custom/path/to/data
```

默认值：`{项目根目录的兄弟目录}/ArcticRoute_data_backup`

## 数据文件结构

当有真实数据时，应按以下结构放置：
```
ArcticRoute_data_backup/
└── data_processed/
    └── newenv/
        ├── env_clean.nc
        ├── grid_spec.nc
        └── land_mask_gebco.nc
```

系统会自动尝试这些文件名。

## 常见场景

### 场景 1：没有真实数据（当前状态）
- CLI 脚本自动回退到 demo
- UI 中选择"演示网格 (demo)"或"真实网格（若可用）"都能工作
- 所有功能正常，使用 demo 数据

### 场景 2：添加真实数据后
1. 将真实 NetCDF 文件放到 `ArcticRoute_data_backup/data_processed/newenv/`
2. 运行 `python -m scripts.check_grid_and_landmask` 验证
3. 在 UI 中选择"真实网格（若可用）"使用真实数据

### 场景 3：自定义数据路径
```bash
export ARCTICROUTE_DATA_ROOT=/my/custom/data/path
python -m scripts.check_grid_and_landmask
```

## 日志输出说明

### [GRID] 前缀
- `[GRID] xarray not available` - 缺少 xarray 库
- `[GRID] real grid file not found` - 找不到网格文件
- `[GRID] successfully loaded real grid` - 成功加载真实网格

### [LANDMASK] 前缀
- `[LANDMASK] real landmask file not found` - 找不到 landmask 文件
- `[LANDMASK] successfully loaded landmask` - 成功加载 landmask
- `[LANDMASK] attempting nearest-neighbor resampling` - 进行形状调整

### [CHECK] 前缀
- `[CHECK] source: demo/real/real_grid_demo_landmask` - 当前使用的数据源

## 测试覆盖

新增 12 个测试：
- 4 个网格加载测试（1D/2D 坐标、缺失文件、缺失变量）
- 4 个 landmask 加载测试（基本加载、缺失文件、缺失变量、形状不匹配）
- 1 个 CLI 脚本测试
- 3 个路径配置测试

所有测试都不依赖真实数据，使用临时 NetCDF 文件。

## 向后兼容性

✅ 所有现有功能保持不变
✅ 现有 35 个测试全部通过
✅ 新增 12 个测试全部通过
✅ 总计 47 个测试通过

## 下一步

当真实数据可用时，系统已准备好无缝集成。无需修改任何代码，只需：
1. 放置真实数据文件
2. 运行脚本/UI
3. 系统自动检测并使用真实数据

如有问题，检查日志输出中的 `[GRID]`、`[LANDMASK]`、`[CHECK]` 前缀的消息。

















