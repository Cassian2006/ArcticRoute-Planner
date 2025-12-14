# Phase 6 完成报告：真实网格/Landmask 适配层

## 概述

Phase 6 成功实现了 ArcticRoute 项目的真实网格和 landmask 加载适配层，同时保证了完全的向后兼容性。在没有真实数据时，所有功能继续使用 demo 模式；当真实数据可用时，系统可以无缝切换到真实网格模式。

## 实现内容

### 1. 新增文件

#### `arcticroute/core/config_paths.py`
- **功能**：统一的数据路径配置模块
- **主要函数**：
  - `get_data_root()`: 返回数据根目录，支持环境变量 `ARCTICROUTE_DATA_ROOT` 覆盖
  - `get_newenv_path()`: 返回处理后的环境数据子目录路径
- **特点**：
  - 纯标准库实现，无第三方依赖
  - 只提供路径查询，不进行 I/O 操作
  - 支持灵活的路径配置

#### `tests/test_real_grid_loader.py`
- **覆盖范围**：12 个新增单元测试
- **测试类**：
  - `TestLoadRealGridFromNC`: 4 个测试，验证 1D/2D 坐标加载、缺失文件处理
  - `TestLoadRealLandmaskFromNC`: 4 个测试，验证 landmask 加载、形状不匹配重采样
  - `TestCheckGridAndLandmaskCLI`: 1 个测试，验证 CLI 脚本行为
  - `TestConfigPaths`: 3 个测试，验证路径配置模块

### 2. 修改的文件

#### `arcticroute/core/grid.py`
- **新增函数**：`load_real_grid_from_nc()`
  - 从 NetCDF 文件加载真实网格坐标
  - 支持 1D 和 2D 坐标格式
  - 自动尝试多个可能的文件名（env_clean.nc, grid_spec.nc, land_mask_gebco.nc）
  - 加载失败时返回 None，不抛异常
  - 包含详细的调试日志输出

#### `arcticroute/core/landmask.py`
- **新增函数**：`load_real_landmask_from_nc()`
  - 从 NetCDF 文件加载与网格对齐的 landmask
  - 支持形状不匹配时的最近邻重采样
  - 返回 bool 数组（True = 陆地）
  - 加载失败时返回 None，不抛异常
  - 包含详细的调试日志输出

#### `scripts/check_grid_and_landmask.py`
- **改进**：
  - 新增真实网格加载尝试逻辑
  - 支持三种 source 标签：
    - `"demo"`: 完全使用 demo 网格和 landmask
    - `"real"`: 使用真实网格和真实 landmask
    - `"real_grid_demo_landmask"`: 混合模式（真实网格 + demo landmask）
  - 自动 fallback 到 demo，不会崩溃

#### `arcticroute/ui/planner_minimal.py`
- **新增功能**：
  - 左侧栏新增"网格配置"部分
  - 网格模式选择框：
    - "演示网格 (demo)": 强制使用 demo 网格
    - "真实网格（若可用）": 尝试加载真实网格，失败时自动回退 demo
  - 规划结果下方显示 "Grid source: {source}" 标签
  - 加载失败时显示友好的 warning 提示
- **保持兼容**：
  - 所有现有功能（3 条路线、ECO、landmask 检查、成本分解）保持不变
  - 现有测试全部通过

## 关键特性

### 1. 完全的向后兼容性
- 没有真实数据时，所有功能继续使用 demo 模式
- 现有的 47 个测试全部通过，无任何破坏

### 2. 优雅的 Fallback 机制
- 每一层都有 fallback：
  - 真实网格加载失败 → 使用 demo 网格
  - 真实 landmask 加载失败 → 使用 demo landmask
  - 文件不存在 → 返回 None，不抛异常

### 3. 灵活的配置
- 支持环境变量 `ARCTICROUTE_DATA_ROOT` 覆盖数据路径
- 自动尝试多个可能的文件名
- 支持 1D 和 2D 坐标格式

### 4. 清晰的调试信息
- 所有加载步骤都有日志输出（[GRID], [LANDMASK], [CHECK] 前缀）
- 便于用户理解系统当前使用的是 demo 还是真实数据

## 测试结果

```
======================== 47 passed, 1 warning in 2.61s ========================
```

### 测试覆盖
- ✅ 4 个 A* 寻路测试
- ✅ 9 个成本分解测试
- ✅ 10 个 ECO 模型测试
- ✅ 3 个网格和 landmask 测试
- ✅ 12 个新增真实网格加载测试
- ✅ 3 个路线 landmask 一致性测试
- ✅ 6 个烟雾测试（导入检查）

## 使用指南

### 1. CLI 脚本验证
```bash
python -m scripts.check_grid_and_landmask
```
输出示例（无真实数据时）：
```
[CHECK] Attempting to load real grid and landmask...
[GRID] real grid file not found in ..., candidates: [...]
[CHECK] Real grid not available, falling back to demo grid and landmask.
[CHECK] source: demo
[CHECK] shape: 40 x 80
[CHECK] frac_land: 0.125
[CHECK] frac_ocean: 0.875
[CHECK] corner lat/lon: (65.000,0.000) -> (80.000,160.000)
```

### 2. UI 使用
```bash
streamlit run run_ui.py
```
- 在左侧栏"网格配置"中选择网格模式
- 选择"演示网格 (demo)"：使用 demo 网格
- 选择"真实网格（若可用）"：尝试加载真实网格，失败自动回退

### 3. 环境变量配置
```bash
# 设置数据根目录
export ARCTICROUTE_DATA_ROOT=/path/to/data

# 然后运行脚本或 UI
python -m scripts.check_grid_and_landmask
streamlit run run_ui.py
```

### 4. 真实数据放置
当有真实数据时，按以下结构放置：
```
ArcticRoute_data_backup/
└── data_processed/
    └── newenv/
        ├── env_clean.nc (或)
        ├── grid_spec.nc (或)
        └── land_mask_gebco.nc
```

## 代码质量

- ✅ 所有代码通过 linting 检查
- ✅ 完整的类型注解（Python 3.9+ 兼容）
- ✅ 详细的 docstring 和注释
- ✅ 异常处理完善，不会因数据缺失而崩溃

## 后续步骤

当真实数据可用时：
1. 将数据放置在 `ArcticRoute_data_backup/data_processed/newenv/` 目录
2. 运行 `python -m scripts.check_grid_and_landmask` 验证加载
3. 在 UI 中选择"真实网格（若可用）"即可使用真实数据

如需自定义数据路径，可通过环境变量 `ARCTICROUTE_DATA_ROOT` 指定。

## 总结

Phase 6 成功实现了一个完整的、可扩展的真实网格加载适配层，同时保证了系统的稳定性和向后兼容性。系统现在可以：
- ✅ 无缝支持 demo 和真实数据的切换
- ✅ 优雅地处理数据缺失情况
- ✅ 提供清晰的用户反馈
- ✅ 为后续的真实数据集成做好准备











