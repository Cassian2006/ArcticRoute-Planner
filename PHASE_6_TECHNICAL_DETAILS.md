# Phase 6 技术细节文档

## 架构设计

### 分层加载策略

```
UI/CLI 层
    ↓
加载函数层 (load_real_grid_from_nc, load_real_landmask_from_nc)
    ↓
配置层 (config_paths.py)
    ↓
文件系统 (NetCDF 文件)
```

每一层都有独立的 fallback 机制，确保系统稳定性。

### 数据流

```
用户请求 (UI/CLI)
    ↓
尝试加载真实网格
    ├─ 成功 → 继续
    └─ 失败 → 返回 None
    ↓
尝试加载真实 landmask
    ├─ 成功 → 返回 (grid, landmask, "real")
    ├─ 失败 → 返回 (grid, demo_landmask, "real_grid_demo_landmask")
    └─ 网格加载失败 → 返回 (demo_grid, demo_landmask, "demo")
```

## 核心函数详解

### `load_real_grid_from_nc()`

**签名**：
```python
def load_real_grid_from_nc(
    nc_path: Optional[Path] = None,
    lat_name: str = "lat",
    lon_name: str = "lon",
) -> Optional[Grid2D]:
```

**工作流程**：
1. 检查 xarray 是否可用
2. 确定文件路径（优先级：传入 → 自动搜索）
3. 打开 NetCDF 文件（decode_times=False）
4. 获取 lat/lon 变量
5. 处理 1D 或 2D 坐标
6. 创建 Grid2D 对象
7. 返回结果或 None

**坐标处理**：
- 1D 情况：使用 `np.meshgrid()` 生成 2D 网格
- 2D 情况：直接使用 `np.broadcast_arrays()` 对齐

**错误处理**：
- 文件不存在 → 打印日志，返回 None
- 变量缺失 → 打印日志，返回 None
- 坐标维度不支持 → 打印日志，返回 None
- 其他异常 → 捕获并打印，返回 None

### `load_real_landmask_from_nc()`

**签名**：
```python
def load_real_landmask_from_nc(
    grid: Grid2D,
    nc_path: Optional[Path] = None,
    var_name: str = "land_mask",
) -> Optional[np.ndarray]:
```

**工作流程**：
1. 检查 xarray 是否可用
2. 确定文件路径
3. 打开 NetCDF 文件
4. 获取 landmask 变量
5. 转换为 bool 数组
6. 检查形状是否匹配
7. 如果不匹配，进行最近邻重采样
8. 返回结果或 None

**形状匹配策略**：
- 精确匹配 → 直接返回
- 不匹配 → 进行最近邻重采样
  ```python
  y_indices = np.round(np.linspace(0, old_ny - 1, ny)).astype(int)
  x_indices = np.round(np.linspace(0, old_nx - 1, nx)).astype(int)
  resampled = land_mask[np.ix_(y_indices, x_indices)]
  ```

**错误处理**：
- 文件不存在 → 打印日志，返回 None
- 变量缺失 → 打印日志，返回 None
- 其他异常 → 捕获并打印，返回 None

## 配置模块设计

### `config_paths.py`

**设计原则**：
- 纯标准库，无外部依赖
- 只提供路径查询，不进行 I/O
- 支持环境变量覆盖
- 返回绝对路径

**路径解析**：
```python
# 获取项目根目录
here = Path(__file__).resolve()  # arcticroute/core/config_paths.py
root = here.parents[2]           # 项目根目录

# 获取数据根目录
data_root = root.parent / "ArcticRoute_data_backup"
```

**环境变量优先级**：
1. `ARCTICROUTE_DATA_ROOT` 环境变量（最高优先级）
2. 默认路径：`{项目根}/ArcticRoute_data_backup`

## UI 集成

### 网格模式选择

**UI 组件**：
```python
grid_mode = st.selectbox(
    "网格模式",
    options=["demo", "real_if_available"],
    format_func=lambda s: "演示网格 (demo)" if s == "demo" else "真实网格（若可用）",
)
```

**逻辑流程**：
```python
if grid_mode == "real_if_available":
    real_grid = load_real_grid_from_nc()
    if real_grid is not None:
        grid = real_grid
        land_mask = load_real_landmask_from_nc(grid)
        if land_mask is not None:
            grid_source_label = "real"
        else:
            st.warning("真实 landmask 不可用，使用演示 landmask。")
            _, land_mask = make_demo_grid(...)
            grid_source_label = "real_grid_demo_landmask"
    else:
        st.warning("真实网格不可用，使用演示网格。")
        grid, land_mask = make_demo_grid()
        grid_source_label = "demo"
else:
    grid, land_mask = make_demo_grid()
    grid_source_label = "demo"
```

**用户反馈**：
- 加载失败时显示 `st.warning()` 提示
- 结果摘要下方显示 `st.caption(f"Grid source: {grid_source_label}")`

## CLI 脚本改进

### 检查流程

```python
print("[CHECK] Attempting to load real grid and landmask...")

# 尝试加载真实网格
real_grid = load_real_grid_from_nc()

if real_grid is not None:
    # 尝试加载真实 landmask
    real_landmask = load_real_landmask_from_nc(real_grid)
    if real_landmask is not None:
        source = "real"
    else:
        # 使用 demo landmask
        _, real_landmask = make_demo_grid(...)
        source = "real_grid_demo_landmask"
else:
    # 完全回退到 demo
    info = load_landmask(prefer_real=False)
    real_grid = info.grid
    real_landmask = info.land_mask
    source = "demo"
```

### 输出格式

```
[CHECK] source: demo/real/real_grid_demo_landmask
[CHECK] shape: 40 x 80
[CHECK] frac_land: 0.125
[CHECK] frac_ocean: 0.875
[CHECK] corner lat/lon: (65.000,0.000) -> (80.000,160.000)
```

## 测试策略

### 单元测试设计

**不依赖真实数据**：
- 使用 `pytest` 的 `tmp_path` fixture 创建临时 NetCDF 文件
- 每个测试都是独立的，不会相互影响
- 测试文件大小小（10x20 网格），执行快速

**测试覆盖**：
```
TestLoadRealGridFromNC
├── test_load_real_grid_from_nc_1d_coords       # 1D 坐标
├── test_load_real_grid_from_nc_2d_coords       # 2D 坐标
├── test_load_real_grid_missing_file_returns_none
└── test_load_real_grid_missing_lat_lon_returns_none

TestLoadRealLandmaskFromNC
├── test_load_real_landmask_from_nc_basic
├── test_load_real_landmask_missing_file_returns_none
├── test_load_real_landmask_missing_var_returns_none
└── test_load_real_landmask_shape_mismatch_resamples

TestCheckGridAndLandmaskCLI
└── test_check_grid_and_landmask_cli_demo_fallback

TestConfigPaths
├── test_get_data_root_returns_path
├── test_get_newenv_path_returns_path
└── test_get_newenv_path_is_subdir_of_data_root
```

## 性能考虑

### 加载性能
- NetCDF 文件打开：~10-100ms（取决于文件大小）
- 坐标处理：O(ny × nx)，通常 <100ms
- Landmask 加载：O(ny × nx)，通常 <100ms

### 内存使用
- Grid2D：2 × ny × nx × 8 字节（float64）
- Landmask：ny × nx × 1 字节（bool）
- 例如 40×80 网格：~50KB

### 优化建议
- 对于大型网格（>10000×10000），考虑分块加载
- 可以缓存已加载的网格，避免重复加载

## 扩展点

### 1. 支持更多文件格式
当前支持 NetCDF，可扩展为：
- HDF5
- GeoTIFF
- 二进制格式

### 2. 高级重采样
当前使用最近邻，可扩展为：
- 双线性插值
- 样条插值
- 面积加权平均

### 3. 数据验证
可添加：
- 坐标范围检查
- Landmask 值范围检查
- 数据完整性检查

### 4. 缓存机制
可添加：
- 内存缓存（避免重复加载）
- 磁盘缓存（预处理数据）

## 兼容性

### Python 版本
- 最低：Python 3.8
- 测试：Python 3.11

### 依赖库
- xarray >= 2023.1.0（可选，缺失时自动 fallback）
- numpy >= 1.24.0（必需）
- netCDF4 >= 1.6.0（可选，xarray 的后端）

### 操作系统
- Windows ✅
- Linux ✅
- macOS ✅

## 日志系统

### 日志前缀
- `[GRID]`: 网格加载相关
- `[LANDMASK]`: Landmask 加载相关
- `[CHECK]`: CLI 脚本相关

### 日志级别
- 成功：`[MODULE] successfully loaded ...`
- 失败：`[MODULE] ... not found at ...`
- 错误：`[MODULE] error processing ...`
- 警告：`[MODULE] ... attempting ...`

### 调试技巧
1. 检查 `[GRID]` 日志确认网格加载状态
2. 检查 `[LANDMASK]` 日志确认 landmask 加载状态
3. 检查 `[CHECK]` 日志确认最终使用的数据源
4. 使用 `ARCTICROUTE_DATA_ROOT` 环境变量测试自定义路径

## 安全性考虑

### 文件操作
- 使用 `Path.exists()` 检查文件存在性
- 使用 `try-except` 捕获所有异常
- 不使用 `os.system()` 或 `eval()`

### 数据验证
- 检查坐标维度
- 检查数组形状
- 检查数据类型

### 错误处理
- 所有错误都返回 None，不抛异常
- 提供详细的日志信息用于调试
- 不会因数据问题导致程序崩溃













