# AIS Grid Signature 快速参考

## 核心概念

### Grid Signature（网格签名）
```
格式：{ny}x{nx}_{lat_min:.4f}_{lat_max:.4f}_{lon_min:.4f}_{lon_max:.4f}
示例：101x1440_60.0000_85.0000_-180.0000_179.7500
用途：唯一标识一个网格，用于 AIS 密度文件匹配
```

### 文件优先级
```
1. 精确匹配 (exact)
   - attrs["grid_signature"] == 当前 grid_signature
   - 最优选择

2. Demo 文件 (demo)
   - 文件名包含 "_demo"
   - 通用备选方案

3. 通用文件 (generic)
   - 其他文件
   - 最后备选
```

---

## API 参考

### 1. 计算网格签名
```python
from arcticroute.core.cost import compute_grid_signature
from arcticroute.core.grid import make_demo_grid

grid, _ = make_demo_grid()
sig = compute_grid_signature(grid)
print(sig)  # "101x1440_60.0000_85.0000_-180.0000_179.7500"
```

### 2. 发现 AIS 密度文件
```python
from arcticroute.core.cost import discover_ais_density_candidates

# 不指定 grid_signature，返回所有文件
candidates = discover_ais_density_candidates()

# 指定 grid_signature，按优先级排序
sig = "101x1440_60.0000_85.0000_-180.0000_179.7500"
candidates = discover_ais_density_candidates(grid_signature=sig)

# 候选文件结构
for cand in candidates:
    print(cand["path"])        # 文件路径
    print(cand["label"])       # 显示标签
    print(cand["match_type"])  # "exact" | "demo" | "generic"
    print(cand["grid_signature"])  # 文件的 grid_signature（可能为 None）
```

### 3. 加载 AIS 密度（自动重采样）
```python
from arcticroute.core.cost import load_ais_density_for_grid
from arcticroute.core.grid import load_real_grid_from_nc

# 加载真实网格
grid = load_real_grid_from_nc(ym="202401")

# 加载 AIS 密度（自动按 grid_signature 匹配）
ais_density = load_ais_density_for_grid(grid=grid, prefer_real=True)

# 如果维度不匹配，自动重采样并保存到：
# data_real/ais/density/derived/ais_density_2024_{grid_signature}.nc
```

### 4. 手动重采样
```python
from arcticroute.core.cost import _regrid_ais_density_to_grid
import xarray as xr

# 加载源 AIS 密度
ds = xr.open_dataset("path/to/ais_density.nc")
ais_da = ds["ais_density"]

# 重采样到目标网格
resampled = _regrid_ais_density_to_grid(ais_da, target_grid)
```

---

## Streamlit UI 使用

### 1. 自动 Grid Signature 计算
```
用户操作：选择 grid_mode (demo/real)
↓
自动计算 current_grid_signature
↓
检测是否发生变化
↓
如果变化，清空 AIS 缓存
↓
重新扫描 AIS 文件
```

### 2. 按优先级选择 AIS 文件
```
UI 显示：
- 自动选择 (推荐)
- ais_density_2024_101x1440_....nc ✓ (精确匹配)
- ais_density_2024_demo.nc (演示)
- ais_density_2024_real.nc
```

### 3. AIS 密度状态
```
✅ AIS density: ais_density_2024_demo.nc [演示文件]

或

⚠ 未找到匹配当前网格的 AIS density，已自动尝试重采样或请运行脚本生成
```

### 4. 重新扫描按钮
```
点击 "🔄 重新扫描 AIS" 按钮
↓
清空缓存
↓
重新扫描文件
↓
UI 刷新
```

---

## 常见场景

### 场景 1：切换网格模式
```
用户：从 demo 切换到 real
↓
系统：
  1. 计算新的 grid_signature
  2. 检测到变化
  3. 清空 AIS 缓存
  4. 重新扫描 AIS 文件
  5. 优先选择匹配 real 网格的文件
```

### 场景 2：维度不匹配
```
用户：选择 AIS 文件，但维度与网格不匹配
↓
系统：
  1. 检测到不匹配
  2. 自动重采样
  3. 保存到 data_real/ais/density/derived/
  4. 后续访问使用缓存
  5. 打印："检测到维度不匹配→已自动重采样→已缓存→AIS 成本已启用"
```

### 场景 3：找不到 AIS 文件
```
用户：选择 AIS 权重 > 0，但没有 AIS 文件
↓
系统：
  1. 显示警告信息
  2. 提示运行脚本生成
  3. 提供"重新扫描"按钮
  4. 用户可手动上传或生成文件
```

---

## 文件位置

### 源文件
```
data_real/ais/density/          # 原始 AIS 密度文件
data_real/ais/derived/          # 衍生 AIS 密度文件
```

### 缓存文件
```
data_real/ais/density/derived/ais_density_2024_{grid_signature}.nc
```

### 示例
```
data_real/ais/density/derived/ais_density_2024_101x1440_60.0000_85.0000_-180.0000_179.7500.nc
```

---

## 调试技巧

### 1. 查看当前网格签名
```python
import streamlit as st
st.write(st.session_state.get("grid_signature", "N/A"))
```

### 2. 查看 AIS 文件列表
```python
from arcticroute.core.cost import discover_ais_density_candidates
candidates = discover_ais_density_candidates()
for c in candidates:
    print(f"{c['label']} - {c['match_type']}")
```

### 3. 查看重采样日志
```
[AIS] resampled density using xarray.interp: (40, 80) -> (101, 1440)
[AIS] saved resampled density to data_real/ais/density/derived/ais_density_2024_101x1440_....nc
[AIS] 检测到维度不匹配→已自动重采样→已缓存→AIS 成本已启用
```

### 4. 清空缓存
```python
import streamlit as st
st.session_state["ais_density_path_selected"] = None
st.session_state["ais_density_cache_key"] = None
st.rerun()
```

---

## 性能指标

| 操作 | 时间 |
|------|------|
| Grid Signature 计算 | < 1ms |
| AIS 文件扫描 | < 100ms |
| 最近邻重采样 (40×80 → 101×1440) | 1-2s |
| 加载缓存文件 | < 100ms |

---

## 故障排除

### 问题 1：找不到 AIS 文件
```
症状：显示 "⚠ 未找到匹配当前网格的 AIS density"
解决：
  1. 检查 data_real/ais/density/ 目录是否存在
  2. 检查文件是否为有效的 NetCDF 格式
  3. 点击"重新扫描 AIS"按钮
  4. 运行 python -m scripts.preprocess_ais_to_density 生成文件
```

### 问题 2：维度不匹配
```
症状：日志显示 "AIS=(40,80) vs GRID=(101,1440)"
解决：
  系统会自动重采样，无需手动操作
  检查 data_real/ais/density/derived/ 中是否有缓存文件
```

### 问题 3：AIS 成本未启用
```
症状：规划结果中没有 AIS 成本分量
解决：
  1. 检查 w_ais 权重是否 > 0
  2. 检查是否选择了 AIS 文件
  3. 查看日志中的警告信息
  4. 点击"重新扫描 AIS"按钮
```

---

## 相关命令

### 生成 AIS 密度文件
```bash
python -m scripts.preprocess_ais_to_density
```

### 清理缓存
```bash
rm -rf data_real/ais/density/derived/ais_density_2024_*.nc
```

### 查看文件信息
```bash
python -c "
import xarray as xr
ds = xr.open_dataset('path/to/ais_density.nc')
print(ds)
print(f'Grid Signature: {ds.attrs.get(\"grid_signature\", \"N/A\")}')
"
```

---

## 相关文件

- `arcticroute/core/cost.py` - 核心实现
- `arcticroute/ui/planner_minimal.py` - UI 集成
- `AIS_GRID_SIGNATURE_IMPLEMENTATION_SUMMARY.md` - 详细文档








