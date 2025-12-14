# AIS 集成总结

本文档概述了 AIS（自动识别系统）数据在北冰洋路由系统中的集成方式，包括数据来源、栅格化处理、成本融合等关键环节。

---

## 1. AIS 数据的来源与格式

### 数据来源
- **路径**: `data_real/ais/raw/*.json`
- **规模**: 约 25 万条记录（取决于具体数据集）
- **格式**: 优先支持 JSON / JSONL 格式，也支持 CSV

### 字段标准化

原始 AIS 数据可能来自不同来源，列名各异。系统通过 **列别名映射** 将其标准化为统一的 7 个字段：

| 标准字段 | 含义 | 可能的别名 |
|---------|------|----------|
| `mmsi` | 船舶 MMSI 号 | MMSI |
| `timestamp` | 时间戳（UTC） | time, datetime, basedatetime, BaseDateTime, utc, ts, postime |
| `lat` | 纬度 | latitude, Lat, LAT, Latitude |
| `lon` | 经度 | longitude, long, lng, Lon, LON, Longitude |
| `sog` | 船速（节） | speed, speed_knots, speedoverground |
| `cog` | 船向（度） | course, heading, hdg |
| `nav_status` | 导航状态 | navstatus, status |

**必需字段**: `mmsi`, `timestamp`, `lat`, `lon`

**可选字段**: `sog`, `cog`, `nav_status`

### 数据清洗规则

在 `arcticroute/core/ais_ingest.py` 中的 `_clean_ais_dataframe()` 函数执行以下清洗步骤：

1. **列映射**: 根据别名和 schema hint 将原始列映射到标准列
2. **数值化**: 将 lat, lon, sog, cog 转换为浮点数
3. **地理范围检查**: 
   - 纬度范围 [-90, 90]
   - 经度范围 [-180, 180]
4. **时间范围检查**: 仅保留 2018-2030 年的数据
5. **缺失值处理**: 删除必需字段缺失的记录

---

## 2. 栅格化过程

### 脚本位置
- **主脚本**: `scripts/preprocess_ais_to_density.py`
- **核心函数**: `arcticroute/core/ais_ingest.py` 中的 `rasterize_ais_density_to_grid()`

### 两种网格模式

#### 2.1 Demo 网格模式
- **网格大小**: 40 × 80（演示用）
- **覆盖范围**: 北冰洋演示区域
- **输出文件**: `data_real/ais/derived/ais_density_2024_demo.nc`
- **命令**: `python scripts/preprocess_ais_to_density.py --grid-mode demo`

#### 2.2 Real 网格模式
- **网格大小**: ~500 × 5333（真实成本网格）
- **覆盖范围**: 完整北冰洋区域
- **输出文件**: `data_real/ais/derived/ais_density_2024_real.nc`
- **命令**: `python scripts/preprocess_ais_to_density.py --grid-mode real`

### 栅格化算法

对于每个 AIS 点 (lat, lon)：

1. 计算该点到所有网格点的距离平方：
   ```
   dist_sq[i,j] = (grid_lat[i,j] - lat)² + (grid_lon[i,j] - lon)²
   ```

2. 找到距离最小的网格点 (i, j)

3. 该网格点的密度计数 +1

4. 最后对整个密度场进行 **max-count 归一化**：
   ```
   density_normalized = density / max(density)
   ```
   结果范围为 [0, 1]，其中 1 表示最高船舶出现频率

### 输出格式

生成的 NetCDF 文件包含：
- **变量**: `ais_density` (float32)
- **维度**: (y, x) 对应网格行列
- **坐标**: 可选的 lat, lon 坐标数组
- **属性**: `source="real_ais"`, `norm="0-1"`

---

## 3. 成本集成逻辑

### 3.1 权重参数

在成本构建函数中，AIS 拥挤度通过权重参数 `w_ais` 控制：

#### Demo 成本 (`build_demo_cost`)
```python
def build_demo_cost(
    grid: Grid2D,
    land_mask: np.ndarray,
    ice_penalty: float = 4.0,
    ice_lat_threshold: float = 75.0,
    w_ais: float = 0.0,  # ← AIS 权重
    ais_density: Optional[np.ndarray | xr.DataArray] = None,
) -> CostField:
```

#### 真实环境成本 (`build_cost_from_real_env`)
```python
def build_cost_from_real_env(
    grid: Grid2D,
    land_mask: np.ndarray,
    env: RealEnvLayers,
    # ... 其他参数 ...
    w_ais: float | None = None,  # ← AIS 权重（优先级最高）
    ais_weight: float = 0.0,      # ← 备选 AIS 权重
    ais_density: np.ndarray | xr.DataArray | None = None,
) -> CostField:
```

**权重含义**:
- `w_ais = 0.0`: 不使用 AIS 拥挤度（默认）
- `w_ais = 0.5`: AIS 拥挤度贡献 50% 的成本增量
- `w_ais = 1.0`: AIS 拥挤度贡献 100% 的成本增量

### 3.2 加载与插值过程

函数 `_add_ais_cost_component()` 执行以下步骤：

1. **加载密度场**（如果未提供）:
   ```python
   if ais_density is None:
       ais_density = load_ais_density_for_grid(grid, prefer_real=True)
   ```
   - 优先加载 `ais_density_2024_real.nc`
   - 回退到 `ais_density_2024_demo.nc`

2. **重采样到目标网格**:
   ```python
   aligned = _regrid_ais_density_to_grid(ais_density, grid)
   ```
   - 若密度场与网格形状相同，直接使用
   - 否则使用 **最近邻插值** (nearest-neighbor interpolation)
   - 若密度场缺少坐标信息，尝试从 demo 网格推断坐标后再插值

3. **归一化**:
   ```python
   normalized = _normalize_ais_density_array(aligned)
   # 结果范围 [0, 1]
   ```

4. **加权累加到总成本**:
   ```python
   cost_increment = w_ais * normalized
   base_cost += cost_increment
   components["ais_density"] = cost_increment
   ```

### 3.3 成本分解中的 AIS 项

在返回的 `CostField.components` 字典中，AIS 拥挤度对应的键为：

```python
components["ais_density"]  # 形状与网格相同，值为 [0, w_ais]
```

**成本分解示例**（假设 w_ais=0.5）:
```
总成本 = base_distance + ice_risk + wave_risk + ais_density + ...
         1.0          + 2.0      + 0.3      + 0.5        + ...
```

其中 `ais_density` 分量的最大值为 `w_ais`，最小值为 0。

---

## 4. 当前局限与 TODO

### 4.1 现有局限

#### 仅基于出现频率
- **现状**: AIS 密度仅统计船舶在各网格点的出现次数
- **问题**: 未区分船型（油轮、集装箱船、渔船等）、航向、船速等特征
- **影响**: 无法精细化评估不同船型的拥挤风险

#### 未融合 EDL 特征
- **现状**: AIS 密度与 EDL（极端依赖学习）模块独立运行
- **问题**: 未将 AIS 密度作为 EDL 输入特征之一
- **影响**: EDL 风险评估缺少船舶活动信息

### 4.2 未来改进方向

#### 短期改进
1. **船型分类**: 在 AIS 数据中添加 ship_type 字段，分别统计不同船型的密度
2. **航向权重**: 根据船舶航向与路由方向的相关性调整密度权重
3. **时间维度**: 按月份或季节分别统计 AIS 密度，捕捉季节性变化

#### 中期改进
1. **EDL 特征融合**: 
   - 将 AIS 密度作为 EDL 模型的输入特征之一
   - 在特征立方体中添加 `ais_density_norm` 维度
   - 重新训练 EDL 模型以学习 AIS 与风险的关系

2. **主航道识别**: 
   - 基于 AIS 轨迹聚类识别主要航道
   - 为主航道区域设置更高的拥挤度权重

#### 长期改进
1. **多源融合**: 结合 AIS、卫星遥感、气象数据等多源信息
2. **动态更新**: 实现 AIS 数据的定期更新机制，保持密度场的时效性
3. **实时预报**: 基于历史 AIS 轨迹和当前环境条件预报未来拥挤度

---

## 5. 使用示例

### 5.1 生成 AIS 密度场

```bash
# 生成 demo 网格密度
python scripts/preprocess_ais_to_density.py --grid-mode demo

# 生成真实网格密度
python scripts/preprocess_ais_to_density.py --grid-mode real
```

### 5.2 在成本构建中使用 AIS

```python
from arcticroute.core.grid import make_demo_grid
from arcticroute.core.cost import build_demo_cost

# 创建网格和陆地掩膜
grid, land_mask = make_demo_grid()

# 构建包含 AIS 拥挤度的成本场（权重 0.5）
cost_field = build_demo_cost(
    grid=grid,
    land_mask=land_mask,
    ice_penalty=4.0,
    ice_lat_threshold=75.0,
    w_ais=0.5,  # 启用 AIS，权重 0.5
)

# 访问成本分解
print(cost_field.components["ais_density"])  # AIS 拥挤度分量
```

### 5.3 在路由规划中使用

```python
from arcticroute.core.planner import RoutePlanner

planner = RoutePlanner(
    cost_field=cost_field,
    start=(75.0, 100.0),
    end=(72.0, 150.0),
)

route = planner.plan()
```

---

## 6. 相关文件索引

| 文件 | 功能 |
|------|------|
| `arcticroute/core/ais_ingest.py` | AIS 数据读取、清洗、栅格化 |
| `scripts/preprocess_ais_to_density.py` | AIS 密度预处理脚本 |
| `arcticroute/core/cost.py` | 成本构建与 AIS 集成逻辑 |
| `data_real/ais/raw/` | 原始 AIS 数据目录 |
| `data_real/ais/derived/` | 生成的 AIS 密度 NetCDF 文件 |

---

## 7. 参考资源

- **AIS 标准**: ITU-R M.1371-5（自动识别系统规范）
- **NetCDF 格式**: https://www.unidata.ucar.edu/software/netcdf/
- **xarray 库**: http://xarray.pydata.org/

---

**文档更新**: 2025-12-10  
**版本**: 1.0




