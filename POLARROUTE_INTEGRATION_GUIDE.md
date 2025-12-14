# PolarRoute 集成指南

## 概述

本指南说明如何将 PolarRoute 路由优化引擎集成到 ArcticRoute 中。

### PolarRoute 工作流程

```
create_mesh → add_vehicle → optimise_routes
```

## 文件结构

```
data_sample/polarroute/
├── vessel_mesh_empty.json      # 空 mesh 示例（关键）
├── config_empty.json           # PolarRoute 配置
└── waypoints_example.json      # 示例 waypoints

scripts/
└── integrate_polarroute.py     # 集成脚本
```

## 快速开始

### 1. 使用 Empty Mesh 运行演示

```bash
cd /path/to/AR_final

# 运行集成演示
python scripts/integrate_polarroute.py --demo --verbose

# 或指定自定义配置
python scripts/integrate_polarroute.py \
  --config data_sample/polarroute/config_empty.json \
  --mesh data_sample/polarroute/vessel_mesh_empty.json \
  --demo
```

### 2. 输出文件

演示运行后会生成：

- `data_sample/polarroute/vessel_mesh_empty.json` - 更新的 mesh（包含规划的路由）
- `data_sample/polarroute/routes_demo.geojson` - GeoJSON 格式的路由

### 3. 验证输出

```bash
# 查看生成的 GeoJSON
cat data_sample/polarroute/routes_demo.geojson | python -m json.tool

# 或在 QGIS/Leaflet 中打开 GeoJSON 文件
```

## vessel_mesh.json 结构说明

### 元数据部分

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Empty Mesh Example for PolarRoute",
    "created": "2025-12-14",
    "source": "ArcticRoute Integration",
    "crs": "EPSG:4326",
    "bounds": {
      "north": 85.0,
      "south": 60.0,
      "east": 180.0,
      "west": -180.0
    }
  }
}
```

### 网格定义

```json
{
  "grid": {
    "type": "regular_latlon",
    "resolution_degrees": 1.0,
    "dimensions": {
      "latitude": 26,
      "longitude": 361
    },
    "origin": {
      "lat": 60.0,
      "lon": -180.0
    }
  }
}
```

### 环境层

```json
{
  "environmental_layers": {
    "ice_concentration": {
      "name": "Sea Ice Concentration",
      "unit": "fraction (0-1)",
      "description": "Sea ice concentration from satellite data",
      "data": []
    },
    "ice_thickness": {
      "name": "Sea Ice Thickness",
      "unit": "meters",
      "description": "Sea ice thickness estimate",
      "data": []
    },
    "wind_speed": {
      "name": "Wind Speed",
      "unit": "m/s",
      "description": "10-meter wind speed",
      "data": []
    },
    "wave_height": {
      "name": "Significant Wave Height",
      "unit": "meters",
      "description": "Significant wave height",
      "data": []
    },
    "current_speed": {
      "name": "Ocean Current Speed",
      "unit": "m/s",
      "description": "Ocean current magnitude",
      "data": []
    }
  }
}
```

### 船舶配置

```json
{
  "vehicles": [
    {
      "id": "vessel_001",
      "type": "handysize",
      "ice_class": "PC7",
      "max_ice_thickness_m": 1.2,
      "design_speed_kn": 14.0,
      "max_draft_m": 10.0,
      "beam_m": 32.0,
      "length_m": 190.0
    }
  ]
}
```

### 路由信息

```json
{
  "routes": [
    {
      "id": "route_001",
      "vessel_id": "vessel_001",
      "waypoints": [
        {
          "id": "wp_000",
          "latitude": 66.0,
          "longitude": 5.0
        },
        {
          "id": "wp_001",
          "latitude": 66.5,
          "longitude": 6.0
        }
      ],
      "distance_nm": 50.0,
      "status": "planned"
    }
  ]
}
```

### 成本函数配置

```json
{
  "cost_function": {
    "type": "weighted_sum",
    "weights": {
      "ice_concentration": 0.4,
      "ice_thickness": 0.3,
      "wind_speed": 0.15,
      "wave_height": 0.1,
      "current_speed": 0.05
    }
  }
}
```

### 约束条件

```json
{
  "constraints": {
    "max_ice_thickness": 2.0,
    "min_water_depth": 5.0,
    "max_wind_speed": 25.0,
    "max_wave_height": 8.0
  }
}
```

## 配置文件说明

### config_empty.json

PolarRoute 配置文件包含：

1. **路由算法**
   - `algorithm`: dijkstra / astar / rrt
   - `optimization_method`: cost_minimization / time_minimization / fuel_minimization

2. **环境权重**
   - 各环境因素对成本的影响权重

3. **船舶默认值**
   - 设计速度、吃水、冰级等

4. **优化目标**
   - 单目标或多目标优化
   - 目标权重配置

5. **约束条件**
   - 硬约束（必须满足）
   - 软约束（违反时施加惩罚）

6. **输出格式**
   - GeoJSON / GPX / KML / CSV

## 集成脚本使用

### PolarRouteIntegration 类

```python
from scripts.integrate_polarroute import PolarRouteIntegration

# 初始化
integration = PolarRouteIntegration(
    config_path="data_sample/polarroute/config_empty.json",
    mesh_path="data_sample/polarroute/vessel_mesh_empty.json"
)

# 加载 ArcticRoute 网格
integration.load_arcticroute_grid()

# 添加船舶
integration.add_vehicle_to_mesh(
    vessel_id="vessel_001",
    vessel_type="handysize",
    ice_class="PC7",
    max_ice_thickness=1.2
)

# 规划路由
path = integration.plan_route(
    start_lat=66.0,
    start_lon=5.0,
    end_lat=78.0,
    end_lon=150.0
)

# 添加到 mesh
if path:
    integration.add_route_to_mesh(
        route_id="route_001",
        vessel_id="vessel_001",
        waypoints=path
    )

# 保存和导出
integration.save_mesh()
integration.export_routes_geojson("output/routes.geojson")
```

## 与真实数据集成

### 步骤 1: 准备环境数据

从 ArcticRoute 的数据管线获取：

```python
from arcticroute.core.grid import load_real_grid
from arcticroute.core.cost import build_cost_function

# 加载真实网格
grid, land_mask = load_real_grid(
    date="2025-12-14",
    region="barents_sea"
)

# 构建成本函数
cost_func = build_cost_function(
    grid=grid,
    land_mask=land_mask,
    weights={
        "ice_concentration": 0.4,
        "ice_thickness": 0.3,
        "wind_speed": 0.15,
        "wave_height": 0.1
    }
)
```

### 步骤 2: 填充 vessel_mesh.json

```python
import xarray as xr
import json

# 加载网格数据
ds = xr.open_dataset("data_processed/grid_2025_12.nc")

# 构建 mesh
mesh = {
    "metadata": {...},
    "grid": {
        "type": "regular_latlon",
        "resolution_degrees": float(ds.attrs.get("resolution", 0.1)),
        "dimensions": {
            "latitude": len(ds.lat),
            "longitude": len(ds.lon)
        }
    },
    "environmental_layers": {
        "ice_concentration": {
            "name": "Sea Ice Concentration",
            "data": ds["ice_concentration"].values.tolist()
        },
        "ice_thickness": {
            "name": "Sea Ice Thickness",
            "data": ds["ice_thickness"].values.tolist()
        },
        # ... 其他层
    }
}

# 保存
with open("data_sample/polarroute/vessel_mesh_real.json", "w") as f:
    json.dump(mesh, f)
```

### 步骤 3: 运行完整集成

```bash
python scripts/integrate_polarroute.py \
  --config data_sample/polarroute/config_real.json \
  --mesh data_sample/polarroute/vessel_mesh_real.json \
  --demo
```

## 常见问题

### Q: vessel_mesh.json 中的 "data" 字段为空是否可以？

**A:** 可以。Empty Mesh 用于演示和测试。当使用真实数据时，需要填充 `environmental_layers` 中的 `data` 字段。

### Q: 如何处理大型网格数据？

**A:** 对于大型网格，建议：
1. 使用 NetCDF 或 HDF5 格式存储原始数据
2. 在 JSON 中仅存储元数据和索引
3. 在运行时动态加载数据

```json
{
  "environmental_layers": {
    "ice_concentration": {
      "name": "Sea Ice Concentration",
      "data_source": "data_processed/ice_concentration_2025_12.nc",
      "data_variable": "ice_concentration",
      "data": null
    }
  }
}
```

### Q: 如何集成多个船舶？

**A:** 在 `vehicles` 数组中添加多个船舶配置：

```python
for vessel_config in vessel_configs:
    integration.add_vehicle_to_mesh(
        vessel_id=vessel_config["id"],
        vessel_type=vessel_config["type"],
        ice_class=vessel_config["ice_class"],
        max_ice_thickness=vessel_config["max_ice_thickness"]
    )
```

### Q: 如何导出为其他格式？

**A:** 扩展 `PolarRouteIntegration` 类：

```python
def export_routes_gpx(self, output_path):
    """导出为 GPX 格式"""
    # 实现 GPX 导出逻辑
    pass

def export_routes_kml(self, output_path):
    """导出为 KML 格式"""
    # 实现 KML 导出逻辑
    pass
```

## 下一步

1. **测试 Empty Mesh**: 运行 `python scripts/integrate_polarroute.py --demo`
2. **准备真实数据**: 从数据管线获取环境数据
3. **填充 vessel_mesh.json**: 使用真实数据更新 mesh
4. **集成 PolarRoute CLI**: 使用 `optimise_routes` 命令
5. **验证结果**: 检查输出的 GeoJSON 和路由质量

## 参考资源

- [PolarRoute 文档](https://github.com/bas-amop/PolarRoute)
- [MeshiPhi 文档](https://github.com/bas-amop/MeshiPhi)
- [ArcticRoute 架构](docs/adr/ADR-0001-layergraph.md)

## 支持

如有问题，请参考：
- 日志输出（使用 `--verbose` 标志）
- 示例文件（`data_sample/polarroute/`）
- 集成脚本（`scripts/integrate_polarroute.py`）

