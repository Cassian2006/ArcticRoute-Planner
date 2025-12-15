# PolarRoute 集成 - 快速开始指南

## 📋 概述

本指南帮助你快速将 PolarRoute 路由优化引擎集成到 ArcticRoute 中。

**关键文件已准备完毕：**
- ✅ `vessel_mesh_empty.json` - Empty Mesh 示例（关键）
- ✅ `config_empty.json` - PolarRoute 配置
- ✅ `waypoints_example.json` - 示例 waypoints
- ✅ 集成脚本和演示脚本

---

## [object Object]分钟快速开始

### 1️⃣ 运行演示

```bash
cd C:\Users\sgddsf\Desktop\AR_final

# 运行简化演示（推荐）
python scripts/demo_polarroute_simple.py

# 或运行完整集成演示（需要 ArcticRoute 环境）
python scripts/integrate_polarroute.py --demo --verbose
```

### 2️⃣ 查看生成的文件

```bash
# 查看生成的 mesh 文件
cat data_sample/polarroute/vessel_mesh_demo.json | python -m json.tool

# 查看生成的 GeoJSON（可在 QGIS/Leaflet 中打开）
cat data_sample/polarroute/routes_demo.geojson | python -m json.tool
```

### 3️⃣ 使用 PolarRoute CLI

```bash
# 使用生成的 mesh 和配置运行 PolarRoute
optimise_routes \
  data_sample/polarroute/config_empty.json \
  data_sample/polarroute/vessel_mesh_demo.json \
  data_sample/polarroute/waypoints_example.json \
  -o output/optimized_routes.json \
  --path_geojson
```

---

## 📁 文件结构

```
data_sample/polarroute/
├── vessel_mesh_empty.json          # ⭐ 空 mesh 示例（关键）
├── vessel_mesh_demo.json           # 演示生成的 mesh
├── config_empty.json               # PolarRoute 配置
├── waypoints_example.json          # 示例 waypoints
└── routes_demo.geojson             # 演示生成的 GeoJSON

scripts/
├── integrate_polarroute.py         # 完整集成脚本
├── demo_polarroute_simple.py       # 简化演示脚本
└── test_polarroute_integration.py  # 测试脚本

docs/
└── POLARROUTE_INTEGRATION_GUIDE.md # 详细指南
```

---

## 🔑 vessel_mesh.json 关键结构

### 最小有效 mesh

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Empty Mesh for PolarRoute",
    "created": "2025-12-14"
  },
  "grid": {
    "type": "regular_latlon",
    "resolution_degrees": 1.0,
    "dimensions": {
      "latitude": 26,
      "longitude": 361
    }
  },
  "environmental_layers": {
    "ice_concentration": {
      "name": "Sea Ice Concentration",
      "unit": "fraction (0-1)",
      "data": []
    },
    "ice_thickness": {
      "name": "Sea Ice Thickness",
      "unit": "meters",
      "data": []
    }
  },
  "vehicles": [],
  "routes": []
}
```

### 添加船舶

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

### 添加路由

```json
{
  "routes": [
    {
      "id": "route_001",
      "vessel_id": "vessel_001",
      "waypoints": [
        {"id": "wp_000", "latitude": 68.95, "longitude": 33.08},
        {"id": "wp_001", "latitude": 69.50, "longitude": 40.00},
        {"id": "wp_002", "latitude": 71.27, "longitude": 72.00}
      ],
      "distance_nm": 500.0,
      "status": "planned"
    }
  ]
}
```

---

## 🔄 PolarRoute 工作流程

```
┌─────────────────────────────────────────────────────────┐
│                  PolarRoute 工作流程                      │
└─────────────────────────────────────────────────────────┘

1. create_mesh
   └─> 从 vessel_mesh.json 加载网格和环境数据

2. add_vehicle
   └─> 添加船舶配置（冰级、速度等）

3. optimise_routes
   └─> 优化路由（考虑环境因素和约束）

4. export
   └─> 导出为 GeoJSON、GPX、KML 等格式
```

---

## 📊 配置说明

### config_empty.json 关键参数

```json
{
  "routing": {
    "algorithm": "dijkstra",           // 路由算法
    "optimization_method": "cost_minimization"  // 优化方法
  },
  "environmental_weights": {
    "ice_concentration": 0.4,          // 冰浓度权重
    "ice_thickness": 0.3,              // 冰厚权重
    "wind_speed": 0.15,                // 风速权重
    "wave_height": 0.1,                // 波高权重
    "current_speed": 0.05              // 洋流权重
  },
  "vessel_defaults": {
    "design_speed_kn": 14.0,           // 设计速度
    "ice_class": "PC7",                // 冰级
    "max_ice_thickness_m": 1.2         // 最大可通行冰厚
  },
  "constraints": {
    "hard_constraints": [
      {
        "name": "ice_thickness_limit",
        "max_value": 2.0,
        "enabled": true
      }
    ]
  }
}
```

---

## 🧪 验证和测试

### 运行测试套件

```bash
python scripts/test_polarroute_integration.py
```

**测试项目：**
- ✓ Mesh 文件结构验证
- ✓ Config 文件结构验证
- ✓ Waypoints 文件验证
- ✓ 集成脚本导入
- ✓ 集成脚本初始化

---

## 🔗 与真实数据集成

### 步骤 1: 准备环境数据

```python
import xarray as xr
import json

# 加载真实网格数据
ds = xr.open_dataset("data_processed/grid_2025_12.nc")

# 构建 mesh
mesh = {
    "metadata": {...},
    "grid": {...},
    "environmental_layers": {
        "ice_concentration": {
            "name": "Sea Ice Concentration",
            "data": ds["ice_concentration"].values.tolist()
        },
        "ice_thickness": {
            "name": "Sea Ice Thickness",
            "data": ds["ice_thickness"].values.tolist()
        }
    }
}

# 保存
with open("vessel_mesh_real.json", "w") as f:
    json.dump(mesh, f)
```

### 步骤 2: 运行 PolarRoute

```bash
optimise_routes \
  config_real.json \
  vessel_mesh_real.json \
  waypoints_real.json \
  -o output/routes_optimized.json \
  --path_geojson
```

### 步骤 3: 验证结果

```bash
# 查看优化后的路由
cat output/routes_optimized.json | python -m json.tool

# 在 GIS 中可视化
# 在 QGIS 中打开 routes_optimized.geojson
```

---

## 📈 性能优化

### 网格大小建议

| 应用场景 | 分辨率 | 网格大小 | 计算时间 |
|---------|--------|---------|---------|
| 演示/测试 | 1.0° | 26×361 | < 1s |
| 区域规划 | 0.5° | 52×722 | 5-10s |
| 详细规划 | 0.1° | 260×3610 | 1-5 min |
| 实时规划 | 0.05° | 520×7220 | > 10 min |

### 优化建议

1. **使用 Empty Mesh 测试**：快速验证工作流程
2. **逐步增加分辨率**：从 1.0° 开始，逐步细化
3. **缓存中间结果**：避免重复计算
4. **并行处理**：多条路由同时优化

---

## 🐛 常见问题

### Q1: vessel_mesh.json 中的 "data" 字段为空可以吗？

**A:** 可以。Empty Mesh 用于演示和测试。当使用真实数据时，需要填充数据。

### Q2: 如何处理大型网格？

**A:** 使用外部存储：

```json
{
  "environmental_layers": {
    "ice_concentration": {
      "name": "Sea Ice Concentration",
      "data_source": "data_processed/ice_concentration.nc",
      "data_variable": "ice_concentration",
      "data": null
    }
  }
}
```

### Q3: 如何添加多个船舶？

**A:** 在 `vehicles` 数组中添加多个配置：

```json
{
  "vehicles": [
    {"id": "vessel_001", "type": "handysize", ...},
    {"id": "vessel_002", "type": "panamax", ...},
    {"id": "vessel_003", "type": "capesize", ...}
  ]
}
```

### Q4: 如何导出为其他格式？

**A:** PolarRoute 支持多种格式：

```bash
# GeoJSON
optimise_routes config.json mesh.json waypoints.json --path_geojson

# GPX
optimise_routes config.json mesh.json waypoints.json --path_gpx

# KML
optimise_routes config.json mesh.json waypoints.json --path_kml

# CSV (Chart Track)
optimise_routes config.json mesh.json waypoints.json --chart_track output/
```

---

## 📚 详细文档

- **完整指南**: `POLARROUTE_INTEGRATION_GUIDE.md`
- **集成脚本**: `scripts/integrate_polarroute.py`
- **演示脚本**: `scripts/demo_polarroute_simple.py`
- **测试脚本**: `scripts/test_polarroute_integration.py`

---

## ✅ 检查清单

- [ ] 运行演示脚本 (`demo_polarroute_simple.py`)
- [ ] 验证生成的 mesh 文件
- [ ] 查看生成的 GeoJSON
- [ ] 运行测试套件 (`test_polarroute_integration.py`)
- [ ] 准备真实环境数据
- [ ] 填充 `vessel_mesh.json` 中的环境数据
- [ ] 运行 PolarRoute CLI 命令
- [ ] 验证优化结果
- [ ] 集成到 ArcticRoute 主管道

---

## 🎯 下一步

1. **立即开始**：运行 `python scripts/demo_polarroute_simple.py`
2. **理解结构**：查看生成的 JSON 文件
3. **准备数据**：从数据管线获取环境数据
4. **集成系统**：将 PolarRoute 集成到完整管道

---

## 📞 支持

遇到问题？

1. 查看日志：使用 `--verbose` 标志
2. 运行测试：`python scripts/test_polarroute_integration.py`
3. 查看示例：`data_sample/polarroute/` 目录
4. 阅读文档：`POLARROUTE_INTEGRATION_GUIDE.md`

---

**最后更新**: 2025-12-14  
**状态**: ✅ 可用  
**测试**: ✅ 5/5 通过


