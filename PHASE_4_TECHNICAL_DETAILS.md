# Phase 4 技术细节文档

## 目录
1. [架构设计](#架构设计)
2. [模块详解](#模块详解)
3. [API 文档](#api-文档)
4. [计算公式](#计算公式)
5. [测试覆盖](#测试覆盖)
6. [集成点](#集成点)

---

## 架构设计

### 整体结构
```
arcticroute/
├── core/
│   ├── eco/
│   │   ├── __init__.py
│   │   ├── vessel_profiles.py    # 船舶参数配置
│   │   └── eco_model.py          # ECO 估算模型
│   ├── grid.py
│   ├── landmask.py
│   ├── cost.py
│   └── astar.py
└── ui/
    └── planner_minimal.py        # 集成 ECO 的 UI
```

### 数据流
```
用户选择船型
    ↓
plan_three_routes() 规划三条路线
    ↓
对每条路线调用 estimate_route_eco()
    ↓
返回 EcoRouteEstimate 对象
    ↓
填入 RouteInfo 数据类
    ↓
UI 表格显示 ECO 指标
```

---

## 模块详解

### 1. vessel_profiles.py

#### VesselProfile 数据类
```python
@dataclass
class VesselProfile:
    key: str                    # 船型标识符，如 "panamax"
    name: str                   # 船型名称，如 "Panamax"
    dwt: float                  # 载重吨数（Deadweight Tonnage）
    design_speed_kn: float      # 设计航速（节）
    base_fuel_per_km: float     # 基础单位油耗（t/km）
```

#### get_default_profiles() 函数
```python
def get_default_profiles() -> Dict[str, VesselProfile]:
    """
    返回内置的默认船型配置字典。
    
    返回值：
    {
        "handy": VesselProfile(...),
        "panamax": VesselProfile(...),
        "ice_class": VesselProfile(...)
    }
    """
```

**内置船型参数：**

| 船型 | key | DWT | 航速 | 油耗 | 说明 |
|-----|-----|-----|------|------|------|
| Handysize | handy | 30,000 t | 13 kn | 0.035 t/km | 小型通用船 |
| Panamax | panamax | 80,000 t | 14 kn | 0.050 t/km | 巴拿马型（默认） |
| Ice-Class Cargo | ice_class | 50,000 t | 12 kn | 0.060 t/km | 破冰型货轮 |

---

### 2. eco_model.py

#### EcoRouteEstimate 数据类
```python
@dataclass
class EcoRouteEstimate:
    distance_km: float      # 航程距离（km）
    travel_time_h: float    # 航行时间（小时）
    fuel_total_t: float     # 总燃油消耗（吨）
    co2_total_t: float      # 总 CO2 排放（吨）
```

#### estimate_route_eco() 函数
```python
def estimate_route_eco(
    route_latlon: List[Tuple[float, float]],
    vessel: VesselProfile,
    co2_per_ton_fuel: float = 3.114,
) -> EcoRouteEstimate:
    """
    估算航程的 ECO（能耗）指标。
    
    参数：
        route_latlon: 路线点列表 [(lat, lon), ...]
        vessel: VesselProfile 船舶参数
        co2_per_ton_fuel: CO2 排放系数（t CO2 / t fuel），默认 3.114
    
    返回：
        EcoRouteEstimate 对象
    
    特殊情况：
        - 空路线或单点路线：返回全 0
    """
```

#### 内部函数：_haversine_km()
```python
def _haversine_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """
    计算两点间的大圆距离（单位：km）。
    
    使用 Haversine 公式：
    a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
    c = 2 * atan2(√a, √(1-a))
    d = R * c
    
    其中：
        φ = 纬度（弧度）
        λ = 经度（弧度）
        R = 地球半径 6371 km
    """
```

---

## API 文档

### 导入方式

#### 获取船型配置
```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles, VesselProfile

profiles = get_default_profiles()
vessel = profiles["panamax"]
```

#### 估算 ECO
```python
from arcticroute.core.eco.eco_model import estimate_route_eco, EcoRouteEstimate

eco = estimate_route_eco(route_latlon, vessel)
```

#### 在 UI 中使用
```python
from arcticroute.core.eco.vessel_profiles import get_default_profiles
from arcticroute.core.eco.eco_model import estimate_route_eco

# UI 中的用法
vessel_profiles = get_default_profiles()
selected_vessel = vessel_profiles[selected_vessel_key]
eco = estimate_route_eco(path, selected_vessel)
```

---

## 计算公式

### 1. 航程距离（Haversine）
```
R = 6371 km（地球平均半径）
φ1, φ2 = 纬度（弧度）
Δφ = φ2 - φ1
Δλ = λ2 - λ1

a = sin²(Δφ/2) + cos(φ1) * cos(φ2) * sin²(Δλ/2)
c = 2 * atan2(√a, √(1-a))
distance = R * c
```

### 2. 航行时间
```
speed_kmh = design_speed_kn * 1.852  （1 节 = 1.852 km/h）
travel_time_h = distance_km / speed_kmh
```

### 3. 燃油消耗
```
fuel_total_t = distance_km * base_fuel_per_km
```

### 4. CO2 排放
```
co2_total_t = fuel_total_t * co2_per_ton_fuel
```

### 示例计算
```
假设：
- 路线：(70.0, 10.0) → (71.0, 10.0)（约 111 km）
- 船型：Panamax
  - design_speed_kn = 14 kn
  - base_fuel_per_km = 0.050 t/km

计算：
1. distance_km ≈ 111 km
2. speed_kmh = 14 * 1.852 = 25.928 km/h
3. travel_time_h = 111 / 25.928 ≈ 4.28 h
4. fuel_total_t = 111 * 0.050 = 5.55 t
5. co2_total_t = 5.55 * 3.114 ≈ 17.28 t
```

---

## 测试覆盖

### 测试文件：tests/test_eco_demo.py

#### 测试分类

**1. 配置测试**
- `test_default_vessels_exist`: 验证 3 种默认船型存在
- `test_default_vessels_have_required_fields`: 验证船型字段完整性

**2. 功能测试**
- `test_eco_scales_with_distance`: 验证 ECO 随距离增加
- `test_empty_route_eco_zero`: 验证空路线返回全 0
- `test_single_point_route_eco_zero`: 验证单点路线返回全 0

**3. 计算正确性测试**
- `test_eco_fuel_calculation`: 验证 fuel = distance * base_fuel_per_km
- `test_eco_co2_calculation`: 验证 co2 = fuel * co2_per_ton_fuel
- `test_eco_travel_time_calculation`: 验证 time = distance / speed

**4. 对比测试**
- `test_eco_different_vessels`: 验证不同船型的差异
- `test_eco_custom_co2_coefficient`: 验证自定义 CO2 系数

#### 测试覆盖率
```
总测试数：10
通过率：100%
覆盖范围：
  - VesselProfile 数据类 ✓
  - get_default_profiles() 函数 ✓
  - EcoRouteEstimate 数据类 ✓
  - estimate_route_eco() 函数 ✓
  - 边界情况（空路线、单点） ✓
  - 计算公式验证 ✓
  - 多船型对比 ✓
```

---

## 集成点

### 1. UI 中的集成（planner_minimal.py）

#### RouteInfo 数据类扩展
```python
@dataclass
class RouteInfo:
    # ... 原有字段 ...
    distance_km: float = 0.0      # 新增
    travel_time_h: float = 0.0    # 新增
    fuel_total_t: float = 0.0     # 新增
    co2_total_t: float = 0.0      # 新增
```

#### plan_three_routes() 函数签名
```python
def plan_three_routes(
    grid,
    land_mask,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    allow_diag: bool = True,
    vessel: VesselProfile | None = None,  # 新增参数
) -> list[RouteInfo]:
```

#### 规划逻辑
```python
if path:
    # ... 现有逻辑 ...
    
    # 新增：计算 ECO 指标
    eco_estimate = None
    if vessel is not None:
        eco_estimate = estimate_route_eco(path, vessel)
    
    route_info = RouteInfo(
        # ... 现有字段 ...
        distance_km=eco_estimate.distance_km if eco_estimate else 0.0,
        travel_time_h=eco_estimate.travel_time_h if eco_estimate else 0.0,
        fuel_total_t=eco_estimate.fuel_total_t if eco_estimate else 0.0,
        co2_total_t=eco_estimate.co2_total_t if eco_estimate else 0.0,
    )
```

### 2. Sidebar 船型选择
```python
st.subheader("船舶配置")
vessel_profiles = get_default_profiles()
vessel_keys = list(vessel_profiles.keys())
selected_vessel_key = st.selectbox(
    "选择船型",
    options=vessel_keys,
    index=vessel_keys.index("panamax") if "panamax" in vessel_keys else 0,
    format_func=lambda k: f"{vessel_profiles[k].name} ({k})",
)
selected_vessel = vessel_profiles[selected_vessel_key]
```

### 3. 摘要表格显示
```python
summary_data.append({
    # ... 现有列 ...
    "distance_km": f"{route_info.distance_km:.1f}" if route_info.distance_km > 0 else "-",
    "travel_time_h": f"{route_info.travel_time_h:.1f}" if route_info.travel_time_h > 0 else "-",
    "fuel_total_t": f"{route_info.fuel_total_t:.2f}" if route_info.fuel_total_t > 0 else "-",
    "co2_total_t": f"{route_info.co2_total_t:.2f}" if route_info.co2_total_t > 0 else "-",
})
```

---

## 扩展建议

### 1. 动态油耗模型
```python
def estimate_fuel_with_conditions(
    distance_km: float,
    vessel: VesselProfile,
    sea_state: float,        # 海况等级 0-8
    wind_speed_kn: float,    # 风速
    ice_concentration: float, # 冰浓度 0-1
) -> float:
    """考虑海况、风向、冰况的动态油耗估算"""
```

### 2. 多目标优化
```python
def optimize_route_eco(
    start: Tuple[float, float],
    end: Tuple[float, float],
    vessel: VesselProfile,
    weights: Dict[str, float],  # {"distance": 0.3, "time": 0.3, "fuel": 0.4}
) -> RouteInfo:
    """基于多个目标的路线优化"""
```

### 3. 路线对比分析
```python
def compare_routes_eco(
    routes: List[List[Tuple[float, float]]],
    vessel: VesselProfile,
) -> pd.DataFrame:
    """对多条路线进行 ECO 对比分析"""
```

---

## 性能考虑

### 计算复杂度
- **Haversine 距离计算**：O(n)，n 为路线点数
- **ECO 估算**：O(n)，主要在距离计算
- **UI 更新**：O(3)，固定三条路线

### 优化空间
1. 缓存 Haversine 计算结果
2. 预计算常用路线的 ECO
3. 使用向量化计算加速多路线对比

---

## 版本信息

| 组件 | 版本 | 说明 |
|-----|------|------|
| Python | 3.11.9 | 项目最低版本 |
| Streamlit | 最新 | UI 框架 |
| NumPy | 最新 | 数值计算 |
| Pandas | 最新 | 数据处理 |

---

**文档版本**: 1.0  
**最后更新**: 2025-12-08  
**作者**: ArcticRoute Team













