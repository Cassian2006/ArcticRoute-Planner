# Phase 7 + 7.5 快速参考指南

## 快速开始

### 1. 使用 POLARIS 计算 RIO 值

```python
from arcticroute.core.constraints.polaris import compute_rio_for_cell

# 计算单个网格点的 RIO
meta = compute_rio_for_cell(
    sic=0.7,              # 海冰浓度 (0-1)
    thickness_m=0.5,      # 冰厚 (米)
    ice_class="PC4",      # 船舶冰级
    use_decayed_table=False  # 是否使用衰减表
)

print(f"RIO: {meta.rio}")
print(f"操作等级: {meta.level}")  # normal, elevated, special
print(f"速度限制: {meta.speed_limit_knots} knots")
print(f"冰型: {meta.ice_type}")
```

### 2. 厚度到冰型的映射

```python
from arcticroute.core.constraints.polaris import thickness_to_ice_type

ice_type = thickness_to_ice_type(thickness_m=0.4, sic=0.8)
# 返回: "thin_fy_1st"
```

### 3. 操作等级分级

```python
from arcticroute.core.constraints.polaris import classify_operation_level

level = classify_operation_level(rio=-5.0, ice_class="PC5")
# 返回: "elevated"
```

### 4. 速度建议查询

```python
from arcticroute.core.constraints.polaris import recommended_speed_limit_knots

speed = recommended_speed_limit_knots(level="elevated", ice_class="PC3")
# 返回: 5.0 (knots)
```

---

## 数据结构

### PolarisMeta
```python
@dataclass(frozen=True)
class PolarisMeta:
    ice_type: IceType                    # 冰型
    rio: float                           # RIO 值
    level: OperationLevel                # 操作等级
    speed_limit_knots: Optional[float]   # 速度限制
    riv_used: str                        # 使用的表版本
```

### IceType 类型
```
"ice_free"          # 无冰
"new_ice"           # 新冰 (< 10 cm)
"grey_ice"          # 灰冰 (10-15 cm)
"grey_white_ice"    # 灰白冰 (15-30 cm)
"thin_fy_1st"       # 薄一年冰 (30-50 cm)
"thin_fy_2nd"       # 薄二年冰 (50-70 cm)
"medium_fy"         # 中一年冰 (70-120 cm)
"thick_fy"          # 厚一年冰 (120-200 cm)
"second_year"       # 二年冰 (200-250 cm)
"multi_year"        # 多年冰 (> 250 cm)
```

### OperationLevel 类型
```
"normal"    # RIO ≥ 0，无限制
"elevated"  # -10 ≤ RIO < 0，有速度限制
"special"   # RIO < -10，特殊操作
```

---

## RIO 公式

```
RIO = (c_open × RIV_open) + (c_ice × RIV_ice)

其中：
  c_open = 10 - c_ice          # 开水十分位
  c_ice = round(10 × SIC)      # 冰十分位
  RIV_open = 3 (总是)          # 开水的 RIV 值
  RIV_ice = 查表(ice_class, ice_type)  # 冰的 RIV 值
```

### 示例计算
```
SIC = 0.6, thickness = 0.4m, ice_class = PC4

c_ice = round(10 × 0.6) = 6
c_open = 10 - 6 = 4
ice_type = thickness_to_ice_type(0.4, 0.6) = "thin_fy_1st"

RIV_open = 3
RIV_ice = Table_1_3[PC4, thin_fy_1st] = 1

RIO = (4 × 3) + (6 × 1) = 12 + 6 = 18
level = classify_operation_level(18, "PC4") = "normal"
```

---

## 操作等级阈值

| 冰级 | RIO ≥ 0 | -10 ≤ RIO < 0 | RIO < -10 |
|------|---------|---------------|-----------|
| PC1-PC7 | normal | elevated | special |
| IC | normal | special | special |
| NOICE | normal | special | special |

---

## 速度限制表（elevated 级别）

| 冰级 | 速度限制 |
|------|---------|
| PC1 | 11.0 knots |
| PC2 | 8.0 knots |
| PC3-PC5 | 5.0 knots |
| Below PC5 | 3.0 knots |

---

## Copernicus 数据拉取

### 基本用法

```bash
python scripts/fetch_copernicus_once.py \
    --outdir data/cmems_cache \
    --bbox -40 60 65 85 \
    --days 2
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--outdir` | `data/cmems_cache` | 输出目录 |
| `--bbox` | `-180 180 60 90` | 地理边界 (MINLON MAXLON MINLAT MAXLAT) |
| `--days` | `2` | 回溯天数 |

### 产品列表

| 产品 ID | 描述 | 更新频率 |
|---------|------|---------|
| SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 | 北极海冰产品 | 日度 (12:00 UTC) |
| ARCTIC_ANALYSIS_FORECAST_WAV_002_014 | 北极波浪预报 | 小时级 |

### 获取产品信息

```bash
copernicusmarine describe --contains SEAICE_ARC_PHY_AUTO_L4_MYNRT_011_024 --include-datasets
copernicusmarine describe --contains ARCTIC_ANALYSIS_FORECAST_WAV_002_014 --include-datasets
```

---

## 单元测试

### 运行所有 POLARIS 测试

```bash
python -m pytest tests/test_polaris_constraints.py -v
```

### 测试覆盖

- ✓ 厚度分段映射 (8 个分段)
- ✓ RIO 公式计算
- ✓ 操作等级分级
- ✓ 速度限制查询

---

## 常见问题

### Q: 如何在成本网格中集成 POLARIS？

A: 在 `polar_rules.py` 的 `apply_soft_penalties()` 中添加：

```python
def apply_soft_penalties(...):
    # 计算 POLARIS RIO
    for i, j in grid_cells:
        meta = compute_rio_for_cell(sic[i,j], thickness[i,j], ice_class)
        
        if meta.level == "special":
            # Hard block
            cost[i,j] = 1e10
        elif meta.level == "elevated":
            # Soft penalty
            cost[i,j] *= (1 + penalty_factor)
```

### Q: Table 1.3 和 Table 1.4 有什么区别？

A: 
- **Table 1.3**: 标准条件（默认）
- **Table 1.4**: 衰减冰条件（仅在确认 decayed ice 时使用）

### Q: RIO 值的范围是多少？

A: RIO 范围取决于冰级和冰型组合，通常在 -8 到 30 之间。

---

## 参考文献

- MSC.1/Circ.1519 - Polar Code 实施指南
- Table 1.1 - 操作等级分级
- Table 1.2 - 速度建议
- Table 1.3 - RIV 值（标准条件）
- Table 1.4 - RIV 值（衰减条件）

---

**最后更新**: 2025-12-14


