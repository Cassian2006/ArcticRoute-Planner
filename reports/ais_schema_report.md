# AIS JSON 结构探测报告 (Phase AIS-A1)

生成时间: 2025-12-09T20:44:30.914894

## 概述

本报告对 `data_real/ais/2024/` 目录下的 5 个 AIS JSON 文件进行了结构探测。
目的是在不假设具体字段名的前提下，理解数据的字段结构，为后续管线设计提供指导。

## 文件清单

| 文件名 | 大小 (MB) | 结构类型 | 记录数 | 错误数 |
|--------|----------|---------|--------|--------|
| AIS of 2023.12.29-2024.03.28.json | 1681.18 | list | 322 | 0 |
| AIS of 2024.03.28-2024.06.26.json | 1916.66 | list | 231 | 0 |
| AIS of 2024.06.24-2024.09.22.json | 1687.88 | list | 254 | 0 |
| AIS of 2024.09.22-2024.12.21.json | 1675.45 | list | 246 | 0 |
| AIS of 2024.12.21-2025.01.01.json | 161.22 | list | 162 | 0 |

## 字段统计

### AIS of 2023.12.29-2024.03.28.json

**抽样记录数**: 322

**字段列表**:

- **code**
  - 类型: str
  - 空值数: 0
  - 示例: `PARAM_INVALID`
- **data**
  - 类型: list
  - 空值数: 0
  - 示例: `N/A`
- **detailMessage**
  - 类型: 
  - 空值数: 1
  - 示例: `N/A`
- **message**
  - 类型: str
  - 空值数: 321
  - 示例: `参数解析异常: [JSON parse error: Cannot deserialize value of type `java.lang.Integer` from String "mmsi": not a valid `java.lang.Integer` value]`
- **status**
  - 类型: int
  - 空值数: 0
  - 示例: `400`
- **success**
  - 类型: bool
  - 空值数: 0
  - 示例: `False`

### AIS of 2024.03.28-2024.06.26.json

**抽样记录数**: 231

**字段列表**:

- **code**
  - 类型: str
  - 空值数: 0
  - 示例: `PARAM_INVALID`
- **data**
  - 类型: list
  - 空值数: 0
  - 示例: `N/A`
- **detailMessage**
  - 类型: 
  - 空值数: 1
  - 示例: `N/A`
- **message**
  - 类型: str
  - 空值数: 230
  - 示例: `参数解析异常: [JSON parse error: Cannot deserialize value of type `java.lang.Integer` from String "mmsi": not a valid `java.lang.Integer` value]`
- **status**
  - 类型: int
  - 空值数: 0
  - 示例: `400`
- **success**
  - 类型: bool
  - 空值数: 0
  - 示例: `False`

### AIS of 2024.06.24-2024.09.22.json

**抽样记录数**: 254

**字段列表**:

- **code**
  - 类型: str
  - 空值数: 0
  - 示例: `PARAM_INVALID`
- **data**
  - 类型: list
  - 空值数: 0
  - 示例: `N/A`
- **detailMessage**
  - 类型: 
  - 空值数: 1
  - 示例: `N/A`
- **message**
  - 类型: str
  - 空值数: 253
  - 示例: `参数解析异常: [JSON parse error: Cannot deserialize value of type `java.lang.Integer` from String "mmsi": not a valid `java.lang.Integer` value]`
- **status**
  - 类型: int
  - 空值数: 0
  - 示例: `400`
- **success**
  - 类型: bool
  - 空值数: 0
  - 示例: `False`

### AIS of 2024.09.22-2024.12.21.json

**抽样记录数**: 246

**字段列表**:

- **code**
  - 类型: str
  - 空值数: 0
  - 示例: `PARAM_INVALID`
- **data**
  - 类型: list
  - 空值数: 0
  - 示例: `N/A`
- **detailMessage**
  - 类型: 
  - 空值数: 1
  - 示例: `N/A`
- **message**
  - 类型: str
  - 空值数: 245
  - 示例: `参数解析异常: [JSON parse error: Cannot deserialize value of type `java.lang.Integer` from String "mmsi": not a valid `java.lang.Integer` value]`
- **status**
  - 类型: int
  - 空值数: 0
  - 示例: `400`
- **success**
  - 类型: bool
  - 空值数: 0
  - 示例: `False`

### AIS of 2024.12.21-2025.01.01.json

**抽样记录数**: 162

**字段列表**:

- **code**
  - 类型: str
  - 空值数: 0
  - 示例: `PARAM_INVALID`
- **data**
  - 类型: list
  - 空值数: 0
  - 示例: `N/A`
- **detailMessage**
  - 类型: 
  - 空值数: 1
  - 示例: `N/A`
- **message**
  - 类型: str
  - 空值数: 161
  - 示例: `参数解析异常: [JSON parse error: Cannot deserialize value of type `java.lang.Integer` from String "mmsi": not a valid `java.lang.Integer` value]`
- **status**
  - 类型: int
  - 空值数: 0
  - 示例: `400`
- **success**
  - 类型: bool
  - 空值数: 0
  - 示例: `False`

## 推荐的统一 Schema

基于对所有文件的分析，推荐以下统一的字段映射：

```json
{}
```

## 统一列定义

基于推荐的 schema，建议的统一数据结构如下：

| 字段名 | 推荐类型 | 说明 |
|--------|---------|------|

## 数据质量检查

### AIS of 2023.12.29-2024.03.28.json

✓ **无解码错误**

### AIS of 2024.03.28-2024.06.26.json

✓ **无解码错误**

### AIS of 2024.06.24-2024.09.22.json

✓ **无解码错误**

### AIS of 2024.09.22-2024.12.21.json

✓ **无解码错误**

### AIS of 2024.12.21-2025.01.01.json

✓ **无解码错误**

## 后续步骤

1. **字段映射**: 根据推荐的 schema，在数据管线中实现字段映射逻辑
2. **数据清洗**: 处理缺失值、异常值、类型转换等
3. **聚合策略**: 决定是否按航次、时间窗口等维度聚合数据
4. **密度/轨迹分析**: 基于清洗后的数据进行后续分析

## 附录：完整字段分析

### AIS of 2023.12.29-2024.03.28.json - 详细字段分析

**语义字段猜测**:

- `timestamp` → [未找到]
- `lat` → [未找到]
- `lon` → [未找到]
- `mmsi` → [未找到]
- `sog` → [未找到]
- `cog` → [未找到]
- `ship_type` → [未找到]

### AIS of 2024.03.28-2024.06.26.json - 详细字段分析

**语义字段猜测**:

- `timestamp` → [未找到]
- `lat` → [未找到]
- `lon` → [未找到]
- `mmsi` → [未找到]
- `sog` → [未找到]
- `cog` → [未找到]
- `ship_type` → [未找到]

### AIS of 2024.06.24-2024.09.22.json - 详细字段分析

**语义字段猜测**:

- `timestamp` → [未找到]
- `lat` → [未找到]
- `lon` → [未找到]
- `mmsi` → [未找到]
- `sog` → [未找到]
- `cog` → [未找到]
- `ship_type` → [未找到]

### AIS of 2024.09.22-2024.12.21.json - 详细字段分析

**语义字段猜测**:

- `timestamp` → [未找到]
- `lat` → [未找到]
- `lon` → [未找到]
- `mmsi` → [未找到]
- `sog` → [未找到]
- `cog` → [未找到]
- `ship_type` → [未找到]

### AIS of 2024.12.21-2025.01.01.json - 详细字段分析

**语义字段猜测**:

- `timestamp` → [未找到]
- `lat` → [未找到]
- `lon` → [未找到]
- `mmsi` → [未找到]
- `sog` → [未找到]
- `cog` → [未找到]
- `ship_type` → [未找到]
