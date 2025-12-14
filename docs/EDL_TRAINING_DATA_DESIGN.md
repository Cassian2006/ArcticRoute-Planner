# EDL 训练数据 Schema 设计

**文档版本**: 1.0  
**创建时间**: 2025-12-11  
**阶段**: EDL-0（训练准备）  
**目标**: 定义 EDL 模型的训练数据格式、特征和标签，为后续数据导出和模型训练奠定基础。

---

## 1. 概述

本文档定义了 EDL（Evidential Deep Learning）模型的训练数据 schema。EDL 模型用于在北冰洋航线规划中进行**多类分类 + 不确定性估计**，帮助决策系统评估航线风险。

### 核心特点
- **输入**: 环境特征 + 船舶特征（7-9 维）
- **输出**: 航线安全等级 + 不确定性（Dirichlet 分布）
- **格式**: Parquet 文件（高效、支持列式存储、易于分布式处理）
- **扩展性**: 支持从二分类逐步扩展到多类分类

---

## 2. 输入特征（Features）

### 2.1 环保特征（Environmental Features）

| 特征名 | 数据类型 | 范围 | 单位 | 说明 |
|--------|---------|------|------|------|
| `lat` | float32 | [-90, 90] | 度 | 纬度（WGS84） |
| `lon` | float32 | [-180, 180] | 度 | 经度（WGS84） |
| `month` | int8 | [1, 12] | - | 月份（1=1月，12=12月） |
| `dayofyear` | int16 | [1, 366] | - | 一年中的第几天（1-366） |
| `sic` | float32 | [0, 100] | % | 海冰浓度（Sea Ice Concentration） |
| `ice_thickness_m` | float32 | [0, 5] | 米 | 海冰厚度 |
| `wave_swh` | float32 | [0, 15] | 米 | 波浪显著波高（Significant Wave Height） |
| `ais_density` | float32 | [0, 1] | 归一化 | AIS 船舶密度（0=无船舶，1=最密集） |

### 2.2 船舶特征（Vessel Features）

| 特征名 | 数据类型 | 范围 | 说明 |
|--------|---------|------|------|
| `vessel_class_id` | int8 | [0, 2] | 船舶等级编码：0=Handy, 1=Panamax, 2=Ice-class |
| `distance_to_coast_m` | float32 | [0, ∞) | 到最近海岸线的距离（米）（可选） |

### 2.3 特征说明

#### `sic`（海冰浓度）
- 来源：NSIDC 或 OSISAF 海冰产品
- 范围：0-100%
- 安全阈值：< 30% 为开阔水域，30-70% 为边际冰区，> 70% 为密集冰区

#### `ice_thickness_m`
- 来源：SMOS / SMAP 或模式预报
- 范围：0-5 米（北冰洋典型值）
- 安全阈值：< 1m 为薄冰，1-2m 为中等冰，> 2m 为厚冰

#### `wave_swh`（显著波高）
- 来源：ECMWF ERA5 或 NOAA 波浪模式
- 范围：0-15 米（北冰洋极端值）
- 安全阈值：< 2m 为平静，2-4m 为中等，> 4m 为恶劣

#### `ais_density`
- 来源：AIS 数据栅格化
- 计算方法：在给定网格内，AIS 点数 / 最大点数（归一化到 [0,1]）
- 含义：高密度 = 更多船舶活动 = 更多参考轨迹

#### `vessel_class_id`
- 编码方案：
  - `0` = Handy（小型通用船，< 50,000 DWT）
  - `1` = Panamax（巴拿马型，50,000-100,000 DWT）
  - `2` = Ice-class（破冰船或冰级船，专为极地设计）

#### `distance_to_coast_m`（可选）
- 来源：自然地球数据或 GEBCO 海岸线
- 用途：评估应急撤离难度
- 可选原因：初期可能不需要，后续可加入

---

## 3. 输出标签（Targets）

### 3.1 简单版本：二分类（Safe / Risky）

```
label_safe_risky: int8 ∈ {0, 1}
  0 = Safe（安全）
  1 = Risky（风险）
```

#### 标签定义规则

**Safe（安全）** 满足以下条件：
- `sic < 30%`（开阔水域）
- `ice_thickness_m < 1.0`（薄冰或无冰）
- `wave_swh < 4.0`（中等及以下波浪）
- `ais_density > 0.1`（有足够船舶活动参考）

**Risky（风险）** 满足以下条件：
- `sic >= 70%`（密集冰区）
- **或** `ice_thickness_m >= 2.0`（厚冰）
- **或** `wave_swh >= 5.0`（恶劣海况）
- **或** `ais_density < 0.05`（极少船舶活动）

**中间状态**（边界情况）：
- 如果不完全满足 Safe 或 Risky，根据**风险评分**判定：
  - 风险评分 = 0.3×(sic/100) + 0.3×(ice_thickness_m/3) + 0.2×(wave_swh/6) + 0.2×(1-ais_density)
  - 评分 < 0.4 → Safe
  - 评分 >= 0.4 → Risky

### 3.2 扩展版本：多类分类（后续）

```
label_ice_zone: int8 ∈ {0, 1, 2}
  0 = Open Water（开阔水域）
  1 = Marginal Ice Zone（边际冰区）
  2 = Heavy Ice（密集冰区）
```

**Open Water**：
- `sic < 30%` 且 `ice_thickness_m < 0.5`

**Marginal Ice Zone**：
- `30% <= sic < 70%` 或 `0.5 <= ice_thickness_m < 2.0`

**Heavy Ice**：
- `sic >= 70%` 或 `ice_thickness_m >= 2.0`

---

## 4. 数据文件格式

### 4.1 推荐格式：Parquet

**优势**：
- 列式存储，压缩率高（相比 CSV 节省 50-80% 空间）
- 支持分布式处理（Spark、Dask）
- 保留数据类型信息，无需再次转换
- 读取速度快（适合大规模训练）

### 4.2 文件组织

```
data/
├── edl_training/
│   ├── train_2024_2025.parquet       # 训练集（2024-2025 年数据）
│   ├── val_2024_2025.parquet         # 验证集
│   ├── test_2024_2025.parquet        # 测试集
│   └── metadata.json                 # 元数据（见 4.4）
```

### 4.3 Parquet 列定义

```
列名                  | 数据类型  | 压缩 | 说明
---------------------|----------|------|-------------------
lat                  | float32  | snappy | 纬度
lon                  | float32  | snappy | 经度
month                | int8     | snappy | 月份
dayofyear            | int16    | snappy | 一年中的第几天
sic                  | float32  | snappy | 海冰浓度 (%)
ice_thickness_m      | float32  | snappy | 冰厚 (m)
wave_swh             | float32  | snappy | 波高 (m)
ais_density          | float32  | snappy | AIS 密度 [0,1]
vessel_class_id      | int8     | snappy | 船舶等级
distance_to_coast_m  | float32  | snappy | 到海岸距离 (m) [可选]
label_safe_risky     | int8     | snappy | 二分类标签
timestamp            | int64    | snappy | Unix 时间戳（用于追溯）
```

### 4.4 元数据文件（metadata.json）

```json
{
  "version": "1.0",
  "created_at": "2025-12-11T00:00:00Z",
  "dataset_name": "EDL_Training_2024_2025",
  "split_info": {
    "train": {
      "file": "train_2024_2025.parquet",
      "num_samples": 50000,
      "date_range": ["2024-01-01", "2025-06-30"]
    },
    "val": {
      "file": "val_2024_2025.parquet",
      "num_samples": 10000,
      "date_range": ["2025-07-01", "2025-09-30"]
    },
    "test": {
      "file": "test_2024_2025.parquet",
      "num_samples": 10000,
      "date_range": ["2025-10-01", "2025-12-31"]
    }
  },
  "features": {
    "environmental": ["lat", "lon", "month", "dayofyear", "sic", "ice_thickness_m", "wave_swh", "ais_density"],
    "vessel": ["vessel_class_id", "distance_to_coast_m"],
    "total_dim": 10
  },
  "targets": {
    "primary": "label_safe_risky",
    "classes": {
      "0": "Safe",
      "1": "Risky"
    },
    "class_distribution": {
      "train": {"0": 0.65, "1": 0.35},
      "val": {"0": 0.63, "1": 0.37},
      "test": {"0": 0.64, "1": 0.36}
    }
  },
  "data_sources": {
    "environmental": "NSIDC/OSISAF/ERA5/ECMWF",
    "ais": "AIS_raw_data",
    "vessel_info": "Manual annotation or AIS metadata"
  },
  "preprocessing": {
    "normalization": "StandardScaler on features",
    "missing_value_strategy": "Forward-fill or interpolation",
    "outlier_removal": "IQR-based (1.5x)"
  },
  "notes": "Initial version with binary classification. Ready for model training."
}
```

---

## 5. 数据生成流程

### 5.1 高层流程

```
原始数据（AIS + 环境场）
    ↓
[Step 1] 数据对齐与栅格化
    - AIS 点 → 网格密度
    - 环境场（SIC、冰厚、波高）→ 网格
    ↓
[Step 2] 特征提取
    - 提取 lat, lon, month, dayofyear, sic, ice_thickness_m, wave_swh, ais_density
    - 添加船舶信息（vessel_class_id）
    ↓
[Step 3] 标签生成
    - 根据规则生成 label_safe_risky
    ↓
[Step 4] 数据清洗
    - 移除异常值
    - 处理缺失值
    - 数据类型转换
    ↓
[Step 5] 数据分割
    - 按时间分割：train / val / test
    ↓
[Step 6] 导出为 Parquet
    - 保存为 train_*.parquet, val_*.parquet, test_*.parquet
    - 生成 metadata.json
```

### 5.2 Python 伪代码框架

```python
import pandas as pd
import numpy as np
from pathlib import Path

def create_edl_training_dataset(
    ais_data_dir: str,
    env_data_dir: str,
    output_dir: str,
    date_range: tuple = ("2024-01-01", "2025-12-31"),
    grid_resolution: float = 0.5,  # 度
    vessel_class_mapping: dict = None,
):
    """
    生成 EDL 训练数据集。
    
    Args:
        ais_data_dir: AIS 原始数据目录
        env_data_dir: 环境数据目录（SIC, 冰厚, 波高）
        output_dir: 输出目录
        date_range: 数据时间范围
        grid_resolution: 网格分辨率（度）
        vessel_class_mapping: 船舶等级映射
    """
    
    # Step 1: 加载并对齐数据
    ais_df = load_ais_data(ais_data_dir, date_range)
    env_data = load_environmental_data(env_data_dir, date_range)
    
    # Step 2: 栅格化 AIS
    ais_density = rasterize_ais_density(ais_df, grid_resolution)
    
    # Step 3: 创建特征矩阵
    features = extract_features(
        ais_density=ais_density,
        env_data=env_data,
        grid_resolution=grid_resolution,
        vessel_class_mapping=vessel_class_mapping,
    )
    
    # Step 4: 生成标签
    labels = generate_labels_safe_risky(features)
    
    # Step 5: 合并
    dataset = pd.concat([features, labels], axis=1)
    
    # Step 6: 清洗
    dataset = clean_dataset(dataset)
    
    # Step 7: 分割
    train, val, test = split_by_time(dataset, date_range)
    
    # Step 8: 导出
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train.to_parquet(output_path / "train_2024_2025.parquet", compression="snappy")
    val.to_parquet(output_path / "val_2024_2025.parquet", compression="snappy")
    test.to_parquet(output_path / "test_2024_2025.parquet", compression="snappy")
    
    # Step 9: 生成元数据
    metadata = generate_metadata(train, val, test)
    save_metadata(metadata, output_path / "metadata.json")
    
    print(f"✓ EDL training dataset created: {output_path}")
    return output_path

def generate_labels_safe_risky(features: pd.DataFrame) -> pd.Series:
    """
    根据特征生成二分类标签。
    """
    # 简单规则
    safe_mask = (
        (features['sic'] < 30) &
        (features['ice_thickness_m'] < 1.0) &
        (features['wave_swh'] < 4.0) &
        (features['ais_density'] > 0.1)
    )
    
    risky_mask = (
        (features['sic'] >= 70) |
        (features['ice_thickness_m'] >= 2.0) |
        (features['wave_swh'] >= 5.0) |
        (features['ais_density'] < 0.05)
    )
    
    # 边界情况：计算风险评分
    risk_score = (
        0.3 * (features['sic'] / 100) +
        0.3 * (features['ice_thickness_m'] / 3) +
        0.2 * (features['wave_swh'] / 6) +
        0.2 * (1 - features['ais_density'])
    )
    
    labels = pd.Series(0, index=features.index, dtype='int8')
    labels[safe_mask] = 0
    labels[risky_mask] = 1
    labels[~(safe_mask | risky_mask)] = (risk_score[~(safe_mask | risky_mask)] >= 0.4).astype('int8')
    
    return labels
```

---

## 6. 数据质量检查清单

在导出数据前，应进行以下检查：

- [ ] **特征范围检查**
  - `lat` ∈ [-90, 90]
  - `lon` ∈ [-180, 180]
  - `sic` ∈ [0, 100]
  - `ice_thickness_m` ∈ [0, 5]
  - `wave_swh` ∈ [0, 15]
  - `ais_density` ∈ [0, 1]

- [ ] **缺失值检查**
  - 所有必需列无 NaN
  - 可选列（如 `distance_to_coast_m`）缺失率 < 5%

- [ ] **标签分布检查**
  - 训练集：Safe:Risky ≈ 65:35
  - 验证集：Safe:Risky ≈ 63:37
  - 测试集：Safe:Risky ≈ 64:36
  - 无严重不平衡（偏差 < 10%）

- [ ] **时间连续性检查**
  - 无重复时间戳
  - 时间范围符合预期

- [ ] **数据类型检查**
  - 所有列数据类型正确
  - 无意外的字符串或对象类型

- [ ] **统计检查**
  - 特征均值、标准差合理
  - 无异常离群值（> 3σ）

---

## 7. 后续扩展方向

### 7.1 特征扩展
- 添加 `wind_speed_ms`（风速）
- 添加 `current_speed_ms`（海流速度）
- 添加 `visibility_km`（能见度）
- 添加 `air_temperature_c`（气温）

### 7.2 标签扩展
- 从二分类扩展到多类（Open Water / Marginal Ice / Heavy Ice）
- 添加连续标签（风险评分 ∈ [0, 1]）
- 添加不确定性标签（用于 EDL 训练）

### 7.3 数据增强
- 时间平移（temporal shift）
- 空间平移（spatial shift）
- 噪声注入（noise injection）

### 7.4 动态数据
- 支持实时数据流（在线学习）
- 支持增量更新（新数据追加）

---

## 8. 参考资源

- **NSIDC 海冰数据**: https://nsidc.org/
- **OSISAF 海冰产品**: https://www.osisaf.org/
- **ERA5 气象数据**: https://cds.climate.copernicus.eu/
- **Parquet 格式**: https://parquet.apache.org/
- **Pandas Parquet 文档**: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html

---

## 9. 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2025-12-11 | 初始版本，定义二分类 schema |
| - | - | 待扩展：多类分类、不确定性标签 |

---

**下一步**: 实现数据导出脚本（E0.2），建立最小训练闭环（E0.3）。



