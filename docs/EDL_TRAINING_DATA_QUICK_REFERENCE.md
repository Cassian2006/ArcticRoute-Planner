# EDL 训练数据 - 快速参考卡片

## 🎯 特征速查表

### 输入特征（10 维）

```python
# 环保特征（8 维）
features = {
    'lat': float32,           # [-90, 90] 度
    'lon': float32,           # [-180, 180] 度
    'month': int8,            # [1, 12]
    'dayofyear': int16,       # [1, 366]
    'sic': float32,           # [0, 100] %
    'ice_thickness_m': float32,  # [0, 5] 米
    'wave_swh': float32,      # [0, 15] 米
    'ais_density': float32,   # [0, 1] 归一化
}

# 船舶特征（2 维）
features.update({
    'vessel_class_id': int8,  # 0=Handy, 1=Panamax, 2=Ice-class
    'distance_to_coast_m': float32,  # [0, ∞) 米（可选）
})
```

---

## 🏷️ 标签速查表

### 二分类（Safe / Risky）

```python
# Safe（安全）
safe_condition = (
    (sic < 30) AND
    (ice_thickness_m < 1.0) AND
    (wave_swh < 4.0) AND
    (ais_density > 0.1)
)

# Risky（风险）
risky_condition = (
    (sic >= 70) OR
    (ice_thickness_m >= 2.0) OR
    (wave_swh >= 5.0) OR
    (ais_density < 0.05)
)

# 边界情况：风险评分
risk_score = (
    0.3 * (sic / 100) +
    0.4 * (ice_thickness_m / 3) +
    0.2 * (wave_swh / 6) +
    0.1 * (1 - ais_density)
)
# risk_score < 0.4 → Safe (0)
# risk_score >= 0.4 → Risky (1)
```

### 多类分类（后续）

```python
# Open Water
open_water = (sic < 30) AND (ice_thickness_m < 0.5)

# Marginal Ice Zone
marginal_ice = (
    (30 <= sic < 70) OR
    (0.5 <= ice_thickness_m < 2.0)
)

# Heavy Ice
heavy_ice = (sic >= 70) OR (ice_thickness_m >= 2.0)
```

---

## 📦 文件格式

### Parquet 列定义

```
lat, lon, month, dayofyear,
sic, ice_thickness_m, wave_swh, ais_density,
vessel_class_id, distance_to_coast_m,
label_safe_risky, timestamp
```

### 文件路径

```
data/edl_training/
├── train_2024_2025.parquet      (50,000 样本)
├── val_2024_2025.parquet        (10,000 样本)
├── test_2024_2025.parquet       (10,000 样本)
└── metadata.json
```

---

## 🔧 数据生成伪代码

```python
import pandas as pd

# 1. 加载原始数据
ais_df = load_ais_data(ais_dir)
env_data = load_environmental_data(env_dir)

# 2. 栅格化 AIS
ais_density = rasterize_ais_density(ais_df, grid_resolution=0.5)

# 3. 提取特征
features = pd.DataFrame({
    'lat': grid_lat,
    'lon': grid_lon,
    'month': env_data['month'],
    'dayofyear': env_data['dayofyear'],
    'sic': env_data['sic'],
    'ice_thickness_m': env_data['ice_thickness_m'],
    'wave_swh': env_data['wave_swh'],
    'ais_density': ais_density,
    'vessel_class_id': vessel_class,
    'distance_to_coast_m': distance_to_coast,
})

# 4. 生成标签
labels = generate_labels_safe_risky(features)

# 5. 合并
dataset = pd.concat([features, labels], axis=1)

# 6. 分割
train, val, test = split_by_time(dataset, split_dates)

# 7. 导出
train.to_parquet('data/edl_training/train_2024_2025.parquet', compression='snappy')
val.to_parquet('data/edl_training/val_2024_2025.parquet', compression='snappy')
test.to_parquet('data/edl_training/test_2024_2025.parquet', compression='snappy')
```

---

## ✅ 数据质量检查清单

```
□ 特征范围检查
  □ lat ∈ [-90, 90]
  □ lon ∈ [-180, 180]
  □ sic ∈ [0, 100]
  □ ice_thickness_m ∈ [0, 5]
  □ wave_swh ∈ [0, 15]
  □ ais_density ∈ [0, 1]

□ 缺失值检查
  □ 必需列无 NaN
  □ 可选列缺失率 < 5%

□ 标签分布检查
  □ 训练集：Safe:Risky ≈ 65:35
  □ 验证集：Safe:Risky ≈ 63:37
  □ 测试集：Safe:Risky ≈ 64:36

□ 时间连续性检查
  □ 无重复时间戳
  □ 时间范围符合预期

□ 数据类型检查
  □ 所有列数据类型正确

□ 统计检查
  □ 无异常离群值（> 3σ）
```

---

## 📊 数据统计示例

```
Dataset: train_2024_2025.parquet
├── Samples: 50,000
├── Features: 10
├── Classes: 2 (Safe: 65%, Risky: 35%)
├── Date Range: 2024-01-01 to 2025-06-30
└── Size: ~50 MB (Parquet compressed)

Feature Statistics:
├── lat: mean=75.5, std=8.2
├── lon: mean=-45.3, std=60.1
├── sic: mean=42.3, std=35.2
├── ice_thickness_m: mean=1.2, std=0.8
├── wave_swh: mean=2.1, std=1.3
└── ais_density: mean=0.35, std=0.28
```

---

## 🔗 相关文档

- **完整设计**: `docs/EDL_TRAINING_DATA_DESIGN.md`
- **任务完成报告**: `PHASE_EDL0_TASK_E0.1_COMPLETION.md`

---

## 💡 常见问题

**Q: 为什么选择 Parquet？**  
A: 列式存储，压缩率高（相比 CSV 节省 50-80%），支持分布式处理，读取速度快。

**Q: 标签如何处理边界情况？**  
A: 使用风险评分（加权组合），评分 < 0.4 为 Safe，>= 0.4 为 Risky。

**Q: 可选特征 distance_to_coast_m 什么时候添加？**  
A: 初期可不包含，后续如需评估应急撤离难度时再加入。

**Q: 如何处理缺失值？**  
A: 使用前向填充（forward-fill）或插值（interpolation），缺失率 < 5% 时可接受。

**Q: 多类分类何时启用？**  
A: 二分类模型训练稳定后，可扩展到 Open Water / Marginal Ice / Heavy Ice。

---

**最后更新**: 2025-12-11  
**版本**: 1.0



