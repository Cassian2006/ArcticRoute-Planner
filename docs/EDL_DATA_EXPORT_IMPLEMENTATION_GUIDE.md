# EDL 数据导出实现指南（E0.2 准备）

**文档版本**: 1.0  
**创建时间**: 2025-12-11  
**目标**: 为实现数据导出脚本提供详细的技术指南

---

## 1. 概述

本指南为 Phase EDL-0 的 E0.2 任务（数据导出脚本实现）提供技术支持。

**目标**: 从原始数据（AIS + 环境场）生成符合 schema 的 Parquet 训练集。

**输入**:
- AIS 原始数据（CSV / JSON）
- 环境数据（SIC、冰厚、波高）
- 船舶信息（等级、位置）

**输出**:
- `train_2024_2025.parquet`（50,000 样本）
- `val_2024_2025.parquet`（10,000 样本）
- `test_2024_2025.parquet`（10,000 样本）
- `metadata.json`（元数据）

---

## 2. 模块设计

### 2.1 模块划分

```
arcticroute/edl/
├── __init__.py
├── data_export.py          # 主导出脚本
├── feature_engineering.py  # 特征提取
├── label_generation.py     # 标签生成
├── data_validation.py      # 数据验证
└── utils.py                # 工具函数
```

### 2.2 核心类和函数

```python
# data_export.py
class EDLDatasetBuilder:
    """EDL 训练数据集构建器"""
    def __init__(self, config: dict):
        pass
    
    def load_ais_data(self) -> pd.DataFrame:
        """加载 AIS 数据"""
        pass
    
    def load_environmental_data(self) -> dict:
        """加载环境数据"""
        pass
    
    def build_dataset(self) -> pd.DataFrame:
        """构建完整数据集"""
        pass
    
    def split_dataset(self, dataset: pd.DataFrame) -> tuple:
        """分割为 train/val/test"""
        pass
    
    def export_to_parquet(self, output_dir: str) -> None:
        """导出为 Parquet 文件"""
        pass

# feature_engineering.py
def rasterize_ais_to_grid(
    ais_df: pd.DataFrame,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> np.ndarray:
    """将 AIS 点栅格化为密度场"""
    pass

def extract_environmental_features(
    env_data: dict,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
) -> pd.DataFrame:
    """提取环境特征"""
    pass

def add_temporal_features(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
) -> pd.DataFrame:
    """添加时间特征（month, dayofyear）"""
    pass

def add_vessel_features(
    df: pd.DataFrame,
    vessel_info: dict,
) -> pd.DataFrame:
    """添加船舶特征"""
    pass

# label_generation.py
def generate_safe_risky_labels(
    features: pd.DataFrame,
) -> pd.Series:
    """生成二分类标签"""
    pass

def compute_risk_score(
    features: pd.DataFrame,
) -> np.ndarray:
    """计算风险评分"""
    pass

# data_validation.py
def validate_features(df: pd.DataFrame) -> dict:
    """验证特征范围和类型"""
    pass

def validate_labels(labels: pd.Series) -> dict:
    """验证标签分布"""
    pass

def generate_validation_report(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> dict:
    """生成完整验证报告"""
    pass
```

---

## 3. 实现步骤

### Step 1: 配置管理

```python
# config.yaml 或 config.json
config = {
    "data_sources": {
        "ais_dir": "data_real/ais_raw",
        "env_dir": "data_real/environment",
        "vessel_info_file": "data_real/vessel_info.csv",
    },
    "output": {
        "output_dir": "data/edl_training",
        "compression": "snappy",
    },
    "grid": {
        "resolution": 0.5,  # 度
        "lat_min": 60,
        "lat_max": 85,
        "lon_min": -180,
        "lon_max": 180,
    },
    "date_range": {
        "start": "2024-01-01",
        "end": "2025-12-31",
    },
    "split": {
        "train_end": "2025-06-30",
        "val_end": "2025-09-30",
        "test_end": "2025-12-31",
    },
    "label_thresholds": {
        "sic_safe": 30,
        "sic_risky": 70,
        "ice_thickness_safe": 1.0,
        "ice_thickness_risky": 2.0,
        "wave_swh_safe": 4.0,
        "wave_swh_risky": 5.0,
        "ais_density_safe": 0.1,
        "ais_density_risky": 0.05,
    },
}
```

### Step 2: AIS 数据加载

```python
def load_ais_data(ais_dir: str, date_range: tuple) -> pd.DataFrame:
    """
    从 ais_dir 加载所有 AIS 文件，返回标准化 DataFrame。
    
    使用现有的 load_ais_from_raw_dir() 函数。
    """
    from arcticroute.core.ais_ingest import load_ais_from_raw_dir
    
    df = load_ais_from_raw_dir(
        ais_dir,
        time_min=pd.to_datetime(date_range[0]),
        time_max=pd.to_datetime(date_range[1]),
    )
    
    # 确保有必需列
    required_cols = ['mmsi', 'timestamp', 'lat', 'lon']
    assert all(col in df.columns for col in required_cols)
    
    return df
```

### Step 3: 环境数据加载

```python
def load_environmental_data(env_dir: str, date_range: tuple) -> dict:
    """
    加载环境数据（SIC、冰厚、波高）。
    
    假设环境数据已栅格化为 NetCDF 或 HDF5 格式。
    """
    import xarray as xr
    
    env_data = {}
    
    # 加载 SIC（海冰浓度）
    sic_files = sorted(Path(env_dir).glob("sic_*.nc"))
    if sic_files:
        ds_sic = xr.open_mfdataset(sic_files, combine='by_coords')
        env_data['sic'] = ds_sic['sic']  # (time, lat, lon)
    
    # 加载冰厚
    ice_thick_files = sorted(Path(env_dir).glob("ice_thickness_*.nc"))
    if ice_thick_files:
        ds_thick = xr.open_mfdataset(ice_thick_files, combine='by_coords')
        env_data['ice_thickness_m'] = ds_thick['ice_thickness_m']
    
    # 加载波高
    wave_files = sorted(Path(env_dir).glob("wave_swh_*.nc"))
    if wave_files:
        ds_wave = xr.open_mfdataset(wave_files, combine='by_coords')
        env_data['wave_swh'] = ds_wave['swh']
    
    return env_data
```

### Step 4: 特征工程

```python
def build_feature_matrix(
    ais_df: pd.DataFrame,
    env_data: dict,
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    vessel_info: dict,
) -> pd.DataFrame:
    """
    构建特征矩阵。
    """
    ny, nx = grid_lat.shape if grid_lat.ndim == 2 else (len(grid_lat), len(grid_lon))
    
    # 1. 栅格化 AIS
    ais_density = rasterize_ais_to_grid(ais_df, grid_lat, grid_lon)
    
    # 2. 提取环境特征
    features = pd.DataFrame({
        'lat': grid_lat.flatten() if grid_lat.ndim == 2 else np.repeat(grid_lat, nx),
        'lon': grid_lon.flatten() if grid_lon.ndim == 2 else np.tile(grid_lon, ny),
        'sic': env_data['sic'].values.flatten(),
        'ice_thickness_m': env_data['ice_thickness_m'].values.flatten(),
        'wave_swh': env_data['wave_swh'].values.flatten(),
        'ais_density': ais_density.flatten(),
    })
    
    # 3. 添加时间特征
    features = add_temporal_features(features)
    
    # 4. 添加船舶特征
    features = add_vessel_features(features, vessel_info)
    
    # 5. 数据清洗
    features = features.dropna()
    features = features[
        (features['lat'] >= 60) & (features['lat'] <= 85) &
        (features['lon'] >= -180) & (features['lon'] <= 180)
    ]
    
    return features
```

### Step 5: 标签生成

```python
def generate_labels(features: pd.DataFrame, thresholds: dict) -> pd.Series:
    """
    根据特征生成二分类标签。
    """
    # 安全条件
    safe_mask = (
        (features['sic'] < thresholds['sic_safe']) &
        (features['ice_thickness_m'] < thresholds['ice_thickness_safe']) &
        (features['wave_swh'] < thresholds['wave_swh_safe']) &
        (features['ais_density'] > thresholds['ais_density_safe'])
    )
    
    # 风险条件
    risky_mask = (
        (features['sic'] >= thresholds['sic_risky']) |
        (features['ice_thickness_m'] >= thresholds['ice_thickness_risky']) |
        (features['wave_swh'] >= thresholds['wave_swh_risky']) |
        (features['ais_density'] < thresholds['ais_density_risky'])
    )
    
    # 边界情况：风险评分
    risk_score = (
        0.3 * (features['sic'] / 100) +
        0.3 * (features['ice_thickness_m'] / 3) +
        0.2 * (features['wave_swh'] / 6) +
        0.2 * (1 - features['ais_density'])
    )
    
    labels = pd.Series(0, index=features.index, dtype='int8')
    labels[safe_mask] = 0
    labels[risky_mask] = 1
    
    # 边界情况
    boundary_mask = ~(safe_mask | risky_mask)
    labels[boundary_mask] = (risk_score[boundary_mask] >= 0.4).astype('int8')
    
    return labels
```

### Step 6: 数据分割

```python
def split_dataset(
    dataset: pd.DataFrame,
    split_dates: dict,
) -> tuple:
    """
    按时间分割为 train/val/test。
    """
    train_end = pd.to_datetime(split_dates['train_end'])
    val_end = pd.to_datetime(split_dates['val_end'])
    
    train = dataset[dataset['timestamp'] <= train_end]
    val = dataset[(dataset['timestamp'] > train_end) & (dataset['timestamp'] <= val_end)]
    test = dataset[dataset['timestamp'] > val_end]
    
    return train, val, test
```

### Step 7: 导出为 Parquet

```python
def export_to_parquet(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    导出为 Parquet 文件。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 选择列
    cols = [
        'lat', 'lon', 'month', 'dayofyear',
        'sic', 'ice_thickness_m', 'wave_swh', 'ais_density',
        'vessel_class_id', 'distance_to_coast_m',
        'label_safe_risky', 'timestamp'
    ]
    
    train[cols].to_parquet(
        output_path / 'train_2024_2025.parquet',
        compression='snappy',
        index=False,
    )
    val[cols].to_parquet(
        output_path / 'val_2024_2025.parquet',
        compression='snappy',
        index=False,
    )
    test[cols].to_parquet(
        output_path / 'test_2024_2025.parquet',
        compression='snappy',
        index=False,
    )
    
    print(f"✓ Exported to {output_path}")
```

### Step 8: 生成元数据

```python
def generate_metadata(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    output_dir: str,
) -> None:
    """
    生成 metadata.json。
    """
    metadata = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "dataset_name": "EDL_Training_2024_2025",
        "split_info": {
            "train": {
                "file": "train_2024_2025.parquet",
                "num_samples": len(train),
                "date_range": [
                    train['timestamp'].min().isoformat(),
                    train['timestamp'].max().isoformat(),
                ],
            },
            "val": {
                "file": "val_2024_2025.parquet",
                "num_samples": len(val),
                "date_range": [
                    val['timestamp'].min().isoformat(),
                    val['timestamp'].max().isoformat(),
                ],
            },
            "test": {
                "file": "test_2024_2025.parquet",
                "num_samples": len(test),
                "date_range": [
                    test['timestamp'].min().isoformat(),
                    test['timestamp'].max().isoformat(),
                ],
            },
        },
        "features": {
            "environmental": ["lat", "lon", "month", "dayofyear", "sic", "ice_thickness_m", "wave_swh", "ais_density"],
            "vessel": ["vessel_class_id", "distance_to_coast_m"],
            "total_dim": 10,
        },
        "targets": {
            "primary": "label_safe_risky",
            "classes": {"0": "Safe", "1": "Risky"},
            "class_distribution": {
                "train": {
                    "0": float((train['label_safe_risky'] == 0).sum() / len(train)),
                    "1": float((train['label_safe_risky'] == 1).sum() / len(train)),
                },
                "val": {
                    "0": float((val['label_safe_risky'] == 0).sum() / len(val)),
                    "1": float((val['label_safe_risky'] == 1).sum() / len(val)),
                },
                "test": {
                    "0": float((test['label_safe_risky'] == 0).sum() / len(test)),
                    "1": float((test['label_safe_risky'] == 1).sum() / len(test)),
                },
            },
        },
    }
    
    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
```

---

## 4. 主脚本框架

```python
# scripts/export_edl_training_data.py

import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from arcticroute.edl.data_export import EDLDatasetBuilder
from arcticroute.edl.data_validation import generate_validation_report

def main():
    # 1. 加载配置
    config_path = Path('configs/edl_data_export.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. 构建数据集
    builder = EDLDatasetBuilder(config)
    dataset = builder.build_dataset()
    
    # 3. 分割
    train, val, test = builder.split_dataset(dataset)
    
    # 4. 验证
    validation_report = generate_validation_report(train, val, test)
    print(json.dumps(validation_report, indent=2))
    
    # 5. 导出
    builder.export_to_parquet(
        train, val, test,
        output_dir=config['output']['output_dir'],
    )
    
    print("✓ EDL training dataset exported successfully!")

if __name__ == '__main__':
    main()
```

---

## 5. 关键考虑事项

### 5.1 内存管理
- 大数据集分批处理
- 使用 Dask 进行分布式处理
- 及时释放不需要的中间结果

### 5.2 数据质量
- 缺失值处理（插值或删除）
- 异常值检测和处理
- 数据类型验证

### 5.3 性能优化
- 使用 Parquet 压缩
- 列式存储减少 I/O
- 并行处理多个文件

### 5.4 可重现性
- 固定随机种子
- 记录所有参数
- 生成详细的元数据

---

## 6. 测试计划

### 6.1 单元测试
```python
def test_rasterize_ais_to_grid():
    """测试 AIS 栅格化"""
    pass

def test_generate_labels():
    """测试标签生成"""
    pass

def test_feature_ranges():
    """测试特征范围"""
    pass
```

### 6.2 集成测试
```python
def test_end_to_end_pipeline():
    """测试完整流程"""
    pass

def test_parquet_export():
    """测试 Parquet 导出"""
    pass
```

### 6.3 数据验证
```python
def test_label_distribution():
    """验证标签分布"""
    pass

def test_no_data_leakage():
    """验证无数据泄露"""
    pass
```

---

## 7. 故障排查

### 常见问题

**Q: 内存溢出**  
A: 使用 Dask 或分批处理，减少一次性加载的数据量。

**Q: 缺失值过多**  
A: 检查数据源，可能需要调整插值策略或扩大数据范围。

**Q: 标签分布不均衡**  
A: 使用加权损失函数或过采样/欠采样技术。

**Q: 导出速度慢**  
A: 使用更高的压缩级别或并行导出。

---

## 8. 后续优化

- [ ] 支持增量更新（新数据追加）
- [ ] 支持多个时间范围的数据集
- [ ] 支持自定义特征和标签
- [ ] 支持数据增强（时间/空间平移）
- [ ] 支持实时数据流

---

**相关文档**:
- `docs/EDL_TRAINING_DATA_DESIGN.md` - 完整 schema 设计
- `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md` - 快速参考

**下一步**: 实现 E0.2 数据导出脚本





