# Phase EDL-0 快速启动指南

**阶段**: EDL-0（EDL 训练准备）  
**当前任务**: E0.1 ✅ 完成 | E0.2 ⏳ 待开始  
**最后更新**: 2025-12-11

---

## [object Object] 分钟快速了解

### 什么是 Phase EDL-0？

EDL-0 是 EDL（Evidential Deep Learning）模型的训练准备阶段，目标是：
- ✅ 定义训练数据 schema（E0.1 已完成）
- ⏳ 实现数据导出脚本（E0.2）
- ⏳ 建立最小训练闭环（E0.3）
- ⏳ 快速评估工具（E0.4）

### E0.1 完成了什么？

定义了 EDL 模型的完整训练数据规范：

```
输入特征（10 维）
├── 环保特征（8 维）: lat, lon, month, dayofyear, sic, ice_thickness_m, wave_swh, ais_density
└── 船舶特征（2 维）: vessel_class_id, distance_to_coast_m

输出标签
├── 二分类（现阶段）: Safe / Risky
└── 多类（扩展）: Open Water / Marginal Ice / Heavy Ice

文件格式: Parquet（高效、可扩展）
```

---

## 📚 文档速查

### 我想快速了解设计
👉 阅读 **`PHASE_EDL0_E0.1_SUMMARY.md`**（5 分钟）

### 我想查阅特征和标签定义
👉 查看 **`docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`**（3 分钟）

### 我想了解完整的设计细节
👉 阅读 **`docs/EDL_TRAINING_DATA_DESIGN.md`**（20 分钟）

### 我想实现数据导出脚本（E0.2）
👉 参考 **`docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`**（30 分钟）

### 我想快速找到相关文档
👉 查看 **`PHASE_EDL0_INDEX.md`**（导航）

---

## [object Object]0.1 核心内容速览

### 输入特征（10 维）

| 特征 | 范围 | 单位 | 说明 |
|------|------|------|------|
| lat | [-90, 90] | 度 | 纬度 |
| lon | [-180, 180] | 度 | 经度 |
| month | [1, 12] | - | 月份 |
| dayofyear | [1, 366] | - | 一年中的第几天 |
| sic | [0, 100] | % | 海冰浓度 |
| ice_thickness_m | [0, 5] | 米 | 冰厚 |
| wave_swh | [0, 15] | 米 | 波高 |
| ais_density | [0, 1] | 归一化 | AIS 密度 |
| vessel_class_id | [0, 2] | - | 船舶等级 |
| distance_to_coast_m | [0, ∞) | 米 | 到海岸距离（可选） |

### 输出标签

```python
# 二分类（现阶段）
label_safe_risky: {0, 1}
  0 = Safe（安全）
  1 = Risky（风险）

# 标签规则
Safe:  sic < 30% AND ice_thick < 1m AND wave_swh < 4m AND ais_density > 0.1
Risky: sic >= 70% OR ice_thick >= 2m OR wave_swh >= 5m OR ais_density < 0.05
边界:  使用风险评分判定
```

### 文件格式

```
格式: Parquet（列式存储）
压缩: Snappy
文件:
├── train_2024_2025.parquet    (50,000 样本)
├── val_2024_2025.parquet      (10,000 样本)
├── test_2024_2025.parquet     (10,000 样本)
└── metadata.json              (元数据)
```

---

## 🔧 后续任务（E0.2-E0.4）

### E0.2: 实现数据导出脚本（⏳ 待开始）

**目标**: 从原始数据生成 Parquet 训练集

**关键步骤**:
1. 实现 `EDLDatasetBuilder` 类
2. 实现特征工程模块
3. 实现标签生成逻辑
4. 实现数据验证和导出
5. 编写单元测试

**预期输出**:
```
arcticroute/edl/
├── data_export.py
├── feature_engineering.py
├── label_generation.py
├── data_validation.py
└── utils.py

scripts/
└── export_edl_training_data.py

configs/
└── edl_data_export.yaml

data/edl_training/
├── train_2024_2025.parquet
├── val_2024_2025.parquet
├── test_2024_2025.parquet
└── metadata.json
```

**参考资源**:
- `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md` - 详细实现指南
- `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md` - 快速参考

### E0.3: 建立最小训练闭环（⏳ 待开始）

**目标**: 数据加载 → 模型训练 → 评估

**关键步骤**:
1. 实现 Parquet 数据加载器
2. 实现 EDL 模型（PyTorch）
3. 实现训练脚本
4. 实现评估指标
5. 编写训练日志和可视化

### E0.4: 快速评估工具（⏳ 待开始）

**目标**: 验证数据质量和模型初步性能

**关键步骤**:
1. 数据质量检查工具
2. 模型性能评估工具
3. 训练日志分析
4. 可视化工具

---

## 📖 学习路径

### 初级（了解设计）- 15 分钟
1. 阅读本文件（5 分钟）
2. 阅读 `PHASE_EDL0_E0.1_SUMMARY.md`（5 分钟）
3. 查看 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`（5 分钟）

### 中级（理解细节）- 45 分钟
1. 详读 `docs/EDL_TRAINING_DATA_DESIGN.md`（30 分钟）
2. 学习数据生成流程（10 分钟）
3. 理解质量检查清单（5 分钟）

### 高级（实现代码）- 2 小时
1. 阅读 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`（30 分钟）
2. 参考代码框架和伪代码（30 分钟）
3. 实现 E0.2 数据导出脚本（60 分钟）

---

## 🎓 关键概念

### 为什么选择这些特征？

```
地理特征 (lat, lon)
  → 位置决定环境条件

时间特征 (month, dayofyear)
  → 季节性变化影响冰况和海况

环保特征 (sic, ice_thickness_m, wave_swh)
  → 直接影响航行安全

AIS 密度 (ais_density)
  → 参考轨迹的可用性

船舶特征 (vessel_class_id)
  → 不同船舶的能力差异
```

### 为什么选择二分类？

```
优势:
  ✅ 简单明确，易于初期验证
  ✅ 规则清晰，可解释性强
  ✅ 数据需求少，快速迭代
  ✅ 支持平滑扩展到多类

扩展路径:
  二分类 (Safe/Risky)
    ↓
  多类 (Open Water / Marginal Ice / Heavy Ice)
    ↓
  连续标签 (风险评分 ∈ [0, 1])
    ↓
  不确定性标签 (Dirichlet 分布，用于 EDL)
```

### 为什么选择 Parquet？

```
相比 CSV:
  ✅ 压缩率高（节省 50-80% 空间）
  ✅ 读取速度快（列式存储）
  ✅ 支持分布式处理（Spark、Dask）
  ✅ 保留数据类型（无需再次转换）
  ✅ 生态完善（工具支持好）
```

---

## 💡 常见问题

### Q: 为什么 distance_to_coast_m 是可选的？
A: 初期可能不需要，后续如需评估应急撤离难度时再加入。

### Q: 标签如何处理边界情况？
A: 使用风险评分（加权组合），评分 < 0.4 为 Safe，>= 0.4 为 Risky。

### Q: 如何处理缺失值？
A: 使用前向填充（forward-fill）或插值（interpolation），缺失率 < 5% 时可接受。

### Q: 多类分类何时启用？
A: 二分类模型训练稳定后，可扩展到 Open Water / Marginal Ice / Heavy Ice。

### Q: 如何避免数据泄露？
A: 按时间顺序分割数据，训练集 < 验证集 < 测试集。

### Q: Parquet 文件如何读取？
A: 使用 `pd.read_parquet('file.parquet')`，简单易用。

---

## 🔗 相关文件

### 核心设计文档
- `docs/EDL_TRAINING_DATA_DESIGN.md` - 完整 schema 设计
- `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md` - 快速参考
- `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md` - 实现指南

### 任务报告
- `PHASE_EDL0_E0.1_SUMMARY.md` - E0.1 总结
- `PHASE_EDL0_TASK_E0.1_COMPLETION.md` - E0.1 完成报告
- `PHASE_EDL0_INDEX.md` - 文档导航

### 代码参考
- `arcticroute/core/ais_ingest.py` - AIS 数据摄取（已有）
- `arcticroute/edl/` - EDL 模块（待实现）

---

## 📊 项目进度

```
Phase EDL-0: EDL 训练准备
├── E0.1: 定义 EDL 训练数据 schema          ✅ 完成
│   ├── 输入特征定义                        ✅
│   ├── 输出标签定义                        ✅
│   ├── 文件格式规范                        ✅
│   ├── 数据生成流程                        ✅
│   └── 质量检查清单                        ✅
├── E0.2: 实现数据导出脚本                  ⏳ 待开始
├── E0.3: 建立最小训练闭环                  ⏳ 待开始
└── E0.4: 快速评估工具                      ⏳ 待开始
```

---

## ✅ 检查清单

### 了解 E0.1 成果
- [ ] 阅读 `PHASE_EDL0_E0.1_SUMMARY.md`
- [ ] 查看 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`
- [ ] 浏览 `docs/EDL_TRAINING_DATA_DESIGN.md`

### 准备 E0.2 实现
- [ ] 阅读 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`
- [ ] 理解模块设计
- [ ] 准备开发环境

### 开始 E0.2 实现
- [ ] 创建 `arcticroute/edl/` 模块
- [ ] 实现 `data_export.py`
- [ ] 实现 `feature_engineering.py`
- [ ] 实现 `label_generation.py`
- [ ] 实现 `data_validation.py`
- [ ] 编写单元测试
- [ ] 编写集成测试

---

## 🎯 下一步

### 立即开始（E0.2）
1. 阅读 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`
2. 创建 `arcticroute/edl/` 模块结构
3. 实现数据导出脚本

### 参考资源
- 完整设计: `docs/EDL_TRAINING_DATA_DESIGN.md`
- 快速参考: `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`
- 实现指南: `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`

### 获取帮助
- 查看 `PHASE_EDL0_INDEX.md` 了解文档结构
- 查看 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md` 的常见问题
- 参考 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md` 的故障排查

---

## 📞 联系方式

有任何问题或建议，请参考：
- 文档导航: `PHASE_EDL0_INDEX.md`
- 常见问题: `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`
- 故障排查: `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`

---

**最后更新**: 2025-12-11  
**状态**: ✅ E0.1 完成，E0.2 待开始  
**下一步**: 实现数据导出脚本（E0.2）





