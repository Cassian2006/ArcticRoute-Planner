# Phase EDL-0 文档索引

**阶段**: EDL-0（EDL 训练准备 - 工程向）  
**目标**: 搭建训练数据出口 + 最小训练闭环 + 快速评估  
**状态**: 进行中（E0.1 ✅ 完成）  

---

## 📑 文档导航

### 🎯 任务总结

| 任务 | 文档 | 状态 | 说明 |
|------|------|------|------|
| E0.1 | `PHASE_EDL0_E0.1_SUMMARY.md` | ✅ 完成 | 任务总结和完成报告 |
| E0.1 | `PHASE_EDL0_TASK_E0.1_COMPLETION.md` | ✅ 完成 | 详细完成报告 |
| E0.2 | - | ⏳ 待开始 | 数据导出脚本实现 |
| E0.3 | - | ⏳ 待开始 | 最小训练闭环 |
| E0.4 | - | ⏳ 待开始 | 快速评估工具 |

### 📚 核心设计文档

#### 1. **EDL 训练数据 Schema 设计** (550+ 行)
📄 `docs/EDL_TRAINING_DATA_DESIGN.md`

**内容**:
- 输入特征定义（10 维）
- 输出标签定义（二分类 + 多类扩展）
- 文件格式规范（Parquet）
- 元数据规范（metadata.json）
- 数据生成流程
- 质量检查清单
- 后续扩展方向

**适用场景**:
- 了解完整的 schema 设计
- 理解特征和标签的含义
- 学习数据生成流程
- 参考数据质量检查清单

#### 2. **EDL 训练数据快速参考** (250+ 行)
📄 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`

**内容**:
- 特征速查表
- 标签速查表
- 文件格式速查
- 伪代码框架
- 常见问题

**适用场景**:
- 快速查阅特征范围和类型
- 查看标签定义规则
- 参考伪代码实现
- 解答常见问题

#### 3. **EDL 数据导出实现指南** (400+ 行)
📄 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`

**内容**:
- 模块设计
- 实现步骤（8 步）
- 代码框架
- 配置管理
- 测试计划
- 故障排查

**适用场景**:
- 实现 E0.2 数据导出脚本
- 学习模块设计方法
- 参考代码框架
- 编写单元测试

### 📋 任务报告

#### 1. **E0.1 任务总结**
📄 `PHASE_EDL0_E0.1_SUMMARY.md`

**内容**:
- 任务目标
- 交付物清单
- 关键设计决策
- 设计规范详情
- 数据生成流程
- 质量检查清单
- 后续任务
- 设计亮点
- 项目进度

**适用场景**:
- 了解 E0.1 任务的完整成果
- 查看设计决策的理由
- 了解后续任务计划

#### 2. **E0.1 完成报告**
📄 `PHASE_EDL0_TASK_E0.1_COMPLETION.md`

**内容**:
- 任务概述
- 设计内容总结
- 设计亮点
- 文档结构
- 后续步骤
- 关键决策
- 相关文件

**适用场景**:
- 快速了解 E0.1 任务成果
- 查看设计亮点
- 了解后续步骤

---

## 🗂️ 文件结构

```
项目根目录/
├── docs/
│   ├── EDL_TRAINING_DATA_DESIGN.md              ← 完整 schema 设计
│   ├── EDL_TRAINING_DATA_QUICK_REFERENCE.md     ← 快速参考
│   ├── EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md  ← 实现指南
│   └── [其他文档...]
├── PHASE_EDL0_INDEX.md                          ← 本文件
├── PHASE_EDL0_E0.1_SUMMARY.md                   ← E0.1 总结
├── PHASE_EDL0_TASK_E0.1_COMPLETION.md           ← E0.1 完成报告
├── arcticroute/
│   ├── core/
│   │   └── ais_ingest.py                        ← AIS 数据摄取
│   ├── edl/
│   │   ├── __init__.py
│   │   ├── data_export.py                       ← [待实现] 数据导出
│   │   ├── feature_engineering.py               ← [待实现] 特征工程
│   │   ├── label_generation.py                  ← [待实现] 标签生成
│   │   ├── data_validation.py                   ← [待实现] 数据验证
│   │   └── utils.py                             ← [待实现] 工具函数
│   └── [其他模块...]
├── scripts/
│   └── export_edl_training_data.py              ← [待实现] 导出脚本
├── configs/
│   └── edl_data_export.yaml                     ← [待实现] 配置文件
└── data/
    └── edl_training/
        ├── train_2024_2025.parquet              ← [待生成]
        ├── val_2024_2025.parquet                ← [待生成]
        ├── test_2024_2025.parquet               ← [待生成]
        └── metadata.json                        ← [待生成]
```

---

## 🎯 快速导航

### 我想...

#### 了解 EDL 训练数据的完整设计
→ 阅读 `docs/EDL_TRAINING_DATA_DESIGN.md`

#### 快速查阅特征和标签定义
→ 查看 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`

#### 实现数据导出脚本
→ 参考 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`

#### 了解 E0.1 任务的完成情况
→ 阅读 `PHASE_EDL0_E0.1_SUMMARY.md`

#### 查看数据生成流程
→ 参考 `docs/EDL_TRAINING_DATA_DESIGN.md` 第 5 章

#### 检查数据质量
→ 参考 `docs/EDL_TRAINING_DATA_DESIGN.md` 第 6 章

#### 了解后续任务计划
→ 查看 `PHASE_EDL0_E0.1_SUMMARY.md` 的"后续任务"部分

---

## 📊 设计概览

### 输入特征（10 维）

```
环保特征（8 维）:
├── lat, lon              (地理坐标)
├── month, dayofyear      (时间特征)
├── sic                   (海冰浓度)
├── ice_thickness_m       (冰厚)
├── wave_swh              (波高)
└── ais_density           (AIS 密度)

船舶特征（2 维）:
├── vessel_class_id       (船舶等级)
└── distance_to_coast_m   (到海岸距离，可选)
```

### 输出标签

```
二分类（现阶段）:
├── 0 = Safe（安全）
└── 1 = Risky（风险）

多类（扩展）:
├── 0 = Open Water（开阔水域）
├── 1 = Marginal Ice Zone（边际冰区）
└── 2 = Heavy Ice（密集冰区）
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

## 🚀 后续任务

### E0.2: 实现数据导出脚本
**状态**: ⏳ 待开始  
**预期产出**:
- `arcticroute/edl/data_export.py`
- `arcticroute/edl/feature_engineering.py`
- `arcticroute/edl/label_generation.py`
- `arcticroute/edl/data_validation.py`
- `scripts/export_edl_training_data.py`
- `configs/edl_data_export.yaml`

**关键步骤**:
1. 实现 `EDLDatasetBuilder` 类
2. 实现特征工程模块
3. 实现标签生成逻辑
4. 实现数据验证和导出
5. 编写单元测试和集成测试

### E0.3: 建立最小训练闭环
**状态**: ⏳ 待开始  
**预期产出**:
- Parquet 数据加载器
- EDL 模型（PyTorch）
- 训练脚本
- 评估指标
- 训练日志和可视化

### E0.4: 快速评估工具
**状态**: ⏳ 待开始  
**预期产出**:
- 数据质量检查工具
- 模型性能评估工具
- 训练日志分析
- 可视化工具

---

## 📈 项目进度

```
Phase EDL-0: EDL 训练准备（工程向）
├── E0.1: 定义 EDL 训练数据 schema
│   ├── ✅ 输入特征定义
│   ├── ✅ 输出标签定义
│   ├── ✅ 文件格式规范
│   ├── ✅ 数据生成流程
│   └── ✅ 质量检查清单
├── E0.2: 实现数据导出脚本
│   ├── ⏳ 特征工程模块
│   ├── ⏳ 标签生成模块
│   ├── ⏳ 数据验证模块
│   └── ⏳ 导出脚本
├── E0.3: 建立最小训练闭环
│   ├── ⏳ 数据加载器
│   ├── ⏳ EDL 模型
│   ├── ⏳ 训练脚本
│   └── ⏳ 评估指标
└── E0.4: 快速评估工具
    ├── ⏳ 数据质量检查
    ├── ⏳ 模型性能评估
    ├── ⏳ 训练日志分析
    └── ⏳ 可视化工具
```

---

## 💡 关键设计决策

### 1. 特征选择
- ✅ 10 维特征（8 环保 + 2 船舶）
- ✅ 覆盖地理、时间、环境、船舶信息
- ✅ 支持后续扩展

### 2. 标签设计
- ✅ 二分类优先（Safe / Risky）
- ✅ 规则清晰，易于验证
- ✅ 支持平滑扩展到多类

### 3. 文件格式
- ✅ Parquet（高效、可扩展）
- ✅ 按时间分割（避免数据泄露）
- ✅ 完整的元数据规范

### 4. 数据质量
- ✅ 详细的检查清单
- ✅ 特征范围验证
- ✅ 标签分布检查

---

## 📚 相关资源

### 数据源
- NSIDC 海冰数据: https://nsidc.org/
- OSISAF 海冰产品: https://www.osisaf.org/
- ERA5 气象数据: https://cds.climate.copernicus.eu/

### 技术文档
- Parquet 格式: https://parquet.apache.org/
- Pandas Parquet: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_parquet.html
- PyTorch: https://pytorch.org/

### 相关论文
- Evidential Deep Learning: [待补充]
- Arctic Route Planning: [待补充]

---

## ✅ 检查清单

### E0.1 完成情况
- [x] 定义输入特征（10 维）
- [x] 定义输出标签（二分类 + 多类）
- [x] 定义文件格式（Parquet）
- [x] 定义元数据规范
- [x] 编写数据生成流程
- [x] 编写质量检查清单
- [x] 编写实现指南
- [x] 编写快速参考
- [x] 编写任务报告

### E0.2 准备情况
- [x] 实现指南已完成
- [x] 代码框架已准备
- [x] 配置模板已定义
- [ ] 代码实现（待开始）
- [ ] 单元测试（待开始）
- [ ] 集成测试（待开始）

---

## 📝 版本历史

| 版本 | 日期 | 变更 |
|------|------|------|
| 1.0 | 2025-12-11 | 初始版本，E0.1 完成 |

---

## 🎓 学习路径

### 初级（了解设计）
1. 阅读 `PHASE_EDL0_E0.1_SUMMARY.md`
2. 查看 `docs/EDL_TRAINING_DATA_QUICK_REFERENCE.md`
3. 浏览 `docs/EDL_TRAINING_DATA_DESIGN.md` 的概述部分

### 中级（理解细节）
1. 详读 `docs/EDL_TRAINING_DATA_DESIGN.md`
2. 学习数据生成流程（第 5 章）
3. 理解质量检查清单（第 6 章）

### 高级（实现代码）
1. 阅读 `docs/EDL_DATA_EXPORT_IMPLEMENTATION_GUIDE.md`
2. 参考代码框架和伪代码
3. 实现 E0.2 数据导出脚本

---

## 🔗 相关链接

### 内部文档
- [AIS 数据摄取](arcticroute/core/ais_ingest.py)
- [项目 README](README.md)

### 外部资源
- [Parquet 官网](https://parquet.apache.org/)
- [Pandas 文档](https://pandas.pydata.org/)
- [NumPy 文档](https://numpy.org/)

---

**最后更新**: 2025-12-11  
**维护者**: Cascade AI  
**状态**: ✅ E0.1 完成，E0.2 待开始



