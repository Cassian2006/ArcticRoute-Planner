# ArcticRoute Final (AR_final)

ArcticRoute 的重构与精简版本，致力于打造一个干净、可长期维护的北极航线规划系统。

## 项目状态

**当前阶段：Phase 0（最小骨架）**

- ✅ 项目目录结构搭建
- ✅ 最小 Streamlit UI（参数输入与回显）
- ✅ 核心模块占位（grid、landmask、cost、astar、eco）
- ⏳ 后续阶段：逐步迁移旧项目的成熟功能

## 项目结构

```
AR_final/
├── arcticroute/                # 主包
│   ├── __init__.py
│   ├── core/                   # 核心功能模块
│   │   ├── __init__.py
│   │   ├── grid.py             # 网格与坐标系工具
│   │   ├── landmask.py         # 陆地掩码加载与质量检查
│   │   ├── cost.py             # 成本构建逻辑
│   │   ├── astar.py            # A* 路由算法
│   │   └── eco/                # ECO 能耗模块
│   │       ├── __init__.py
│   │       ├── eco_model.py    # 简化 ECO 模型
│   │       └── vessel_profiles.py  # 船舶配置
│   └── ui/                     # 前端 UI 模块
│       ├── __init__.py
│       └── planner_minimal.py  # 极简规划器 UI
├── tests/                      # 测试包
│   ├── __init__.py
│   └── test_smoke_import.py    # 烟雾测试
├── data_sample/                # 样本数据目录（暂空）
├── run_ui.py                   # Streamlit 入口
├── requirements.txt            # 依赖清单
├── .gitignore                  # Git 忽略规则
└── README.md                   # 本文件
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动 UI

```bash
streamlit run run_ui.py
```

浏览器会自动打开 `http://localhost:8501`，你将看到：
- 标题：**ArcticRoute Planner (minimal skeleton)**
- 左侧参数输入：环境时间 (ym)、起终点坐标
- 主区域：参数回显（JSON 格式）

### 3. 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行多源回归测试套件
pytest -q tests/test_cost_multisource_sensitivity.py \
       tests/test_cost_sit_drift_effect.py \
       tests/test_planner_selection_traceability.py
```

**多源回归测试说明**：
- `test_cost_multisource_sensitivity.py`: 验证 AIS、浅水、POLARIS 等多源数据对成本的影响
- `test_cost_sit_drift_effect.py`: 验证海冰厚度（SIT）、漂移（Drift）对成本的影响及缺失数据处理
- `test_planner_selection_traceability.py`: 验证规划器选择的可追溯性和回退机制

这些测试使用真实的 vessel profiles 和黑盒测试方式（通过 demo_end_to_end 脚本），确保系统的稳定性和可追溯性。

## 开发计划

### Phase 0（当前）
- [x] 项目骨架搭建
- [x] 最小 UI 实现
- [x] 包结构与导入测试

### Phase 1（后续）
- [ ] 网格初始化与坐标系工具
- [ ] 陆地掩码加载与验证
- [ ] 成本网格构建

### Phase 2（后续）
- [ ] A* 寻路算法实现
- [ ] ECO 能耗模型集成
- [ ] 路由结果可视化

### Phase 3（后续）
- [ ] 性能优化
- [ ] 完整功能测试
- [ ] 文档完善

## 注意事项

- **暂时不引入复杂依赖**：后续根据需要逐步添加 geopandas、shapely 等重依赖
- **不迁移旧项目复杂逻辑**：Phase 0 只做占位，后续分阶段迁移
- **保持代码简洁**：优先考虑可维护性和可读性

## 贡献指南

在提交代码前，请：
1. 运行 `pytest tests/` 确保测试通过
2. 遵循项目的目录结构与命名规范
3. 为新功能添加相应的测试用例

## 许可证

待定

---

**最后更新**：2024-12-14
