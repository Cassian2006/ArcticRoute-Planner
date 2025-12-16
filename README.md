# ArcticRoute-Planner (AR_final)

北极航线智能规划系统：在 **SIC（海冰）/ SWH（风浪）/ AIS 拥挤度 / EDL 风险与不确定性 / 规则约束（POLARIS/Polar Code）** 等多源信息约束下，生成可解释的候选航线，并提供 UI 诊断与演示闭环。

## 主要能力
- **多模态成本场**：SIC / SWH / AIS density（含选择、匹配、重采样、缓存与提示）
- **规则与阈值框架**：可配置 hard-block + soft-penalty（已集成 **POLARIS RIO**）
- **近实时数据**：支持从 **Copernicus Marine (CMEMS)** 拉取 NetCDF，并摄入到环境层
- **可解释性**：成本分解、沿程诊断（RIO / level / speed_limit）、回退提示与诊断元数据
- **规划内核可切换**：默认 A*；可选 PolarRoute / pipeline（不装也不影响主流程）

## 一键运行 UI
```bash
pip install -r requirements.txt
streamlit run run_ui.py
```

## CMEMS 近实时数据（可选）

支持 cmems_latest → newenv → load_environment(use_newenv_for_cost=True) 的无侵入接线

离线也能跑：若本地已有缓存 nc，则优先使用；若下载失败会清晰提示并回退到 archive/demo

常用命令：

```bash
python -m scripts.phase9_quick_check
python -m scripts.phase9_validation
python -m scripts.cmems_refresh_and_export --days 5 --bbox -40 60 65 85
python -m scripts.cmems_newenv_sync
```

## POLARIS（RIO）解释面板

在 UI 的规划结果区，打开：

🧊 POLARIS 沿程解释（RIO/等级/建议航速）

全局统计：rio_min / rio_mean / special_fraction / elevated_fraction / riv_table_used

沿程表：RIO / level / speed_limit

分段聚合：每 10 个采样点（可选）

## 开发与测试（门禁）

只跑主门禁（不含 legacy）：

```bash
python -m pytest -q tests
```

演示闭环（离线兜底）：

```bash
python -m scripts.demo_end_to_end --outdir reports/demo_run
```

## 常见问题（Windows 导入污染）

如果出现 ModuleNotFoundError、导入跑到旧工程路径、或"null bytes"等异常，优先检查并清理：

- 仓库根是否存在 ArcticRoute/（大写目录）
- 仓库根是否存在 arcticroute.py（会 shadow package）
- 清理 __pycache__/

推荐先运行：

```bash
python -m scripts.import_sanity_check
```

## License / 引用

待定

---

## 附录（历史版本说明）

以下为历史版本的项目说明（保留以供参考）：

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
pytest tests/
```

验证包结构与基本导入功能。

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
