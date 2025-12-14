# 仓库检查 - 快速查看指南

## 🎯 核心应用入口

### Streamlit UI 应用
**主应用**: `arcticroute/ui/planner_minimal.py` (2,631 行)
- 完整的 Streamlit 应用
- 包含路由规划、成本计算、结果展示等功能

**运行方式**:
```bash
streamlit run arcticroute/ui/planner_minimal.py
```

**相关页面**:
- `arcticroute/ui/home.py` - 首页
- `arcticroute/ui/eval_results.py` - 评估结果
- `arcticroute/ui/components/pipeline_flow.py` - 管道流程
- `arcticroute/ui/components/pipeline_timeline.py` - 时间线

---

## 📚 核心模块概览

### 1. 成本计算 (`arcticroute/core/cost.py`) - 1,581 行
**主要功能**:
- 计算路由成本（燃料、时间、冰级等）
- AIS 密度集成
- EDL（Evidential Deep Learning）成本模型
- 网格签名匹配和自动重采样

**关键函数**:
- `compute_cost()` - 计算单条路由的成本
- `compute_grid_signature()` - 生成网格唯一签名
- `discover_ais_density_candidates()` - 发现 AIS 密度文件
- `load_ais_density_for_grid()` - 加载 AIS 密度（支持自动重采样）
- `_regrid_ais_density_to_grid()` - 重采样 AIS 数据

### 2. 路径规划 (`arcticroute/core/astar.py`) - 304 行
**主要功能**:
- A* 算法实现
- 网格上的最优路径搜索
- 启发式函数优化

**关键函数**:
- `astar_plan()` - 执行 A* 规划

### 3. AIS 数据摄取 (`arcticroute/core/ais_ingest.py`) - 746 行
**主要功能**:
- 解析 AIS 消息
- 生成 AIS 密度网格
- 数据验证和清理

**关键函数**:
- `ingest_ais_messages()` - 摄取 AIS 消息
- `rasterize_ais_to_density()` - 栅格化 AIS 数据

### 4. 环境模型 (`arcticroute/core/env_real.py`) - 514 行
**主要功能**:
- 加载真实环境数据
- 冰况、风速、洋流等环境因素
- 网格管理

**关键函数**:
- `load_real_env()` - 加载真实环境
- `get_ice_class_cost()` - 获取冰级成本

### 5. 生态模型 (`arcticroute/core/eco/vessel_profiles.py`) - 541 行
**主要功能**:
- 船舶性能配置
- 多目标优化（燃料、时间、排放等）
- 船舶特性参数

**关键类**:
- `VesselProfile` - 船舶配置类

### 6. 机器学习 (`arcticroute/ml/edl_core.py`) - 233 行
**主要功能**:
- Evidential Deep Learning 模型
- 不确定性估计
- 成本预测

---

## 🧪 测试框架

**测试目录**: `tests/` (50+ 个测试文件)

### 关键测试
| 测试文件 | 功能 |
|---------|------|
| `test_cost_real_env_edl.py` | EDL 成本计算 |
| `test_edl_mode_strength.py` | EDL 模式强度 |
| `test_ais_density_rasterize.py` | AIS 栅格化 |
| `test_real_env_cost.py` | 真实环境成本 |
| `test_route_scoring.py` | 路由评分 |
| `test_vessel_profiles.py` | 船舶配置 |

**运行所有测试**:
```bash
pytest tests/ -v
```

**运行特定测试**:
```bash
pytest tests/test_cost_real_env_edl.py -v
```

---

## 🛠️ 实用脚本

### 数据处理
- `scripts/export_edl_training_dataset.py` - 导出 EDL 训练数据
- `scripts/preprocess_ais_to_density.py` - 预处理 AIS 数据
- `scripts/inspect_ais_json.py` - 检查 AIS JSON 数据

### 模型训练
- `scripts/edl_train_torch.py` - EDL 模型训练
- `scripts/calibrate_env_exponents.py` - 校准环境指数
- `scripts/fit_speed_exponents.py` - 拟合速度指数

### 场景和评估
- `scripts/run_scenario_suite.py` - 运行场景套件
- `scripts/run_edl_sensitivity_study.py` - EDL 敏感性分析
- `scripts/eval_scenario_results.py` - 评估场景结果

### 系统检查
- `scripts/system_health_check.py` - 系统健康检查
- `scripts/check_real_edl_task.py` - 检查真实 EDL 任务

---

## 📋 配置文件

### 场景配置 (`configs/scenarios.yaml`) - 114 行
定义路由规划场景：
- 起点、终点、船舶类型
- 成本权重（燃料、时间等）
- 环境参数

### 船舶配置 (`configs/vessel_profiles.yaml`) - 301 行
定义船舶性能参数：
- 燃料消耗率
- 速度范围
- 冰级能力

### EDL 配置
- `configs/edl_train.yaml` - 训练参数
- `configs/edl_dataset.yaml` - 数据集参数

---

## 📊 数据流

```
AIS 原始数据
    ↓
ais_ingest.py (摄取和栅格化)
    ↓
AIS 密度网格 (NetCDF)
    ↓
cost.py (加载和重采样)
    ↓
路由成本计算
    ↓
planner_minimal.py (UI 展示)
```

---

## 🔧 开发工作流

### 1. 添加新的成本因素
编辑 `arcticroute/core/cost.py`:
```python
def compute_cost(route, grid, **kwargs):
    # 添加新的成本分量
    new_cost = compute_new_factor(route, grid)
    total_cost += new_cost
    return total_cost
```

### 2. 添加新的环境数据
编辑 `arcticroute/core/env_real.py`:
```python
def load_real_env():
    # 加载新的环境变量
    new_data = load_new_environmental_factor()
    return env_with_new_data
```

### 3. 添加新的 UI 页面
在 `arcticroute/ui/` 创建新文件:
```python
import streamlit as st

def show_new_page():
    st.title("New Page")
    # 页面内容
```

### 4. 运行测试
```bash
pytest tests/test_your_feature.py -v
```

---

## 📈 项目统计

| 类别 | 数量 |
|------|------|
| Python 文件 | 100+ |
| 测试文件 | 50+ |
| 脚本工具 | 30+ |
| 文档文件 | 100+ |
| 总行数 (Python) | 15,000+ |

---

## 🚀 快速启动

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 运行 UI
```bash
streamlit run arcticroute/ui/planner_minimal.py
```

### 3. 运行测试
```bash
pytest tests/ -v
```

### 4. 生成报告
```bash
python scripts/repo_inspect.py
```

---

## 📖 文档资源

### 实现总结
- `AIS_GRID_SIGNATURE_IMPLEMENTATION_SUMMARY.md` - 网格签名实现
- `VESSEL_PROFILES_DOCUMENTATION.md` - 船舶配置文档
- `PYTORCH_EDL_FIX_SUMMARY.md` - PyTorch EDL 修复

### 快速参考
- `AIS_GRID_SIGNATURE_QUICK_REFERENCE.md` - 网格签名快速参考
- `QUICK_REFERENCE.md` - 项目快速参考
- `PHASE_4_QUICK_REFERENCE.md` - 第 4 阶段快速参考

### 阶段报告
- `PHASE_EVAL_1_START_HERE.md` - 评估阶段入口
- `FINAL_DELIVERY_REPORT.md` - 最终交付报告
- `PROJECT_COMPLETION_SUMMARY.md` - 项目完成总结

---

## 🔍 关键代码位置

| 功能 | 文件 | 行数 |
|------|------|------|
| 成本计算核心 | `cost.py` | 1,581 |
| UI 主应用 | `planner_minimal.py` | 2,631 |
| AIS 摄取 | `ais_ingest.py` | 746 |
| 路径规划 | `astar.py` | 304 |
| 真实环境 | `env_real.py` | 514 |
| 船舶配置 | `vessel_profiles.py` | 541 |
| EDL 模型 | `edl_core.py` | 233 |
| 训练脚本 | `train_small_edl.py` | 254 |

---

## 💡 常见任务

### 添加新的路由场景
1. 编辑 `configs/scenarios.yaml`
2. 添加新的场景定义
3. 在 UI 中选择新场景

### 优化成本权重
1. 编辑 `configs/scenarios.yaml` 中的权重
2. 运行 `scripts/run_scenario_suite.py` 评估
3. 查看 `reports/scenario_results.csv`

### 训练新的 EDL 模型
1. 准备训练数据（使用 `export_edl_training_dataset.py`）
2. 运行 `scripts/edl_train_torch.py`
3. 验证模型性能

### 分析路由质量
1. 运行 `scripts/evaluate_routes_vs_ais.py`
2. 查看 `reports/` 中的结果
3. 使用 UI 的评估页面可视化

---

**最后更新**: 2025-12-14  
**报告位置**: `reports/repo_report.md` (21,315 行)  
**清单位置**: `reports/repo_manifest.json` (920 KB)

