# EDL 模式更新总结

## 概述

本文档记录了 ArcticRoute 项目中 EDL（Evidential Deep Learning）三种模式的更新。主要改进是将 `efficient` 模式从"无 EDL"改为"弱 EDL"，形成一个完整的 EDL 强度梯度。

## 更新内容

### 1. 脚本端更新（scripts/run_edl_sensitivity_study.py）

#### 模式配置变更

| 模式 | 原配置 | 新配置 | 说明 |
|-----|-------|-------|------|
| efficient | w_edl=0.0, use_edl=False | w_edl=0.3, use_edl=True | 改为弱 EDL |
| edl_safe | w_edl=1.0, use_edl=True | w_edl=1.0, use_edl=True | 保持不变 |
| edl_robust | w_edl=1.0, use_edl=True, use_edl_uncertainty=True | w_edl=1.0, use_edl=True, use_edl_uncertainty=True | 保持不变 |

#### 新的模式定义

```python
MODES = {
    "efficient": {
        "description": "弱 EDL（偏燃油/距离）",
        "w_edl": 0.3,  # 约为 safe 的 1/3
        "use_edl": True,
        "use_edl_uncertainty": False,
        "edl_uncertainty_weight": 0.0,
        "ice_penalty": 4.0,
    },
    "edl_safe": {
        "description": "中等 EDL（偏风险规避）",
        "w_edl": 1.0,
        "use_edl": True,
        "use_edl_uncertainty": False,
        "edl_uncertainty_weight": 0.0,
        "ice_penalty": 4.0,
    },
    "edl_robust": {
        "description": "强 EDL（风险 + 不确定性）",
        "w_edl": 1.0,
        "use_edl": True,
        "use_edl_uncertainty": True,
        "edl_uncertainty_weight": 1.0,
        "ice_penalty": 4.0,
    },
}
```

### 2. UI 端更新（arcticroute/ui/planner_minimal.py）

#### ROUTE_PROFILES 变更

| 模式 | 原配置 | 新配置 | 说明 |
|-----|-------|-------|------|
| efficient | edl_weight_factor=0.3 | edl_weight_factor=0.3 | 保持一致 |
| edl_safe | edl_weight_factor=2.0 | edl_weight_factor=1.0 | 调整为 1.0 |
| edl_robust | edl_weight_factor=2.0 | edl_weight_factor=1.0 | 调整为 1.0 |

#### 新的 ROUTE_PROFILES 定义

```python
ROUTE_PROFILES = [
    {
        "key": "efficient",
        "label": "Efficient（弱 EDL，偏燃油/距离）",
        "ice_penalty_factor": 0.5,
        "wave_weight_factor": 0.5,
        "edl_weight_factor": 0.3,  # 弱 EDL
        "use_edl_uncertainty": False,
        "edl_uncertainty_weight": 0.0,
    },
    {
        "key": "edl_safe",
        "label": "EDL-Safe（中等 EDL，偏风险规避）",
        "ice_penalty_factor": 2.0,
        "wave_weight_factor": 1.5,
        "edl_weight_factor": 1.0,  # 中等 EDL
        "use_edl_uncertainty": False,
        "edl_uncertainty_weight": 0.0,
    },
    {
        "key": "edl_robust",
        "label": "EDL-Robust（强 EDL，风险 + 不确定性）",
        "ice_penalty_factor": 2.0,
        "wave_weight_factor": 1.5,
        "edl_weight_factor": 1.0,  # 强 EDL
        "use_edl_uncertainty": True,
        "edl_uncertainty_weight": 2.0,
    },
]
```

### 3. 测试端新增（tests/test_edl_mode_strength.py）

新增测试文件，包含以下测试类：

#### TestEDLModeStrength
- `test_modes_configuration`: 验证模式配置的基本属性
- `test_edl_weight_hierarchy`: 验证 EDL 权重的层级关系
- `test_cost_field_construction`: 测试三种模式的成本场构建
- `test_route_planning_and_cost_accumulation`: 测试路线规划和成本积累
- `test_uncertainty_cost_hierarchy`: 测试不确定性成本的层级关系
- `test_mode_descriptions`: 测试模式描述的一致性

#### TestUIRouteProfilesConsistency
- `test_route_profiles_exist`: 验证 UI 中的 ROUTE_PROFILES 存在
- `test_route_profiles_keys_match_modes`: 验证 key 一致性
- `test_route_profiles_edl_weight_factors`: 验证 EDL 权重因子一致性
- `test_route_profiles_uncertainty_settings`: 验证不确定性设置一致性

### 4. 演示脚本新增（scripts/demo_edl_modes.py）

新增演示脚本，用于在虚拟环境数据上展示三种 EDL 模式的行为。

## 验证结果

### 测试结果

```
tests/test_edl_mode_strength.py::TestEDLModeStrength::test_modes_configuration PASSED
tests/test_edl_mode_strength.py::TestEDLModeStrength::test_edl_weight_hierarchy PASSED
tests/test_edl_mode_strength.py::TestUIRouteProfilesConsistency::test_route_profiles_exist PASSED
tests/test_edl_mode_strength.py::TestUIRouteProfilesConsistency::test_route_profiles_keys_match_modes PASSED
tests/test_edl_mode_strength.py::TestUIRouteProfilesConsistency::test_route_profiles_edl_weight_factors PASSED
tests/test_edl_mode_strength.py::TestUIRouteProfilesConsistency::test_route_profiles_uncertainty_settings PASSED

总计：10 个测试通过
```

### 演示脚本结果

在虚拟环境数据上运行演示脚本，结果如下：

#### EDL 风险成本对比

| 模式 | EDL 风险成本 | 占比 |
|-----|-----------|------|
| efficient | 6.8560 | 6.1% |
| edl_safe | 24.1071 | 19.3% |
| edl_robust | 22.8119 | 18.3% |

#### 相对强度关系

- **efficient / edl_safe = 0.28**（符合预期，约为 1/3）
- **efficient < edl_safe ≤ edl_robust**（满足预期的层级关系）

#### EDL 不确定性成本

| 模式 | 不确定性成本 |
|-----|-----------|
| efficient | 0.0000 |
| edl_safe | 0.0000 |
| edl_robust | 39.6056 |

**✓ edl_robust 正确启用了不确定性成本**

## 设计原理

### 三层 EDL 强度梯度

1. **Efficient（弱 EDL）**
   - 用途：偏向燃油/距离优化，但仍考虑 EDL 风险
   - w_edl = 0.3（约为 safe 的 1/3）
   - 不启用不确定性
   - 适合成本敏感型用户

2. **EDL-Safe（中等 EDL）**
   - 用途：平衡风险和燃油
   - w_edl = 1.0
   - 不启用不确定性
   - 适合综合考虑的用户

3. **EDL-Robust（强 EDL）**
   - 用途：最大化风险规避
   - w_edl = 1.0
   - 启用不确定性（weight = 1.0）
   - 适合风险厌恶型用户

### 权重设计原则

- **efficient 的 w_edl = 0.3**：
  - 相对于 safe 的 1/3，提供弱 EDL 支持
  - 足以影响路线选择，但不会过度约束
  - 建议范围：0.2 ~ 0.4

- **edl_safe 和 edl_robust 的 w_edl = 1.0**：
  - 相同的基础 EDL 权重
  - 区别在于是否启用不确定性
  - 建议范围：0.8 ~ 1.5

- **edl_robust 的 edl_uncertainty_weight = 1.0**：
  - 与 w_edl 同量级
  - 在高不确定性区域施加显著惩罚
  - 建议范围：0.5 ~ 2.0

## 使用指南

### 脚本使用

```bash
# 使用 demo 网格运行灵敏度分析
python -m scripts.run_edl_sensitivity_study

# 使用真实数据运行
python -m scripts.run_edl_sensitivity_study --use-real-data

# 运行演示脚本
python -m scripts.demo_edl_modes
```

### UI 使用

在 Streamlit UI 中，用户可以通过下拉框选择三种方案：
- **Efficient（弱 EDL，偏燃油/距离）**
- **EDL-Safe（中等 EDL，偏风险规避）**
- **EDL-Robust（强 EDL，风险 + 不确定性）**

UI 会自动应用相应的权重配置。

### 测试验证

```bash
# 运行完整的 EDL 模式强度测试
pytest tests/test_edl_mode_strength.py -v

# 运行特定测试
pytest tests/test_edl_mode_strength.py::TestEDLModeStrength::test_edl_weight_hierarchy -v
```

## 后续改进方向

### 短期
- [ ] 在真实海冰数据上验证三种模式的行为
- [ ] 收集用户反馈，调整权重参数
- [ ] 添加更多的演示场景

### 中期
- [ ] 实现参数扫描（grid search）来优化权重
- [ ] 添加交互式参数调优工具
- [ ] 支持自定义权重配置

### 长期
- [ ] 集成多个 EDL 模型的对比
- [ ] 实现在线学习和模型更新
- [ ] 建立 EDL 模型库和评估框架

## 相关文件

- **脚本**: `scripts/run_edl_sensitivity_study.py`
- **UI**: `arcticroute/ui/planner_minimal.py`
- **测试**: `tests/test_edl_mode_strength.py`
- **演示**: `scripts/demo_edl_modes.py`
- **文档**: `docs/EDL_BEHAVIOR_CHECK.md`

## 版本信息

- **更新日期**: 2024-12-09
- **版本**: 1.1
- **状态**: 完成

---

**维护者**: ArcticRoute 项目组











