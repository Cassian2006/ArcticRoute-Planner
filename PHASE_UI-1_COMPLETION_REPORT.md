# Phase UI-1 完成报告

## 任务目标

将 planner_minimal 侧边栏的所有已做能力同步到 UI，实现：
- 四大区块统一管理：数据源/约束/成本组件/规划器
- 所有开关写入 st.session_state 和 cost_breakdown.json["meta"]
- 运行摘要面板显示数据层/fallback/planner/cost components
- 支持下载 cost_breakdown.json / summary.txt / polaris_diagnostics.csv

## 实施内容

### 1. 新增模块：arcticroute/ui/sidebar_config.py

创建了统一的侧边栏配置模块，包含四大区块：

#### 📊 数据源配置 (render_data_source_section)
- **env_source**: demo / cmems_latest / manual_nc
- **CMEMS 数据层开关**:
  - enable_sic (海冰浓度) - 默认 true
  - enable_swh (有效波高) - 默认 true
  - enable_sit (海冰厚度) - 默认 false
  - enable_drift (海冰漂移) - 默认 false
- **newenv sync**: 一键同步到 newenv + 显示 newenv_index.json
- **grid_mode**: demo / real
- **grid_signature**: 自动计算并存储
- **cost_mode**: demo_icebelt / real_sic_if_available

#### ⚠️ 约束配置 (render_constraints_section)
- **POLARIS 冰级约束**:
  - enabled: 启用/禁用
  - use_decayed_table: 使用衰减表
  - hard_block_level: 硬禁区等级 (0-5)
  - elevated_penalty_scale: 提升惩罚系数 (0-10)
- **浅水约束**:
  - shallow_enabled: 启用/禁用
  - min_depth_m: 最小水深 (m)
  - w_shallow: 浅水惩罚权重
  - 需要 bathymetry 数据可用

#### 💰 成本组件 (render_cost_components_section)
- **AIS 密度成本**:
  - w_ais_corridor: 主航线偏好 (0-10)
  - w_ais_congestion: 拥挤惩罚 (0-10)
  - w_ais: 旧版权重 (deprecated)
  - ais_density_path: 自动选择或手动指定
  - 显示当前选中的密度文件与 grid_signature
- **波浪成本**: wave_penalty (0-10)
- **EDL 风险成本**:
  - w_edl: EDL 权重 (0-10)
  - edl_uncertainty_weight: 不确定性权重 (0-10)

#### 🎯 规划器后端 (render_planner_backend_section)
- **planner_backend**: auto / astar / polarroute_pipeline / polarroute_external
- 显示当前选择的规划器

### 2. 修改 planner_minimal.py

#### 集成新侧边栏
- 添加 `use_unified_sidebar` 开关 (默认 true)
- 调用 `render_sidebar_unified()` 渲染四大区块
- 保持向后兼容：可切换回原有侧边栏
- 在侧边栏底部添加场景选择、起止点输入、船舶选择、EDL 模式选择

#### 添加运行摘要面板
- 在规划完成后调用 `render_run_summary_panel()`
- 将所有配置写入 `cost_meta`:
  - env_source, cmems_layers
  - polaris_enabled, use_decayed_table, hard_block_level, elevated_penalty_scale
  - shallow_enabled, min_depth_m, w_shallow
  - planner_backend
  - w_ais_corridor, w_ais_congestion, w_ais
  - wave_penalty, w_edl, edl_uncertainty_weight
  - grid_signature

### 3. 运行摘要面板功能 (render_run_summary_panel)

#### 数据层状态
- 显示已加载的数据层：sic, swh, sit, drift, ais_density, bathymetry
- 使用 ✅/❌ 图标标识加载状态

#### Fallback 信息
- 显示 fallback_reason
- 若无 fallback 则显示 ✅

#### 规划器信息
- 显示 planner_used 和 polaris_enabled 状态

#### 成本组件统计
- 显示每个成本组件的：
  - 总值
  - 均值
  - 是否全零

#### 导出数据
- **cost_breakdown.json**: 下载成本元数据 JSON
- **summary.txt**: 下载运行摘要文本
- **polaris_diagnostics.csv**: 下载 POLARIS 诊断数据（如果存在）

## 技术亮点

### 1. 模块化设计
- 将侧边栏配置独立为单独模块
- 每个区块独立渲染，便于维护和扩展
- 统一的配置字典返回格式

### 2. 向后兼容
- 保留原有侧边栏代码
- 通过 `use_unified_sidebar` 开关切换
- 确保现有功能不受影响

### 3. 状态管理
- 所有配置同步到 st.session_state
- 写入 cost_breakdown.json["meta"] 用于持久化
- 支持配置的导出和审计

### 4. 用户体验
- 清晰的四大区块组织
- 直观的图标和标签
- 实时状态反馈
- 一键导出功能

## 测试结果

```bash
python -m pytest tests/ -q --tb=short
```

**结果**: 322 passed, 6 skipped, 4 warnings in 31.39s ✅

## Git 提交

```bash
git checkout -b feat/ui-sync-minimal
git add -A
git commit -m "feat(ui): sync minimal planner sidebar with multisource layers + rules + planner backend (no shell)"
git push -u origin feat/ui-sync-minimal
```

**提交 SHA**: 608590e

## 文件清单

### 新增文件
- `arcticroute/ui/sidebar_config.py` (约 400 行)

### 修改文件
- `arcticroute/ui/planner_minimal.py` (添加约 150 行)

## 后续建议

### 短期优化
1. 实现 newenv sync 功能的实际逻辑
2. 添加 bathymetry 数据加载和浅水约束的实际计算
3. 完善 POLARIS 冰级约束的集成
4. 支持 polarroute_pipeline 和 polarroute_external 规划器

### 中期扩展
1. 添加配置预设保存/加载功能
2. 支持批量场景运行
3. 添加配置对比功能
4. 实现配置历史记录

### 长期规划
1. 开发独立的配置管理系统
2. 支持多用户配置共享
3. 添加配置验证和推荐
4. 集成配置优化建议

## 使用说明

### 启动 UI
```bash
streamlit run run_ui.py
```

### 切换侧边栏模式
在代码中修改：
```python
use_unified_sidebar = st.session_state.get("use_unified_sidebar", True)
```
- `True`: 使用新的四大区块侧边栏
- `False`: 使用原有侧边栏

### 查看运行摘要
规划完成后，展开 "📋 运行摘要面板" expander 即可查看：
- 数据层加载状态
- Fallback 信息
- 规划器使用情况
- 成本组件统计
- 导出数据按钮

### 导出配置
点击运行摘要面板中的导出按钮：
- 📥 cost_breakdown.json: 完整的成本元数据
- 📥 summary.txt: 简洁的文本摘要
- 📥 polaris_diagnostics.csv: POLARIS 诊断数据

## 总结

Phase UI-1 成功完成了以下目标：

✅ 将侧边栏重组为四大区块：数据源/约束/成本组件/规划器  
✅ 所有开关同步到 session_state 和 cost_breakdown.json["meta"]  
✅ 添加运行摘要面板显示数据层/fallback/planner/cost components  
✅ 支持下载 cost_breakdown.json / summary.txt / polaris_diagnostics.csv  
✅ 保持向后兼容，不破坏原流程  
✅ 所有测试通过  

新的 UI 结构更加清晰、模块化，为后续的功能扩展奠定了良好的基础。

