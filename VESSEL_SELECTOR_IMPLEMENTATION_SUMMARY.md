# 船舶配置选择器实现总结

## 任务目标

将船舶配置选择器（10x8 个选项）同步到 UI，并与三策略（efficient/edl_safe/edl_robust）分离。

## 实施内容

### 1. 新增模块：arcticroute/ui/vessel_selector.py

创建了独立的船舶选择器组件：

```python
def render_vessel_selector(key_prefix: str = "vessel") -> Tuple[str, str, Dict[str, Any]]:
    """
    Returns (profile_key, ice_class_label, meta_dict)
    profile_key: the chosen key in get_default_profiles()
    ice_class_label: best-effort label parsed from key
    meta_dict: serializable info for cost_breakdown/meta
    """
```

**功能特点**：
- 从 `get_default_profiles()` 获取所有可用的船舶配置（10x8 个选项）
- 自动解析冰级标签（PC1-PC7）
- 返回可序列化的元数据字典
- 支持通过 `key_prefix` 避免 widget key 冲突
- 从 session_state 读取默认值，保持用户选择

**返回值**：
- `profile_key`: 选中的船舶配置键（如 "panamax", "PC6_tanker" 等）
- `ice_class_label`: 解析出的冰级标签（如 "PC6"）
- `meta_dict`: 包含完整船舶配置信息的字典
  - `vessel_profile_key`: 配置键
  - `ice_class_label`: 冰级标签
  - `vessel_profile`: 船舶配置的完整数据（dataclass 转 dict）

### 2. 修改 planner_minimal.py

#### 2.1 添加导入
```python
from arcticroute.ui.vessel_selector import render_vessel_selector
```

#### 2.2 集成到统一侧边栏
在 `use_unified_sidebar=True` 分支中：
```python
# 船舶选择 - 使用新的 vessel_selector
profile_key, ice_label, vessel_meta = render_vessel_selector(key_prefix="unified_vessel")
st.session_state["vessel_profile_key"] = profile_key
st.session_state["vessel_profile"] = profile_key  # 保持向后兼容
st.session_state["vessel_meta"] = vessel_meta
st.session_state["ice_class_label"] = ice_label

# 获取船舶对象
vessel_profiles = get_default_profiles()
selected_vessel_key = profile_key
selected_vessel = vessel_profiles.get(profile_key, list(vessel_profiles.values())[0])
```

#### 2.3 集成到原有侧边栏
在 `use_unified_sidebar=False` 分支中也进行了相同的集成。

#### 2.4 将船舶信息写入 cost_meta
```python
# 添加船舶信息到 cost_meta
vessel_meta = st.session_state.get('vessel_meta', {})
cost_meta.update({
    'vessel_profile_key': st.session_state.get('vessel_profile_key', ''),
    'ice_class_label': st.session_state.get('ice_class_label', ''),
    'vessel_profile': vessel_meta.get('vessel_profile', None),
})
```

### 3. 保持三策略独立

**三策略**（efficient / edl_safe / edl_robust）保持为**规划策略**，不再与船舶类型混淆：
- **策略**：决定风险权重、EDL 使用、不确定性处理等
- **船舶**：决定船舶物理特性、冰级能力、燃油消耗等

**分离的好处**：
- 用户可以选择任意船舶配置 + 任意策略的组合
- 例如：PC6 船舶 + efficient 策略
- 例如：panamax 船舶 + edl_robust 策略
- 总共支持 10x8 个船舶 × 3 个策略 = 240+ 种组合

### 4. 数据持久化

船舶配置信息会被写入多个地方：

#### 4.1 Session State
```python
st.session_state["vessel_profile_key"]  # 船舶配置键
st.session_state["vessel_profile"]      # 向后兼容
st.session_state["vessel_meta"]         # 完整元数据
st.session_state["ice_class_label"]     # 冰级标签
```

#### 4.2 cost_breakdown.json["meta"]
```json
{
  "vessel_profile_key": "PC6_tanker",
  "ice_class_label": "PC6",
  "vessel_profile": {
    "name": "PC6 Tanker",
    "max_ice_thickness_m": 1.2,
    "ice_margin_factor": 0.8,
    ...
  },
  ...
}
```

#### 4.3 导出的 CSV/JSON
在导出规划结果时，`vessel_profile` 字段会包含选中的船舶配置键。

## 技术亮点

### 1. 模块化设计
- 船舶选择器独立为单独模块
- 可在多个地方复用（统一侧边栏、原有侧边栏）
- 通过 `key_prefix` 避免 widget key 冲突

### 2. 向后兼容
- 保留 `st.session_state["vessel_profile"]` 以兼容旧代码
- `plan_three_routes` 返回列表以兼容现有测试
- 在 UI 层将列表转换为字典方便使用

### 3. 类型安全
- 使用类型注解 `Tuple[str, str, Dict[str, Any]]`
- dataclass 自动转换为可序列化的字典
- 支持 None 值的安全处理

### 4. 用户体验
- 自动解析冰级标签（PC1-PC7）
- 显示友好的船舶名称
- 保持用户的选择状态
- 完整的元数据用于审计和复现

## 测试结果

```bash
python -m pytest tests/ -q --tb=short
```

**结果**: 322 passed, 6 skipped, 4 warnings ✅

**关键测试**：
- `test_three_routes_are_reachable`: ✅ 通过
- `test_efficient_vs_robust_costs_differ`: ✅ 通过
- 所有多目标配置测试：✅ 通过

## Git 提交

```bash
git checkout -b feat/ui-vessel-profile-selector
git add -A
git commit -m "feat(ui): sync VesselProfile selector (10x8) into planner UI and persist to meta"
git push -u origin feat/ui-vessel-profile-selector
```

**提交 SHA**: 3bbbfe6

## 文件清单

### 新增文件
- `arcticroute/ui/vessel_selector.py` (约 55 行)
- `VESSEL_SELECTOR_IMPLEMENTATION_SUMMARY.md` (本文档)

### 修改文件
- `arcticroute/ui/planner_minimal.py` (修改约 30 行)

## 使用示例

### 在 UI 中使用

```python
from arcticroute.ui.vessel_selector import render_vessel_selector

# 渲染船舶选择器
profile_key, ice_label, vessel_meta = render_vessel_selector(key_prefix="my_vessel")

# 获取船舶对象
from arcticroute.core.eco.vessel_profiles import get_default_profiles
vessel_profiles = get_default_profiles()
selected_vessel = vessel_profiles[profile_key]

# 使用船舶进行规划
routes = plan_three_routes(
    ...,
    vessel=selected_vessel,
    ...
)

# 将船舶信息写入元数据
cost_meta.update(vessel_meta)
```

### 在代码中访问船舶信息

```python
# 从 session_state 读取
vessel_key = st.session_state.get("vessel_profile_key")
ice_class = st.session_state.get("ice_class_label")
vessel_meta = st.session_state.get("vessel_meta")

# 从 cost_meta 读取
vessel_key = cost_meta["vessel_profile_key"]
vessel_profile = cost_meta["vessel_profile"]
```

## 可用的船舶配置

从 `get_default_profiles()` 获取的船舶配置包括（但不限于）：

### 按冰级分类
- **PC1-PC3**: 极地船舶，最强冰级
- **PC4-PC5**: 中等冰级
- **PC6-PC7**: 轻冰级
- **1A/1B/1C**: 芬兰-瑞典冰级
- **Non-ice**: 无冰级船舶

### 按船型分类
- **Tanker**: 油轮
- **Bulk Carrier**: 散货船
- **Container**: 集装箱船
- **Panamax**: 巴拿马型船
- **LNG Carrier**: LNG 运输船

### 示例配置键
```
panamax
PC6_tanker
PC5_bulk
PC3_icebreaker
1A_container
non_ice_cargo
...
```

## 后续优化建议

### 短期
1. 添加船舶配置的详细说明（tooltip）
2. 支持按冰级、船型筛选
3. 添加船舶对比功能
4. 显示船舶的关键参数（吃水、速度等）

### 中期
1. 支持自定义船舶配置
2. 船舶配置的导入/导出
3. 船舶配置的版本管理
4. 船舶性能的可视化对比

### 长期
1. 基于实际航行数据的船舶性能校准
2. 船舶配置的推荐系统
3. 多船舶协同规划
4. 船舶配置的优化建议

## 总结

✅ 成功实现了船舶配置选择器（10x8 个选项）  
✅ 与三策略（efficient/edl_safe/edl_robust）完全分离  
✅ 船舶信息持久化到 session_state 和 cost_meta  
✅ 保持向后兼容，所有测试通过  
✅ 模块化设计，易于复用和扩展  

新的船舶选择器为用户提供了更大的灵活性，可以自由组合船舶配置和规划策略，满足不同场景的需求。

