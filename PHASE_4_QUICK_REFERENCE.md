## Phase 4 快速参考指南

### 📋 核心改动

#### 1. 新建配置模块
```
arcticroute/config/
├── __init__.py          # 统一导出
├── edl_modes.py         # EDL 模式配置
└── scenarios.py         # 场景预设配置
```

#### 2. 修改的文件
- `scripts/run_edl_sensitivity_study.py` - 使用共享配置
- `arcticroute/ui/planner_minimal.py` - 添加下拉框

#### 3. 新增测试
- `tests/test_edl_config_and_scenarios.py` - 20 个配置测试
- `tests/test_ui_edl_comparison.py` - 7 个 UI 集成测试

---

### 🎯 三种 EDL 模式

| 模式 | w_edl | 不确定性 | 用途 |
|------|-------|--------|------|
| **Efficient** | 0.3 | ❌ | 偏燃油/距离 |
| **EDL-Safe** | 1.0 | ❌ | 平衡风险 |
| **EDL-Robust** | 1.0 | ✅ | 最保守 |

---

### 🗺️ 四个预设场景

| 场景 | 起点 | 终点 | 描述 |
|------|------|------|------|
| **barents_to_chukchi** | 69°N, 33°E | 70.5°N, 170°E | 高冰区长距离 |
| **kara_short** | 73°N, 60°E | 76°N, 120°E | 中等冰区 |
| **southern_route** | 60°N, 30°E | 68°N, 90°E | 低冰区 |
| **west_to_east_demo** | 66°N, 5°E | 78°N, 150°E | 全程高纬 |

---

### 💻 使用示例

#### CLI 使用
```bash
# 运行灵敏度分析（自动使用三种模式）
python -m scripts.run_edl_sensitivity_study

# 指定输出路径
python -m scripts.run_edl_sensitivity_study \
  --output-csv reports/results.csv \
  --output-dir reports/charts
```

#### Python 代码使用
```python
from arcticroute.config import EDL_MODES, SCENARIOS, get_scenario_by_name

# 获取 EDL 模式配置
config = EDL_MODES["edl_safe"]
print(f"w_edl: {config['w_edl']}")

# 获取场景
scenario = get_scenario_by_name("west_to_east_demo")
print(f"起点: {scenario.start_lat}, {scenario.start_lon}")
```

#### UI 使用
1. 打开 Streamlit UI
2. 左侧栏选择"场景预设"
3. 左侧栏选择"规划风格"
4. 点击"规划三条方案"
5. 查看对比结果

---

### 🔍 参数验证

所有参数都经过验证：
- ✅ EDL 模式参数单调递增
- ✅ 场景坐标在北极地区
- ✅ 参数范围合理
- ✅ CLI 和 UI 使用相同配置

---

### 📊 测试覆盖

```
总计: 205 passed, 5 skipped
新增: 27 个测试（20 + 7）
覆盖: 配置、场景、单调性、一致性
```

---

### 🚀 关键特性

1. **参数统一化**
   - 一个配置源
   - 多个使用点（CLI、UI）
   - 易于维护和更新

2. **场景预设**
   - 四个标准场景
   - 自动填充坐标
   - 快速测试

3. **规划风格**
   - 三种预设模式
   - 自动参数调整
   - 简化用户界面

4. **完整测试**
   - 配置完整性检查
   - 参数单调性验证
   - UI 集成测试

---

### 📝 配置文件位置

```
arcticroute/config/edl_modes.py
  ├── EDL_MODES (dict)
  ├── get_edl_mode_config(mode)
  ├── list_edl_modes()
  └── validate_edl_mode_config(config)

arcticroute/config/scenarios.py
  ├── SCENARIOS (list)
  ├── Scenario (dataclass)
  ├── get_scenario_by_name(name)
  ├── list_scenarios()
  └── list_scenario_descriptions()

arcticroute/config/__init__.py
  └── 统一导出接口
```

---

### 🔗 导入方式

```python
# 推荐方式
from arcticroute.config import EDL_MODES, SCENARIOS

# 或者
from arcticroute.config import get_edl_mode_config, get_scenario_by_name

# 或者
from arcticroute.config.edl_modes import list_edl_modes
from arcticroute.config.scenarios import list_scenario_descriptions
```

---

### ⚙️ 参数调优

如需修改参数，只需编辑：
```
arcticroute/config/edl_modes.py
```

例如，增加 EDL 权重：
```python
EDL_MODES["edl_safe"]["w_edl"] = 1.5  # 从 1.0 改为 1.5
```

修改会自动应用到 CLI 和 UI。

---

### ✅ 验收清单

- [x] 创建 EDL 模式配置模块
- [x] 创建场景预设配置模块
- [x] 修改 CLI 使用共享配置
- [x] 修改 UI 添加下拉框
- [x] 添加配置测试
- [x] 添加 UI 集成测试
- [x] 验证参数单调性
- [x] 所有测试通过

---

### 📞 常见问题

**Q: 如何添加新的 EDL 模式？**
A: 在 `arcticroute/config/edl_modes.py` 中的 `EDL_MODES` 字典中添加新条目。

**Q: 如何添加新的场景？**
A: 在 `arcticroute/config/scenarios.py` 中的 `SCENARIOS` 列表中添加新的 `Scenario` 对象。

**Q: CLI 和 UI 使用的参数是否相同？**
A: 是的，它们都从 `arcticroute.config` 导入配置，确保完全一致。

**Q: 如何验证参数的正确性？**
A: 运行测试：`pytest tests/test_edl_config_and_scenarios.py -v`

---

**更新日期**: 2024-12-09  
**版本**: 1.0











