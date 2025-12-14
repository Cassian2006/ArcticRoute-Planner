# 任务 U1 & U2 快速参考指南

## 📋 修复清单

### ✅ 任务 U1：乱码修复
- [x] 确认所有文件都使用 UTF-8 编码
- [x] scenarios.py 正确读取 YAML（`encoding="utf-8"`）
- [x] planner_minimal.py 中文标签正确显示
- [x] 添加乱码检测测试（防复发）

### ✅ 任务 U2：地图配置
- [x] 添加 ARCTIC_VIEW 配置（北极固定视角）
- [x] 添加 MAP_CONTROLLER 配置（禁止拖动）
- [x] 更新两处 ViewState 定义
- [x] 更新两处 pydeck_chart 调用

---

## 🔍 关键代码位置

### planner_minimal.py 中的新配置

**第 63-70 行**：北极固定视角
```python
ARCTIC_VIEW = {
    "latitude": 75.0,      # 北极中心
    "longitude": 30.0,     # 北冰洋中心
    "zoom": 2.6,           # 默认缩放
    "min_zoom": 2.2,       # 最小缩放限制
    "max_zoom": 6.0,       # 最大缩放限制
    "pitch": 0,            # 俯视角度
}
```

**第 73-81 行**：地图控制器配置
```python
MAP_CONTROLLER = {
    "dragPan": False,      # ✅ 禁止拖动
    "dragRotate": False,
    "scrollZoom": True,    # ✅ 允许滚轮缩放
    "doubleClickZoom": True,
    "touchZoom": True,
    "keyboard": False,
}
```

---

## 🧪 验证步骤

### 1. 运行乱码检测测试
```bash
python -m pytest tests/test_mojibake_detection.py -v
```
**预期结果**：4 passed

### 2. 运行修复验证脚本
```bash
python verify_fixes.py
```
**预期结果**：所有修复都已成功应用

### 3. 启动 UI 进行手动测试
```bash
streamlit run run_ui.py
```

**在"航线规划驾驶舱"中检查**：
- ✅ 左侧预设/模式文字不乱码（"效率优先"、"风险均衡"、"稳健安全"）
- ✅ 地图默认显示北极区域
- ✅ 地图无法拖到赤道/南半球
- ✅ 地图无法缩放到无限小（min_zoom=2.2）
- ✅ 地图无法缩放到无限大（max_zoom=6.0）

---

## 📊 修改统计

| 文件 | 修改类型 | 行数 | 状态 |
|------|---------|------|------|
| arcticroute/ui/planner_minimal.py | 添加配置 + 更新代码 | 63-81, 1265-1280, 2175-2190 | ✅ |
| arcticroute/core/scenarios.py | 验证编码 | 54 | ✅ |
| tests/test_mojibake_detection.py | 新增测试 | 全新 | ✅ |
| verify_fixes.py | 验证脚本 | 全新 | ✅ |

---

## 🎯 预期用户体验

### 地图行为
1. **打开驾驶舱** → 地图自动显示北极区域
2. **尝试拖动** → 地图不会移动（dragPan=False）
3. **滚轮缩放** → 可以缩放，但受 min/max_zoom 限制
4. **查看标签** → 所有中文标签正确显示，无乱码

### 地图范围
- **最北**：可显示到 90°N（北极点）
- **最南**：受 min_zoom 限制，无法显示到赤道
- **东西**：可显示整个北冰洋（0°-360°E）

---

## 🔧 如需调整

### 改变北极中心位置
编辑 `ARCTIC_VIEW` 中的 `longitude`：
```python
"longitude": 0.0,   # 改为 0 或 20 或其他值
```

### 调整缩放范围
编辑 `ARCTIC_VIEW` 中的 `min_zoom` 和 `max_zoom`：
```python
"min_zoom": 1.5,    # 允许更小的缩放
"max_zoom": 8.0,    # 允许更大的缩放
```

### 允许小范围拖动（可选）
如果需要在北极框内允许拖动，修改 `MAP_CONTROLLER`：
```python
"dragPan": True,    # 改为 True
# 并在 Deck 中添加 maxBounds
maxBounds=[[-180, 55], [180, 90]],  # [[minLon,minLat],[maxLon,maxLat]]
```

---

## 📝 测试命令速查

```bash
# 乱码检测
python -m pytest tests/test_mojibake_detection.py -v

# 修复验证
python verify_fixes.py

# 启动 UI
streamlit run run_ui.py

# 检查特定配置
python -c "from pathlib import Path; content = Path('arcticroute/ui/planner_minimal.py').read_text(encoding='utf-8'); print('ARCTIC_VIEW found:', 'ARCTIC_VIEW' in content); print('MAP_CONTROLLER found:', 'MAP_CONTROLLER' in content)"
```

---

## ✨ 完成标志

- ✅ 乱码测试通过
- ✅ 修复验证通过
- ✅ 地图固定在北极
- ✅ 缩放和拖动受限
- ✅ 中文标签正确显示

**所有修复都已完成！** [object Object]



