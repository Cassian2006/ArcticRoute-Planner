# Bug 修复报告

## 问题描述

**错误类型**: `UnboundLocalError`

**错误信息**:
```
UnboundLocalError: cannot access local variable 'routes_info' where it is not assigned
```

**发生位置**: `arcticroute/ui/planner_minimal.py`, 第 1076 行

**根本原因**: 代码的执行顺序错误，导致在 `routes_info` 被定义之前就试图访问它。

## 问题分析

### 错误的代码顺序

原始代码中存在以下问题：

```python
# ❌ 错误的顺序
pipeline.done('cost_build')
pipeline.done('snap')
num_reachable = sum(1 for r in routes_info.values() if r.reachable)  # routes_info 还未定义！
pipeline.done('astar', extra_info=f'routes reachable={num_reachable}/3')
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)

pipeline.start('cost_build')
pipeline.start('snap')
pipeline.start('astar')

# 这里才定义 routes_info
routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(...)
```

### 问题原因

1. 在规划流程的修改脚本中，意外地添加了重复的 `start()` 和 `done()` 调用
2. `done()` 调用被放在了 `plan_three_routes()` 之前
3. 导致 `routes_info` 在被使用时还未被定义

## 解决方案

### 修复步骤

1. **删除重复的代码**
   - 删除了在 `plan_three_routes()` 之前的所有 `start()` 和 `done()` 调用
   - 保留了 `pipeline.start()` 调用（在 `plan_three_routes()` 之前）

2. **重新排序代码**
   - 将 `plan_three_routes()` 调用移到前面
   - 在 `plan_three_routes()` 之后添加 `done()` 调用

### 修复后的代码顺序

```python
# ✅ 正确的顺序
# 启动后续 stages
pipeline.start('cost_build')
pipeline.start('snap')
pipeline.start('astar')

# 执行规划
routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(...)

# 完成 stages（此时 routes_info 已定义）
pipeline.done('cost_build')
pipeline.done('snap')
num_reachable = sum(1 for r in routes_info.values() if r.reachable)
pipeline.done('astar', extra_info=f'routes reachable={num_reachable}/3')
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)
```

## 修复验证

### 修复前
```
❌ UnboundLocalError: cannot access local variable 'routes_info'
```

### 修复后
```
✅ 所有验证通过
✅ 语法检查通过
✅ 集成测试通过
```

## 测试结果

### 文件验证
- ✅ planner_minimal.py (136176 bytes)

### 语法验证
- ✅ arcticroute/ui/components/pipeline_timeline.py
- ✅ arcticroute/ui/planner_minimal.py
- ✅ test_pipeline_integration.py

### 集成验证
- ✅ Pipeline 导入
- ✅ Pipeline 初始化
- ✅ Pipeline stages
- ✅ Pipeline start
- ✅ Pipeline done
- ✅ render_pipeline
- ✅ session_state 控制
- ✅ st.rerun()

## 修复工具

**脚本**: `fix_routes_info_order.py`

**功能**:
1. 查找有问题的代码段
2. 删除重复的 `start()` 和 `done()` 调用
3. 在正确的位置添加 `done()` 调用
4. 保存修改

## 影响范围

- **修改文件**: `arcticroute/ui/planner_minimal.py`
- **修改行数**: ~10 行
- **影响功能**: Pipeline Timeline 的 astar stage 完成逻辑

## 相关代码

### 修复前
```python
# 第 1072-1092 行
# 完成 cost_build/snap/astar stages
pipeline.done('cost_build')
pipeline.done('snap')
num_reachable = sum(1 for r in routes_info.values() if r.reachable)  # ❌ 错误
pipeline.done('astar', extra_info=f'routes reachable={num_reachable}/3')
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)

pipeline.start('cost_build')
pipeline.start('snap')
pipeline.start('astar')

routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(...)
```

### 修复后
```python
# 第 1072-1092 行
# 启动后续 stages
pipeline.start('cost_build')
pipeline.start('snap')
pipeline.start('astar')

routes_info, cost_fields, cost_meta, scores_by_key, recommended_key = plan_three_routes(...)

# 完成 cost_build/snap/astar stages
pipeline.done('cost_build')
pipeline.done('snap')
num_reachable = sum(1 for r in routes_info.values() if r.reachable)  # ✅ 正确
pipeline.done('astar', extra_info=f'routes reachable={num_reachable}/3')
render_pipeline(pipeline.get_stages_list(), pipeline_placeholder)
```

## 总结

✅ **Bug 已修复**

- **问题**: `routes_info` 在定义前被使用
- **原因**: 代码执行顺序错误
- **解决**: 重新排序代码，将 `plan_three_routes()` 调用移到前面
- **验证**: 所有测试通过

**状态**: ✅ RESOLVED

---

**修复日期**: 2025-12-12
**修复人**: AI Assistant
**状态**: ✅ 完成


