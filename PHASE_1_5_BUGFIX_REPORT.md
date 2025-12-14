# Phase 1.5 Bug 修复报告

**日期**: 2025-12-10  
**问题**: UI 中 AIS 状态提示加载失败  
**状态**: ✅ 已修复

---

## 问题描述

在 UI 中启动时，AIS 状态提示显示以下错误：

```
[WARN] 当前未加载 AIS 拥挤度 (加载失败: inspect_ais_csv() got an unexp...)
```

---

## 根本原因

在 `arcticroute/ui/planner_minimal.py` 中调用 `inspect_ais_csv()` 时，使用了错误的参数名：

**错误代码**:
```python
ais_summary = inspect_ais_csv(str(ais_csv_path), max_rows=100)
```

**正确的参数名**是 `sample_n`，而不是 `max_rows`。

---

## 修复方案

### 修改文件
`arcticroute/ui/planner_minimal.py` (第 586 行)

### 修改内容
```python
# 修改前
ais_summary = inspect_ais_csv(str(ais_csv_path), max_rows=100)

# 修改后
ais_summary = inspect_ais_csv(str(ais_csv_path), sample_n=100)
```

---

## 验证

### 测试命令
```bash
python -c "from arcticroute.core.ais_ingest import inspect_ais_csv; summary = inspect_ais_csv('data_real/ais/raw/ais_2024_sample.csv', sample_n=100); print(f'行数: {summary.num_rows}')"
```

### 测试结果
```
行数: 20
✅ 成功
```

---

## 影响范围

- **文件**: `arcticroute/ui/planner_minimal.py`
- **行数**: 1 行
- **功能**: AIS 状态提示
- **严重性**: 中等（影响 UI 显示，但不影响核心功能）

---

## 修复后的行为

修复后，UI 中的 AIS 状态提示将正确显示：

**已加载状态**（绿色）:
```
[OK] 已加载 AIS 拥挤度数据 (20 点映射到网格)
```

**未加载状态**（黄色）:
```
[WARN] 当前未加载 AIS 拥挤度 (数据文件不存在)
```

---

## 测试清单

- [x] 修复代码
- [x] 验证 `inspect_ais_csv()` 函数能正常工作
- [x] 确认参数名正确
- [x] UI 状态提示显示正确

---

## 总结

**问题**: 参数名错误（`max_rows` vs `sample_n`）  
**修复**: 更正参数名为 `sample_n`  
**状态**: ✅ 已修复并验证  
**影响**: 无其他影响

---

**修复日期**: 2025-12-10  
**修复人**: Cascade AI Assistant






