# Phase 9 执行指南

## 📋 概述

本指南提供了 Phase 9 收口和 Phase 9.1 诊断的完整执行步骤。

**预计时间**：10 分钟（包括 PR 创建）  
**难度**：简单（大部分由脚本自动化）  
**前置条件**：Git、Python、pytest 已安装

---

## 🚀 快速开始（推荐）

### 步骤 1：运行收口脚本（5 分钟）

```powershell
# 打开 PowerShell，进入项目根目录
cd C:\Users\sgddsf\Desktop\AR_final

# 运行自动化脚本
.\scripts\phase9_closure.ps1
```

**脚本会自动执行**：
- ✅ 检查数据文件是否误提交
- ✅ 显示 diff 统计
- ✅ 询问是否还原 __init__.py
- ✅ 运行完整测试
- ✅ 提交并推送到远程

**预期输出**：
```
========================================
Phase 9 收口：PR 合并前检查
========================================

[1/5] 检查是否有误提交的数据文件...
✓ 确认：没有误提交的数据文件

[2/5] 检查 diff 统计...
当前分支相对于 origin/main 的改动：
399 行统计信息
摘要：399 files changed, 34527 insertions(+), 884 deletions(-)

[3/5] 检查 __init__.py 改动...
✓ 没有 __init__.py 改动需要还原

[4/5] 运行完整测试...
执行: python -m pytest -q
✓ 所有测试通过

[5/5] 提交并推送...
✓ 没有待提交的改动
✓ 已推送到远程

========================================
Phase 9 收口完成！
========================================

后续步骤：
1. 访问 GitHub: https://github.com/Cassian2006/ArcticRoute-Planner
2. 创建 PR 从当前分支到 main
3. 填写 PR 描述（包含验收点、测试结果等）
```

### 步骤 2：创建 PR（2 分钟）

1. 访问 GitHub：https://github.com/Cassian2006/ArcticRoute-Planner

2. 点击 "New Pull Request"

3. 选择：
   - Base: `main`
   - Compare: 当前分支

4. 填写 PR 标题：
   ```
   Phase 9: Multi-objective Route Planning with CMEMS Integration
   ```

5. 复制以下描述到 PR 正文：

```markdown
## 概述

完成 Phase 9 多目标路由规划与 CMEMS 数据集成。

## 主要改动

- 集成 CMEMS 海冰浓度（SIC）和波浪高度（SWH）数据源
- 实现多目标 Pareto 前沿计算
- 添加 AIS 密度分析和约束规则引擎
- 完善 UI 面板和诊断工具

## 验收点

- ✅ 没有误提交数据文件
- ✅ 所有 399 个改动文件来自功能实现
- ✅ 完整测试套件通过
- ✅ CMEMS 数据加载和解析正常
- ✅ Pareto 前沿计算可用
- ✅ UI 集成完整

## 数据不入库策略

- 所有 CMEMS 数据缓存存储在 `data/cmems_cache/`（已 .gitignore）
- 所有生成的报告存储在 `reports/`（已 .gitignore）
- 仅提交代码和配置文件

## 测试结果

```
$ python -m pytest -q
[所有测试通过]
```

## 后续计划

- Phase 9.1：诊断和改进 nextsim HM describe 稳定性
- Phase 10：性能优化和缓存策略
```

6. 点击 "Create Pull Request"

### 步骤 3：诊断 Phase 9.1（可选，3 分钟）

如果需要诊断 nextsim HM describe 问题：

```powershell
# 运行诊断脚本
.\scripts\phase91_diagnose_nextsim.ps1

# 查看诊断结果
Get-Content reports\cmems_sic_describe.nextsim.exitcode.txt
Get-Content reports\cmems_sic_describe.nextsim.log | Select-Object -First 50
```

---

## 📝 详细步骤（如果脚本失败）

### 手动执行 Phase 9 收口

#### 1. 检查数据文件

```bash
# 检查是否有误提交的数据文件
git ls-files | grep -E "data/cmems_cache|ArcticRoute/data_processed|reports/cmems_"

# 应该没有输出
```

#### 2. 检查 diff 统计

```bash
# 查看改动统计
git diff --stat origin/main...HEAD

# 预期输出：
# 399 files changed, 34527 insertions(+), 884 deletions(-)
```

#### 3. 检查 __init__.py 改动

```bash
# 查看 __init__.py 改动
git diff origin/main...HEAD -- ArcticRoute/__init__.py ArcticRoute/core/__init__.py ArcticRoute/core/eco/__init__.py

# 如果只是格式调整，还原它们
git checkout -- ArcticRoute/__init__.py ArcticRoute/core/__init__.py ArcticRoute/core/eco/__init__.py
```

#### 4. 运行测试

```bash
# 运行完整测试
python -m pytest -q

# 预期输出：所有测试通过（返回码 0）
```

#### 5. 提交并推送

```bash
# 添加改动
git add -A

# 提交（如果有改动）
git commit -m "chore: reduce diff noise (revert formatting-only __init__ changes)" || true

# 推送到远程
git push
```

### 手动执行 Phase 9.1 诊断

#### 使用 PowerShell 脚本

```powershell
# 运行诊断脚本
.\scripts\phase91_diagnose_nextsim.ps1

# 查看诊断结果
Get-Content reports\cmems_sic_describe.nextsim.exitcode.txt
Get-Content reports\cmems_sic_describe.nextsim.log
Get-Content reports\cmems_sic_describe.nextsim.stderr.txt
```

#### 使用 Python 脚本

```bash
# 运行改进的脚本
python scripts/cmems_refresh_and_export.py --describe-only

# 查看诊断文件
cat reports/cmems_sic_describe.exitcode.txt
cat reports/cmems_sic_describe.stderr.txt
cat reports/cmems_swh_describe.exitcode.txt
cat reports/cmems_swh_describe.stderr.txt
```

---

## 🔍 故障排除

### 问题 1：脚本权限不足

**错误信息**：
```
cannot be loaded because running scripts is disabled on this system
```

**解决方案**：
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题 2：找不到 git 命令

**错误信息**：
```
git: command not found
```

**解决方案**：
- 安装 Git for Windows：https://git-scm.com/download/win
- 或检查 PATH 环境变量

### 问题 3：pytest 失败

**错误信息**：
```
FAILED tests/test_xxx.py::test_yyy
```

**解决方案**：
```bash
# 运行详细模式查看失败原因
python -m pytest -v

# 修复代码后重新运行
python -m pytest -q
```

### 问题 4：describe 命令仍然失败

**错误信息**：
```
cmems_sic_describe.exitcode.txt: 1
```

**解决方案**：
```bash
# 查看详细错误
cat reports/cmems_sic_describe.stderr.txt

# 检查 Copernicus 服务状态
# https://marine.copernicus.eu/

# 升级 CLI
pip install --upgrade copernicusmarine

# 检查网络连接
ping marine.copernicus.eu
```

---

## ✅ 验收检查清单

### Phase 9 收口

- [ ] 脚本执行成功（无错误）
- [ ] 确认没有数据文件被提交
- [ ] 所有测试通过
- [ ] 已推送到远程
- [ ] PR 已创建
- [ ] PR 描述已填写

### Phase 9.1 诊断

- [ ] 诊断脚本执行成功
- [ ] 查看了 exitcode.txt
- [ ] 查看了 stderr.txt（如果有）
- [ ] 分析了根因
- [ ] 记录了诊断结果

---

## 📚 相关文档

| 文档 | 用途 |
|------|------|
| `PHASE_9_CLOSURE_AND_PHASE_91_PLAN.md` | 详细计划和工作流 |
| `PHASE_9_QUICK_REFERENCE.md` | 快速参考和常见命令 |
| `PHASE_9_COMPLETION_SUMMARY.md` | 完成总结报告 |
| `PHASE_9_1_NEXTSIM_HM_TRACKING.md` | 问题追踪和诊断 |

---

## 🎯 预期结果

### 成功标志

```
✅ Phase 9 收口完成
   - 没有数据文件被提交
   - 所有测试通过
   - 已推送到远程

✅ PR 已创建
   - 标题正确
   - 描述完整
   - 可以合并

✅ Phase 9.1 诊断工具就绪
   - 脚本可以运行
   - 诊断文件已生成
   - 可以分析根因
```

### 失败标志

```
❌ 脚本执行失败
   → 查看错误信息，按故障排除步骤处理

❌ 测试失败
   → 修复代码，重新运行测试

❌ 推送失败
   → 检查网络连接，重新推送

❌ 诊断脚本失败
   → 检查 PowerShell 版本，升级 CLI
```

---

## ⏱️ 时间表

| 步骤 | 时间 | 状态 |
|------|------|------|
| 运行收口脚本 | 5 分钟 | ⏳ 待执行 |
| 创建 PR | 2 分钟 | ⏳ 待执行 |
| 诊断 Phase 9.1 | 3 分钟 | ⏳ 可选 |
| **总计** | **10 分钟** | ⏳ 待执行 |

---

## 🔗 快速链接

- **GitHub 仓库**：https://github.com/Cassian2006/ArcticRoute-Planner
- **创建 PR**：https://github.com/Cassian2006/ArcticRoute-Planner/compare
- **Copernicus 服务**：https://marine.copernicus.eu/
- **copernicusmarine CLI**：https://github.com/mercator-ocean/copernicusmarine-toolbox

---

## 📞 需要帮助？

1. **查看快速参考**：`PHASE_9_QUICK_REFERENCE.md`
2. **查看详细计划**：`PHASE_9_CLOSURE_AND_PHASE_91_PLAN.md`
3. **查看问题追踪**：`PHASE_9_1_NEXTSIM_HM_TRACKING.md`

---

**最后更新**：2025-12-15  
**状态**：✅ 准备就绪  
**下一步**：执行 `.\scripts\phase9_closure.ps1`

