# Phase 9 快速参考指南

## 一句话总结

**Phase 9 收口**：运行一个脚本完成 PR 合并前的所有检查；**Phase 9.1 诊断**：用新脚本诊断 nextsim HM describe 问题。

---

## Phase 9 收口（5 分钟）

### 快速执行

```powershell
# 在项目根目录执行
.\scripts\phase9_closure.ps1
```

脚本会自动：
1. ✅ 检查没有误提交数据
2. ✅ 显示 diff 统计
3. ✅ 询问是否还原 __init__.py
4. ✅ 运行测试
5. ✅ 提交并推送

### 手动步骤（如果脚本失败）

```bash
# 1. 检查数据
git ls-files | grep -E "data/cmems_cache|reports/cmems_"
# 应该没有输出

# 2. 查看改动
git diff --stat origin/main...HEAD

# 3. 还原 __init__.py（可选）
git checkout -- ArcticRoute/__init__.py ArcticRoute/core/__init__.py ArcticRoute/core/eco/__init__.py

# 4. 测试
python -m pytest -q

# 5. 推送
git add -A && git commit -m "chore: reduce diff noise" && git push
```

### 创建 PR

访问：https://github.com/Cassian2006/ArcticRoute-Planner

创建 PR，标题：
```
Phase 9: Multi-objective Route Planning with CMEMS Integration
```

---

## Phase 9.1 诊断（3 分钟）

### 快速诊断

```powershell
# 运行诊断脚本
.\scripts\phase91_diagnose_nextsim.ps1

# 查看结果
Get-Content reports\cmems_sic_describe.nextsim.exitcode.txt
Get-Content reports\cmems_sic_describe.nextsim.log | Select-Object -First 50
```

### 或用 Python

```bash
# 运行改进的脚本
python scripts/cmems_refresh_and_export.py --describe-only

# 查看诊断文件
cat reports/cmems_sic_describe.exitcode.txt
cat reports/cmems_sic_describe.stderr.txt
```

### 诊断结果解读

| 退出码 | 含义 |
|--------|------|
| 0 | ✅ 成功 |
| 1 | ❌ API 错误或网络问题 |
| -1 | ⏱️ 超时（60秒） |
| -2 | 💥 异常 |

### 根因排查

```powershell
# 查看具体错误
Get-Content reports\cmems_sic_describe.stderr.txt

# 查看兜底检索结果
Get-Content reports\cmems_sic_probe_nextsim.txt
Get-Content reports\cmems_sic_probe_product.txt

# 检查 CLI 版本
copernicusmarine --version

# 升级 CLI（如果需要）
pip install --upgrade copernicusmarine
```

---

## 文件清单

### 新增脚本

```
scripts/
├── phase9_closure.ps1              # Phase 9 收口脚本
└── phase91_diagnose_nextsim.ps1    # Phase 9.1 诊断脚本
```

### 改进的脚本

```
scripts/
└── cmems_refresh_and_export.py     # 添加 stderr + exit code 捕获
```

### 文档

```
├── PHASE_9_CLOSURE_AND_PHASE_91_PLAN.md    # 详细计划
├── PHASE_9_QUICK_REFERENCE.md              # 本文件
└── PHASE_9_1_NEXTSIM_HM_TRACKING.md        # 问题追踪（已更新）
```

---

## 常见命令

### Git 相关

```bash
# 查看当前分支
git branch -v

# 查看改动
git diff origin/main...HEAD --stat

# 查看具体改动（某个文件）
git diff origin/main...HEAD -- ArcticRoute/__init__.py

# 还原某个文件
git checkout -- ArcticRoute/__init__.py

# 查看日志
git log --oneline -10
```

### 测试相关

```bash
# 运行所有测试
python -m pytest -q

# 运行特定测试
python -m pytest tests/test_cmems_loader.py -v

# 运行并显示输出
python -m pytest -s

# 跳过某些测试
python -m pytest -k "not slow" -q
```

### 诊断相关

```bash
# 查看 reports 目录
ls -la reports/

# 查看诊断文件
cat reports/cmems_sic_describe.exitcode.txt
cat reports/cmems_sic_describe.stderr.txt
cat reports/cmems_sic_describe.nextsim.log

# 查看 Copernicus 配置
cat reports/cmems_resolved.json
```

---

## 故障排除

### 问题 1：脚本权限不足

```powershell
# 允许执行脚本
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 问题 2：找不到 git 命令

```powershell
# 检查 git 是否安装
git --version

# 如果没有，安装 Git for Windows
# https://git-scm.com/download/win
```

### 问题 3：pytest 找不到

```bash
# 安装 pytest
pip install pytest

# 或升级
pip install --upgrade pytest
```

### 问题 4：describe 仍然失败

```bash
# 检查 Copernicus 服务状态
# https://marine.copernicus.eu/

# 检查网络连接
ping marine.copernicus.eu

# 升级 CLI
pip install --upgrade copernicusmarine

# 查看详细错误
cat reports/cmems_sic_describe.stderr.txt
```

---

## 时间表

| 任务 | 时间 | 状态 |
|------|------|------|
| Phase 9 收口 | 5 分钟 | 准备就绪 |
| Phase 9.1 诊断 | 3 分钟 | 准备就绪 |
| PR 创建 | 2 分钟 | 待执行 |
| Code Review | 待定 | 待执行 |

---

## 下一步

1. ✅ 运行 `.\scripts\phase9_closure.ps1`
2. ✅ 在 GitHub 创建 PR
3. ⏳ 等待 code review
4. ⏳ 运行 `.\scripts\phase91_diagnose_nextsim.ps1`（诊断）
5. ⏳ 根据诊断结果改进脚本

---

## 联系方式

有问题？查看详细文档：
- `PHASE_9_CLOSURE_AND_PHASE_91_PLAN.md` - 完整计划
- `PHASE_9_1_NEXTSIM_HM_TRACKING.md` - 问题追踪

---

**最后更新**：2025-12-15
**状态**：✅ 准备就绪

