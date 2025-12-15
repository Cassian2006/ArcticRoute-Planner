# Phase 9 收口 + Phase 9.1 诊断计划

## 概述

本文档描述了如何完成 Phase 9 的 PR 收口工作，以及如何诊断和改进 Phase 9.1 中 nextsim HM describe 空文件问题。

---

## Phase 9 收口：PR 合并前检查

### 目标
确保 PR 满足以下条件，可以安全合并到 main：
1. ✅ 没有误提交数据/缓存文件
2. ✅ diff 统计清晰，文件来源明确
3. ✅ 不必要的格式调整已还原
4. ✅ 所有测试通过
5. ✅ 已推送到远程，可创建 PR

### 执行步骤

#### 方式 1：使用自动化脚本（推荐）

```powershell
# 在项目根目录执行
.\scripts\phase9_closure.ps1
```

脚本会自动执行以下步骤：
1. 检查是否有误提交的数据文件（data/cmems_cache, reports/cmems_*）
2. 显示 diff 统计（相对于 origin/main）
3. 检查 __init__.py 改动，询问是否还原
4. 运行完整测试 (`pytest -q`)
5. 提交并推送到远程

#### 方式 2：手动执行

```bash
# 1. 检查误提交数据
git ls-files | grep -E "data/cmems_cache|ArcticRoute/data_processed|reports/cmems_"
# 应该没有输出

# 2. 检查 diff 统计
git diff --stat origin/main...HEAD
# 查看改动的文件数和行数

# 3. 检查 __init__.py 改动
git diff origin/main...HEAD -- ArcticRoute/__init__.py ArcticRoute/core/__init__.py ArcticRoute/core/eco/__init__.py

# 4. 如果只是格式调整，还原它们
git checkout -- ArcticRoute/__init__.py ArcticRoute/core/__init__.py ArcticRoute/core/eco/__init__.py

# 5. 运行测试
python -m pytest -q

# 6. 提交并推送
git add -A
git commit -m "chore: reduce diff noise (revert formatting-only __init__ changes)" || true
git push
```

### 验收点

- [ ] 没有数据文件被追踪
- [ ] diff 统计显示 ~399 个文件，~34k 行新增
- [ ] 所有测试通过（pytest -q 返回 0）
- [ ] 已推送到远程

### PR 创建

完成上述步骤后，访问 GitHub：

```
https://github.com/Cassian2006/ArcticRoute-Planner
```

创建 PR，填写以下内容：

**标题**：
```
Phase 9: Multi-objective Route Planning with CMEMS Integration
```

**描述**：
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

---

## Phase 9.1：nextsim HM describe 问题诊断

### 问题描述

在 Phase 9 中，`copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm` 命令经常返回空输出或超时，导致无法获取该数据集的变量列表。

**症状**：
- 输出为 0 字节或 < 100 字节
- 无法判断是网络错误、API 错误还是其他问题
- 当前脚本只捕获 stdout，stderr 被忽略

### 改进方案

#### 1. 诊断脚本（PowerShell）

使用 `scripts/phase91_diagnose_nextsim.ps1` 进行诊断：

```powershell
.\scripts\phase91_diagnose_nextsim.ps1
```

**功能**：
- 执行 describe 命令并同时捕获 stdout 和 stderr
- 记录退出码到 `reports/cmems_sic_describe.nextsim.exitcode.txt`
- 记录日志到 `reports/cmems_sic_describe.nextsim.log`
- 执行兜底检索（nextsim 关键词、产品 ID）
- 生成诊断报告

**输出文件**：
```
reports/
├── cmems_sic_describe.nextsim.json          # 如果成功
├── cmems_sic_describe.nextsim.tmp.txt       # 如果输出过短
├── cmems_sic_describe.nextsim.log           # 完整日志
├── cmems_sic_describe.nextsim.exitcode.txt  # 退出码
├── cmems_sic_probe_nextsim.txt              # nextsim 关键词检索结果
└── cmems_sic_probe_product.txt              # 产品 ID 检索结果
```

#### 2. 改进的 Python 脚本

已更新 `scripts/cmems_refresh_and_export.py` 的 describe-only 模式：

```bash
python scripts/cmems_refresh_and_export.py --describe-only
```

**改进点**：
- 不再使用 `check=True`，改为捕获所有返回码
- 记录 exit code 到 `reports/cmems_*_describe.exitcode.txt`
- 记录 stderr 到 `reports/cmems_*_describe.stderr.txt`
- 添加 60 秒超时控制
- 详细的错误日志

**输出文件**：
```
reports/
├── cmems_sic_describe.json                  # stdout（如果 >= 1000 字节）
├── cmems_sic_describe.exitcode.txt          # 退出码
├── cmems_sic_describe.stderr.txt            # stderr（如果有）
├── cmems_swh_describe.json                  # stdout（如果 >= 1000 字节）
├── cmems_swh_describe.exitcode.txt          # 退出码
└── cmems_swh_describe.stderr.txt            # stderr（如果有）
```

### 诊断工作流

#### 步骤 1：运行诊断脚本

```powershell
.\scripts\phase91_diagnose_nextsim.ps1
```

#### 步骤 2：检查诊断结果

```powershell
# 查看退出码
Get-Content reports\cmems_sic_describe.nextsim.exitcode.txt

# 查看错误日志
Get-Content reports\cmems_sic_describe.nextsim.log | Select-Object -First 50

# 查看兜底检索结果
Get-Content reports\cmems_sic_probe_nextsim.txt
Get-Content reports\cmems_sic_probe_product.txt
```

#### 步骤 3：分析根因

| 退出码 | 含义 | 可能原因 |
|--------|------|--------|
| 0 | 成功 | 命令执行成功，输出有效 |
| 1 | 一般错误 | API 错误、网络问题、关键词不匹配 |
| 2 | 误用 | 命令行参数错误 |
| 124 | 超时 | 命令执行超过 60 秒 |
| -1 | 超时（脚本） | Python 脚本捕获的超时 |
| -2 | 异常（脚本） | Python 脚本捕获的其他异常 |

#### 步骤 4：根据根因采取行动

**如果是网络问题**：
- 检查 Copernicus 服务状态：https://marine.copernicus.eu/
- 检查代理/防火墙设置
- 尝试使用 VPN

**如果是关键词不匹配**：
- 查看 `cmems_sic_probe_nextsim.txt` 了解 nextsim 相关数据集
- 查看 `cmems_sic_probe_product.txt` 了解产品 ID 相关数据集
- 更新配置中的数据集 ID

**如果是 CLI 版本问题**：
```bash
copernicusmarine --version
pip install --upgrade copernicusmarine
```

### 改进的脚本特性

#### 安全写入机制

```python
def _safe_atomic_write_text(target_path: Path, content: str, min_bytes: int = 1000) -> bool:
    """仅当内容 >= min_bytes 时才原子替换目标文件"""
    data = content.encode("utf-8")
    if len(data) < min_bytes:
        return False  # 输出过短，保留旧文件
    # 原子替换
    with tempfile.NamedTemporaryFile(...) as tf:
        tf.write(data)
        os.replace(temp_name, target_path)
    return True
```

**优点**：
- 防止 0 字节 JSON 文件覆盖有效数据
- 保留旧数据以便诊断
- 原子操作，避免部分写入

#### 完整的错误捕获

```python
try:
    res = subprocess.run(..., capture_output=True, timeout=60)
    # 记录 exit code
    (reports_dir / "cmems_sic_describe.exitcode.txt").write_text(str(res.returncode))
    # 记录 stderr
    if res.stderr:
        (reports_dir / "cmems_sic_describe.stderr.txt").write_text(res.stderr)
except subprocess.TimeoutExpired:
    # 记录超时
    (reports_dir / "cmems_sic_describe.exitcode.txt").write_text("-1")
except Exception as e:
    # 记录其他异常
    (reports_dir / "cmems_sic_describe.exitcode.txt").write_text("-2")
```

### 后续改进计划

#### 短期（Phase 9.1）
- ✅ 实现 stderr + exit code 捕获
- ✅ 创建诊断脚本
- [ ] 执行诊断并记录根因
- [ ] 更新 Phase 9.1 跟踪文档

#### 中期（Phase 9.2）
- [ ] 实现自动重试机制（指数退避）
- [ ] 添加缓存策略（避免频繁调用）
- [ ] 支持手动变量名配置

#### 长期（Phase 10+）
- [ ] 向 Copernicus 报告问题
- [ ] 评估替代数据源
- [ ] 实现离线数据同步

---

## 文件清单

### 新增脚本

| 文件 | 用途 |
|------|------|
| `scripts/phase9_closure.ps1` | Phase 9 收口自动化脚本 |
| `scripts/phase91_diagnose_nextsim.ps1` | Phase 9.1 诊断脚本 |

### 改进的脚本

| 文件 | 改进 |
|------|------|
| `scripts/cmems_refresh_and_export.py` | 添加 stderr + exit code 捕获 |

### 文档

| 文件 | 内容 |
|------|------|
| `PHASE_9_CLOSURE_AND_PHASE_91_PLAN.md` | 本文档 |
| `PHASE_9_1_NEXTSIM_HM_TRACKING.md` | 问题追踪（已更新） |

---

## 执行检查清单

### Phase 9 收口

- [ ] 运行 `.\scripts\phase9_closure.ps1`
- [ ] 确认所有测试通过
- [ ] 确认已推送到远程
- [ ] 在 GitHub 创建 PR
- [ ] 填写 PR 描述（包含验收点、测试结果）
- [ ] 请求 code review

### Phase 9.1 诊断

- [ ] 运行 `.\scripts\phase91_diagnose_nextsim.ps1`
- [ ] 检查诊断文件（exitcode, stderr, log）
- [ ] 分析根因
- [ ] 记录诊断结果到 PHASE_9_1_NEXTSIM_HM_TRACKING.md
- [ ] 根据根因采取行动

---

## 常见问题

### Q1: 脚本执行失败，提示找不到 git 命令

**A**: 确保 Git 已安装并在 PATH 中。在 PowerShell 中执行：
```powershell
git --version
```

### Q2: pytest 失败，如何调试？

**A**: 运行详细模式：
```bash
python -m pytest -v
```

查看失败的测试，修复代码后重新运行。

### Q3: describe 命令仍然返回空输出

**A**: 
1. 检查 Copernicus 服务状态
2. 检查网络连接和代理设置
3. 尝试升级 copernicusmarine CLI：`pip install --upgrade copernicusmarine`
4. 查看 `reports/cmems_sic_describe.stderr.txt` 了解具体错误

### Q4: 如何跳过某些测试？

**A**: 使用 `-k` 选项：
```bash
python -m pytest -k "not slow" -q
```

---

## 参考资源

- [Copernicus Marine Service](https://marine.copernicus.eu/)
- [copernicusmarine CLI 文档](https://github.com/mercator-ocean/copernicusmarine-toolbox)
- [Git 工作流](https://git-scm.com/book/en/v2)
- [pytest 文档](https://docs.pytest.org/)

---

**最后更新**：2025-12-15
**状态**：准备就绪

