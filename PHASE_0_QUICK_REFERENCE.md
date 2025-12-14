# Phase 0：基线稳定化 - 快速参考

## 🚀 快速开始

### 1. 验证环境
```bash
python -m scripts.env_doctor --fail-on-contamination
# 预期：Exit code: 0
```

### 2. 运行测试
```bash
python -m pytest
# 预期：34 failed, 293 passed, 6 skipped
```

### 3. 检查特定测试
```bash
python -m pytest tests/test_ais_density_rasterize.py -v
```

## 📋 关键文件

| 文件 | 用途 | 位置 |
|------|------|------|
| pytest.ini | pytest 配置 | 项目根目录 |
| tests/conftest.py | pytest 钩子 | tests/ |
| scripts/env_doctor.py | 环境自检 | scripts/ |

## 🔍 常见问题

### Q1: 如何清理缓存？
```bash
# PowerShell
Remove-Item -Recurse -Force .pytest_cache -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Force -Directory -Filter __pycache__ | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
```

### Q2: 如何检查路径污染？
```bash
python -m scripts.env_doctor --fail-on-contamination
```

### Q3: 如何只运行通过的测试？
```bash
python -m pytest -m "not (failed or skipped)"
```

### Q4: 如何获取详细的失败信息？
```bash
python -m pytest --tb=long -v
```

## 📊 测试统计

```
总计：333 个测试
├─ 通过：293 个 (88.0%)
├─ 失败：34 个 (10.2%)
└─ 跳过：6 个 (1.8%)
```

## ✅ 验收标准

### 标准 1：env_doctor 退出码
```bash
python -m scripts.env_doctor --fail-on-contamination
# 期望：Exit code: 0
```

### 标准 2：pytest 无 collection error
```bash
python -m pytest --collect-only
# 期望：0 errors
```

## 🔧 配置说明

### pytest.ini
```ini
[pytest]
testpaths = tests              # 只收集 tests/ 目录
addopts = -q --import-mode=importlib  # 安静模式 + importlib
norecursedirs = .* build dist node_modules .venv venv minimum legacy  # 排除目录
```

### conftest.py 的作用
1. 清理 sys.path 中的 minimum 污染
2. 强制重新加载错误导入的模块
3. 确保项目根目录优先级最高

### env_doctor.py 的作用
1. 检查 Python 环境
2. 检查 sys.path 污染
3. 验证导入源位置
4. 支持 CI/CD 集成

## 📈 性能指标

| 指标 | 值 |
|------|-----|
| 测试总数 | 333 |
| 执行时间 | ~41 秒 |
| 吞吐量 | ~8 个测试/秒 |
| 通过率 | 88.0% |

## 🎯 下一步

### 立即行动
- [ ] 验证环境：`python -m scripts.env_doctor --fail-on-contamination`
- [ ] 运行测试：`python -m pytest`
- [ ] 查看报告：`PHASE_0_COMPLETION_REPORT.md`

### Phase 1 准备
- [ ] 修复 34 个失败的测试
- [ ] 完善 VesselProfile 实现
- [ ] 补充缺失的函数导出

## 📚 文档索引

| 文档 | 内容 |
|------|------|
| PHASE_0_COMPLETION_REPORT.md | 完整的完成报告 |
| PHASE_0_中文总结.md | 中文总结 |
| PHASE_0_FINAL_SUMMARY.txt | 最终总结 |
| PHASE_0_VERIFICATION_REPORT.md | 验证报告 |
| PHASE_0_执行总结.md | 执行总结 |
| PHASE_0_QUICK_REFERENCE.md | 快速参考（本文件） |

## 🔗 相关命令

### 测试相关
```bash
# 运行所有测试
python -m pytest

# 运行特定测试
python -m pytest tests/test_ais_density_rasterize.py

# 运行特定测试类
python -m pytest tests/test_ais_density_rasterize.py::test_rasterize_ais_density_basic

# 显示详细输出
python -m pytest -v

# 显示失败的测试
python -m pytest --tb=short

# 只运行失败的测试
python -m pytest --lf

# 运行最后失败的测试
python -m pytest --ff
```

### 环境相关
```bash
# 检查环境
python -m scripts.env_doctor

# 检查环境（失败时退出）
python -m scripts.env_doctor --fail-on-contamination

# 清理缓存
python -m pytest --cache-clear
```

### 收集相关
```bash
# 只收集测试，不运行
python -m pytest --collect-only

# 显示收集的测试数
python -m pytest --collect-only -q
```

## 💡 提示

1. **快速验证**：运行 `python -m scripts.env_doctor --fail-on-contamination` 确保环境正确

2. **调试失败**：使用 `python -m pytest --tb=long -v` 获取详细信息

3. **性能监控**：使用 `python -m pytest --durations=10` 查看最慢的 10 个测试

4. **并行运行**：安装 `pytest-xdist` 后使用 `python -m pytest -n auto` 并行运行测试

## 🆘 故障排除

### 问题：collection error
**解决**：检查 pytest.ini 配置和 conftest.py 是否正确

### 问题：导入错误
**解决**：运行 `python -m scripts.env_doctor` 检查环境

### 问题：测试失败
**解决**：查看 PHASE_0_COMPLETION_REPORT.md 了解已知的失败原因

---

**最后更新**：2024-12-14  
**分支**：feat/pareto-front  
**状态**：✅ Phase 0 完成

