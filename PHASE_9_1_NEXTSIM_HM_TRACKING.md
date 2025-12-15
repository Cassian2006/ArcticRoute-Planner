# Phase 9.1 - nextsim HM describe 问题追踪

## 问题概述

在 Phase 9 中，CMEMS 数据集集成过程中发现 `cmems_mod_arc_phy_anfc_nextsim_hm` 数据集的 `describe` 命令存在不稳定性：

- **现象**：`copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm` 经常返回空输出或超时
- **影响**：无法获取该数据集的变量列表，导致无法自动配置 SIC（海冰浓度）数据源
- **当前状态**：已切换到观测数据 `cmems_obs-si_arc_phy_my_l4_P1D` 作为主要 SIC 源

## 复现条件

### 环境信息
- **系统**：Windows 11
- **Python**：3.11.9
- **copernicusmarine CLI**：已安装（版本待确认）
- **网络**：可能存在代理/防火墙限制

### 复现步骤
```bash
# 直接运行 describe 命令
copernicusmarine describe --contains cmems_mod_arc_phy_anfc_nextsim_hm --return-fields all

# 或通过脚本
python scripts/cmems_refresh_and_export.py --describe-only --sic-dataset-id cmems_mod_arc_phy_anfc_nextsim_hm
```

### 观察到的行为
1. 命令执行时间过长（>30秒）
2. 返回空 JSON 或不完整输出
3. 偶尔返回有效数据，但不稳定

## 已采取的缓解措施

### 1. 安全写入机制（scripts/cmems_refresh_and_export.py）
- 实现了 `_safe_atomic_write_text()` 函数
- 仅当输出 ≥ 1000 字节时才原子替换目标文件
- describe-only 模式下，如果输出过短则保留旧文件并返回非零退出码
- 防止 0 字节 JSON 文件覆盖有效数据

### 2. 数据源切换
- 主 SIC 源：`cmems_obs-si_arc_phy_my_l4_P1D`（观测数据，稳定）
- 备用源：`cmems_mod_arc_phy_anfc_nextsim_hm`（模式数据，待稳定）
- 配置文件：`reports/cmems_resolved.json`

### 3. 诊断脚本
- `scripts/force_sic_describe.py`：强制刷新 SIC describe 输出
- `scripts/phase9_quick_check.py`：快速验证集成状态
- `scripts/phase9_validation.py`：完整验证流程

## 后续计划

### 短期（不阻塞 Phase 9 合并）
1. **网络诊断**
   - 检查 Copernicus 服务可用性
   - 记录 CLI 返回码和错误日志
   - 验证代理/防火墙配置

2. **CLI 版本检查**
   - 确认 copernicusmarine 版本
   - 测试不同版本的 describe 命令行为
   - 查看是否有已知 bug 修复

3. **变量名锁定**
   - 手动验证 nextsim_hm 的有效变量名
   - 记录到 `reports/cmems_nextsim_hm_variables.json`
   - 作为后备配置

### 中期（Phase 9.2 或后续）
1. **自动重试机制**
   - 在 describe 失败时自动重试（指数退避）
   - 实现超时控制和降级策略

2. **缓存策略**
   - 缓存已成功的 describe 输出
   - 定期刷新（如每周一次）
   - 避免频繁调用不稳定的 API

3. **配置灵活性**
   - 允许手动指定变量名（不依赖 describe）
   - 支持多个备用数据源配置
   - 提供 UI 选项切换数据源

### 长期（Phase 10+）
1. **Copernicus 反馈**
   - 向 Copernicus 报告 nextsim_hm describe 问题
   - 跟踪官方修复进度

2. **替代方案评估**
   - 评估其他北极海冰数据源
   - 考虑本地缓存策略
   - 研究离线数据同步方案

## 当前配置状态

### reports/cmems_resolved.json
```json
{
  "sic": {
    "dataset_id": "cmems_obs-si_arc_phy_my_l4_P1D",
    "variables": ["sic", "uncertainty_sic"]
  },
  "wav": {
    "dataset_id": "dataset-wam-arctic-1hr3km-be",
    "variables": [...]
  }
}
```

### 测试覆盖
- ✅ `tests/test_cmems_loader.py`：CMEMS 数据加载
- ✅ `tests/test_cmems_planner_integration.py`：规划器集成
- ⚠️ nextsim_hm 特定测试：待补充

## 相关文件
- `scripts/cmems_refresh_and_export.py`：主刷新脚本（包含安全写入）
- `scripts/force_sic_describe.py`：强制 describe 刷新
- `reports/cmems_resolved.json`：当前配置
- `.gitignore`：大文件忽略规则
- `PHASE_9_1_NEXTSIM_HM_TRACKING.md`：本文档

## 状态标记
- **Phase 9 完成条件**：✅ 已满足（使用观测数据）
- **Phase 9.1 阻塞**：❌ 不阻塞（nextsim_hm 为可选优化）
- **优先级**：中等（改进数据源多样性，但不影响核心功能）

---

**最后更新**：2025-12-15
**负责人**：AI Assistant
**状态**：进行中

