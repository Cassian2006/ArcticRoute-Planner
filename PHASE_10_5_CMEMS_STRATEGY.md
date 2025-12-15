# Phase 10.5：CMEMS 数据源策略

## 概述

Phase 10.5 定义了 CMEMS（Copernicus Marine Environmental Monitoring Service）数据源的长期策略，特别是关于 nextsim HM 数据集的集成方式。

## 核心原则

### 1. 保持当前策略不变

**主要 SIC 数据源**（稳定且可靠）：
- 数据集 ID：`cmems_obs-si_arc_phy_my_l4_P1D`
- 类型：观测数据（Level 4）
- 覆盖范围：北极地区
- 更新频率：每日
- 状态：**生产环境就绪**

该数据源已在 Phase 8/9 中充分验证，具有：
- ✅ 稳定的 API 接口
- ✅ 完整的变量定义
- ✅ 可靠的数据质量
- ✅ 充分的历史数据

### 2. nextsim HM 作为"可用则优先"的增强

**备用/增强数据源**（待稳定）：
- 数据集 ID：`cmems_mod_arc_phy_anfc_nextsim_hm`
- 类型：模式数据（高分辨率预报）
- 覆盖范围：北极地区
- 分辨率：更高（相比观测数据）
- 状态：**有条件启用**

#### 集成策略

```
┌─────────────────────────────────────────────────────────┐
│ 数据源选择流程（Phase 10.5+）                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. 尝试加载 nextsim HM 数据                             │
│     ├─ 成功 → 使用 nextsim HM（高分辨率）               │
│     └─ 失败 → 继续步骤 2                                 │
│                                                          │
│  2. 加载观测数据（cmems_obs-si）                         │
│     ├─ 成功 → 使用观测数据（稳定）                      │
│     └─ 失败 → 返回错误                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### 实现位置

在 `arcticroute/io/cmems_loader.py` 中实现：

```python
def load_sic_with_fallback(
    ym: str,
    prefer_nextsim: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    加载 SIC 数据，优先尝试 nextsim HM，失败则回退到观测数据。
    
    Args:
        ym: 年月字符串 (YYYYMM)
        prefer_nextsim: 是否优先尝试 nextsim（默认 True）
    
    Returns:
        (sic_2d, metadata)
    """
    if prefer_nextsim:
        try:
            sic_2d, meta = load_sic_from_nextsim_hm(ym)
            meta["source"] = "nextsim_hm"
            logger.info(f"使用 nextsim HM 数据源 (ym={ym})")
            return sic_2d, meta
        except Exception as e:
            logger.warning(f"nextsim HM 加载失败: {e}，回退到观测数据")
    
    # 回退到观测数据
    sic_2d, meta = load_sic_from_obs(ym)
    meta["source"] = "cmems_obs-si"
    logger.info(f"使用观测数据源 (ym={ym})")
    return sic_2d, meta
```

## 配置管理

### 配置文件位置

`reports/cmems_resolved.json`

### 配置结构

```json
{
  "sic": {
    "primary": {
      "dataset_id": "cmems_obs-si_arc_phy_my_l4_P1D",
      "variable": "sea_ice_concentration",
      "status": "stable"
    },
    "enhanced": {
      "dataset_id": "cmems_mod_arc_phy_anfc_nextsim_hm",
      "variable": "sea_ice_concentration",
      "status": "conditional",
      "fallback_to_primary": true
    }
  },
  "swh": {
    "dataset_id": "cmems_obs-waves_arc_phy_l4_my_P1D",
    "variable": "sea_surface_wave_significant_height",
    "status": "stable"
  }
}
```

## 状态追踪

### nextsim HM 已知问题

参考：`PHASE_9_1_NEXTSIM_HM_TRACKING.md`

**问题**：`copernicusmarine describe` 命令不稳定
- 返回空输出或超时
- 影响自动变量发现

**缓解措施**：
1. 实现了安全写入机制（防止 0 字节文件）
2. 切换到观测数据作为主源
3. nextsim 作为可选增强

**后续计划**：
- 监控 Copernicus 服务稳定性
- 定期测试 nextsim HM 可用性
- 一旦稳定，自动切换为主源

## 使用指南

### 对于用户

1. **默认行为**：系统自动使用观测数据（稳定）
2. **启用 nextsim**：在配置中设置 `prefer_nextsim=True`
3. **监控日志**：查看日志了解实际使用的数据源

### 对于开发者

1. **添加新数据源**：
   - 在 `cmems_loader.py` 中添加 `load_xxx_from_yyy()` 函数
   - 更新 `load_sic_with_fallback()` 的优先级列表
   - 更新 `reports/cmems_resolved.json`

2. **测试数据源**：
   ```bash
   python scripts/test_cmems_pipeline.py --test-nextsim
   ```

3. **诊断问题**：
   ```bash
   python scripts/phase9_quick_check.py
   ```

## 性能考虑

### nextsim HM 优势

- ✅ 更高的空间分辨率（可能 ~2km vs ~25km）
- ✅ 更频繁的更新（可能每 6 小时 vs 每天）
- ✅ 物理模式更新的预报能力

### 观测数据优势

- ✅ 稳定的 API 接口
- ✅ 可靠的数据质量
- ✅ 充分的历史数据
- ✅ 无需担心模式偏差

## 时间表

| 阶段 | 目标 | 预期时间 |
|------|------|---------|
| Phase 10 | POLARIS 集成完成 | 当前 |
| Phase 10.5 | CMEMS 策略定义 | 当前 |
| Phase 11 | nextsim HM 稳定性评估 | TBD |
| Phase 12+ | 条件切换为 nextsim 主源 | TBD |

## 相关文档

- `PHASE_8_CMEMS_INGESTION_SUMMARY.md`：CMEMS 数据摄入总结
- `PHASE_9_1_NEXTSIM_HM_TRACKING.md`：nextsim HM 问题追踪
- `CMEMS_QUICK_START.md`：CMEMS 快速开始指南
- `docs/CMEMS_WORKFLOW.md`：CMEMS 工作流文档

## 审批与反馈

- **审批人**：[待指定]
- **审批日期**：[待指定]
- **反馈渠道**：GitHub Issues / Pull Requests

---

**文档版本**：1.0  
**最后更新**：2024-12-15  
**状态**：草稿 → 待审批

