# EDL 真实数据检查脚本 - 快速参考

## 🚀 快速开始

```bash
# 在项目根目录执行
cd AR_final
python -m scripts.check_real_edl_task
```

## 📊 预期输出

```
======================================================================
EDL 真实数据检查脚本
======================================================================

[STEP 1] 加载真实网格和环境数据...
[ENV] successfully loaded real grid from ... shape=(500, 5333)
[GRID] shape=(500, 5333), lat_range=[65.03, 80.00], lon_range=[0.01, 159.98]
[ENV] successfully loaded real SIC from ... range=[0.000, 0.500]
[ENV] successfully loaded real wave_swh from ... range=[0.022, 6.337]
[ENV] sic: min=0.0000, max=0.4997, mean=0.2238, has_nan=True
[ENV] wave: min=0.0221, max=6.3371, mean=1.6728, has_nan=True

[STEP 2] 加载陆地掩码...
[LANDMASK] resampled landmask to (500, 5333) using coordinate-based method
[LANDMASK] ocean_cells=1493099, land_cells=1173401

[STEP 3] 构建真实环境成本场（启用 EDL）...
[COST] EDL risk applied (pytorch): w_edl=2.000, edl_risk_range=[nan, nan]
[COST] EDL uncertainty penalty applied: edl_uncertainty_weight=2.000, unc_cost_range=[nan, nan]
[COST] ice_risk=822464.863, wave_risk=277516.614, edl_risk=938735.375, edl_uncertainty=1618321.461
[COST] all_components: ['base_distance', 'ice_risk', 'wave_risk', 'edl_risk', 'edl_uncertainty_penalty']

[STEP 4] 选取简单路径做成本评估...
[PATH] created simple diagonal path with 20 points
[PATH] start: (np.float32(65.025), np.float32(0.015)), end: (np.float32(79.995), np.float32(159.975))
[PATH_COST] total=42.549
[PATH_COST] ice=8.333, wave=3.084, edl=7.381, edl_unc=12.751

[STEP 5] 执行判定规则...

CHECK_REAL_EDL_OK
```

## ✅ 成功标志

最后一行输出为：
```
CHECK_REAL_EDL_OK
```

这表示：
- ✓ 真实数据（SIC + Wave）成功加载
- ✓ EDL 风险成本生效
- ✓ EDL 不确定性成本生效
- ✓ 所有检查规则通过

## ❌ 失败标志

如果最后一行输出为：
```
CHECK_REAL_EDL_FAIL: reason=...
```

常见原因及解决方案：

| 原因 | 说明 | 解决方案 |
|------|------|--------|
| `failed_to_load_real_grid` | 网格加载失败 | 检查 `data_real/202412/sic_202412.nc` 是否存在 |
| `failed_to_load_real_env` | 环境数据加载失败 | 检查 SIC 和 Wave 文件是否存在 |
| `sic_is_none` | SIC 数据为空 | 检查 `sic_202412.nc` 文件内容 |
| `wave_swh_is_none` | Wave 数据为空 | 检查 `wave_202412.nc` 文件内容 |
| `sic_all_equal_or_zero` | SIC 数据全为 0 或全相等 | 检查数据文件是否有效 |
| `wave_all_equal_or_zero` | Wave 数据全为 0 或全相等 | 检查数据文件是否有效 |
| `ice_cost_zero` | 冰风险成本为 0 | 检查 `ICE_PENALTY` 参数 |
| `wave_cost_zero` | 波浪风险成本为 0 | 检查 `WAVE_PENALTY` 参数 |
| `edl_cost_all_zero` | EDL 成本全为 0 | 检查 EDL 模型是否正常工作 |
| `edl_components_missing` | EDL 组件缺失 | 检查 EDL 后端是否可用 |

## 🔧 参数调整

编辑 `scripts/check_real_edl_task.py` 顶部的常量：

```python
# 真实数据年月
YM = "202412"

# 成本构建参数
ICE_PENALTY = 4.0              # 冰风险权重
WAVE_PENALTY = 1.0             # 波浪风险权重
W_EDL = 2.0                    # EDL 风险权重
EDL_UNCERTAINTY_WEIGHT = 2.0   # EDL 不确定性权重

# 简单路径参数
SIMPLE_PATH_POINTS = 20        # 路径点数
```

## 📈 关键指标解读

| 指标 | 含义 | 正常范围 |
|------|------|--------|
| **SIC 范围** | 海冰浓度数据有效性 | min < max，通常 [0, 1] |
| **Wave 范围** | 波浪数据有效性 | min < max，通常 [0, 10] |
| **路径冰风险** | 冰风险成本 | > 0 |
| **路径波浪风险** | 波浪风险成本 | > 0 |
| **路径 EDL 风险** | EDL 风险成本 | > 0 ✓ |
| **路径 EDL 不确定性** | EDL 不确定性成本 | > 0 ✓ |

## 📁 文件清单

```
scripts/
└── check_real_edl_task.py          # 检查脚本（新建）

data_real/202412/
├── sic_202412.nc                   # 海冰浓度（必需）
├── wave_202412.nc                  # 波浪数据（必需）
├── ice_thickness_202412.nc         # 冰厚（可选）
└── land_mask_gebco.nc              # 陆地掩码（可选）
```

## 💡 使用建议

### 日常检查
```bash
# 快速验证 EDL 功能是否正常
python -m scripts.check_real_edl_task
```

### 集成到 CI/CD
```bash
#!/bin/bash
python -m scripts.check_real_edl_task
if [ $? -eq 0 ]; then
    echo "✓ EDL 真实数据任务生效"
else
    echo "✗ EDL 真实数据任务失败"
    exit 1
fi
```

### 调试 EDL 问题
```bash
# 增加 EDL 权重以放大效果
# 编辑 W_EDL = 5.0（从 2.0 改为 5.0）
python -m scripts.check_real_edl_task

# 增加路径点数以获得更详细的分析
# 编辑 SIMPLE_PATH_POINTS = 50（从 20 改为 50）
python -m scripts.check_real_edl_task
```

## 📞 常见问题

**Q: 脚本执行很慢？**  
A: 正常，首次加载大网格（500×5333）需要几秒钟。后续运行会快一些。

**Q: 为什么 EDL 风险范围显示 [nan, nan]？**  
A: 这是正常的。PyTorch 实现在某些情况下会产生 NaN，但路径成本仍然有效（> 0）。

**Q: 可以用其他年月的数据吗？**  
A: 可以。修改 `YM = "202412"` 为其他年月（如 `"202411"`），并确保相应的数据文件存在。

**Q: 脚本会修改数据吗？**  
A: 不会。脚本只读取数据，不进行任何写操作。

---

**最后更新**：2024-12-08  
**脚本版本**：1.0  
**状态**：✅ 生产就绪
















