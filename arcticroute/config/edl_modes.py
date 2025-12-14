"""
EDL 模式配置模块。

定义三种规划模式的参数：
  - efficient: 弱 EDL，偏燃油/距离
  - edl_safe: 中等 EDL，偏风险规避
  - edl_robust: 强 EDL，风险 + 不确定性

这个模块被 CLI 脚本和 UI 共同使用，确保参数一致性。
"""

from __future__ import annotations

from typing import Dict, Any, Optional


# ============================================================================
# EDL 模式定义
# ============================================================================

EDL_MODES: Dict[str, Dict[str, Any]] = {
    "efficient": {
        "name": "Efficient",
        "description": "弱 EDL，偏燃油/距离",
        "display_name": "Efficient（弱 EDL，偏燃油/距离）",
        
        # EDL 相关参数
        "w_edl": 0.3,  # EDL 风险权重（相对较小）
        "use_edl": True,  # 启用 EDL
        "use_edl_uncertainty": False,  # 不考虑不确定性
        "edl_uncertainty_weight": 0.0,
        
        # 其他成本权重
        "ice_penalty": 4.0,  # 冰风险权重
        "wave_penalty": 0.0,  # 波浪权重
        
        # 相对因子（用于 UI 中的参数调整）
        "ice_penalty_factor": 0.5,  # 相对于基础值的倍率
        "wave_weight_factor": 0.5,
        "edl_weight_factor": 0.3,
    },
    
    "edl_safe": {
        "name": "EDL-Safe",
        "description": "中等 EDL，偏风险规避",
        "display_name": "EDL-Safe（中等 EDL，偏风险规避）",
        
        # EDL 相关参数
        "w_edl": 1.0,  # EDL 风险权重（中等）
        "use_edl": True,  # 启用 EDL
        "use_edl_uncertainty": False,  # 不考虑不确定性
        "edl_uncertainty_weight": 0.0,
        
        # 其他成本权重
        "ice_penalty": 4.0,  # 冰风险权重
        "wave_penalty": 0.0,  # 波浪权重
        
        # 相对因子
        "ice_penalty_factor": 2.0,
        "wave_weight_factor": 1.5,
        "edl_weight_factor": 1.0,
    },
    
    "edl_robust": {
        "name": "EDL-Robust",
        "description": "强 EDL，风险 + 不确定性",
        "display_name": "EDL-Robust（强 EDL，风险 + 不确定性）",
        
        # EDL 相关参数
        "w_edl": 1.0,  # EDL 风险权重（中等，但配合不确定性）
        "use_edl": True,  # 启用 EDL
        "use_edl_uncertainty": True,  # 考虑不确定性
        "edl_uncertainty_weight": 1.0,  # 不确定性权重
        
        # 其他成本权重
        "ice_penalty": 4.0,  # 冰风险权重
        "wave_penalty": 0.0,  # 波浪权重
        
        # 相对因子
        "ice_penalty_factor": 2.0,
        "wave_weight_factor": 1.5,
        "edl_weight_factor": 1.0,
    },
}


# ============================================================================
# 工具函数
# ============================================================================

def get_edl_mode_config(mode: str) -> Dict[str, Any]:
    """
    获取指定 EDL 模式的配置。
    
    Args:
        mode: 模式名称（"efficient", "edl_safe", "edl_robust"）
    
    Returns:
        模式配置字典
    
    Raises:
        ValueError: 如果模式不存在
    """
    if mode not in EDL_MODES:
        raise ValueError(
            f"Unknown EDL mode: {mode}. "
            f"Available modes: {', '.join(EDL_MODES.keys())}"
        )
    return EDL_MODES[mode].copy()


def list_edl_modes() -> list[str]:
    """
    列出所有可用的 EDL 模式。
    
    Returns:
        模式名称列表
    """
    return list(EDL_MODES.keys())


def get_edl_mode_display_name(mode: str) -> str:
    """
    获取 EDL 模式的显示名称。
    
    Args:
        mode: 模式名称
    
    Returns:
        显示名称
    
    Raises:
        ValueError: 如果模式不存在
    """
    config = get_edl_mode_config(mode)
    return config.get("display_name", config.get("name", mode))


def validate_edl_mode_config(config: Dict[str, Any]) -> bool:
    """
    验证 EDL 模式配置的完整性。
    
    Args:
        config: 配置字典
    
    Returns:
        True 如果配置有效，否则 False
    """
    required_keys = {
        "w_edl",
        "use_edl",
        "use_edl_uncertainty",
        "edl_uncertainty_weight",
        "ice_penalty",
        "wave_penalty",
    }
    return all(key in config for key in required_keys)


# ============================================================================
# 参数调优建议（文档）
# ============================================================================

"""
参数调优指南：

1. w_edl（EDL 风险权重）
   - 当前设置：efficient=0.3, edl_safe=1.0, edl_robust=1.0
   - 调优范围：0.0 ~ 2.0
   - 观察指标：EDL 风险成本占总成本的比例
   - 建议：
     * 若占比 < 2%，增加 w_edl（1.5~2.0）
     * 若占比 > 30%，减少 w_edl（0.5~0.7）
     * 若占比 5%~15%，保持当前值

2. edl_uncertainty_weight（不确定性权重）
   - 当前设置：edl_robust=1.0
   - 调优范围：0.0 ~ 3.0
   - 观察指标：不确定性成本占总成本的比例
   - 建议：
     * 若占比 < 1%，增加权重（2.0~3.0）
     * 若占比 > 20%，减少权重（0.3~0.5）
     * 若占比 5%~10%，保持当前值

3. ice_penalty（冰风险权重）
   - 当前设置：4.0（所有模式相同）
   - 调优范围：2.0 ~ 10.0
   - 观察指标：冰风险成本占总成本的比例
   - 建议：
     * 若占比 > 60%，降低 ice_penalty（2.0~3.0）
     * 若占比 < 20%，提高 ice_penalty（5.0~8.0）
"""















