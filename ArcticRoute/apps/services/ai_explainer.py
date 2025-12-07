"""
AI explainer service for routes, backed by Moonshot K2.
- 复用 legacy 中已存在的 MoonshotK2Client，不重复造轮子。
- 只负责构造 prompt 与调用，环境变量缺失时抛出可读异常。
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict

try:
    # 复用已存在的客户端
    from ArcticRoute.legacy.ai.providers.moonshot_k2 import (
        MoonshotK2Client,
        MoonshotK2Error,
    )
except Exception:  # pragma: no cover - 环境不完整时的兜底
    MoonshotK2Client = None  # type: ignore
    MoonshotK2Error = Exception  # type: ignore


class AIExplainError(RuntimeError):
    """Raised when AI explanation fails in a controlled, user-friendly way."""


def _get_model_name() -> str:
    # 优先读取项目 README 中提到的变量名；多种兼容
    return (
        os.getenv("MOONSHOT_MODEL")
        or os.getenv("MOONSHOT_MODEL_NAME")
        or os.getenv("AI_MODEL_NAME")
        or "moonshot-k2"
    )


def generate_route_explanation(payload: Dict[str, Any]) -> str:
    """
    调用 Moonshot K2 接口，对单条航线进行自然语言解释。
    输入 payload 为结构化信息（距离、风险分解、Eco 指标等）。
    返回一段人类可读的中文解释文本。

    注意：
    - 不在此函数内做任何网络外的副作用；失败抛 AIExplainError。
    - 若未配置 MOONSHOT_API_KEY，将抛出可读错误。
    """
    api_key = os.getenv("MOONSHOT_API_KEY")
    if not api_key:
        raise AIExplainError("未配置 AI 解释器：缺少环境变量 MOONSHOT_API_KEY")

    if MoonshotK2Client is None:
        raise AIExplainError("未找到 MoonshotK2 客户端模块（ArcticRoute.legacy.ai.providers.moonshot_k2）")

    # 严格的系统提示：限定单位与字段含义，避免编造；要求“结构化分析 + 建议”而非“罗列数值”
    system_prompt = (
        "你是一个北极航线规划分析官，面向评委和非技术听众，用通俗但专业的中文解释一次航线规划结果。\n"
        "你会收到一份 JSON，包含：距离、步骤数、各类风险积分、燃油与排放、Eco 模式、护航与历史主航线先验等。\n"
        "你的任务：先给整体简短概览，再分别从安全性、效率/距离、绿色性三个角度做定性分析，最后给出 1–2 条建议。\n"
        "必须严格只依据 JSON 事实，可做合理定性判断，但不能编造字段或背景。\n\n"
        "重要约束：\n"
        " - 距离单位 km；可用‘约’做近似描述。\n"
        " - total_cost 是无量纲代价积分，不是货币，禁止写成美元/人民币。\n"
        " - 燃油/排放单位为吨；fuel_cost_estimate 的币种见 fuel_cost_currency，禁止自行猜测或换算。\n"
        " - risk_layers.ice/accident/interact 属于安全/环境/交通风险；prior_penalty 代表偏离历史主航道的惩罚，不等同于自然或交通危险。\n"
        " - risk_layers.*.level 为 none/low/medium/high；none 或 share=0 应表述为‘积分为 0 或可忽略’，不可说‘绝对没有风险’。\n"
        " - quality.cost_fallback_used=True 时，必须提示‘当前基于简化/兜底风险图，解释偏示意性’；False 可说‘解释具有一定参考价值’。\n"
        " - 禁止出现拼音碎片或奇怪英文（如 kan'yi），全部使用自然中文。\n\n"
        "整体风险表述规则：\n"
        " - 若 metrics.risk_overall_level=low，用‘整体风险水平偏低’或‘总体风险压力不大’。\n"
        " - 若为 medium，用‘整体风险处于中等水平/中等偏上’等表述。\n"
        " - 只有当为 high 时，才可使用‘高风险水平’。禁止在 total_cost 落在约 0.7~1.3 区间时直接称为‘高风险’。\n\n"
        "风格要求：使用小标题（如‘整体概览’‘安全性分析’‘绿色性与经济性’‘建议与适用场景’）；\n"
        "每节最多点出 1–2 个关键数字，其余用‘约/大致/偏高/偏低/适中’等定性表述；\n"
        "禁止连续使用‘该航线总长度为…属于…’‘风险值为…占比…’这类模板句式；\n"
        "避免机械罗列字段名与原始数值，不要输出 JSON。"
    )

    # 面向任务的用户提示：明确结构，要求用 length_level / fuel_level_hint 等辅助标签
    user_prompt = (
        "下面是本次北极航线规划的结构化结果 JSON。\n"
        "请按以下结构输出解释：\n"
        "1. 整体概览：用 2-3 句话概括航线长度（结合 metrics.distance_km 与 metrics.length_level）和总体风险印象，"
        "   并简单提及 Eco 模式（eco.mode）。\n"
        "2. 安全性分析：结合 risk_layers 的 value/share/level，说明冰风险、事故风险、拥挤/碰撞风险分别处于什么水平，"
        "   哪一类是主要风险来源。可用‘主要压力来自…，其他维度相对温和/可忽略’等表达，避免逐条照抄数字。\n"
        "3. 绿色性与经济性：结合 eco 的燃油、排放与成本，使用 fuel_level_hint（low/medium/high）判断大致水平，"
        "   说明‘在该里程下属于偏高/中等/偏低’，并简要解释 Eco 模式（eco_model vs simple_distance）的含义。\n"
        "4. 额外因素：若 quality.escort_applied 为 true，说明护航可能起到的作用；若 quality.prior_weight>0，提到‘参考了历史主航线偏好’；"
        "   若 quality.cost_fallback_used 为 True，提醒读者本次解释基于简化风险图更偏示意；若为 False，可说‘解释具有一定参考价值’。\n"
        "5. 建议与适用场景：用 1-2 句话，从‘安全 vs 距离 vs 燃油’的权衡角度给出简要点评，例如‘稳健保守’或‘偏重效率但需关注冰风险’。\n\n"
        "请注意：所有分析必须从 JSON 中可合理推导，不要虚构地理/天气/历史对比；数字仅作支撑，不要机械罗列。\n\n"
        "以下是 JSON：```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )

    prompt = f"[系统说明]\n{system_prompt}\n\n[输入数据]\n{user_prompt}"

    try:
        client = MoonshotK2Client(model=_get_model_name(), timeout=float(os.getenv("MOONSHOT_TIMEOUT", "30")))
        text = client.complete(
            prompt,
            max_tokens=int(os.getenv("MOONSHOT_MAX_TOKENS", "800")),
            temperature=float(os.getenv("MOONSHOT_TEMPERATURE", "0.5")),
        )
        return (text or "").strip()
    except MoonshotK2Error as e:  # 来自客户端的受控失败
        raise AIExplainError(f"K2 调用失败：{e}") from e
    except Exception as e:  # 其它异常统一转成受控异常
        raise AIExplainError(f"AI 解释器未知错误：{e}") from e


__all__ = ["AIExplainError", "generate_route_explanation"]
