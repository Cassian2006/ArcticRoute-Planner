"""Natural language explainers for planner reports."""

from __future__ import annotations

import json
from time import perf_counter
from typing import Any, Dict

from .config import load_ai_config
from .providers.common import (
    add_safety_guardrails,
    log_ai_call,
    redact_paths,
    trim_dict_for_llm,
)
from .providers.moonshot_k2 import MoonshotK2Client, MoonshotK2Error
from .schema import ExplainerOutput


def explain_single(
    run_report: Dict[str, Any],
    params: Dict[str, Any],
    *,
    use_llm: bool = True,
) -> ExplainerOutput:
    """Explain a single run using LLM with template fallback."""
    fallback = _template_single(run_report, params)
    if not use_llm:
        return fallback
    prompt = _build_single_prompt(run_report, params)
    try:
        data = _call_llm(prompt, action="explainer.single")
        output = ExplainerOutput.model_validate(data)
        setattr(output, "_from_llm", True)
        output.key_deltas.setdefault("_source", "llm")
        return output
    except (MoonshotK2Error, ValueError, RuntimeError, json.JSONDecodeError):
        fallback.key_deltas["_source"] = "template"
        setattr(fallback, "_from_llm", False)
        return fallback


def explain_compare(
    report_a: Dict[str, Any],
    report_b: Dict[str, Any],
    *,
    use_llm: bool = True,
) -> ExplainerOutput:
    """Explain differences between two runs with LLM fallback."""
    fallback = _template_compare(report_a, report_b)
    if not use_llm:
        return fallback
    prompt = _build_compare_prompt(report_a, report_b)
    try:
        data = _call_llm(prompt, action="explainer.compare")
        output = ExplainerOutput.model_validate(data)
        setattr(output, "_from_llm", True)
        output.key_deltas.setdefault("_source", "llm")
        return output
    except (MoonshotK2Error, ValueError, RuntimeError, json.JSONDecodeError):
        fallback.key_deltas["_source"] = "template"
        setattr(fallback, "_from_llm", False)
        return fallback


# ----------------------------
# Prompt and LLM interactions
# ----------------------------


def _call_llm(prompt: str, *, action: str) -> Dict[str, Any]:
    cfg = load_ai_config()
    client = MoonshotK2Client(
        model=cfg.model_name,
        timeout=cfg.timeout_s,
    )
    start = perf_counter()
    try:
        raw = client.complete(
            prompt,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
        )
        data = _parse_llm_json(raw)
        meta = getattr(client, "_last_meta", {})
        retries = max(0, int(meta.get("attempts", 1)) - 1)
        duration = perf_counter() - start
        log_ai_call(action, duration_s=duration, retries=retries, degraded=False)
        return data
    except (MoonshotK2Error, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        meta = getattr(client, "_last_meta", {})
        retries = max(0, int(meta.get("attempts", 1)) - 1)
        duration = perf_counter() - start
        log_ai_call(action, duration_s=duration, retries=retries, degraded=True, error=str(exc))
        raise


def _build_single_prompt(report: Dict[str, Any], params: Dict[str, Any]) -> str:
    sanitized_report = trim_dict_for_llm(redact_paths(report))
    sanitized_params = trim_dict_for_llm(redact_paths(params))
    payload = {
        "run_report": sanitized_report,
        "params": sanitized_params,
        "requirements": {
            "language": "zh",
            "markdown_length_chars": "300-500",
            "sections": ["概览", "风险表现", "走廊覆盖", "建议"],
            "output_schema": {
                "markdown": "str",
                "bullets": "list[str]",
                "key_deltas": "dict[str, number|str]",
            },
        },
    }
    prompt = (
        "你是航线规划解释助手。请根据输入总结关键指标并输出严格 JSON，字段为："
        "markdown（约300-500字中文Markdown）、bullets（3-6条要点列表）、key_deltas（关键数值字典）。\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )
    return add_safety_guardrails(prompt)


def _build_compare_prompt(report_a: Dict[str, Any], report_b: Dict[str, Any]) -> str:
    sanitized_a = trim_dict_for_llm(redact_paths(report_a))
    sanitized_b = trim_dict_for_llm(redact_paths(report_b))
    payload = {
        "report_a": sanitized_a,
        "report_b": sanitized_b,
        "focus_metrics": ["total_cost", "mean_risk", "geodesic_length_m", "nearest_accident_km"],
        "requirements": {
            "language": "zh",
            "markdown_length_chars": "300-500",
            "output_schema": {
                "markdown": "str",
                "bullets": "list[str]",
                "key_deltas": "dict[str, number|str|dict]",
            },
        },
    }
    prompt = (
        "你是航线规划解析助手。比较两次运行，描述成本、风险、航程与事故距离的变化及原因。"
        "请输出严格 JSON，字段：markdown（300-500字中文Markdown）、bullets（概括要点列表）、key_deltas（包含主要指标差异）。\n"
        "```json\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
        "```"
    )
    return add_safety_guardrails(prompt)


def _parse_llm_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if "```" in cleaned:
        segments = cleaned.split("```")
        for segment in segments:
            segment = segment.strip()
            if segment.startswith("{") and segment.endswith("}"):
                cleaned = segment
                break
    return json.loads(cleaned)


# ----------------------------
# Template fallbacks
# ----------------------------


def _template_single(report: Dict[str, Any], params: Dict[str, Any]) -> ExplainerOutput:
    total_cost = float(report.get("total_cost", 0.0))
    mean_risk = float(report.get("mean_risk", report.get("risk_mean", 0.0)))
    max_risk = float(report.get("max_risk", 0.0))
    length_m = float(report.get("geodesic_length_m", 0.0))
    length_nm = length_m / 1852 if length_m else 0.0
    corr_stats = report.get("corridor_stats", {}) or {}
    corridor_coverage = float(corr_stats.get("coverage", 0.0))
    corridor_mean = float(corr_stats.get("mean", 0.0))
    accident_stats = report.get("accident_stats", {}) or {}
    accident_mean = float(accident_stats.get("mean", 0.0))
    nearest = report.get("nearest_accident_km", {}) or {}
    nearest_min = float(nearest.get("min", 0.0))

    beta = float(params.get("beta", 3.0))
    gamma = float(params.get("gamma", 0.3))
    p_exp = float(params.get("p", 1.0))
    beta_a = float(params.get("beta_a", 0.0))

    markdown = (
        f"本次航线规划在当前参数 β={beta:.2f}、γ={gamma:.2f}、p={p_exp:.2f}、beta_a={beta_a:.2f} 下完成。"
        f" 线路总成本约 {total_cost:,.0f}，等效航程约 {length_nm:.1f} 海里。平均风险 {mean_risk:.3f}，"
        f"峰值风险 {max_risk:.3f}。走廊覆盖率 {corridor_coverage:.2%}、平均走廊概率 {corridor_mean:.3f}，"
        f"说明航线大部分时间处于走廊之外，建议持续观察。事故概率均值 {accident_mean:.6f}，"
        f"最近事故距离 {nearest_min:.1f} 公里，属于可接受区间。综合来看，"
        f"当前参数能在维持风险水平的同时控制成本，但仍可通过提升走廊约束或优化事故权重来改善航迹稳定性。"
    )
    bullets = [
        f"总成本 {total_cost:,.0f}，平均风险 {mean_risk:.3f}。",
        f"走廊覆盖率 {corridor_coverage:.1%}，建议关注 γ 调节。",
        f"最近事故距离 {nearest_min:.1f} 公里，事故均值 {accident_mean:.6f}。",
        "若希望提升走廊依从，可提高 γ 并加强走廊权重。",
    ]
    key_deltas = {
        "total_cost": total_cost,
        "mean_risk": mean_risk,
        "max_risk": max_risk,
        "corridor_coverage": corridor_coverage,
        "nearest_accident_min_km": nearest_min,
    }
    return ExplainerOutput(markdown=markdown, bullets=bullets, key_deltas=key_deltas)


def _template_compare(report_a: Dict[str, Any], report_b: Dict[str, Any]) -> ExplainerOutput:
    def _metric(report: Dict[str, Any], key: str) -> float:
        value = report.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        return 0.0

    cost_a, cost_b = _metric(report_a, "total_cost"), _metric(report_b, "total_cost")
    mean_risk_a = float(report_a.get("mean_risk", report_a.get("risk_mean", 0.0)))
    mean_risk_b = float(report_b.get("mean_risk", report_b.get("risk_mean", 0.0)))
    length_a = float(report_a.get("geodesic_length_m", 0.0))
    length_b = float(report_b.get("geodesic_length_m", 0.0))
    near_a = (report_a.get("nearest_accident_km") or {}).get("min", 0.0) or 0.0
    near_b = (report_b.get("nearest_accident_km") or {}).get("min", 0.0) or 0.0

    delta_cost = cost_b - cost_a
    delta_risk = mean_risk_b - mean_risk_a
    delta_length = length_b - length_a
    delta_nearest = float(near_b) - float(near_a)

    markdown = (
        f"对比两次运行，方案A 成本 {cost_a:,.0f}，方案B 为 {cost_b:,.0f}，差值 {delta_cost:,.0f}。"
        f"平均风险由 {mean_risk_a:.3f} 调整至 {mean_risk_b:.3f}（Δ={delta_risk:+.3f}），"
        f"航程由 {length_a/1852:.1f} 海里变为 {length_b/1852:.1f} 海里。"
        f"最近事故距离从 {float(near_a):.1f} km 变为 {float(near_b):.1f} km（Δ={delta_nearest:+.1f}）。"
        f"若成本上升同时风险下降，说明更强的走廊或事故权重起效；若成本下降但风险上升，则需回顾参数取舍，"
        f"确保安全与效率之间的平衡。建议结合任务目标进一步调优参数，并复核对应的走廊覆盖表现。"
    )
    bullets = [
        f"成本差值：{delta_cost:,.0f}",
        f"平均风险变化：{delta_risk:+.3f}",
        f"航程变化：{delta_length/1852:+.1f} 海里",
        f"最近事故距离变化：{delta_nearest:+.1f} 公里",
    ]
    key_deltas = {
        "total_cost_delta": delta_cost,
        "mean_risk_delta": delta_risk,
        "geodesic_length_delta_m": delta_length,
        "nearest_accident_min_delta_km": delta_nearest,
    }
    return ExplainerOutput(markdown=markdown, bullets=bullets, key_deltas=key_deltas)


__all__ = ["explain_single", "explain_compare"]
