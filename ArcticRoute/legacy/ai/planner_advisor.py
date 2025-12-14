"""Advisor utilities combining rule-based and LLM-driven recommendations."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, Optional, Tuple

from .config import load_ai_config
from time import perf_counter

from .providers.common import (
    add_safety_guardrails,
    log_ai_call,
    redact_paths,
    trim_dict_for_llm,
)
from .providers.moonshot_k2 import MoonshotK2Client, MoonshotK2Error
from .schema import AdvisorInput, AdvisorOutput


def rule_advice(advisor_input: AdvisorInput) -> AdvisorOutput:
    """Generate heuristic advice based on risk percentiles and recent metrics."""
    beta = advisor_input.beta if advisor_input.beta is not None else 3.0
    gamma = advisor_input.gamma if advisor_input.gamma is not None else 0.3
    p_exp = advisor_input.p if advisor_input.p is not None else 1.0
    beta_a = advisor_input.beta_a if advisor_input.beta_a is not None else 0.0

    percentiles = advisor_input.risk_env_percentiles or {}
    p50 = percentiles.get("p50", 0.0)
    p95 = percentiles.get("p95", 0.0)
    p5 = percentiles.get("p5", 0.0)

    # Adjust beta based on percentile thresholds.
    if p50 > 0.6:
        beta += 0.5
    if p95 > 0.9:
        beta += 0.5
    if p5 < 0.4 and beta > 2.0:
        beta -= 0.3

    # Recent metrics insights.
    corridor_coverages = []
    if advisor_input.recent_metrics:
        for item in advisor_input.recent_metrics:
            value = item.get("corridor_coverage")
            if isinstance(value, (int, float)):
                corridor_coverages.append(float(value))
        if corridor_coverages:
            last_avg = sum(corridor_coverages[-3:]) / min(3, len(corridor_coverages))
            if last_avg < 0.4:
                gamma = max(0.0, gamma - 0.1)
            elif last_avg > 0.75:
                gamma = min(0.8, gamma + 0.1)

    # Encourage beta_a when high-risk segments exist.
    if percentiles.get("p75", 0.0) > 0.85:
        beta_a = max(beta_a, 0.2)

    confidence = 0.45
    if corridor_coverages:
        confidence = min(0.65, confidence + 0.2)

    output = AdvisorOutput(
        beta=round(beta, 3),
        gamma=round(gamma, 3),
        p=round(p_exp, 3),
        beta_a=round(beta_a, 3),
        rationale="基于近期风险百分位与走廊覆盖率的规则建议。",
        confidence=confidence,
    )
    setattr(output, "_from_llm", False)
    return output


def build_llm_prompt(advisor_input: AdvisorInput) -> str:
    payload = trim_dict_for_llm(redact_paths(advisor_input.model_dump()))
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)
    prompt = (
        "你是北极航线规划助手，请根据以下输入生成推荐参数。\n"
        "必须返回严格的 JSON，对应字段：beta,gamma,p,beta_a,rationale (中文),confidence (0-1)。\n"
        "```json\n"
        f"{serialized}\n"
        "```"
    )
    return add_safety_guardrails(prompt)


def llm_advice(advisor_input: AdvisorInput, *, config=None) -> Tuple[AdvisorOutput, str]:
    ai_config = config or load_ai_config()
    prompt = build_llm_prompt(advisor_input)
    client = MoonshotK2Client(
        model=ai_config.model_name,
        timeout=ai_config.timeout_s,
    )
    start = perf_counter()
    try:
        raw = client.complete(
            prompt,
            max_tokens=ai_config.max_tokens,
            temperature=ai_config.temperature,
        )
        data = _parse_llm_json(raw)
        result = AdvisorOutput.model_validate(data)
        setattr(result, "_from_llm", True)
        meta = getattr(client, "_last_meta", {})
        retries = max(0, int(meta.get("attempts", 1)) - 1)
        duration = perf_counter() - start
        log_ai_call("advisor", duration_s=duration, retries=retries, degraded=False)
        return result, prompt
    except (MoonshotK2Error, ValueError) as exc:
        meta = getattr(client, "_last_meta", {})
        retries = max(0, int(meta.get("attempts", 1)) - 1)
        duration = perf_counter() - start
        log_ai_call("advisor", duration_s=duration, retries=retries, degraded=True, error=str(exc))
        fallback = rule_advice(advisor_input)
        setattr(fallback, "_from_llm", False)
        return fallback, prompt


def advise(advisor_input: AdvisorInput, *, use_llm: bool = False) -> AdvisorOutput:
    if use_llm:
        result, _ = llm_advice(advisor_input)
        return result
    return rule_advice(advisor_input)


def _parse_llm_json(text: str) -> Dict[str, object]:
    cleaned = text.strip()
    if "```" in cleaned:
        segments = cleaned.split("```")
        for segment in segments:
            segment = segment.strip()
            if segment.startswith("{") and segment.endswith("}"):
                cleaned = segment
                break
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse LLM JSON: {exc}") from exc


__all__ = [
    "rule_advice",
    "llm_advice",
    "advise",
    "build_llm_prompt",
]
