"""Pydantic models for AI advisor and explainer workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import json
from pydantic import BaseModel, Field, ValidationError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


class AdvisorInput(BaseModel):
    beta: Optional[float] = Field(None, description="当前风险权重 β")
    gamma: Optional[float] = Field(None, description="走廊折扣 γ")
    p: Optional[float] = Field(None, description="风险幂指数 p")
    beta_a: Optional[float] = Field(None, description="事故密度权重 β_a")
    risk_env_percentiles: Dict[str, float] = Field(..., description="风险场百分位信息")
    recent_metrics: Optional[List[Dict[str, float]]] = Field(
        None, description="近期运行指标列表"
    )


class AdvisorOutput(BaseModel):
    beta: float = Field(..., description="推荐的风险权重 β")
    gamma: float = Field(..., description="推荐的走廊折扣 γ")
    p: float = Field(..., description="推荐的风险幂指数 p")
    beta_a: float = Field(..., description="推荐的事故密度权重 β_a")
    rationale: str = Field(..., description="中文短文解释推荐原因")
    confidence: float = Field(..., ge=0.0, le=1.0, description="置信度 0-1")


class ExplainerInput(BaseModel):
    run_report: Dict[str, object] = Field(..., description="目标运行报告数据")
    params: Dict[str, object] = Field(..., description="运行参数信息")
    baseline_report: Optional[Dict[str, object]] = Field(
        None, description="基线运行报告（可选）"
    )


class ExplainerOutput(BaseModel):
    markdown: str = Field(..., description="中文 Markdown 总结")
    bullets: List[str] = Field(..., description="简要要点列表")
    key_deltas: Dict[str, object] = Field(..., description="关键差异信息")


def _load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_report(tag: str) -> Dict[str, object]:
    """Load a single run report by tag (without prefix)."""
    candidates = [
        OUTPUTS_DIR / f"run_report_{tag}.json",
        *OUTPUTS_DIR.glob(f"*/run_report_{tag}.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return _load_json(candidate)
    raise FileNotFoundError(f"run_report with tag '{tag}' not found")


def advisor_input_from_files(
    *,
    parameters: Dict[str, float],
    run_report_tag: Optional[str] = None,
    metrics_path: Optional[Path] = None,
) -> AdvisorInput:
    """Build AdvisorInput using optional run report and metrics references."""
    risk_env_percentiles: Dict[str, float] = {}
    recent_metrics: Optional[List[Dict[str, float]]] = None

    if run_report_tag:
        report = load_run_report(run_report_tag)
        risk_env_percentiles = _extract_percentiles(report)
    elif run_report_tag is None:
        risk_env_percentiles = parameters.get("risk_env_percentiles", {})  # type: ignore[assignment]
        if not isinstance(risk_env_percentiles, dict):
            risk_env_percentiles = {}

    if metrics_path is None:
        metrics_path = OUTPUTS_DIR / "metrics.csv"
    if metrics_path.exists():
        recent_metrics = _load_metrics(metrics_path)

    payload = {**parameters, "risk_env_percentiles": risk_env_percentiles}
    if recent_metrics:
        payload["recent_metrics"] = recent_metrics

    try:
        return AdvisorInput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid AdvisorInput payload: {exc}") from exc


def advisor_output_from_dict(data: Dict[str, object]) -> AdvisorOutput:
    try:
        return AdvisorOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid AdvisorOutput payload: {exc}") from exc


def explainer_input_from_files(
    *,
    run_report_tag: str,
    params: Dict[str, object],
    baseline_tag: Optional[str] = None,
) -> ExplainerInput:
    payload = {
        "run_report": load_run_report(run_report_tag),
        "params": params,
    }
    if baseline_tag:
        payload["baseline_report"] = load_run_report(baseline_tag)
    try:
        return ExplainerInput.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"Invalid ExplainerInput payload: {exc}") from exc


def explainer_output_from_dict(data: Dict[str, object]) -> ExplainerOutput:
    try:
        return ExplainerOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid ExplainerOutput payload: {exc}") from exc


def _extract_percentiles(report: Dict[str, object]) -> Dict[str, float]:
    percentiles = report.get("risk_env_percentiles")
    if isinstance(percentiles, dict):
        return {k: float(v) for k, v in percentiles.items() if isinstance(v, (int, float))}
    return {}


def _load_metrics(path: Path) -> List[Dict[str, float]]:
    try:
        import csv
    except ImportError:  # pragma: no cover
        return []

    metrics: List[Dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            cleaned: Dict[str, float] = {}
            for key, value in row.items():
                if value is None:
                    continue
                try:
                    cleaned[key] = float(value)
                except ValueError:
                    continue
            if cleaned:
                metrics.append(cleaned)
    return metrics


__all__ = [
    "AdvisorInput",
    "AdvisorOutput",
    "ExplainerInput",
    "ExplainerOutput",
    "advisor_input_from_files",
    "advisor_output_from_dict",
    "explainer_input_from_files",
    "explainer_output_from_dict",
    "load_run_report",
]

