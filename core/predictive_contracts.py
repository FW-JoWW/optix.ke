from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ReadinessWarning(BaseModel):
    code: str
    message: str
    severity: Literal["low", "medium", "high"]


class PredictiveMetrics(BaseModel):
    values: Dict[str, float | None] = Field(default_factory=dict)


class ConfidenceAssessment(BaseModel):
    score: int = Field(ge=0, le=100)
    label: Literal["low", "moderate", "high"]
    explanation: str
    factors: Dict[str, Any] = Field(default_factory=dict)


class ModelResult(BaseModel):
    model_name: str
    problem_type: Literal["regression", "classification", "forecasting"]
    metrics: PredictiveMetrics
    feature_importance: List[Dict[str, float | str]] = Field(default_factory=list)
    validation_notes: List[str] = Field(default_factory=list)
    validation_summary: Dict[str, Any] = Field(default_factory=dict)


class PredictiveResult(BaseModel):
    tool: str = "predictive_analysis"
    problem_type: Literal["regression", "classification", "forecasting"]
    target_column: str
    chosen_model: str
    model_comparison: List[ModelResult]
    readiness_warnings: List[ReadinessWarning] = Field(default_factory=list)
    leakage_columns: List[str] = Field(default_factory=list)
    feature_columns: List[str] = Field(default_factory=list)
    top_drivers: List[Dict[str, float | str]] = Field(default_factory=list)
    predictions_preview: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: PredictiveMetrics
    confidence_level: Literal["low", "moderate", "high"]
    confidence: ConfidenceAssessment | None = None
    validation_summary: Dict[str, Any] = Field(default_factory=dict)
    truthfulness_flags: List[str] = Field(default_factory=list)
    no_reliable_recommendation: bool = False
    limitations: List[str] = Field(default_factory=list)


class PrescriptiveAction(BaseModel):
    action: str
    rationale: str
    estimated_uplift: float | None = None
    estimated_uplift_range: Dict[str, float] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "medium"
    feasibility: Literal["low", "medium", "high"] = "medium"
    affected_segments: List[str] = Field(default_factory=list)
    requires_experiment: bool = True
    causal_safety_note: str | None = None
    evidence_summary: List[str] = Field(default_factory=list)
    monitoring_kpis: List[str] = Field(default_factory=list)
    downside_risks: List[str] = Field(default_factory=list)
    failure_conditions: List[str] = Field(default_factory=list)
    reliability: Literal["low", "moderate", "high"] = "low"
    safety_grade: Literal["guarded", "controlled", "expansion_ready"] = "guarded"


class PrescriptiveResult(BaseModel):
    tool: str = "prescriptive_analysis"
    based_on_target: str
    objective: str
    scenario_summary: List[Dict[str, Any]] = Field(default_factory=list)
    decision_paths: List[Dict[str, Any]] = Field(default_factory=list)
    recommended_actions: List[PrescriptiveAction] = Field(default_factory=list)
    estimated_upside: float | None = None
    assumptions: List[str] = Field(default_factory=list)
    confidence_level: Literal["low", "moderate", "high"]
    confidence: ConfidenceAssessment | None = None
    operational_confidence: ConfidenceAssessment | None = None
    truthfulness_notes: List[str] = Field(default_factory=list)
