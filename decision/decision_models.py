from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class DecisionTrace(BaseModel):
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    originating_signals: List[str] = Field(default_factory=list)


class CleaningDecision(BaseModel):
    target: Optional[str]
    action_type: Literal[
        "forward_fill",
        "drop_rows",
        "convert_type",
        "standardize_categories",
        "recompute_if_possible",
        "leave_unchanged",
    ]
    confidence_score: float = Field(ge=0.0, le=1.0)
    justification: str
    trace: DecisionTrace
    requires_validation: bool = False
    deferred: bool = False


class ComputationStep(BaseModel):
    operation: str
    column: Optional[str] = None
    columns: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    justification: str
    trace: DecisionTrace


class ComputationPlanModel(BaseModel):
    steps: List[ComputationStep] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    justification: str
    deferred: bool = False


class AnalysisAbstractionModel(BaseModel):
    capability_family: str = "unknown"
    dimensions: List[str] = Field(default_factory=list)
    measures: List[str] = Field(default_factory=list)
    aggregation: Optional[str] = None
    comparison_type: Optional[str] = None
    temporal_behavior: Optional[str] = None
    statistical_operation: Optional[str] = None
    presentation_strategy: Optional[str] = None
    confidence_score: float = Field(ge=0.0, le=1.0, default=0.0)
    justification: str = ""


class AnalysisOperation(BaseModel):
    tool: str
    columns: List[str] = Field(default_factory=list)
    computation_refs: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    valid: bool = True
    justification: str
    trace: DecisionTrace
    fallback_tool: Optional[str] = None


class AnalysisPlanModel(BaseModel):
    analytical_strategy: Literal[
        "aggregation",
        "comparison",
        "relationship",
        "distribution",
        "outliers",
        "data_quality",
        "predictive",
        "unknown",
    ]
    operations: List[AnalysisOperation] = Field(default_factory=list)
    confidence_score: float = Field(ge=0.0, le=1.0)
    justification: str
    deferred: bool = False


class DecisionEngineOutput(BaseModel):
    cleaning_decisions: List[CleaningDecision] = Field(default_factory=list)
    analysis_abstraction: AnalysisAbstractionModel
    computation_plan: ComputationPlanModel
    analysis_plan: AnalysisPlanModel
    decision_notes: List[str] = Field(default_factory=list)
    retry_hints: Dict[str, Any] = Field(default_factory=dict)
