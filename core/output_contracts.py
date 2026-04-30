from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class RelationshipStats(BaseModel):
    coefficient: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    test_statistic: Optional[float] = None
    confidence_interval_95: Optional[Dict[str, Optional[float]]] = None
    additional_metrics: Dict[str, Any] = Field(default_factory=dict)


class NonlinearSignal(BaseModel):
    detected: bool = False
    mutual_information: Optional[float] = None
    tree_importance: Optional[float] = None
    polynomial_gain_r2: Optional[float] = None
    warning: Optional[str] = None


class PartialCorrelationResult(BaseModel):
    computed: bool = False
    controls: List[str] = Field(default_factory=list)
    coefficient: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: int = 0
    note: Optional[str] = None


class SegmentRelationship(BaseModel):
    segment_column: str
    segment_value: str
    method_used: str
    coefficient: Optional[float] = None
    p_value: Optional[float] = None
    sample_size: int = 0


class TemporalSignal(BaseModel):
    applicable: bool = False
    timestamp_column: Optional[str] = None
    best_lag: Optional[int] = None
    best_lag_correlation: Optional[float] = None
    lag_direction: Optional[Literal["x_precedes_y", "y_precedes_x", "same_time", "unclear"]] = None
    reverse_causality_warning: Optional[str] = None


class ConfounderCandidate(BaseModel):
    column: str
    score: float = Field(ge=0.0, le=1.0)
    relates_to_x: Optional[float] = None
    relates_to_y: Optional[float] = None
    reason: str


class BiasCheckResult(BaseModel):
    bias_risks: List[str] = Field(default_factory=list)
    simpsons_paradox_detected: bool = False
    outlier_domination_detected: bool = False
    small_sample_risk: bool = False
    missingness_bias_risk: bool = False
    selection_bias_risk: bool = False
    leakage_risk: bool = False
    warnings: List[str] = Field(default_factory=list)


class CausalEvidenceResult(BaseModel):
    grade: Literal["LOW", "MODERATE", "STRONG"] = "LOW"
    score: int = Field(ge=0, le=100)
    rubric: Dict[str, int] = Field(default_factory=dict)
    rationale: List[str] = Field(default_factory=list)


class RelationshipEvidenceReport(BaseModel):
    question: str
    relationship_found: str
    method_used: str
    stats: RelationshipStats
    assumptions: List[str] = Field(default_factory=list)
    assumption_checks_passed: bool = False
    warnings: List[str] = Field(default_factory=list)
    bias_risks: List[str] = Field(default_factory=list)
    confounders: List[ConfounderCandidate] = Field(default_factory=list)
    nonlinear_signal: Optional[NonlinearSignal] = None
    partial_correlation: Optional[PartialCorrelationResult] = None
    segment_correlations: List[SegmentRelationship] = Field(default_factory=list)
    temporal_signal: Optional[TemporalSignal] = None
    causal_evidence: CausalEvidenceResult
    recommended_next_step: str
    confidence: int = Field(ge=0, le=100)
    human_summary: str
    structured_summary: Dict[str, Any] = Field(default_factory=dict)
