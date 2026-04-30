from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ImpactAssessment(BaseModel):
    impact_level: Literal["none", "uncertain", "low", "medium", "high"]
    estimated_direction: Literal["positive", "negative", "unclear"]
    confidence_adjusted_impact: float = Field(ge=0.0, le=1.0)


class PriorityAssessment(BaseModel):
    priority_score: int = Field(ge=0, le=100)
    priority_level: Literal["low", "medium", "high", "critical"]


class DecisionRecord(BaseModel):
    story_signature: str
    insight: str
    action_type: Literal["no_action", "investigation", "experiment", "optimization", "strategic"]
    possible_actions: List[str]
    impact_assessment: ImpactAssessment
    priority: PriorityAssessment
    decision_summary: str
    recommended_action: str
    confidence_in_action: int = Field(ge=0, le=100)
    relationship_type: Optional[str] = None
    validity: Optional[bool] = None
    recommendation_restrictions: List[str] = Field(default_factory=list)
