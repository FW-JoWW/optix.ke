from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CollaborativeTask:
    task_id: str
    title: str
    request: str
    status: str = "queued"
    version: int = 1
    dependencies: List[str] = field(default_factory=list)
    parent_task_id: Optional[str] = None
    created_at: str = field(default_factory=utc_now)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    failed_at: Optional[str] = None
    failure_reason: Optional[str] = None
    execution_metadata: Dict[str, Any] = field(default_factory=dict)
    evidence_ids: List[str] = field(default_factory=list)
    result_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "request": self.request,
            "status": self.status,
            "version": self.version,
            "dependencies": list(self.dependencies),
            "parent_task_id": self.parent_task_id,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "failed_at": self.failed_at,
            "failure_reason": self.failure_reason,
            "execution_metadata": dict(self.execution_metadata),
            "evidence_ids": list(self.evidence_ids),
            "result_summary": dict(self.result_summary),
        }


@dataclass
class EvidenceRecord:
    evidence_id: str
    task_source: str
    evidence_type: str
    statement: str
    statistical_support: Dict[str, Any] = field(default_factory=dict)
    confidence: Any = None
    method: str = ""
    supporting_visualizations: List[Any] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    timestamp: str = field(default_factory=utc_now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "task_source": self.task_source,
            "evidence_type": self.evidence_type,
            "statement": self.statement,
            "statistical_support": dict(self.statistical_support),
            "confidence": self.confidence,
            "method": self.method,
            "supporting_visualizations": list(self.supporting_visualizations),
            "dependencies": list(self.dependencies),
            "quality_score": self.quality_score,
            "timestamp": self.timestamp,
            "metadata": dict(self.metadata),
        }


@dataclass
class HypothesisRecord:
    hypothesis: str
    status: str = "inconclusive"
    confidence: Any = None
    supporting_evidence: List[str] = field(default_factory=list)
    conflicting_evidence: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=utc_now)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis": self.hypothesis,
            "status": self.status,
            "confidence": self.confidence,
            "supporting_evidence": list(self.supporting_evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
            "timestamp": self.timestamp,
            "notes": list(self.notes),
        }


@dataclass
class InvestigationSession:
    investigation_id: str
    original_question: str
    current_status: str = "active"
    task_graph: Dict[str, Any] = field(default_factory=dict)
    evidence_store: Dict[str, EvidenceRecord] = field(default_factory=dict)
    investigation_memory: Dict[str, Any] = field(default_factory=dict)
    decision_log: List[Dict[str, Any]] = field(default_factory=list)
    progressive_narrative: List[str] = field(default_factory=list)
    final_executive_report: Optional[str] = None
    tasks: Dict[str, CollaborativeTask] = field(default_factory=dict)
    queued_tasks: List[str] = field(default_factory=list)
    running_tasks: List[str] = field(default_factory=list)
    completed_tasks: List[str] = field(default_factory=list)
    failed_tasks: List[str] = field(default_factory=list)
    cancelled_tasks: List[str] = field(default_factory=list)
    hypotheses: Dict[str, HypothesisRecord] = field(default_factory=dict)
    task_comparisons: List[Dict[str, Any]] = field(default_factory=list)
    ai_suggestions: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "investigation_id": self.investigation_id,
            "original_question": self.original_question,
            "current_status": self.current_status,
            "task_graph": dict(self.task_graph),
            "evidence_store": {key: value.to_dict() for key, value in self.evidence_store.items()},
            "investigation_memory": dict(self.investigation_memory),
            "decision_log": list(self.decision_log),
            "progressive_narrative": list(self.progressive_narrative),
            "final_executive_report": self.final_executive_report,
            "tasks": {key: value.to_dict() for key, value in self.tasks.items()},
            "queued_tasks": list(self.queued_tasks),
            "running_tasks": list(self.running_tasks),
            "completed_tasks": list(self.completed_tasks),
            "failed_tasks": list(self.failed_tasks),
            "cancelled_tasks": list(self.cancelled_tasks),
            "hypotheses": {key: value.to_dict() for key, value in self.hypotheses.items()},
            "task_comparisons": list(self.task_comparisons),
            "ai_suggestions": list(self.ai_suggestions),
        }
