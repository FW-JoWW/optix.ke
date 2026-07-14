from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any, Dict, Iterable, List, Sequence

from .models import CollaborativeTask, EvidenceRecord, InvestigationSession, utc_now


class TaskManager:
    def __init__(self, session: InvestigationSession):
        self.session = session
        self._task_sequence = self._initial_sequence()
        self._evidence_sequence = self._initial_evidence_sequence()

    def _initial_sequence(self) -> int:
        prefix = "task-"
        highest = 0
        for task_id in self.session.tasks:
            if task_id.startswith(prefix):
                suffix = task_id.removeprefix(prefix)
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return highest

    def _initial_evidence_sequence(self) -> int:
        prefix = "evidence-"
        highest = 0
        for evidence_id in self.session.evidence_store:
            if evidence_id.startswith(prefix):
                suffix = evidence_id.removeprefix(prefix)
                if suffix.isdigit():
                    highest = max(highest, int(suffix))
        return highest

    def _next_task_id(self) -> str:
        self._task_sequence += 1
        return f"task-{self._task_sequence}"

    def _next_evidence_id(self) -> str:
        self._evidence_sequence += 1
        return f"evidence-{self._evidence_sequence}"

    def create_task(
        self,
        request: str,
        title: str | None = None,
        dependencies: Sequence[str] | None = None,
        parent_task_id: str | None = None,
        version: int = 1,
        metadata: Dict[str, Any] | None = None,
    ) -> CollaborativeTask:
        task_id = self._next_task_id()
        task = CollaborativeTask(
            task_id=task_id,
            title=title or request,
            request=request,
            status="queued",
            version=version,
            dependencies=list(dependencies or []),
            parent_task_id=parent_task_id,
            execution_metadata=dict(metadata or {}),
        )
        self.session.tasks[task_id] = task
        self.session.queued_tasks.append(task_id)
        self.session.task_graph[task_id] = {
            "dependencies": list(task.dependencies),
            "parent_task_id": parent_task_id,
        }
        self.session.investigation_memory.setdefault("task_history", []).append(task.to_dict())
        return task

    def queue_task(self, task: CollaborativeTask) -> CollaborativeTask:
        self.session.tasks[task.task_id] = task
        if task.task_id not in self.session.queued_tasks and task.status == "queued":
            self.session.queued_tasks.append(task.task_id)
        return task

    def enqueue_request(
        self,
        request: str,
        title: str | None = None,
        dependencies: Sequence[str] | None = None,
        parent_task_id: str | None = None,
        metadata: Dict[str, Any] | None = None,
    ) -> CollaborativeTask:
        return self.create_task(
            request=request,
            title=title,
            dependencies=dependencies,
            parent_task_id=parent_task_id,
            metadata=metadata,
        )

    def reorder_queue(self, task_ids: Sequence[str]) -> List[str]:
        queue = [task_id for task_id in task_ids if task_id in self.session.queued_tasks]
        remaining = [task_id for task_id in self.session.queued_tasks if task_id not in queue]
        self.session.queued_tasks = queue + remaining
        return list(self.session.queued_tasks)

    def remove_queued_task(self, task_id: str) -> bool:
        if task_id in self.session.queued_tasks:
            self.session.queued_tasks.remove(task_id)
            self.session.cancelled_tasks.append(task_id)
            task = self.session.tasks.get(task_id)
            if task:
                task.status = "cancelled"
                task.completed_at = utc_now()
            return True
        return False

    def cancel_queued_task(self, task_id: str) -> bool:
        return self.remove_queued_task(task_id)

    def dequeue_next_task(self) -> CollaborativeTask | None:
        if not self.session.queued_tasks:
            return None
        task_id = self.session.queued_tasks.pop(0)
        task = self.session.tasks[task_id]
        task.status = "running"
        task.started_at = utc_now()
        self.session.running_tasks.append(task_id)
        return task

    def mark_completed(
        self,
        task_id: str,
        result_state: Dict[str, Any],
        evidence: EvidenceRecord,
        summary: Dict[str, Any],
    ) -> None:
        task = self.session.tasks[task_id]
        task.status = "completed"
        task.completed_at = utc_now()
        task.result_summary = dict(summary)
        task.evidence_ids.append(evidence.evidence_id)
        self.session.evidence_store[evidence.evidence_id] = evidence
        if task_id in self.session.running_tasks:
            self.session.running_tasks.remove(task_id)
        if task_id not in self.session.completed_tasks:
            self.session.completed_tasks.append(task_id)
        self.session.decision_log.append(
            {
                "task_id": task_id,
                "status": "completed",
                "timestamp": task.completed_at,
                "summary": dict(summary),
                "metadata": deepcopy(task.execution_metadata),
            }
        )
        self.session.investigation_memory["last_result"] = dict(summary)
        self.session.investigation_memory["last_completed_task"] = task_id
        self.session.investigation_memory.setdefault("task_summaries", {})[task_id] = dict(summary)

    def mark_failed(self, task_id: str, reason: str, metadata: Dict[str, Any] | None = None) -> None:
        task = self.session.tasks[task_id]
        task.status = "failed"
        task.failed_at = utc_now()
        task.failure_reason = reason
        task.execution_metadata.update(metadata or {})
        if task_id in self.session.running_tasks:
            self.session.running_tasks.remove(task_id)
        if task_id not in self.session.failed_tasks:
            self.session.failed_tasks.append(task_id)
        self.session.decision_log.append(
            {
                "task_id": task_id,
                "status": "failed",
                "reason": reason,
                "timestamp": task.failed_at,
            }
        )

    def refine_task(self, task_id: str, request: str, metadata: Dict[str, Any] | None = None) -> CollaborativeTask:
        original = self.session.tasks[task_id]
        return self.create_task(
            request=request,
            title=f"{original.title} (Version {original.version + 1})",
            dependencies=original.dependencies,
            parent_task_id=task_id,
            version=original.version + 1,
            metadata=metadata or {"refines": task_id},
        )

    def compare_tasks(self, left_task_id: str, right_task_id: str) -> Dict[str, Any]:
        left = self.session.tasks.get(left_task_id)
        right = self.session.tasks.get(right_task_id)
        comparison = {
            "comparison_id": self._next_evidence_id(),
            "left_task_id": left_task_id,
            "right_task_id": right_task_id,
            "left_summary": dict(left.result_summary) if left else {},
            "right_summary": dict(right.result_summary) if right else {},
            "shared_dependencies": sorted(set((left.dependencies if left else [])).intersection(right.dependencies if right else [])),
            "timestamp": utc_now(),
        }
        self.session.task_comparisons.append(comparison)
        self.session.investigation_memory.setdefault("comparisons", []).append(comparison)
        return comparison

    def record_evidence(
        self,
        task_id: str,
        evidence_type: str,
        statement: str,
        statistical_support: Dict[str, Any] | None = None,
        confidence: Any = None,
        method: str = "",
        supporting_visualizations: Sequence[Any] | None = None,
        dependencies: Sequence[str] | None = None,
        quality_score: float = 0.0,
        metadata: Dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        record = EvidenceRecord(
            evidence_id=self._next_evidence_id(),
            task_source=task_id,
            evidence_type=evidence_type,
            statement=statement,
            statistical_support=dict(statistical_support or {}),
            confidence=confidence,
            method=method,
            supporting_visualizations=list(supporting_visualizations or []),
            dependencies=list(dependencies or []),
            quality_score=quality_score,
            metadata=dict(metadata or {}),
        )
        return record

    def active_queue(self) -> List[Dict[str, Any]]:
        return [self.session.tasks[task_id].to_dict() for task_id in self.session.queued_tasks]

    def running(self) -> List[Dict[str, Any]]:
        return [self.session.tasks[task_id].to_dict() for task_id in self.session.running_tasks]

    def completed(self) -> List[Dict[str, Any]]:
        return [self.session.tasks[task_id].to_dict() for task_id in self.session.completed_tasks]

    def failed(self) -> List[Dict[str, Any]]:
        return [self.session.tasks[task_id].to_dict() for task_id in self.session.failed_tasks]

    def cancelled(self) -> List[Dict[str, Any]]:
        return [self.session.tasks[task_id].to_dict() for task_id in self.session.cancelled_tasks if task_id in self.session.tasks]
