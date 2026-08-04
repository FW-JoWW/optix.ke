from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from io import StringIO
import re
from typing import Any, Callable, Dict, Iterable, List, Sequence

import pandas as pd

from graph.analyst_graph import graph
from collaborative_mode.answer_synthesis import synthesize_answer
from nodes.intent_parser_node import classify_analytic_intent
from nodes.report_node import report_node
from state.state import AnalystState

from .narrative_composer import compose_checkpoint_narrative
from .integrity import build_traceability_record, evaluate_investigation_integrity
from .presentation import render_collaborative_desk_view
from .narration import format_suggestion_line, humanize_columns, humanize_text, suggestion_impact_percent
from .models import InvestigationSession
from .registry import get_investigation, list_investigations, register_investigation, unregister_investigation
from .orchestrator import (
    CollaborativeRunResult,
    _build_desk_view,
    _build_evidence_record,
    _build_task_state,
    _derive_hypothesis,
    _inject_collaborative_context,
    _short_text,
    _suggest_next_investigations,
    _summarize_task_result,
    create_investigation_session,
)
from .task_manager import TaskManager


PromptFn = Callable[[str], str]


def _build_base_state(question: str, dataset_path: str | None, dataframe: Any | None) -> AnalystState:
    if dataframe is not None:
        df = dataframe
    elif dataset_path:
        df = pd.read_csv(dataset_path, low_memory=False)
    else:
        from scripts.guided_mode_harness import build_guided_sample_dataframe

        df = build_guided_sample_dataframe()

    return {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": "collaborative",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_evidence": {},
    }


def _default_checkpoint_tasks(question: str) -> List[Dict[str, Any]]:
    return [
        {
            "title": "Primary investigation",
            "request": question,
        }
    ]


def _format_checkpoint_summary(checkpoint: Dict[str, Any], session: Dict[str, Any] | None = None, question: str | None = None) -> List[str]:
    lines = compose_checkpoint_narrative(checkpoint, original_question=question, session=session)
    if lines:
        return lines
    return ["No completed checkpoint yet. The investigation is still gathering evidence."]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_task_ids(text: str) -> List[str]:
    ids = re.findall(r"\btask[-\s]?(\d+)\b", text or "", flags=re.IGNORECASE)
    return [f"task-{task_id}" for task_id in ids]


def _looks_like_query(text: str) -> bool:
    lowered = (text or "").strip().lower()
    return lowered.endswith("?") or lowered.startswith(
        (
            "what ",
            "which ",
            "show ",
            "summarize ",
            "continue ",
            "generate ",
            "explain ",
            "recommend ",
            "why ",
            "how ",
        )
    ) or "?" in lowered


def _infer_action_from_text(text: str) -> tuple[str | None, str]:
    raw = (text or "").strip()
    lowered = raw.lower()
    if not raw:
        return None, ""

    task_ids = _extract_task_ids(raw)
    analytic_intent = classify_analytic_intent(raw)

    if _looks_like_query(raw):
        return "query", raw

    if lowered.startswith(("compare ", "compare task", "compare version")) or task_ids or analytic_intent == "comparison":
        if task_ids:
            return "compare results", ", ".join(task_ids[:2])
        return "compare results", raw.removeprefix("compare").strip()

    if lowered.startswith(("challenge ", "challenge finding", "challenge findings", "can you ", "could ", "test whether ", "falsify ")) or analytic_intent in {"investigative", "relationship"}:
        return "challenge finding", raw

    if lowered.startswith(("investigate ", "analyze ", "analyse ", "build ", "repeat ", "rerun ", "re-run ", "run ", "use ", "test ", "question ")) or analytic_intent in {
        "temporal",
        "composition",
        "relationship",
        "extremes",
        "profiling",
        "outliers",
        "data_quality",
        "predictive",
    }:
        if task_ids and ("repeat" in lowered or "rerun" in lowered or "re-run" in lowered or "version" in lowered):
            return "refine task", raw
        return "new investigation", raw

    if lowered.startswith(("resume previous investigation", "show all active investigations", "what is the objective", "summarize everything discovered", "what should we investigate next")):
        return "query", raw

    return None, raw


def _title_from_description(details: str, fallback: str = "New investigation") -> str:
    raw = (details or "").strip()
    if not raw:
        return fallback

    task_ids = _extract_task_ids(raw)
    intent = classify_analytic_intent(raw)
    if task_ids and any(term in raw.lower() for term in ["repeat", "rerun", "re-run", "version"]):
        return f"Refined {task_ids[0]}"

    title_map = {
        "comparison": "Comparison investigation",
        "temporal": "Trend investigation",
        "composition": "Composition investigation",
        "relationship": "Relationship investigation",
        "extremes": "Ranking investigation",
        "profiling": "Profile investigation",
        "outliers": "Anomaly investigation",
        "data_quality": "Data quality investigation",
        "investigative": "Exploratory investigation",
        "predictive": "Predictive investigation",
    }
    if intent in title_map:
        return title_map[intent]
    if len(raw) <= 48:
        return raw[:1].upper() + raw[1:]
    return raw[:45].rstrip() + "..."


def _capability_panel_lines(controller: "CollaborativeSessionController") -> List[str]:
    session = controller.session
    objective = session.objective or controller.question
    suggestions = session.ai_suggestions or []
    dataframe = controller.base_state.get("dataframe")
    ranked_suggestions = controller._rank_suggestions(suggestions)
    top_suggestion = ranked_suggestions[0] if ranked_suggestions else {}
    lines = [
        f"Investigation objective: {objective}",
        "Active capabilities:",
        "1. New investigation - create a completely new investigation task.",
        "2. Refine task - modify and rerun a previous task while preserving versions.",
        "3. Compare results - compare completed tasks and generate comparison evidence.",
        "4. Challenge finding - challenge a conclusion and turn it into analytical tasks.",
        "5. Accept AI suggestion - execute the highest-confidence AI recommendation.",
        "6. Finish investigation - close the investigation and generate the final report.",
        "Natural language examples:",
        " - 'Investigate customer churn by region'",
        " - 'Compare Task 2 and Task 5'",
        " - 'Re-run Task 4 using Spearman correlation'",
        " - 'Could age explain this instead?'",
        " - 'What should we investigate next?'",
    ]
    if suggestions:
        lines.append("AI recommendations:")
        for index, suggestion in enumerate(ranked_suggestions, start=1):
            lines.append(f" - {format_suggestion_line(suggestion, index=index, dataframe=dataframe)}")
    if top_suggestion:
        impact = suggestion_impact_percent(top_suggestion)
        impact_text = f"{impact}%" if impact is not None else "unknown"
        lines.append(f"Top AI suggestion: {top_suggestion.get('title')} | impact={impact_text} | confidence={top_suggestion.get('confidence')}")
    return lines


CAPABILITY_ACTIONS: Dict[str, Dict[str, Any]] = {
    "1": {"action": "new investigation", "label": "New investigation", "needs_details": True, "run_next": True},
    "2": {"action": "refine task", "label": "Refine task", "needs_details": True, "run_next": True},
    "3": {"action": "compare results", "label": "Compare results", "needs_details": True, "run_next": False},
    "4": {"action": "challenge finding", "label": "Challenge finding", "needs_details": True, "run_next": True},
    "5": {"action": "accept ai suggestion", "label": "Accept AI suggestion", "needs_details": False, "run_next": True},
    "6": {"action": "finish investigation", "label": "Finish investigation", "needs_details": False, "run_next": False},
}


def _resolve_menu_choice(choice: str) -> Dict[str, Any] | None:
    raw = (choice or "").strip().lower()
    if not raw:
        return None
    if raw in CAPABILITY_ACTIONS:
        return CAPABILITY_ACTIONS[raw]
    normalized = raw.replace("  ", " ")
    for option in CAPABILITY_ACTIONS.values():
        if normalized == option["action"]:
            return option
        if normalized == option["label"].lower():
            return option
    return None


@dataclass
class CollaborativeSessionController:
    question: str
    session: InvestigationSession
    manager: TaskManager
    base_state: AnalystState
    build_final_report: bool = True
    queue_paused: bool = False
    task_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_successful_state: AnalystState | None = None
    finished: bool = False

    @classmethod
    def create(
        cls,
        question: str,
        dataset_path: str | None = None,
        dataframe: Any | None = None,
        initial_tasks: Sequence[Dict[str, Any] | str] | None = None,
        build_final_report: bool = True,
    ) -> "CollaborativeSessionController":
        base_state = _build_base_state(question, dataset_path, dataframe)
        session = create_investigation_session(question)
        manager = TaskManager(session)

        if not initial_tasks:
            initial_tasks = _default_checkpoint_tasks(question)

        for task_spec in initial_tasks:
            if isinstance(task_spec, str):
                manager.enqueue_request(task_spec)
                continue
            manager.enqueue_request(
                request=str(task_spec.get("request") or question),
                title=str(task_spec.get("title") or task_spec.get("request") or question),
                dependencies=task_spec.get("dependencies") or [],
                parent_task_id=task_spec.get("parent_task_id"),
                metadata=task_spec.get("metadata") or {},
            )

        return cls(
            question=question,
            session=session,
            manager=manager,
            base_state=base_state,
            build_final_report=build_final_report,
        )

    def register(self) -> None:
        register_investigation(self)

    def _refresh_registry(self) -> None:
        register_investigation(self)

    def _snapshot(self) -> CollaborativeRunResult:
        if self.last_successful_state is not None:
            final_state = deepcopy(self.last_successful_state)
        else:
            final_state = deepcopy(self.base_state)
        _inject_collaborative_context(final_state, self.session)
        final_state.setdefault("analysis_evidence", {})
        final_state["analysis_evidence"]["collaborative_session"] = self.session.to_dict()
        final_state["analysis_evidence"]["collaborative_task_outputs"] = dict(self.task_outputs)
        final_state["analysis_evidence"]["collaborative_desk"] = _build_desk_view(self.session)
        final_state["analysis_evidence"]["collaborative_last_failure"] = self.session.investigation_memory.get("last_failure")
        final_state["collaborative_final_report"] = final_state.get("final_report")
        final_state["awaiting_user"] = self.session.current_status != "completed"
        return CollaborativeRunResult(
            final_state=final_state,
            session=self.session.to_dict(),
            desk=_build_desk_view(self.session),
            task_outputs=dict(self.task_outputs),
        )

    def _record_failure(self, action: str, message: str, details: str = "", reason: str | None = None) -> Dict[str, Any]:
        failure = {
            "action": action,
            "details": details,
            "message": message,
            "reason": reason or message,
            "input_needed": "Please correct the request or choose another capability.",
            "timestamp": _utc_now(),
        }
        self.session.investigation_memory["last_failure"] = failure
        self.session.decision_log.append(
            {
                "action": action,
                "status": "failed",
                "details": details,
                "reason": reason or message,
            }
        )
        self._refresh_registry()
        return {"status": "failed", "message": message, "run_next": False, "failure": failure}

    def process_next_task(self) -> CollaborativeRunResult:
        if self.finished:
            return self._snapshot()

        task = self.manager.dequeue_next_task()
        if task is None:
            self.session.current_status = "awaiting_user"
            return self._snapshot()

        task_state = _build_task_state(self.base_state, self.session, task.request)
        try:
            with redirect_stdout(StringIO()):
                final_state = graph.invoke(task_state)
        except Exception as exc:  # pragma: no cover - defensive safety net
            self.manager.mark_failed(task.task_id, str(exc), metadata={"request": task.request})
            self.session.progressive_narrative.append(f"{task.title} failed: {exc}")
            self.session.investigation_memory["last_failure"] = {
                "action": "execute task",
                "task_id": task.task_id,
                "title": task.title,
                "details": task.request,
                "message": str(exc),
                "reason": f"Task execution raised an exception: {exc}",
                "input_needed": "Please revise the task request or select a different investigation path.",
                "timestamp": _utc_now(),
            }
            self.session.checkpoint_summaries.append(
                {
                    "checkpoint_id": f"checkpoint-{len(self.session.checkpoint_summaries) + 1}",
                    "task_id": task.task_id,
                    "task_title": task.title,
                    "task_request": task.request,
                    "version": task.version,
                    "status": "failed",
                    "current_understanding": f"Task execution failed: {exc}",
                    "narrative": f"{task.title} failed: {exc}",
                    "analysis_story": f"The task could not complete because the analytical graph raised an exception while processing: {task.request}.",
                    "analysis_steps": [],
                    "analysis_signals": [],
                    "confidence": "unknown",
                    "selected_columns": [],
                    "analysis_plan": [],
                    "tool_results": [],
                    "visualizations": [],
                    "report_excerpt": "",
                    "timestamp": _utc_now(),
                    "next_investigations": list(self.session.ai_suggestions),
                    "failure_message": str(exc),
                    "failure_reason": f"Task execution raised an exception: {exc}",
                    "failure_input_needed": "Please revise the task request or select a different investigation path.",
                }
            )
            self.session.current_status = "awaiting_user"
            return self._snapshot()

        summary = _summarize_task_result(task.request, final_state, task.task_id, task.version)
        evidence = _build_evidence_record(task.task_id, final_state, summary)
        self.manager.mark_completed(task.task_id, final_state, evidence, summary)
        hypothesis = _derive_hypothesis(task.request, final_state, evidence)
        current_hypothesis = self.session.investigation_memory.get("current_understanding") or self.question
        answer_synthesis = synthesize_answer(
            business_question=task.request,
            evidence=final_state.get("analysis_evidence", {}) or {},
            hypotheses=[hypothesis.to_dict()],
            current_understanding=summary.get("current_understanding") or summary.get("task_finding") or task.request,
            confidence=summary.get("confidence"),
            knowledge_gaps=self.session.investigation_memory.get("knowledge_gaps") or [],
            investigation_memory=self.session.investigation_memory,
            dataframe=self.base_state.get("dataframe"),
        )
        decision = (
            final_state.get("investigation_decision")
            or final_state.get("analysis_evidence", {}).get("investigation_decision")
            or answer_synthesis.get("investigation_decision")
            or {}
        )
        integrity = evaluate_investigation_integrity(
            original_question=self.session.original_question,
            task_request=task.request,
            summary=summary,
            current_hypothesis=current_hypothesis,
            prior_findings=self.session.investigation_memory.get("previous_findings") or [],
            dataframe=self.base_state.get("dataframe"),
        )
        summary["integrity"] = integrity
        summary["traceability"] = build_traceability_record(integrity)
        summary["investigation_decision"] = decision
        summary["task_finding"] = summary.get("task_finding") or summary.get("current_understanding")
        summary["current_understanding"] = integrity["promoted_understanding"] if integrity["should_promote"] else (self.session.investigation_memory.get("current_understanding") or self.question)
        summary["integrity_status"] = integrity["overall"]["level"]
        hypothesis.notes.append(
            f"Integrity: question relevance={integrity['question_relevance']['level']}, continuity={integrity['continuity']['level']}, information gain={integrity['information_gain']['level']}, validity={integrity['analytical_validity']['level']}."
        )
        if not integrity["should_promote"]:
            hypothesis.status = "inconclusive"
        self.session.hypotheses[f"{task.task_id}:v{task.version}"] = hypothesis
        self.session.progressive_narrative.append(summary["narrative"])
        self.session.investigation_memory["last_integrity"] = integrity
        self.session.investigation_memory["last_traceability"] = summary["traceability"]
        self.session.investigation_memory["investigation_decision"] = decision
        self.session.investigation_memory["current_decision"] = decision.get("decision")
        self.session.investigation_memory["question_for_user"] = decision.get("question_for_user") or self.session.investigation_memory.get("question_for_user")
        self.session.investigation_memory.setdefault("integrity_history", []).append(
            {
                "task_id": task.task_id,
                "task_request": task.request,
                "traceability": summary["traceability"],
                "integrity": integrity,
                "decision": decision,
            }
        )
        if integrity["should_promote"]:
            self.session.investigation_memory["current_understanding"] = summary["current_understanding"]
        else:
            self.session.investigation_memory.setdefault("supporting_findings", []).append(summary["task_finding"])
            if "current_understanding" not in self.session.investigation_memory:
                self.session.investigation_memory["current_understanding"] = self.question
        self.session.investigation_memory.setdefault("previous_findings", []).append(summary["task_finding"])
        self.session.investigation_memory.setdefault("task_references", []).append(task.task_id)
        self.session.investigation_memory["objective"] = self.session.objective or self.question
        next_suggestions = _suggest_next_investigations(self.session, final_state, task.task_id)
        self.session.ai_suggestions = next_suggestions
        self.session.checkpoint_summaries.append(
            {
                "checkpoint_id": f"checkpoint-{len(self.session.checkpoint_summaries) + 1}",
                "task_id": task.task_id,
                "task_title": task.title,
                "task_request": task.request,
                "version": task.version,
                "status": summary.get("status", "completed"),
                "current_understanding": summary.get("current_understanding"),
                "narrative": summary.get("narrative"),
                "analysis_story": summary.get("analysis_story"),
                "analysis_steps": list(summary.get("analysis_steps") or []),
                "analysis_signals": list(summary.get("analysis_signals") or []),
                "confidence": summary.get("confidence"),
                "integrity": integrity,
                "traceability": summary["traceability"],
                "selected_columns": list(summary.get("selected_columns") or []),
                "analysis_plan": list(summary.get("analysis_plan") or []),
                "tool_results": list(summary.get("tool_results") or []),
                "visualizations": summary.get("visualizations"),
                "report_excerpt": summary.get("report_excerpt"),
                "timestamp": summary.get("timestamp"),
                "next_investigations": list(next_suggestions),
            }
        )
        if len(self.session.completed_tasks) >= 2:
            comparison = self.manager.compare_tasks(self.session.completed_tasks[-2], self.session.completed_tasks[-1])
            self.session.investigation_memory.setdefault("comparison_history", []).append(comparison)
        self.task_outputs[task.task_id] = summary
        final_state.setdefault("analysis_evidence", {})
        final_state["analysis_evidence"]["investigation_decision"] = decision
        final_state["investigation_decision"] = decision
        self.last_successful_state = final_state
        self.session.current_status = "awaiting_user"
        self._refresh_registry()
        return self._snapshot()

    def apply_action(self, action: str, details: str | None = None) -> Dict[str, Any]:
        normalized = (action or "").strip().lower()
        details = (details or "").strip()
        if not normalized:
            return {"status": "ignored", "message": "No action supplied.", "run_next": False}

        self.session.decision_log.append(
            {
                "action": normalized,
                "details": details,
            }
        )

        if normalized in {"finish", "finish investigation", "complete", "close"}:
            self.finished = True
            self.session.current_status = "completed"
            self.session.closed_at = self.session.closed_at or _utc_now()
            unregister_investigation(self.session.investigation_id)
            self._refresh_registry()
            return {"status": "completed", "message": "Investigation finished.", "run_next": False}

        if normalized in {"archive", "archive this investigation"}:
            self.finished = True
            self.session.current_status = "archived"
            self.session.archived_at = self.session.archived_at or _utc_now()
            unregister_investigation(self.session.investigation_id)
            self._refresh_registry()
            return {"status": "archived", "message": "Investigation archived.", "run_next": False}

        if normalized in {"rename", "rename investigation"}:
            if details:
                self.session.investigation_title = details
            self._refresh_registry()
            return {"status": "updated", "message": "Investigation renamed.", "run_next": False}

        if normalized in {"set objective", "update objective", "objective"}:
            if details:
                self.session.objective = details
                self.session.investigation_memory["objective"] = details
            self._refresh_registry()
            return {"status": "updated", "message": "Objective updated.", "run_next": False}

        if normalized in {"pause queue", "pause"}:
            self.queue_paused = True
            self.session.decision_log.append({"action": "pause queue", "details": details})
            self._refresh_registry()
            return {"status": "updated", "message": "Queue paused.", "run_next": False}

        if normalized in {"resume queue", "resume"}:
            self.queue_paused = False
            self.session.decision_log.append({"action": "resume queue", "details": details})
            self._refresh_registry()
            return {"status": "updated", "message": "Queue resumed.", "run_next": False}

        if normalized in {"execute next task", "next task", "run next task"}:
            self.queue_paused = False
            self._refresh_registry()
            return {"status": "updated", "message": "Ready to execute next task.", "run_next": True}

        if normalized in {"reorder queue", "reorder"}:
            task_ids = _extract_task_ids(details or normalized)
            if task_ids:
                self.manager.reorder_queue(task_ids)
                self.session.decision_log.append({"action": "reorder queue", "details": details, "task_ids": task_ids})
            self._refresh_registry()
            return {"status": "updated", "message": f"Queue reordered: {task_ids or 'no task ids found'}.", "run_next": False}

        if normalized in {"cancel task", "remove queued investigation", "remove this queued investigation", "cancel queued tasks"}:
            task_ids = _extract_task_ids(details or normalized)
            if task_ids:
                for task_id in task_ids:
                    self.manager.cancel_queued_task(task_id)
                self.session.decision_log.append({"action": "cancel task", "details": details, "task_ids": task_ids})
            self._refresh_registry()
            return {"status": "updated", "message": f"Cancelled: {task_ids or 'no task ids found'}.", "run_next": False}

        if normalized in {"resume previous investigation"}:
            task_ids = _extract_task_ids(details or normalized)
            if details and not task_ids:
                if self.resume_previous_investigation(details):
                    self.session.decision_log.append({"action": "resume previous investigation", "details": details})
                    self._refresh_registry()
                    return {"status": "completed", "message": f"Resumed investigation {details}.", "run_next": False}
            elif task_ids and self.resume_previous_investigation(task_ids[0]):
                self.session.decision_log.append({"action": "resume previous investigation", "details": task_ids[0]})
                self._refresh_registry()
                return {"status": "completed", "message": f"Resumed investigation {task_ids[0]}.", "run_next": False}
            return self._record_failure(
                "resume previous investigation",
                "Could not find a previous investigation to resume.",
                details=details,
                reason="No matching investigation id was found in the live registry.",
            )

        if normalized in {"new investigation", "new", "new task"}:
            request = details or (self.session.ai_suggestions[0]["request"] if self.session.ai_suggestions else self.question)
            title = _title_from_description(details or request, "New investigation")
            self.manager.enqueue_request(request=request, title=title, metadata={"source_action": "new investigation"})
            self.session.current_status = "active"
            return {"status": "queued", "message": f"Queued new investigation: {title}.", "run_next": True}

        if normalized in {"refine task", "refine"}:
            task_id, request = self._parse_refine_details(details)
            if task_id and request and task_id in self.session.tasks:
                refined = self.manager.refine_task(task_id, request, metadata={"source_action": "refine task"})
                self.session.current_status = "active"
                self.session.decision_log[-1]["created_task_id"] = refined.task_id
                return {"status": "queued", "message": f"Queued refinement for {task_id} as {refined.task_id}.", "run_next": True}
            reason = "The refinement needs both a valid task id and a supported rerun instruction."
            if not task_id:
                reason = "I could not find a task id in the refinement request."
            elif task_id not in self.session.tasks:
                reason = f"Task id {task_id} is not known in this investigation."
            elif not request:
                reason = "The refinement text is missing the rerun instruction."
            return self._record_failure(
                "refine task",
                f"Refine task failed: {reason}",
                details=details,
                reason=reason,
            )

        if normalized in {"compare results", "compare"}:
            left_task_id, right_task_id = self._parse_compare_details(details)
            if not left_task_id or not right_task_id:
                completed = list(self.session.completed_tasks)
                if len(completed) >= 2:
                    left_task_id, right_task_id = completed[-2], completed[-1]
            if left_task_id and right_task_id:
                comparison = self.manager.compare_tasks(left_task_id, right_task_id)
                self.session.current_status = "active"
                self.session.decision_log[-1]["comparison_id"] = comparison["comparison_id"]
                return {"status": "completed", "message": f"Compared {left_task_id} and {right_task_id}.", "run_next": False}
            return self._record_failure(
                "compare results",
                "Compare results failed: provide two completed task ids that already finished successfully.",
                details=details,
                reason="No valid pair of completed task ids was supplied.",
            )

        if normalized in {"challenge finding", "challenge", "challenge findings"}:
            if not details and (len(self.session.checkpoint_summaries) > 1 or len(self.session.hypotheses) > 1):
                active_findings = []
                for checkpoint in self.session.checkpoint_summaries[-3:]:
                    finding = checkpoint.get("current_understanding") or checkpoint.get("narrative") or checkpoint.get("task_title")
                    if finding:
                        active_findings.append(humanize_text(finding, dataframe=self.base_state.get("dataframe")))
                active_findings = [finding for finding in active_findings if finding]
                options = "; ".join(active_findings[:3]) if active_findings else "the current active finding"
                return self._record_failure(
                    "challenge finding",
                    "Challenge finding needs a specific finding to challenge.",
                    details=details,
                    reason=f"Ambiguous reference detected. Please name which active finding to challenge. Current active findings include: {options}.",
                    input_needed=f"Specify which finding to challenge, for example one of: {options}.",
                )
            challenge_text = details or self.session.investigation_memory.get("current_understanding") or self.question
            request = f"Challenge the leading conclusion for: {challenge_text}"
            self.manager.enqueue_request(
                request=request,
                title=_title_from_description(challenge_text, "Challenge finding"),
                metadata={"source_action": "challenge finding"},
            )
            self.session.current_status = "active"
            return {"status": "queued", "message": "Queued challenge investigation.", "run_next": True}

        if normalized in {"accept ai suggestion", "accept", "accept ai suggestions"}:
            suggestion = self._pick_ai_suggestion(details)
            if suggestion:
                if suggestion.get("terminal"):
                    return self._record_failure(
                        "accept ai suggestion",
                        "No further distinct AI suggestion is available right now.",
                        details=details,
                        reason="The recommendation list has converged, so the next step is to consolidate or finish the investigation.",
                    )
                self.manager.enqueue_request(
                    request=str(suggestion.get("request") or self.question),
                    title=str(suggestion.get("title") or "Accepted AI suggestion"),
                    dependencies=suggestion.get("depends_on") or [],
                    metadata={"source_action": "accept ai suggestion", "suggestion": suggestion.get("title")},
                )
                self.session.current_status = "active"
                return {"status": "queued", "message": f"Accepted AI suggestion: {suggestion.get('title')}.", "run_next": True}
            return self._record_failure(
                "accept ai suggestion",
                "No AI suggestion is currently available to accept.",
                details=details,
                reason="The suggestion list is empty.",
            )

        if normalized in {"queue", "enqueue"} and details:
            self.manager.enqueue_request(request=details, title=_title_from_description(details, "Queued investigation"))
            self.session.current_status = "active"
            self._refresh_registry()
            return {"status": "queued", "message": "Queued investigation.", "run_next": False}

        if normalized in {"queue three investigations"}:
            for suffix in ("1", "2", "3"):
                self.manager.enqueue_request(
                    request=f"{self.question} - queued investigation {suffix}",
                    title=f"Queued investigation {suffix}",
                    metadata={"source_action": "queue three investigations"},
                )
            self.session.current_status = "active"
            self._refresh_registry()
            return {"status": "queued", "message": "Queued three investigations.", "run_next": False}

        return self._record_failure(
            action,
            f"Unsupported action: {action}.",
            details=details,
            reason="This collaborative capability is not wired to a handler yet.",
        )

    def _pick_ai_suggestion(self, details: str) -> Dict[str, Any] | None:
        suggestions = self._rank_suggestions(self.session.ai_suggestions or [])
        if not suggestions:
            return None

        if not details:
            return suggestions[0]
        if details.isdigit():
            index = int(details) - 1
            if 0 <= index < len(suggestions):
                return suggestions[index]
        lowered = details.lower()
        for suggestion in suggestions:
            combined = f"{suggestion.get('title', '')} {suggestion.get('request', '')}".lower()
            if lowered in combined:
                return suggestion
        return suggestions[0]

    @staticmethod
    def _rank_suggestions(items: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def _confidence_score(item: Dict[str, Any]) -> int:
            confidence = item.get("confidence")
            if isinstance(confidence, (int, float)):
                value = float(confidence)
                return int(value * 100 if value <= 1 else value)
            if isinstance(confidence, dict):
                score = confidence.get("score")
                if isinstance(score, (int, float)):
                    value = float(score)
                    return int(value * 100 if value <= 1 else value)
            if isinstance(confidence, str):
                lowered = confidence.strip().lower()
                if lowered in {"high", "strong", "very high"}:
                    return 90
                if lowered in {"moderate", "medium"}:
                    return 60
                if lowered in {"low", "weak"}:
                    return 30
            return 0

        def _impact_score(item: Dict[str, Any]) -> int:
            impact = suggestion_impact_percent(item)
            return int(impact) if isinstance(impact, (int, float)) else 0

        def _sort_key(item: Dict[str, Any]) -> tuple[bool, int, int, str, str]:
            title = str(item.get("title") or "").strip().lower()
            request = str(item.get("request") or item.get("description") or "").strip().lower()
            return (
                bool(item.get("terminal")),
                -_confidence_score(item),
                -_impact_score(item),
                title,
                request,
            )

        return sorted(list(items), key=_sort_key)

    @staticmethod
    def _parse_refine_details(details: str) -> tuple[str | None, str | None]:
        if not details:
            return None, None
        match = re.search(r"\btask[-\s]?(\d+)\b", details, flags=re.IGNORECASE)
        if match:
            task_id = f"task-{match.group(1)}"
            request = details.strip()
            if ":" in details:
                _, right = details.split(":", 1)
                request = right.strip() or request
            return task_id, request
        if ":" in details:
            left, right = details.split(":", 1)
            return left.strip() or None, right.strip() or None
        return None, None

    @staticmethod
    def _parse_compare_details(details: str) -> tuple[str | None, str | None]:
        if not details:
            return None, None
        cleaned = details.replace("|", ",")
        parts = [part.strip() for part in cleaned.split(",") if part.strip()]
        if len(parts) < 2 and " and " in cleaned.lower():
            parts = [part.strip() for part in cleaned.replace(" and ", ",").split(",") if part.strip()]
        if len(parts) >= 2:
            return parts[0], parts[1]
        return None, None

    def list_active_investigations(self) -> List[Dict[str, Any]]:
        return list_investigations(include_closed=False)

    def resume_previous_investigation(self, investigation_id: str) -> bool:
        other = get_investigation(investigation_id)
        if other is None:
            return False
        session = getattr(other, "session", None)
        if session is None:
            return False
        self.session = session
        self.manager = other.manager
        self.base_state = other.base_state
        self.task_outputs = dict(other.task_outputs)
        self.last_successful_state = deepcopy(other.last_successful_state) if other.last_successful_state is not None else None
        self.finished = other.finished
        self.queue_paused = other.queue_paused
        return True

    def summarize_progress(self) -> str:
        checkpoints = self.session.checkpoint_summaries
        current_understanding = self.session.investigation_memory.get("current_understanding") or "The investigation is still inconclusive."
        question_text = humanize_text(self.question, dataframe=self.base_state.get("dataframe"))
        current_focus = "No checkpoint has been completed yet."
        remaining_gap = "The next task should reduce uncertainty around the original question."
        if checkpoints:
            latest = checkpoints[-1]
            current_focus = humanize_text(latest.get("task_title") or latest.get("task_request") or "the current investigation", dataframe=self.base_state.get("dataframe"))
            next_investigations = latest.get("next_investigations") or []
            if next_investigations:
                best_next = next_investigations[0]
                remaining_gap = humanize_text(best_next.get("request") or best_next.get("description") or remaining_gap, dataframe=self.base_state.get("dataframe"))
        lines = [
            f"Direct answer to the question '{question_text}': {humanize_text(current_understanding, dataframe=self.base_state.get('dataframe'))}",
            f"Objective: {self.session.objective or self.question}",
            f"Current status: {self.session.current_status}",
            f"Current focus: {current_focus}",
            f"Remaining gap: {remaining_gap}",
            f"Completed tasks: {len(self.session.completed_tasks)}",
            f"Queued tasks: {len(self.session.queued_tasks)}",
            f"Evidence items: {len(self.session.evidence_store)}",
            f"Hypotheses: {len(self.session.hypotheses)}",
            f"Current understanding: {self.session.investigation_memory.get('current_understanding') or 'None yet'}",
        ]
        if checkpoints:
            latest = checkpoints[-1]
            lines.append(f"Latest checkpoint: {humanize_text(latest.get('task_title') or latest.get('task_id'))}")
            if latest.get("status") == "failed":
                lines.append(f"Latest failure: {humanize_text(latest.get('failure_message') or latest.get('failure_reason') or 'Task failed.')}")
            if latest.get("next_investigations"):
                suggestion_text = format_suggestion_line(latest["next_investigations"][0], dataframe=self.base_state.get("dataframe"))
                lines.append(f"Current best next step: {suggestion_text}")
        return "\n".join(lines)

    def summarize_everything_discovered(self) -> str:
        checkpoints = self.session.checkpoint_summaries
        lines = [self.summarize_progress(), "", "Discoveries:"]
        if checkpoints:
            for checkpoint in checkpoints:
                lines.append(
                    f"- {humanize_text(checkpoint.get('task_title') or checkpoint.get('task_id'))}: {humanize_text(checkpoint.get('current_understanding') or checkpoint.get('narrative'))}"
                )
        else:
            lines.append("- No completed investigations yet.")
        return "\n".join(lines)

    def summarize_history(self) -> str:
        return self.session.final_executive_report or self.summarize_everything_discovered()

    def summarize_memory(self) -> str:
        memory = self.session.investigation_memory or {}
        lines = [
            f"Accepted assumptions: {memory.get('accepted_assumptions') or []}",
            f"Rejected hypotheses: {memory.get('rejected_hypotheses') or []}",
            f"Previous findings: {memory.get('previous_findings') or []}",
            f"Selected variables: {memory.get('selected_variables') or []}",
            f"Datasets used: {memory.get('datasets_used') or [self.session.investigation_memory.get('dataset_path') or self.base_state.get('dataset_path')] }",
            f"Cleaning version: {memory.get('cleaning_version') or 'unknown'}",
            f"Pending tasks: {self.session.queued_tasks or []}",
            f"Earlier conclusion: {memory.get('current_understanding') or 'None yet'}",
        ]
        return "\n".join(lines)

    def _checkpoint_explainer(self) -> List[str]:
        checkpoint = self.session.checkpoint_summaries[-1] if self.session.checkpoint_summaries else None
        failure = self.session.investigation_memory.get("last_failure") or {}
        lines: List[str] = []

        if checkpoint:
            lines.extend(
                compose_checkpoint_narrative(
                    checkpoint,
                    original_question=self.session.original_question or self.question,
                    session=self.session.to_dict(),
                )
            )

        if failure and failure is not checkpoint:
            lines.append(f"Last failure: {humanize_text(failure.get('message') or 'A recent action failed.')}")
            if failure.get("reason"):
                lines.append(f"Failure reason: {humanize_text(failure.get('reason'))}")
            if failure.get("input_needed"):
                lines.append(f"Next input needed: {humanize_text(failure.get('input_needed'))}")

        if self.session.ai_suggestions:
            top = self.session.ai_suggestions[0]
            lines.append(f"Suggested next step: {format_suggestion_line(top, dataframe=self.base_state.get('dataframe'))}")

        if not lines:
            lines.append("No completed checkpoint yet. The investigation is still gathering evidence.")

        return lines

    def _with_explainer(self, answer: str) -> str:
        explainer = self._checkpoint_explainer()
        return "\n".join([answer, "", "Explainer:"] + [f"- {line}" for line in explainer])

    def _query_hypotheses(self) -> str:
        hypotheses = list(self.session.hypotheses.values())
        supported = [item for item in hypotheses if str(item.status).lower() == "supported"]
        rejected = [item for item in hypotheses if str(item.status).lower() == "rejected"]
        inconclusive = [item for item in hypotheses if str(item.status).lower() == "inconclusive"]
        lines = [
            f"Supported hypotheses: {len(supported)}",
            f"Rejected hypotheses: {len(rejected)}",
            f"Inconclusive hypotheses: {len(inconclusive)}",
        ]
        for item in sorted(hypotheses, key=lambda value: float(value.confidence) if isinstance(value.confidence, (int, float)) else -1, reverse=True):
            lines.append(f"- {humanize_text(item.hypothesis)} | status={item.status} | confidence={item.confidence}")
        return self._with_explainer("\n".join(lines))

    def _query_evidence(self, task_id: str | None = None) -> str:
        evidence_items = list(self.session.evidence_store.values())
        if task_id:
            evidence_items = [item for item in evidence_items if item.task_source == task_id]
        if not evidence_items:
            return self._with_explainer("No evidence recorded yet.")
        strongest = sorted(evidence_items, key=lambda item: item.quality_score or 0.0, reverse=True)
        lines = [f"Evidence count: {len(evidence_items)}", "Strongest evidence:"]
        for item in strongest[:5]:
            lines.append(
                f"- {humanize_text(item.statement)} | task={item.task_source} | type={item.evidence_type} | confidence={item.confidence}"
            )
        weak = [item for item in evidence_items if (item.quality_score or 0.0) < 0.5]
        if weak:
            lines.append(f"Weak evidence: {[item.evidence_id for item in weak]}")
        return self._with_explainer("\n".join(lines))

    def _query_planning(self) -> str:
        lines = [
            "What to investigate next:",
        ]
        if self.session.ai_suggestions:
            for index, suggestion in enumerate(self._rank_suggestions(self.session.ai_suggestions), start=1):
                lines.append(f"- {format_suggestion_line(suggestion, index=index, dataframe=self.base_state.get('dataframe'))}")
        else:
            lines.append("- No AI suggestion is available yet.")
        if self.session.queued_tasks:
            lines.append(f"Queued tasks: {self.session.queued_tasks}")
        lines.append("Highest priority investigations are the queued tasks that unlock unresolved hypotheses.")
        lines.append("Missing information usually centers on weak evidence, unanswered hypotheses, or untested alternative explanations.")
        return self._with_explainer("\n".join(lines))

    def _query_memory(self, text: str) -> str:
        return self._with_explainer(self.summarize_memory())

    def _ambiguous_reference_response(self) -> str:
        active_findings = []
        for checkpoint in self.session.checkpoint_summaries[-3:]:
            finding = checkpoint.get("current_understanding") or checkpoint.get("task_finding") or checkpoint.get("narrative") or checkpoint.get("task_title")
            if finding:
                active_findings.append(humanize_text(finding, dataframe=self.base_state.get("dataframe")))
        active_findings = [finding for finding in active_findings if finding]
        if active_findings:
            options = "; ".join(active_findings[:3])
            return self._with_explainer(
                f"Please name which active finding you want to refer to. Current active findings include: {options}."
            )
        return self._with_explainer("Please name which active finding you want to refer to. The investigation does not yet have a stable set of findings to disambiguate.")

    def respond_to_query(self, text: str) -> str:
        query = (text or "").strip().lower()
        if not query:
            return "No query supplied."

        ambiguous_tokens = {"that", "this", "those", "these", "why", "it", "they", "them"}
        if query in ambiguous_tokens or (len(query.split()) <= 4 and any(token in query.split() for token in ambiguous_tokens)):
            return self._ambiguous_reference_response()

        if "show all active investigations" in query:
            investigations = self.list_active_investigations()
            if not investigations:
                return self._with_explainer("No active investigations.")
            return self._with_explainer("\n".join(
                f"- {item['investigation_id']} | {item['title']} | status={item['status']} | understanding={item['current_understanding'] or 'None yet'}"
                for item in investigations
            ))

        if "resume previous investigation" in query:
            return self._with_explainer("Use resume_previous_investigation(investigation_id) to attach a prior investigation into the current controller.")

        if "what is the objective" in query or query.startswith("objective"):
            return self._with_explainer(f"Objective: {self.session.objective or self.question}")

        if "summarize everything discovered so far" in query or "summarize everything" in query:
            return self._with_explainer(self.summarize_everything_discovered())

        if "what should we investigate next" in query or "recommend the next analytical task" in query:
            return self._with_explainer(self._query_planning())

        if "what are the highest priority investigations" in query:
            return self._with_explainer(self._query_planning())

        if "what questions remain unanswered" in query or "what information is still missing" in query:
            return self._with_explainer(self._query_planning())

        if "which investigation would provide the greatest business value" in query:
            return self._with_explainer(self._query_planning())

        if "show queued tasks" in query:
            return self._with_explainer("\n".join(
                f"- {humanize_text(self.session.tasks[task_id].title, dataframe=self.session.investigation_memory.get('dataframe'))}"
                for task_id in self.session.queued_tasks
                if task_id in self.session.tasks
            ) or "No queued tasks.")

        if "queue three investigations" in query:
            return self._with_explainer("Use the queue three investigations action to add three queued investigation tasks.")

        if "reorder the queue" in query or "reorder queue" in query:
            return self._with_explainer("Use reorder_queue(task_ids) on the task manager with the desired task order.")

        if "cancel task" in query or "remove this queued investigation" in query or "cancel queued tasks" in query:
            return self._with_explainer("Use cancel_queued_task(task_id) or remove_queued_task(task_id) on the task manager.")

        if "pause the queue" in query or "pause queue" in query:
            return self._with_explainer("Use the pause queue action to stop the queue from advancing.")

        if "resume the queue" in query or "resume queue" in query:
            return self._with_explainer("Use the resume queue action to continue from the queue.")

        if "execute the next task" in query or "next task" in query:
            return self._with_explainer("Use the execute next task action to run one queued investigation.")

        if "what have we already analyzed" in query or "continue from where we stopped" in query:
            return self._with_explainer(self.summarize_memory())

        if "which variables have already been tested" in query or "which models have already been built" in query:
            return self._with_explainer(self.summarize_memory())

        if "show the strongest evidence" in query or "which evidence is statistically significant" in query:
            return self._with_explainer(self._query_evidence())

        if "which findings support this conclusion" in query or "which findings contradict it" in query:
            return self._with_explainer(self._query_evidence())

        if "compare" in query and ("task" in query or "version" in query or "model" in query or "region" in query):
            comparisons = self.session.task_comparisons or []
            if not comparisons:
                return self._with_explainer("No task comparisons have been recorded yet.")
            lines = ["Comparisons:"]
            for comparison in comparisons:
                lines.append(
                    f"- {humanize_text(comparison.get('left_task_id'))} vs {humanize_text(comparison.get('right_task_id'))} | comparison_id={comparison.get('comparison_id')}"
                )
            return self._with_explainer("\n".join(lines))

        if "show all current hypotheses" in query or "which hypotheses are supported" in query or "which hypotheses remain inconclusive" in query:
            return self._query_hypotheses()

        if "what did we conclude earlier" in query or "generate the current narrative" in query or "what is the current business story" in query:
            return self._with_explainer(self.summarize_progress())

        if "summarize every successful task" in query or "show the complete investigation history" in query:
            return self._with_explainer(self.summarize_history())

        if "show the evidence trail" in query or "show the evidence trail behind every recommendation" in query:
            return self._query_evidence()

        if "evidence generated by task" in query or "show evidence generated by" in query:
            import re

            match = re.search(r"task\s+(\d+)", query)
            if match:
                return self._with_explainer(self._query_evidence(task_id=f"task-{match.group(1)}"))
            match = re.search(r"(task-\d+)", query)
            if match:
                return self._with_explainer(self._query_evidence(task_id=match.group(1)))
            return self._with_explainer(self._query_evidence())

        return self._with_explainer(self.summarize_progress())

    def finalize(self) -> CollaborativeRunResult:
        if self.finished is False:
            self.session.current_status = "completed"
            self.finished = True
        if self.last_successful_state is None:
            final_state = deepcopy(self.base_state)
            final_state["analysis_evidence"] = {
                "collaborative_session": self.session.to_dict(),
                "final_output": [
                    "No collaborative tasks completed successfully.",
                    "The investigation preserved the session and can be resumed.",
                ],
            }
            final_state["final_report"] = "\n".join(final_state["analysis_evidence"]["final_output"])
            self.session.final_executive_report = final_state["final_report"]
            desk = _build_desk_view(self.session)
            return CollaborativeRunResult(
                final_state=final_state,
                session=self.session.to_dict(),
                desk=desk,
                task_outputs=dict(self.task_outputs),
            )

        final_state = deepcopy(self.last_successful_state)
        _inject_collaborative_context(final_state, self.session)
        final_state["analysis_evidence"]["collaborative_session"] = self.session.to_dict()
        final_state["analysis_evidence"]["collaborative_task_outputs"] = dict(self.task_outputs)
        final_state["analysis_evidence"]["collaborative_desk"] = _build_desk_view(self.session)
        final_state["analysis_evidence"]["investigation_decision"] = self.session.investigation_memory.get("investigation_decision") or {}
        final_state["investigation_decision"] = self.session.investigation_memory.get("investigation_decision") or {}

        if self.build_final_report:
            final_state = report_node(final_state)
            self.session.final_executive_report = final_state.get("final_report")
        else:
            report_lines = [
                "================ COLLABORATIVE INVESTIGATION ================",
                f"Investigation ID: {self.session.investigation_id}",
                f"Original question: {self.session.original_question}",
                f"Completed tasks: {', '.join(self.session.completed_tasks) if self.session.completed_tasks else 'None'}",
                f"Current understanding: {self.session.investigation_memory.get('current_understanding') or 'None'}",
                "Task findings:",
            ]
            for task_id in self.session.completed_tasks:
                task_summary = self.task_outputs.get(task_id, {})
                report_lines.append(
                    f"- {task_id}: {task_summary.get('current_understanding') or task_summary.get('narrative') or 'Task completed.'}"
                )
            report_lines.extend(
                [
                    "Evidence store:",
                    f"- {len(self.session.evidence_store)} evidence items",
                    "Hypotheses:",
                    f"- {len(self.session.hypotheses)} hypotheses tracked",
                ]
            )
            final_state["final_report"] = "\n".join(report_lines)
            self.session.final_executive_report = final_state["final_report"]

        final_state["collaborative_final_report"] = final_state.get("final_report")
        return CollaborativeRunResult(
            final_state=final_state,
            session=self.session.to_dict(),
            desk=_build_desk_view(self.session),
            task_outputs=dict(self.task_outputs),
        )


def run_interactive_collaborative_session(
    question: str,
    dataset_path: str | None = None,
    dataframe: Any | None = None,
    initial_tasks: Sequence[Dict[str, Any] | str] | None = None,
    responses: Sequence[str] | None = None,
    build_final_report: bool = True,
    input_fn: PromptFn | None = None,
    print_fn: Callable[[str], None] | None = None,
) -> CollaborativeRunResult:
    """
    Run collaborative mode as an investigation that pauses after each checkpoint.

    The function accepts optional scripted responses for testing, but when no
    responses are provided it falls back to interactive console prompts.
    """
    prompt = input_fn or input
    printer = print_fn or print
    controller = CollaborativeSessionController.create(
        question=question,
        dataset_path=dataset_path,
        dataframe=dataframe,
        initial_tasks=initial_tasks,
        build_final_report=build_final_report,
    )
    controller.register()

    response_queue = list(responses or [])

    def ask(message: str) -> str:
        if response_queue:
            return response_queue.pop(0)
        return prompt(message)

    result = controller.process_next_task()

    def _show_failure(action_name: str, action_result: Dict[str, Any]) -> str:
        failure = action_result.get("failure") or controller.session.investigation_memory.get("last_failure") or {}
        printer("\n===== ACTION FAILED =====")
        printer(f"Action: {action_name}")
        printer(action_result.get("message", "The action failed."))
        reason = failure.get("reason")
        if reason and reason != action_result.get("message"):
            printer(f"Reason: {reason}")
        details = failure.get("details")
        if details:
            printer(f"Details: {details}")
        printer("Choose what to do next:")
        printer("1. Retry this action with corrected details")
        printer("2. Choose another capability")
        printer("3. Finish investigation")
        return ask("Select 1-3:\n> ").strip().lower()

    def _execute_action(action_name: str, details: str = "") -> tuple[Dict[str, Any], bool]:
        nonlocal result
        controller.session.investigation_memory.pop("last_failure", None)
        action_result = controller.apply_action(action_name, details)
        printer("\n===== ACTION RESULT =====")
        printer(action_result.get("message", action_result.get("status", "No result returned.")))

        if action_name == "finish investigation":
            final_result = controller.finalize()
            printer("\n===== FINAL REPORT =====")
            printer(final_result.final_state.get("final_report", "No report generated"))
            result = final_result
            return action_result, True

        if action_result.get("status") == "failed":
            return action_result, False

        if action_result.get("run_next"):
            if controller.queue_paused:
                printer("\nThe queue is paused, so the new request was saved but not executed yet.")
                result = controller._snapshot()
            else:
                result = controller.process_next_task()
                last_failure = controller.session.investigation_memory.get("last_failure") or {}
                if last_failure.get("action") == "execute task":
                    task_id = last_failure.get("task_id") or "unknown task"
                    failure_message = last_failure.get("message") or last_failure.get("reason") or "Task execution failed."
                    return {
                        "status": "failed",
                        "message": f"{task_id} failed: {failure_message}",
                        "run_next": False,
                        "failure": last_failure,
                    }, False
        else:
            result = controller._snapshot()

        return action_result, False

    while True:
        desk = result.desk
        checkpoint_lines = []
        if controller.session.checkpoint_summaries:
            checkpoint_lines = _format_checkpoint_summary(
                controller.session.checkpoint_summaries[-1],
                session=controller.session.to_dict(),
                question=controller.session.original_question or controller.question,
            )

        printer("\n===== INVESTIGATION DESK =====")
        printer(render_collaborative_desk_view(controller.session.to_dict(), dataframe=controller.base_state.get("dataframe")))
        printer("\n===== CAPABILITY PANEL =====")
        printer("\n".join(_capability_panel_lines(controller)))
        if checkpoint_lines:
            printer("\n===== CHECKPOINT =====")
            printer("\n".join(checkpoint_lines))

        if controller.finished:
            final_result = controller.finalize()
            printer("\n===== FINAL REPORT =====")
            printer(final_result.final_state.get("final_report", "No report generated"))
            return final_result

        selection = ask(
            "\nChoose an action (1-6) or ask a capability question:\n> "
        ).strip()
        if not selection:
            continue

        menu_choice = _resolve_menu_choice(selection)
        if menu_choice is not None:
            action = menu_choice["action"]
            details = ""
            if menu_choice.get("needs_details"):
                if action == "new investigation":
                    details = ask(
                        "Describe the new investigation you want to run:\n> "
                    ).strip()
                elif action == "refine task":
                    details = ask(
                        "Describe the task refinement. Include the task id and the change you want, for example 'Task 4 using Spearman correlation':\n> "
                    ).strip()
                elif action == "compare results":
                    details = ask(
                        "Name two completed tasks to compare, for example 'Task 2 and Task 5':\n> "
                    ).strip()
                elif action == "challenge finding":
                    details = ask(
                        "Describe the challenge or alternate explanation:\n> "
                    ).strip()
            elif action == "accept ai suggestion":
                details = ask(
                    "Choose an AI suggestion number or describe which suggestion to accept. The list already shows impact percentages, and leaving this blank uses the highest-impact suggestion:\n> "
                ).strip()

            action_result, finished = _execute_action(action, details)
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure(action, action_result)
                if recovery_choice in {"1", "retry", "r", "retry this action"}:
                    corrected_details = ask("Enter the corrected details:\n> ").strip()
                    retry_result, finished = _execute_action(action, corrected_details)
                    if finished:
                        return result
                    if retry_result.get("status") == "failed":
                        continue
                elif recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue

        normalized = selection.lower()
        inferred_action, inferred_details = _infer_action_from_text(selection)

        if inferred_action == "query":
            printer("\n===== CAPABILITY RESPONSE =====")
            printer(controller.respond_to_query(selection))
            continue

        if inferred_action in {"new investigation", "refine task", "compare results", "challenge finding"}:
            action_result, finished = _execute_action(inferred_action, inferred_details)
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure(inferred_action, action_result)
                if recovery_choice in {"1", "retry", "r", "retry this action"}:
                    corrected_details = ask("Enter the corrected details:\n> ").strip()
                    retry_result, finished = _execute_action(inferred_action, corrected_details)
                    if finished:
                        return result
                    if retry_result.get("status") == "failed":
                        continue
                elif recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue

        if normalized.startswith("cancel task") or normalized.startswith("remove "):
            action_result, finished = _execute_action("cancel task", selection)
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("cancel task", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue
        if normalized.startswith("reorder ") or normalized.startswith("prioritize "):
            action_result, finished = _execute_action("reorder queue", selection)
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("reorder queue", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue
        if normalized.startswith("resume previous investigation"):
            action_result, finished = _execute_action("resume previous investigation", selection)
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("resume previous investigation", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue

        if _looks_like_query(selection) and normalized not in {"finish", "finish investigation", "complete", "close"}:
            printer("\n===== CAPABILITY RESPONSE =====")
            printer(controller.respond_to_query(selection))
            continue

        if normalized in {"pause queue", "pause"}:
            action_result, finished = _execute_action("pause queue")
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("pause queue", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue
        if normalized in {"resume queue", "resume"}:
            action_result, finished = _execute_action("resume queue")
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("resume queue", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue
        if normalized in {"execute next task", "next task", "run next task"}:
            action_result, finished = _execute_action("execute next task")
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("execute next task", action_result)
                if recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue
        if normalized in {"show all active investigations", "what is the objective of this investigation", "summarize everything discovered so far"}:
            printer("\n===== CAPABILITY RESPONSE =====")
            printer(controller.respond_to_query(selection))
            continue
        if normalized in {"finish", "finish investigation", "complete", "close"}:
            action_result, finished = _execute_action("finish investigation")
            if finished:
                return result
            if action_result.get("status") == "failed":
                recovery_choice = _show_failure("finish investigation", action_result)
                if recovery_choice in {"1", "retry", "r", "retry this action"}:
                    corrected_details = ask("Enter the corrected details:\n> ").strip()
                    retry_result, finished = _execute_action("finish investigation", corrected_details)
                    if finished:
                        return result
                    if retry_result.get("status") == "failed":
                        continue
                elif recovery_choice in {"3", "finish", "finish investigation", "close"}:
                    final_result = controller.finalize()
                    printer("\n===== FINAL REPORT =====")
                    printer(final_result.final_state.get("final_report", "No report generated"))
                    return final_result
            continue

        printer("\n===== CAPABILITY RESPONSE =====")
        printer(controller.respond_to_query(selection))
