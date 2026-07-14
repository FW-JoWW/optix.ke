from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import List, Sequence

import pandas as pd

from graph.analyst_graph import graph
from state.state import AnalystState


@dataclass
class GuidedHarnessResult:
    final_state: AnalystState
    prompts: List[str]
    responses: List[str]
    dataset_path: str


class ScriptedInput:
    def __init__(self, responses: Sequence[str]):
        self._responses = list(responses)
        self.prompts: List[str] = []
        self.used_responses: List[str] = []
        self._index = 0

    def __call__(self, prompt: str = "") -> str:
        self.prompts.append(prompt)
        if self._index >= len(self._responses):
            raise RuntimeError(
                "Scripted input exhausted before the workflow finished. "
                "Add more responses to the scenario."
            )
        response = self._responses[self._index]
        self._index += 1
        self.used_responses.append(response)
        print(prompt, end="")
        print(response)
        return response


@contextmanager
def scripted_input(responses: Sequence[str]):
    import builtins

    original_input = builtins.input
    responder = ScriptedInput(responses)
    builtins.input = responder
    try:
        yield responder
    finally:
        builtins.input = original_input


def build_guided_sample_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [21, 34, 34, 45, 52, None],
            "Revenue": [100, 150, 180, 220, 260, 290],
            "Profit": [20, 33, 41, 50, 59, 65],
            "Region": ["North", "South", "South", "West", "West", "North"],
            "Category": ["A", "A", "B", "B", "C", "C"],
            "Score": [0.10, 0.25, 0.40, 0.30, 0.55, 0.60],
        }
    )


def _write_temp_csv(df: pd.DataFrame) -> str:
    fd, path = tempfile.mkstemp(prefix="guided_mode_", suffix=".csv")
    os.close(fd)
    df.to_csv(path, index=False)
    return path


def build_guided_state(
    question: str,
    dataset_path: str | None = None,
    dataframe: pd.DataFrame | None = None,
) -> tuple[AnalystState, str]:
    df = dataframe if dataframe is not None else build_guided_sample_dataframe()
    if dataset_path is None:
        dataset_path = _write_temp_csv(df)

    state: AnalystState = {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": "guided",
        "enable_llm_reasoning": False,
        "disable_llm_reasoning": True,
        "disable_semantic_matcher": True,
        "analysis_evidence": {},
    }
    return state, dataset_path


def run_guided_workflow(
    question: str,
    responses: Sequence[str],
    dataset_path: str | None = None,
    dataframe: pd.DataFrame | None = None,
) -> GuidedHarnessResult:
    state, resolved_path = build_guided_state(
        question=question,
        dataset_path=dataset_path,
        dataframe=dataframe,
    )

    try:
        with scripted_input(responses) as responder:
            final_state = graph.invoke(state)
        return GuidedHarnessResult(
            final_state=final_state,
            prompts=list(responder.prompts),
            responses=list(responder.used_responses),
            dataset_path=resolved_path,
        )
    finally:
        if dataset_path is None:
            try:
                Path(resolved_path).unlink(missing_ok=True)
            except Exception:
                pass


def summarize_guided_result(result: GuidedHarnessResult) -> dict:
    evidence = result.final_state.get("analysis_evidence", {})
    return {
        "awaiting_user": result.final_state.get("awaiting_user", False),
        "final_report_available": bool(result.final_state.get("final_report")),
        "final_output": evidence.get("final_output"),
        "guided_decision_log": result.final_state.get("guided_decision_log", []),
        "human_in_loop": evidence.get("human_in_loop"),
        "analysis_plan": evidence.get("analysis_plan"),
        "tool_result_keys": list((evidence.get("tool_results") or {}).keys()),
        "visualization_count": len(evidence.get("visualizations") or []),
    }


def default_guided_responses() -> List[str]:
    return ["continue", "continue", "continue", "continue"]


def scenario_responses(name: str) -> List[str]:
    normalized = (name or "").strip().lower()
    if normalized in {"continue", "happy_path", "default"}:
        return default_guided_responses()
    if normalized in {"fallback", "unsupported"}:
        return [
            "modify",
            "use mean imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "modify",
            "use Kruskal-Wallis",
            "continue",
            "continue",
        ]
    if normalized in {"column_specific", "column", "median_age"}:
        return [
            "modify",
            "use median imputation for Age, keep outliers, do not remove duplicates",
            "continue",
            "continue",
            "continue",
            "continue",
        ]
    if normalized in {"cancel", "stop"}:
        return ["continue", "continue", "continue", "cancel"]
    raise ValueError(f"Unknown guided scenario: {name}")


def main() -> None:
    question = os.getenv("GUIDED_TEST_QUESTION", "What is the relationship between Revenue and Profit by Region?")
    scenario = os.getenv("GUIDED_TEST_SCENARIO", "continue")
    responses = scenario_responses(scenario)
    result = run_guided_workflow(question=question, responses=responses)
    summary = summarize_guided_result(result)
    print("\n===== GUIDED HARNESS SUMMARY =====")
    for key, value in summary.items():
        print(f"{key}: {value}")
    if summary["final_report_available"]:
        print("\n===== FINAL REPORT =====\n")
        print(result.final_state.get("final_report", "No report generated"))


if __name__ == "__main__":
    main()
