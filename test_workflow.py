import os
import pprint
import io
import warnings
from contextlib import redirect_stdout
from time import perf_counter

import pandas as pd

from collaborative_mode.session_runner import run_interactive_collaborative_session
from collaborative_mode.presentation import render_collaborative_analyst_view, render_debug_collaborative_view
from graph.analyst_graph import graph
from state.state import AnalystState
from utils.openai_runtime import get_openai_runtime_info
from scripts.collaborative_mode_harness import build_guided_sample_dataframe
from scripts.guided_mode_harness import (
    default_guided_responses,
    run_guided_workflow,
    scenario_responses,
    summarize_guided_result,
)


def load_default_dataframe(dataset_path: str) -> pd.DataFrame:
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    return pd.read_csv(dataset_path, low_memory=False)


def _run_collaborative(question: str, workflow_mode: str, dataset_path: str, df: pd.DataFrame) -> None:
    collab_responses_env = os.getenv("COLLABORATIVE_TEST_RESPONSES", "").strip()
    collab_test_mode = os.getenv("COLLABORATIVE_TEST_MODE", "").strip().lower()
    collab_report_mode = os.getenv("COLLABORATIVE_REPORT_MODE", "").strip().lower()
    use_scripted_responses = collab_test_mode in {"scripted", "1", "true", "yes"}
    responses = [item.strip() for item in collab_responses_env.split("|") if item.strip()] if (collab_responses_env and use_scripted_responses) else None
    build_final_report = collab_report_mode != "preview"

    if responses is None:
        print("\n[Agent] Running collaborative workflow in interactive test mode.")
        print("[Agent] This path uses the same collaborative session runner as the live mode, so it keeps going until the decision layer asks for human guidance or you select a human action.")
        if collab_responses_env and not use_scripted_responses:
            print("[Agent] Scripted collaborative responses were detected, but they are being ignored because COLLABORATIVE_TEST_MODE is not set to scripted.")
    else:
        print("\n[Agent] Running collaborative workflow in scripted test mode.")
    initial_tasks = [
        {
            "title": "Primary investigation",
            "request": question,
        }
    ]
    result = run_interactive_collaborative_session(
        question=question,
        responses=responses,
        dataset_path=dataset_path,
        dataframe=df,
        initial_tasks=initial_tasks,
        build_final_report=build_final_report,
    )
    print("\n===== COLLABORATIVE TEST SUMMARY =====")
    if collab_report_mode == "debug":
        print(render_debug_collaborative_view(result))
    else:
        print(render_collaborative_analyst_view(result))
    if result.final_state.get("final_report"):
        print("\n===== COLLABORATIVE FINAL REPORT =====")
        print(result.final_state.get("final_report"))


def main() -> None:
    print("\n===== DATA ANALYST AGENT =====\n")

    question = input("Enter your business question:\n> ").strip()
    mode = input("\nChoose mode (autonomous / guided / collaborative):\n> ").strip().lower() or "autonomous"

    workflow_mode = os.getenv("WORKFLOW_TEST_MODE", "").strip().lower()
    guided_scenario = os.getenv("GUIDED_TEST_SCENARIO", "").strip().lower()
    guided_responses_env = os.getenv("GUIDED_TEST_RESPONSES", "").strip()

    if mode == "collaborative":
        dataset_path = "data/olist_merged_dataset.csv"
        df = load_default_dataframe(dataset_path)
        use_sample_dataset = os.getenv("COLLABORATIVE_USE_SAMPLE_DATASET", "").strip().lower() in {"1", "true", "yes"}
        if use_sample_dataset:
            df = build_guided_sample_dataframe()
            dataset_path = "data/collaborative_test_dataset.csv"
            print("\n[Agent] Collaborative test dataset override enabled.")
        _run_collaborative(question, workflow_mode, dataset_path, df)
        return

    if mode == "guided" and workflow_mode in {"guided", "guided-test"}:
        responses = default_guided_responses()
        if guided_scenario:
            responses = scenario_responses(guided_scenario)
        elif guided_responses_env:
            responses = [item.strip() for item in guided_responses_env.split("|") if item.strip()]
        print("\n[Agent] Running guided workflow in scripted test mode.")
        result = run_guided_workflow(
            question=question,
            responses=responses,
            dataframe=build_guided_sample_dataframe(),
        )
        summary = summarize_guided_result(result)
        print("\n===== GUIDED TEST SUMMARY =====")
        pprint.pprint(summary)
        if summary.get("final_report_available"):
            print("\n===== FINAL REPORT =====\n")
            print(result.final_state.get("final_report", "No report generated"))
        else:
            print("\n===== FINAL OUTPUT =====\n")
            pprint.pprint(summary.get("final_output"))
        return

    dataset_path = "data/olist_merged_dataset.csv"  # "data/Car Dataset 1945-2020.csv"
    df = load_default_dataframe(dataset_path)

    runtime_info = get_openai_runtime_info()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    semantic_matcher_disabled = os.getenv("DISABLE_SEMANTIC_MATCHER", "").strip().lower() in {"1", "true", "yes"}

    if not openai_api_key:
        print("[WARN] OPENAI_API_KEY is not set. LLM nodes may fall back or skip.")

    state: AnalystState = {
        "business_question": question,
        "dataset_path": dataset_path,
        "dataframe": df,
        "mode": mode,
        "enable_llm_reasoning": True,
        "disable_llm_reasoning": False,
        "disable_semantic_matcher": semantic_matcher_disabled,
        "analysis_evidence": {},
    }

    print("\n[Agent] Workflow configuration:")
    print(f"- LLM requested: {state['enable_llm_reasoning'] and not state['disable_llm_reasoning']}")
    print(f"- Semantic matcher requested: {not state['disable_semantic_matcher']}")
    print(f"- OPENAI_API_KEY configured: {bool(openai_api_key)}")
    print(f"- OpenAI proxy env detected: {runtime_info['proxy_env_present']}")
    print(f"- OpenAI client ignores proxy env: {runtime_info['trust_env_for_openai'] is False}")

    print("\n[Agent] Starting analysis...\n")

    show_trace = os.getenv("SHOW_WORKFLOW_TRACE", "").strip().lower() in {"1", "true", "yes"}
    if mode == "guided" and workflow_mode not in {"guided", "guided-test"}:
        show_trace = True
    trace_buffer = io.StringIO()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        started_at = perf_counter()
        if show_trace:
            final_state = graph.invoke(state)
        else:
            with redirect_stdout(trace_buffer):
                final_state = graph.invoke(state)
        elapsed_seconds = round(perf_counter() - started_at, 2)

    evidence = final_state.get("analysis_evidence", {})

    if show_trace:
        print("\n===== INTERNAL TRACE =====")
    else:
        trace_output = trace_buffer.getvalue()
        if trace_output.strip():
            trace_lines = [line for line in trace_output.splitlines() if line.strip()]
            tail = trace_lines[-20:]
            if tail:
                print("\n===== INTERNAL TRACE TAIL =====")
                print("\n".join(tail))

    print("\n===== HUMAN IN LOOP =====")
    pprint.pprint(evidence.get("human_in_loop"))
    print(f"\n===== ELAPSED SECONDS =====\n{elapsed_seconds}")

    if final_state.get("awaiting_user") or evidence.get("final_output") is not None:
        print("\n===== FINAL OUTPUT =====")
        pprint.pprint(evidence.get("final_output"))
        raise SystemExit(0)

    print("\n===== LLM STATUS =====")
    print(f"Reasoning: {final_state.get('llm_reasoning_status', 'unknown')}")
    print(f"Synthesis: {evidence.get('llm_synthesis_status', 'unknown')}")

    print("\n===== ANALYSIS PLAN =====")
    pprint.pprint(evidence.get("analysis_plan"))

    print("\n===== DECISION ENGINE =====")
    pprint.pprint(evidence.get("analysis_decisions") or final_state.get("decision_output"))

    print("\n===== COMPUTATION PLAN =====")
    pprint.pprint(evidence.get("computation_plan"))

    print("\n===== TOOL RESULTS =====")
    pprint.pprint(evidence.get("tool_results"))

    print("\n===== STORY CANDIDATES =====")
    pprint.pprint(evidence.get("story_candidates"))

    print("\n===== TOP STORIES =====")
    pprint.pprint(evidence.get("top_stories"))

    print("\n===== DECISION PRIORITIES =====")
    pprint.pprint(evidence.get("decision_priority_ranking"))

    print("\n===== ALL DECISION RECORDS =====")
    pprint.pprint(evidence.get("decision_recommendations"))

    print("\n===== JUDGMENT SUMMARY =====")
    pprint.pprint(evidence.get("judgment_summary"))

    print("\n===== RECOMMENDED FIRST ACTION =====")
    pprint.pprint(evidence.get("decision_recommended_first"))

    print("\n===== CLARIFICATION QUESTIONS =====")
    pprint.pprint(final_state.get("clarification_questions") or evidence.get("clarification_questions"))

    print("\n===== LLM INSIGHTS =====")
    pprint.pprint(final_state.get("llm_insights") or evidence.get("llm_insights"))

    print("\n===== FINAL REPORT =====")
    print(final_state.get("final_report", "No report generated"))


if __name__ == "__main__":
    main()
