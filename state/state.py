#state/state.py
from typing import TypedDict, Optional, Dict, Any, List
import pandas as pd

class AnalystState(TypedDict):

    # user input
    business_question: str

    # dataset
    dataset_path: Optional[str]
    dataframe: Optional[pd.DataFrame]
    data_quality_issues: Optional[Dict[str, Any]]
    cleaning_plan: Optional[List[Dict[str, Any]]]
    cleaned_data: Optional[pd.DataFrame]
    active_dataset: Optional[str]
    analysis_dataset: Optional[pd.DataFrame]
    raw_analysis_dataset: Optional[pd.DataFrame]

    # dataset understanding
    dataset_profile: Optional[Dict[str, Any]]
    data_validation: Optional[Dict[str, Any]]
    profile_ambiguity: Optional[Dict[str, Any]]
    structural_signals: Optional[Dict[str, Any]]
    context_inference: Optional[Dict[str, Any]]
    cleaning_constraints: Optional[Dict[str, Any]]
    decision_output: Optional[Dict[str, Any]]
    cleaning_validation: Optional[Dict[str, Any]]
    relationship_signals: Optional[Dict[str, Any]]
    column_registry: Optional[Dict[str, Any]]
    relevant_columns: Optional[List[str]]
    selected_columns: Optional[List[str]]
        
    # intent
    intent: Optional[Dict[str, Any]]
    intent_confidence: Optional[float]
    llm_reasoning: Optional[Dict[str, Any]]
    llm_reasoning_status: Optional[str]
    
    # analysis plan
    analysis_plan: Optional[List[str]]

    # analysis outputs
    eda_results: Optional[Dict[str, Any]]
    statistical_results: Optional[Dict[str, Any]]
    stat_test: Optional[str]

    # tools result
    tool_plan: Optional[List[Dict[str, Any]]]
    tool_results: Optional[Dict[str, Any]]

    # reasoning
    insights: Optional[str]

    # modes
    mode: str
    output_mode: Optional[str]
    awaiting_user: bool
    question_for_user: str
    user_response: str
    enable_llm_reasoning: bool
    disable_llm_reasoning: bool
    disable_semantic_matcher: bool

    # analysis evidence container
    analysis_evidence: Optional[Dict[str, Any]]
    human_in_loop: Optional[Dict[str, Any]]

    # dataset normalization pipeline
    normalization_schema: Optional[Any]
    normalization_schema_name: Optional[str]
    normalization_source_type: Optional[str]
    normalization_synonyms: Optional[Dict[str, List[str]]]
    normalization_output: Optional[Dict[str, Any]]

    # agent messages
    clarification_questions: Optional[list]
    llm_insights: Optional[str]

    # final report
    final_report: Optional[str]
    decision_context: Optional[Any]
