from typing import TypedDict, Optional, Dict, Any, List
import pandas as pd

class AnalystState(TypedDict):

    # user input
    business_question: str

    # dataset
    dataset_path: Optional[str]
    dataframe: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]

    # dataset understanding
    dataset_profile: Optional[Dict[str, Any]]
    data_validation: Optional[Dict[str, Any]]
    relevant_columns: Optional[List[str]]
    
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
    awaiting_user: bool
    question_for_user: str
    user_response: str

    # analysis evidence container
    analysis_evidence: Optional[Dict[str, Any]]

    # agent messages
    clarification_questions: Optional[list]
    llm_insights: Optional[str]

    # final report
    final_report: Optional[str]