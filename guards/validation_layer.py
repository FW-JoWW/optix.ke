# guards/validation_layer.py

import re

def parse_comparison_insight(insight: str):
    """
    Extract structured meaning from insight.
    Example:
    'Diesel has higher mileage than Petrol by 1000'
    """

    pattern = r"(\w+)\s+(has\s+)?(higher|lower)\s+(\w+)\s+(than|compared\s+to)\s+(\w+)(\s+by\s+([\d\.]+))?"

    match = re.search(pattern, insight.lower())

    if not match:
        return None

    return {
        "group_1": match.group(1),
        "direction": match.group(3),
        "metric": match.group(4),
        "group_2": match.group(6),
        "value": match.group(8),
    }


# -----------------------------
# 1. SCHEMA VALIDATION
# -----------------------------
def validate_schema(output: dict):
    required_keys = [
        "insight",
        "evidence_used",
        "assumptions",
        "uncertainties",
        "confidence"
    ]

    for key in required_keys:
        if key not in output:
            return False, f"Missing key: {key}"

    if not isinstance(output["evidence_used"], list):
        return False, "evidence_used must be a list"

    if not (0.0 <= output["confidence"] <= 1.0):
        return False, "Confidence must be between 0 and 1"

    return True, "Schema valid"


# -----------------------------
# 2. EVIDENCE VALIDATION
# -----------------------------
def validate_evidence(output: dict, tool_results: dict):
    for ref in output.get("evidence_used", []):
        if ref not in tool_results:
            return False, f"Missing evidence: {ref}"

    return True, "Evidence valid"


# -----------------------------
# 3. HALLUCINATION CHECK
# -----------------------------
def extract_columns(text: str):
    # extract words safely
    return set(re.findall(r"\b[a-zA-Z_]+\b", text.lower()))



def validate_no_hallucination(output, dataset_columns, category_values):
    insight = output.get("insight", "").lower()

    parsed = parse_comparison_insight(insight)

    if not parsed:
        return False, "Insight structure not recognized"

    metric = parsed["metric"]
    group_1 = parsed["group_1"]
    group_2 = parsed["group_2"]

    # Normalize
    dataset_columns_lower = [c.lower() for c in dataset_columns]

    # Flatten categories
    all_categories = set()
    for values in category_values.values():
        all_categories.update([str(v).lower() for v in values])

    # ✅ Check metric (must be a real column)
    if metric not in dataset_columns_lower:
        return False, f"Invalid metric: {metric}"

    # ✅ Check groups (must be real category values)
    if group_1 not in all_categories:
        return False, f"Invalid category: {group_1}"

    if group_2 not in all_categories:
        return False, f"Invalid category: {group_2}"

    return True, "Valid structured insight"


# -----------------------------
# 4. CONFIDENCE VALIDATION
# -----------------------------
def compute_confidence(p_value: float, effect_size: float):
    if p_value < 0.05 and effect_size > 0.1:
        return 0.9
    elif p_value < 0.05:
        return 0.7
    else:
        return 0.4


def validate_confidence(output: dict, computed_conf: float):
    llm_conf = output.get("confidence", 0)

    if abs(llm_conf - computed_conf) > 0.2:
        return False, f"Confidence mismatch: LLM={llm_conf}, computed={computed_conf}"

    return True, "Confidence valid"


# -----------------------------
# 5. CRITIC PASS (SECOND LLM)
# -----------------------------
def critic_check(critic_output: dict):
    issues = critic_output.get("issues", [])

    if len(issues) > 0:
        return False, f"Issues found: {issues}"

    return True, "Critic passed"


# -----------------------------
# 6. MASTER VALIDATOR
# -----------------------------
def run_full_validation(
    output,
    tool_results,
    dataset_columns,
    category_values,
    computed_conf,
    critic_output
):
    checks = []

    checks.append(validate_schema(output))
    checks.append(validate_evidence(output, tool_results))
    checks.append(validate_no_hallucination(output, dataset_columns, category_values))
    checks.append(validate_confidence(output, computed_conf))
    checks.append(critic_check(critic_output))

    failed = [msg for success, msg in checks if not success]

    if failed:
        return False, failed

    return True, "All checks passed"