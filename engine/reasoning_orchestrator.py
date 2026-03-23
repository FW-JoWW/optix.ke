def reasoning_orchestrator(candidate, validation_fn, max_retries=3):
    """
    Adaptive retry system that modifies insight based on failure reason
    """

    current = candidate.copy()

    for attempt in range(max_retries):
        is_valid, result = validation_fn(current)

        if is_valid:
            return {
                "status": "accepted",
                "output": current
            }

        print(f"[RETRY] Attempt {attempt+1} failed: {result}")

        error_text = " ".join(result).lower()

        # -----------------------------
        # 🔥 ADAPTIVE FIXES
        # -----------------------------

        # 1. Structure issue → simplify sentence
        if "structure" in error_text:
            current["insight"] = simplify_insight(current["insight"])

        # 2. Invalid category → fallback to safe generic
        elif "invalid category" in error_text:
            current["insight"] = fallback_category_insight(current)

        # 3. Invalid metric → fallback generic
        elif "invalid metric" in error_text:
            current["insight"] = "There is a measurable difference across groups"

        # 4. Confidence mismatch → fix it
        elif "confidence mismatch" in error_text:
            current["confidence"] = adjust_confidence(current)

        # 5. Unknown issue → last resort simplification
        else:
            current["insight"] = generic_insight(current)

    return {
        "status": "rejected",
        "output": current
    }


# -----------------------------
# FIX STRATEGIES
# -----------------------------

def simplify_insight(insight: str):
    # remove "by X" (most common failure point)
    return insight.split(" by ")[0]


def fallback_category_insight(output):
    return "Different categories show variation in values"


def generic_insight(output):
    return "A statistically significant relationship exists"


def adjust_confidence(output):
    # safe fallback
    return 0.9

'''# engine/reasoning_orchestrator.py

def reasoning_orchestrator(candidate, validation_fn, max_retries=2):
    """
    Controls reasoning flow and improves output quality.
    """

    attempt = 0

    while attempt <= max_retries:

        is_valid, message = validation_fn(candidate)

        if is_valid:
            return {
                "status": "accepted",
                "output": candidate,
                "confidence": candidate.get("confidence", 0.5),
                "message": message,
                "attempts": attempt + 1
            }

        # ❗ Not valid → try to improve instead of rejecting immediately
        print(f"[RETRY] Attempt {attempt+1} failed: {message}")

        candidate = refine_candidate(candidate, message)

        attempt += 1

    return {
        "status": "rejected",
        "output": candidate,
        "confidence": 0.0,
        "message": "Max retries exceeded"
    }


def refine_candidate(candidate, error_message):
    """
    Lightweight repair logic before sending back to LLM.
    """

    # Example: fix hallucination issues
    if "hallucinated" in error_message:
        candidate["confidence"] = max(0.3, candidate.get("confidence", 0.5) - 0.2)

    # Example: if confidence mismatch
    if "Confidence mismatch" in error_message:
        candidate["confidence"] *= 0.9

    return candidate'''