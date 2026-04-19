from __future__ import annotations

import re

import pandas as pd


def normalize_numeric_token(value) -> float | None:
    if pd.isna(value):
        return None

    s = str(value).strip()
    if not s:
        return None

    if any(ch.isalpha() for ch in s):
        return None

    if "/" in s:
        return None

    s = re.sub(r"[\s$€£¥%]", "", s)
    s = re.sub(r"[^\d,.\-]", "", s)
    if not s:
        return None

    if re.match(r"^\d{1,3}(\.\d{3})+,\d+$", s):
        s = s.replace(".", "").replace(",", ".")
    elif re.match(r"^\d{1,3}(,\d{3})+$", s):
        s = s.replace(",", "")
    elif "," in s and "." not in s:
        s = s.replace(",", ".")

    try:
        return float(s)
    except (TypeError, ValueError):
        return None
