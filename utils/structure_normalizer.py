from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass
class NormalizationResult:
    dataframe: pd.DataFrame
    applied: bool
    strategy: str
    details: dict


def _is_blank(value) -> bool:
    if pd.isna(value):
        return True
    return not str(value).strip()


def _clean_header_value(value, fallback: str) -> str:
    if _is_blank(value):
        return fallback
    return str(value).strip()


def _dedupe_headers(headers: List[str]) -> List[str]:
    counts = {}
    deduped = []
    for header in headers:
        base = header.strip() or "column"
        count = counts.get(base, 0)
        if count == 0:
            deduped.append(base)
        else:
            deduped.append(f"{base}_{count + 1}")
        counts[base] = count + 1
    return deduped


def _score_header_row(raw_df: pd.DataFrame, row_idx: int) -> float:
    row = raw_df.iloc[row_idx]
    non_empty = [str(v).strip() for v in row.tolist() if not _is_blank(v)]
    if len(non_empty) < 2:
        return -1.0

    unique_ratio = len(set(non_empty)) / max(len(non_empty), 1)
    alpha_like = sum(any(ch.isalpha() for ch in value) for value in non_empty) / len(non_empty)
    numeric_like = sum(value.replace(".", "", 1).isdigit() for value in non_empty) / len(non_empty)

    sample_following = raw_df.iloc[row_idx + 1: row_idx + 6]
    row_density = 0.0
    if not sample_following.empty:
        row_density = float(sample_following.notna().mean(axis=1).mean())

    return (len(non_empty) * 1.5) + (unique_ratio * 4.0) + (alpha_like * 2.0) + row_density - (numeric_like * 3.0)


def _find_header_row(raw_df: pd.DataFrame, max_scan_rows: int = 15) -> int | None:
    limit = min(max_scan_rows, len(raw_df))
    best_idx = None
    best_score = -1.0
    for row_idx in range(limit):
        score = _score_header_row(raw_df, row_idx)
        if score > best_score:
            best_score = score
            best_idx = row_idx
    return best_idx


def _header_blocks(header_row: pd.Series, min_block_width: int = 2) -> List[tuple[int, int]]:
    blocks: List[tuple[int, int]] = []
    start = None
    blank_run = 0

    for idx, value in enumerate(header_row.tolist()):
        is_blank = _is_blank(value)
        if start is None:
            if not is_blank:
                start = idx
            continue

        if is_blank:
            blank_run += 1
            if blank_run >= 2:
                end = idx - blank_run
                if end - start + 1 >= min_block_width:
                    blocks.append((start, end))
                start = None
                blank_run = 0
        else:
            blank_run = 0

    if start is not None:
        end = len(header_row) - 1
        while end >= start and _is_blank(header_row.iloc[end]):
            end -= 1
        if end - start + 1 >= min_block_width:
            blocks.append((start, end))

    return blocks


def _normalize_block(raw_df: pd.DataFrame, header_idx: int, start: int, end: int) -> pd.DataFrame:
    block = raw_df.iloc[header_idx:, start:end + 1].copy()
    raw_headers = [_clean_header_value(v, f"column_{i}") for i, v in enumerate(block.iloc[0].tolist(), start=1)]
    headers = _dedupe_headers(raw_headers)
    block = block.iloc[1:].copy()
    block.columns = headers
    block = block.dropna(axis=1, how="all")
    block = block.dropna(axis=0, how="all")

    if block.empty:
        return block

    min_non_null = max(1, min(3, len(block.columns)))
    block = block[block.notna().sum(axis=1) >= min_non_null]
    return block.reset_index(drop=True)


def _stack_similar_blocks(blocks: List[pd.DataFrame]) -> pd.DataFrame:
    non_empty_blocks = [block for block in blocks if not block.empty]
    if not non_empty_blocks:
        return pd.DataFrame()

    combined_columns = []
    for block in non_empty_blocks:
        for col in block.columns:
            if col not in combined_columns:
                combined_columns.append(col)

    aligned = [block.reindex(columns=combined_columns) for block in non_empty_blocks]
    combined = pd.concat(aligned, ignore_index=True)
    combined = combined.dropna(axis=1, how="all")
    combined = combined.dropna(axis=0, how="all")
    return combined.reset_index(drop=True)


def _frame_quality(df: pd.DataFrame) -> float:
    if df is None or df.empty:
        return -1.0
    cols = list(df.columns)
    valid_headers = [c for c in cols if isinstance(c, str) and c.strip() and not c.lower().startswith("unnamed:")]
    unnamed_penalty = sum(str(c).lower().startswith("unnamed:") for c in cols)
    duplicate_penalty = len(cols) - len(set(cols))
    row_density = float(df.notna().mean(axis=1).mean()) if len(df) else 0.0
    return (len(valid_headers) * 2.0) + (row_density * 5.0) - (unnamed_penalty * 1.5) - (duplicate_penalty * 2.0)


def normalize_report_like_dataframe(raw_df: pd.DataFrame) -> NormalizationResult:
    header_idx = _find_header_row(raw_df)
    if header_idx is None:
        return NormalizationResult(raw_df, False, "none", {"reason": "no_header_row"})

    blocks = _header_blocks(raw_df.iloc[header_idx])
    if not blocks:
        return NormalizationResult(raw_df, False, "none", {"reason": "no_blocks", "header_row": header_idx})

    normalized_blocks = [_normalize_block(raw_df, header_idx, start, end) for start, end in blocks]
    normalized_df = _stack_similar_blocks(normalized_blocks)
    if normalized_df.empty:
        return NormalizationResult(raw_df, False, "none", {"reason": "empty_after_normalization", "header_row": header_idx})

    return NormalizationResult(
        normalized_df,
        True,
        "header_blocks",
        {
            "header_row": header_idx,
            "blocks": blocks,
            "original_shape": tuple(raw_df.shape),
            "normalized_shape": tuple(normalized_df.shape),
        },
    )


def choose_best_dataframe(standard_df: pd.DataFrame, raw_df: pd.DataFrame | None = None) -> NormalizationResult:
    best = NormalizationResult(standard_df, False, "standard", {"quality": _frame_quality(standard_df)})
    if raw_df is None:
        return best

    normalized = normalize_report_like_dataframe(raw_df)
    normalized_quality = _frame_quality(normalized.dataframe)
    standard_quality = _frame_quality(standard_df)

    if normalized.applied and normalized_quality > standard_quality + 2.0:
        normalized.details["quality"] = normalized_quality
        return normalized

    best.details["quality"] = standard_quality
    return best
