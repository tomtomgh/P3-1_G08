from __future__ import annotations
import pandas as pd
from typing import Iterable
from .constants import ALL_STRATEGIES

def enforce_inactive_guard(df: pd.DataFrame, pred_col_names: Iterable[str] | None = None) -> pd.DataFrame:
    """
    Overwrite strategy predictions & probabilities for rows that are inactive.
    Inactive = num_actions == 0 OR has_events == 0 (handles missing cols safely).
    Returns modified DataFrame (inplace operations used but returns df for convenience).
    """
    df = df.copy()
    num_actions = df.get("num_actions")
    has_events = df.get("has_events")

    inactive_mask = pd.Series(False, index=df.index)
    if num_actions is not None:
        inactive_mask = inactive_mask | (num_actions.astype(int) == 0)
    if has_events is not None:
        inactive_mask = inactive_mask | (has_events.astype(int) == 0)

    if not inactive_mask.any():
        return df

    # zero out strategy probability columns
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in df.columns:
            df.loc[inactive_mask, col] = 0.0

    # canonical prediction column names: pred_strategy, pred, pred_label
    candidates = ["pred_strategy", "pred", "pred_label"]
    if pred_col_names:
        candidates = list(pred_col_names) + candidates

    for c in candidates:
        if c in df.columns:
            df.loc[inactive_mask, c] = "No Strategy"

    # fallback: if none of the pred cols exist, create pred_strategy
    if not any(c in df.columns for c in candidates):
        df.loc[inactive_mask, "pred_strategy"] = "No Strategy"

    return df