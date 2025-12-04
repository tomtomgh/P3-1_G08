from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
from .segmentation import build_segments_from_speed_csv, SpeedSegment
from .features import build_feature_table_for_session
from .trees import apply_all_strategies
from .prediction_utils import enforce_inactive_guard

def run_full_pipeline_for_session(
    events: List[Dict[str,Any]],
    speed_csv_path: str | Path,
    session_id: str,
    dull_max_duration: float = 20.0,
    dull_window: float = 10.0,
) -> Tuple[pd.DataFrame, List[SpeedSegment]]:
    segments = build_segments_from_speed_csv(speed_csv_path, dull_max_duration, dull_window)
    df_feats = build_feature_table_for_session(events, segments, session_id=session_id)

    # run rule-based strategy prediction / prob assignment
    df_pred = apply_all_strategies(df_feats.copy())

    # enforce inactive-rows guard AFTER predictions/probabilities are assigned
    df_pred = enforce_inactive_guard(df_pred)

    return df_pred, segments
