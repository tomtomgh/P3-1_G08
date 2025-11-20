from __future__ import annotations

import os
from typing import Dict
import pandas as pd

from InputOutput import load_user_logs, compute_parameter_change_segments, time_to_seconds
from Analysis.TimeSeries import TimeSeriesConfig, build_timeseries
from Analysis.UserStats import compute_user_activity, compute_global_carefulness
from Strategies.Fighting import FightingConfig, StrategyWindow
from Strategies.Carefulness import CarefulnessConfig
from Strategies.TrialError import TrialErrorConfig, detect_trial_error, aggregate_trial_error_scores
from Strategies.Manager import StrategyManager
from UI.Dashboard import show_dashboard

OUTPUT_CSV = "parameter_changes_summary.csv"
TIME_RESOLUTION = 0.1
SHARED_PARAMS = ["frequency"]

def main() -> None:
    df = load_user_logs()
    summary = compute_parameter_change_segments(df)
    summary.to_csv(OUTPUT_CSV, index=False)

    ts_cfg = TimeSeriesConfig(time_resolution=TIME_RESOLUTION, shared_params=SHARED_PARAMS)
    ts = build_timeseries(df, ts_cfg)

    user_stats, total_duration = compute_user_activity(
        df=df,
        user_at_time=ts.user_at_time,
        shared_params=SHARED_PARAMS,
        time_resolution=TIME_RESOLUTION,
    )

    global_carefulness = compute_global_carefulness(df)

    fighting_cfg = FightingConfig()
    carefulness_cfg = CarefulnessConfig()
    trial_cfg = TrialErrorConfig()

    mgr = StrategyManager.default(
        fighting_cfg=fighting_cfg,
        carefulness_cfg=carefulness_cfg,
        trial_error_cfg=trial_cfg,
    )

    strategy_windows = mgr.run_all(df)

    te_windows = strategy_windows.get("trial_error", [])
    trial_scores = aggregate_trial_error_scores(te_windows)


    show_dashboard(
        ts=ts,
        user_stats=user_stats,
        total_duration=total_duration,
        strategy_windows=strategy_windows,
        trial_scores=trial_scores,
        carefulness_summary=global_carefulness,
    )




# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
