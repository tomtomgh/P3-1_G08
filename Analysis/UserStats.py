# analysis/user_stats.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple, List

import numpy as np
import pandas as pd

# Tunables kept as constants to avoid magic numbers littered through the code.
LONG_GAP_THRESHOLD = 10.0

__all__ = ["UserActivityStats", "compute_user_activity", "compute_global_carefulness"]


@dataclass
class UserActivityStats:
    """
    Per-user activity, dominance and timing metrics.

    This is a structured version of what your old `analyze_user_activity`
    function computed and stuffed into `user_stats[user]`.
    """
    total_changes: int
    changes_per_minute: float
    param_changes: Dict[str, int] = field(default_factory=dict)
    time_in_control: Dict[str, float] = field(default_factory=dict)

    max_inactivity_gap: float = 0.0
    avg_gap_between_changes: float = 0.0
    long_inactivity_periods: int = 0

    first_action_time: float = 0.0
    last_action_time: float = 0.0
    active_duration: float = 0.0
    activity_percentage: float = 0.0


def compute_user_activity(
    df: pd.DataFrame,
    user_at_time: Dict[str, np.ndarray],
    *,
    shared_params: Iterable[str],
    time_resolution: float,
) -> Tuple[Dict[int, UserActivityStats], float]:
    """
    Compute user-level activity/dominance metrics.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: 'user', 'param', 'time_sec'.
    user_at_time : dict[param -> np.ndarray]
        For each shared parameter, an array (len = len(timeline)) with the
        user-id that 'owns' that sample (or -1 if no value).
        This is exactly what TimeSeriesResult.user_at_time contains for shared params.
    shared_params : iterable of str
        Parameters that are considered shared (e.g. 'frequency').
    time_resolution : float
        Time step of the timeline (used to convert control points -> seconds).

    Returns
    -------
    (user_stats, total_duration):
        user_stats : dict[user_id -> UserActivityStats]
        total_duration : float (seconds between first and last event in df)
    """
    required_cols = {"user", "param", "time_sec"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    if df.empty:
        return {}, 0.0

    df = df.copy()
    df["param"] = df["param"].astype(str).str.lower().str.strip()
    df = df.sort_values("time_sec").reset_index(drop=True)

    users: List[int] = sorted(df["user"].unique())
    params: List[str] = sorted(df["param"].unique())

    t_min = float(df["time_sec"].min())
    t_max = float(df["time_sec"].max())
    total_duration = max(t_max - t_min, 0.0)

    user_stats: Dict[int, UserActivityStats] = {}

    shared_params_set = set(p.lower() for p in shared_params)

    for user in users:
        user_df = df[df["user"] == user]

        total_changes = int(len(user_df))
        if total_duration > 0:
            changes_per_minute = (total_changes / total_duration) * 60.0
        else:
            changes_per_minute = 0.0

        # Per-parameter change counts
        param_changes: Dict[str, int] = {}
        for p in params:
            param_changes[p] = int((user_df["param"] == p).sum())

        # Time in control for shared parameters using user_at_time
        time_in_control: Dict[str, float] = {}
        for p in shared_params_set:
            arr = user_at_time.get(p)
            if arr is None:
                continue
            control_points = int(np.sum(arr == user))
            time_controlled = control_points * time_resolution
            time_in_control[p] = float(time_controlled)

        # Inactivity gaps
        if len(user_df) > 1:
            times = user_df["time_sec"].sort_values().to_numpy()
            gaps = np.diff(times)
            if gaps.size > 0:
                max_gap = float(gaps.max())
                avg_gap = float(gaps.mean())
                num_long_gaps = int(np.sum(gaps > LONG_GAP_THRESHOLD))  # > LONG_GAP_THRESHOLD seconds
            else:
                max_gap = 0.0
                avg_gap = 0.0
                num_long_gaps = 0
        else:
            max_gap = 0.0
            avg_gap = 0.0
            num_long_gaps = 0

        # First / last actions and active duration
        if len(user_df) > 0:
            first_action = float(user_df["time_sec"].min())
            last_action = float(user_df["time_sec"].max())
            active_duration = max(last_action - first_action, 0.0)
        else:
            first_action = 0.0
            last_action = 0.0
            active_duration = 0.0

        if total_duration > 0:
            activity_percentage = (active_duration / total_duration) * 100.0
        else:
            activity_percentage = 0.0

        user_stats[user] = UserActivityStats(
            total_changes=total_changes,
            changes_per_minute=changes_per_minute,
            param_changes=param_changes,
            time_in_control=time_in_control,
            max_inactivity_gap=max_gap,
            avg_gap_between_changes=avg_gap,
            long_inactivity_periods=num_long_gaps,
            first_action_time=first_action,
            last_action_time=last_action,
            active_duration=active_duration,
            activity_percentage=activity_percentage,
        )

    return user_stats, total_duration


# -------- OPTIONAL: global carefulness summary (not sliding-window strategy) --------

def compute_global_carefulness(
    df: pd.DataFrame,
    *,
    small_step_ratio: float = 0.10,
    min_points: int = 2,
) -> Dict[int, Dict[str, Dict[str, float | str]]]:
    """
    Static (non-windowed) carefulness metrics per user & parameter.

    This is essentially your old `analyze_carefulness` logic, but returning a
    nested dict instead of printing. It is **separate** from the sliding-window
    Carefulness strategy in strategies/Carefulness.py.

    Returns
    -------
    {
        user_id: {
            param_name: {
                'avg_step': float | nan,
                'max_step': float | nan,
                'careful_ratio': float | nan,
                'behavior_type': 'Careful' | 'Reckless' | 'Insufficient data' | 'Careful (no variation)',
            },
            ...
        },
        ...
    }
    """
    required_cols = {"user", "param", "time_sec", "value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        raise ValueError(f"DataFrame missing required columns: {missing}")

    df = df.copy()
    df["param"] = df["param"].astype(str).str.lower().str.strip()

    users = sorted(df["user"].unique())
    params = sorted(df["param"].unique())

    results: Dict[int, Dict[str, Dict[str, float | str]]] = {}

    for user in users:
        user_dict: Dict[str, Dict[str, float | str]] = {}

        for param in params:
            param_df = df[(df["user"] == user) & (df["param"] == param)].sort_values(
                "time_sec"
            )
            if len(param_df) < min_points:
                user_dict[param] = {
                    "avg_step": float("nan"),
                    "max_step": float("nan"),
                    "careful_ratio": float("nan"),
                    "behavior_type": "Insufficient data",
                }
                continue

            values = param_df["value"].to_numpy()
            diffs = np.abs(np.diff(values))

            if diffs.size == 0 or np.all(diffs == 0):
                user_dict[param] = {
                    "avg_step": 0.0,
                    "max_step": 0.0,
                    "careful_ratio": 0.0,
                    "behavior_type": "Careful (no variation)",
                }
                continue

            avg_step = float(np.mean(diffs))
            max_step = float(np.max(diffs))

            param_range = float(np.ptp(values))
            threshold = small_step_ratio * (param_range if param_range > 0 else 1.0)

            small_steps = int(np.sum(diffs <= threshold))
            careful_ratio = 100.0 * small_steps / diffs.size if diffs.size > 0 else 0.0

            behavior = "Careful" if careful_ratio > 70.0 else "Reckless"

            user_dict[param] = {
                "avg_step": avg_step,
                "max_step": max_step,
                "careful_ratio": careful_ratio,
                "behavior_type": behavior,
            }

        results[user] = user_dict

    return results
