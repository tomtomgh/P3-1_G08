# analysis/timeseries.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Iterable

import numpy as np
import pandas as pd


@dataclass
class TimeSeriesConfig:
    """
    Configuration for building dense time series from event logs.
    """
    time_resolution: float = 0.1
    # Which parameters are "shared" and should get merged_series/user_at_time
    shared_params: Iterable[str] = ("frequency",)


@dataclass
class TimeSeriesResult:
    """
    Result of build_timeseries():
    - timeline: common time axis
    - series:   per-parameter, per-user trajectories
    - merged_series: per-parameter trajectories merged across users (for shared params)
    - user_at_time:  for each shared param, which user 'owns' that time sample
    - users, params, t_min, t_max, time_resolution
    """
    timeline: np.ndarray
    series: Dict[str, Dict[int, np.ndarray]]
    merged_series: Dict[str, np.ndarray]
    user_at_time: Dict[str, np.ndarray]
    users: List[int]
    params: List[str]
    t_min: float
    t_max: float
    time_resolution: float


def build_timeseries(
    df: pd.DataFrame,
    cfg: TimeSeriesConfig,
) -> TimeSeriesResult:
    """
    Build dense time series from the raw change events.

    Expects df with at least: 'time_sec', 'user', 'param', 'value'.

    Returns a TimeSeriesResult you can pass to the plotting/UI layer.
    """
    # small helper used only for clearer structure
    def _ensure_required_columns(dataframe: pd.DataFrame, required: set) -> None:
        if not required.issubset(dataframe.columns):
            missing = required - set(dataframe.columns)
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    _ensure_required_columns(df, {"time_sec", "user", "param", "value"})

    # Normalise / sort
    df = df.copy()
    df["param"] = df["param"].astype(str).str.lower().str.strip()
    df = df.sort_values(["time_sec", "user", "param"]).reset_index(drop=True)

    if df.empty:
        raise ValueError("No data in df; cannot build time series.")

    users: List[int] = sorted(df["user"].unique())
    params: List[str] = sorted(df["param"].unique())

    t_min: float = float(df["time_sec"].min())
    t_max: float = float(df["time_sec"].max())

    # Same behaviour as your old code: np.arange [t_min, t_max) with step = time_resolution
    timeline = np.arange(t_min, t_max, cfg.time_resolution)
    if timeline.size == 0:
        # Degenerate case: all events at exactly the same time → create single sample
        timeline = np.array([t_min], dtype=float)

    # Initialise containers
    series: Dict[str, Dict[int, np.ndarray]] = {
        p: {u: np.full_like(timeline, np.nan, dtype=float) for u in users}
        for p in params
    }

    # Only create merged_series / user_at_time for shared params
    shared_params_set = set(p.lower() for p in cfg.shared_params)
    merged_series: Dict[str, np.ndarray] = {}
    user_at_time: Dict[str, np.ndarray] = {}

    for p in params:
        # All changes for this parameter (all users)
        all_changes = df[df["param"] == p].sort_values("time_sec")

        # Per-user trajectories
        for u in users:
            d = df[(df["user"] == u) & (df["param"] == p)].sort_values("time_sec")

            val = np.nan
            idx = 0
            for i, t in enumerate(timeline):
                while idx < len(d) and t >= d.iloc[idx]["time_sec"]:
                    val = d.iloc[idx]["value"]
                    idx += 1
                series[p][u][i] = val

        # Shared parameter logic (frequency, etc.)
        if p in shared_params_set:
            merged_data = np.full_like(timeline, np.nan, dtype=float)
            owner = np.full_like(timeline, -1, dtype=int)

            val = np.nan
            idx = 0
            last_user = -1

            for i, t in enumerate(timeline):
                while idx < len(all_changes) and t >= all_changes.iloc[idx]["time_sec"]:
                    val = all_changes.iloc[idx]["value"]
                    last_user = int(all_changes.iloc[idx]["user"])
                    idx += 1

                merged_data[i] = val
                owner[i] = last_user if not np.isnan(val) else -1

            merged_series[p] = merged_data
            user_at_time[p] = owner

    return TimeSeriesResult(
        timeline=timeline,
        series=series,
        merged_series=merged_series,
        user_at_time=user_at_time,
        users=users,
        params=params,
        t_min=t_min,
        t_max=t_max,
        time_resolution=cfg.time_resolution,
    )


# Public API
__all__ = ["TimeSeriesConfig", "TimeSeriesResult", "build_timeseries"]
