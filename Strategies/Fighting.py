from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pandas as pd

from .Utils import StrategyWindow, sliding_time_windows, merge_windows

@dataclass
class FightingConfig:
    time_window: float = 5.0
    min_changes: int = 4
    min_users: int = 2
    merge_gap: float = 2.0
    param_name: str = "frequency"


def detect_fighting(df: pd.DataFrame, cfg: FightingConfig) -> List[StrategyWindow]:
    param_df = df[df["param"] == cfg.param_name].sort_values("time_sec").copy()
    if len(param_df) < cfg.min_changes:
        return []

    times = param_df["time_sec"].to_numpy()

    r_windows: List[StrategyWindow] = []

    for i, j in sliding_time_windows(times, cfg.time_window):
        window_changes = param_df.iloc[i:j]
        num_changes = len(window_changes)
        if num_changes < cfg.min_changes:
            continue

        users = window_changes["user"].unique().tolist()
        if len(users) < cfg.min_users:
            continue

        w = StrategyWindow(
            start_time=float(window_changes["time_sec"].iloc[0]),
            end_time=float(window_changes["time_sec"].iloc[-1]),
            users=[int(u) for u in users],
            param=cfg.param_name,
            tag="fighting",
            meta={"num_changes": num_changes},
        )
        r_windows.append(w)

    if not r_windows:
        return []

    def group_key(w: StrategyWindow):
        return tuple(sorted(w.users)), w.param, w.tag

    def combine(target: StrategyWindow, other: StrategyWindow):
        target.end_time = max(target.end_time, other.end_time)
        target.meta["num_changes"] = target.meta.get("num_changes", 0) + other.meta.get("num_changes", 0)

    merged = merge_windows(r_windows, cfg.merge_gap, group_key, combine)
    return merged

# Public API
__all__ = ["FightingConfig", "detect_fighting"]

