from __future__ import annotations
from dataclasses import dataclass
from typing import List, Iterable
import numpy as np
import pandas as pd

from .Utils import StrategyWindow, sliding_time_windows, merge_windows

@dataclass
class CarefulnessConfig:
    window_size: float = 5.0
    min_steps: int = 3
    small_step: float = 0.10
    min_careful_ratio: float = 70.0
    merge_gap: float = 1.0
    params: Iterable[str] | None = None

def detect_carefulness(df: pd.DataFrame, cfg: CarefulnessConfig) -> List[StrategyWindow]:
    users = sorted(df["user"].unique())
    params_all = sorted(df["param"].unique())
    params = list(cfg.params) if cfg.params is not None else params_all
    r_windows: List[StrategyWindow] = []
    for user in users:
        for param in params:
            param_df = df[(df["user"] == user) & (df["param"] == param)].sort_values("time_sec")
            if len(param_df) < 2:
                continue
            values = param_df["value"].to_numpy()
            times = param_df["time_sec"].to_numpy()
            steps = np.abs(np.diff(values))
            step_times = times[1:]
            if len(steps) == 0:
                continue
            value_range = float(np.ptp(values))
            step_threshold = cfg.small_step * (value_range if value_range > 0 else 1.0)
            for i_start, j_end in sliding_time_windows(step_times, cfg.window_size, start_idx=0):
                window_steps = steps[i_start:j_end]
                num_steps = len(window_steps)
                if num_steps < cfg.min_steps:
                    continue
                small_steps = int(np.sum(window_steps <= step_threshold))
                careful_ratio = (small_steps / num_steps) * 100.0
                behavior_type = "Careful" if careful_ratio >= cfg.min_careful_ratio else "Reckless"
                w = StrategyWindow(
                    start_time=float(step_times[i_start]),
                    end_time=float(step_times[j_end - 1]),
                    users=[int(user)],
                    param=param,
                    tag=f"carefulness:{behavior_type.lower()}",
                    meta={
                        "num_steps": num_steps,
                        "small_steps": small_steps,
                        "careful_ratio": careful_ratio,
                        "behavior_type": behavior_type,
                    },
                )
                r_windows.append(w)
    if not r_windows:
        return []
    def group_key(w: StrategyWindow):
        behavior = w.meta.get("behavior_type")
        return tuple(w.users), w.param, behavior
    threshold = cfg.min_careful_ratio
    def combine(target: StrategyWindow, other: StrategyWindow):
        target_steps = target.meta.get("num_steps", 0)
        target_small = target.meta.get("small_steps", 0)
        other_steps = other.meta.get("num_steps", 0)
        other_small = other.meta.get("small_steps", 0)
        total_steps = target_steps + other_steps
        total_small = target_small + other_small
        target.meta["num_steps"] = total_steps
        target.meta["small_steps"] = total_small
        ratio = 100.0 * total_small / total_steps if total_steps > 0 else 0.0
        behavior = "Careful" if ratio >= threshold else "Reckless"
        target.meta["careful_ratio"] = ratio
        target.meta["behavior_type"] = behavior
        target.tag = f"carefulness:{behavior.lower()}"
    merged = merge_windows(r_windows, cfg.merge_gap, group_key, combine)
    return merged

# Public API
__all__ = ["CarefulnessConfig", "detect_carefulness"]

