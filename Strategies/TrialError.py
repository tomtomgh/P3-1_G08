from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import math
import numpy as np
import pandas as pd

from .Utils import StrategyWindow, sliding_time_windows, merge_windows


# ===================== CONFIG =====================

@dataclass
class TrialErrorConfig:
    """
    Configuration for Trial & Error strategy detection.
    """
    window_size: float = 20.0
    min_changes: int = 4
    time_window_simul: float = 1.0
    high_score_threshold: float = 30.0
    merge_gap: float = 2.0


# ===================== HELPERS =====================

def _entropy(params: np.ndarray) -> float:
    if params.size == 0:
        return 0.0
    vals, counts = np.unique(params, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-(p * np.log2(p + 1e-12)).sum())


def _alternation_index(params: np.ndarray) -> float:
    n = len(params)
    if n < 2:
        return float("nan")
    switches = np.sum(params[1:] != params[:-1])
    return float(switches) / float(n - 1)


def _simultaneous_exploration(
    times: np.ndarray,
    params: np.ndarray,
    time_window: float,
) -> tuple[int, float]:
    n = len(times)
    if n == 0:
        return 0, 0.0

    count = 0
    for i in range(n):
        touched = {params[i]}
        j = i + 1
        while j < n and (times[j] - times[i]) <= time_window:
            touched.add(params[j])
            j += 1
        if len(touched) >= 2:
            count += 1

    rate = float(count) / float(n)
    return int(count), rate


def _compute_te_metrics(
    params: np.ndarray,
    values: np.ndarray,
    times: np.ndarray,
    cfg: TrialErrorConfig,
    max_bits: float,
) -> Dict[str, Any]:
    alt_index = _alternation_index(params)
    H = _entropy(params)
    entropy_norm = float(H) / float(max_bits) if max_bits > 0 else 0.0

    simul_events, simul_rate = _simultaneous_exploration(
        times=times,
        params=params,
        time_window=cfg.time_window_simul,
    )

    alt_n = 0.0 if not np.isfinite(alt_index) else float(np.clip(alt_index, 0.0, 1.0))
    ent_n = float(np.clip(entropy_norm, 0.0, 1.0))
    simul_n = float(np.clip(simul_rate, 0.0, 1.0))

    score_0to1 = 0.4 * alt_n + 0.4 * ent_n + 0.2 * simul_n
    score_0to1 = float(np.clip(score_0to1, 0.0, 1.0))
    score_0to100 = 100.0 * score_0to1

    return {
        "alt_index": float(alt_index),
        "param_entropy_bits": float(H),
        "entropy_norm": float(entropy_norm),
        "simul_events": int(simul_events),
        "simul_rate": float(simul_rate),
        "te_score_0to1": score_0to1,
        "te_score_0to100": score_0to100,
    }


# ===================== MAIN DETECTOR =====================

def detect_trial_error(
    df: pd.DataFrame,
    cfg: TrialErrorConfig,
) -> List[StrategyWindow]:
    required = {"user", "time_sec", "param", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"detect_trial_error: missing columns {missing}")

    df = df.copy()
    df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
    df = df.dropna(subset=["time_sec"]).sort_values(["user", "time_sec"])

    all_params = (
        df["param"].astype(str).str.lower().str.strip().unique()
    )
    max_bits = math.log2(len(all_params)) if len(all_params) > 0 else 1.0

    windows: List[StrategyWindow] = []

    for user, user_df in df.groupby("user"):
        user_df = user_df.sort_values("time_sec")
        if len(user_df) < cfg.min_changes:
            continue

        times = user_df["time_sec"].to_numpy()
        if len(times) == 0:
            continue

        for i_start, j_end in sliding_time_windows(times, cfg.window_size):
            window_df = user_df.iloc[i_start:j_end]
            num_changes = len(window_df)
            if num_changes < cfg.min_changes:
                continue

            params_w = (
                window_df["param"]
                .astype(str)
                .str.lower()
                .str.strip()
                .to_numpy()
            )
            values_w = window_df["value"].to_numpy(dtype=float)
            times_w = window_df["time_sec"].to_numpy(dtype=float)

            metrics = _compute_te_metrics(
                params=params_w,
                values=values_w,
                times=times_w,
                cfg=cfg,
                max_bits=max_bits,
            )
            score_100 = metrics["te_score_0to100"]

            # Only keep HIGH windows; drop low/medium entirely
            if score_100 < cfg.high_score_threshold:
                continue
            meta: Dict[str, Any] = dict(metrics)
            meta["num_changes"] = int(num_changes)
            w = StrategyWindow(
                start_time=float(times_w[0]),
                end_time=float(times_w[-1]),
                users=[int(user)],
                param=None,
                tag="trial_error:high",
                meta=meta,
            )
            windows.append(w)

    if not windows:
        return []

    def group_key(w: StrategyWindow):
        return (tuple(w.users), w.tag)

    def combine(target: StrategyWindow, other: StrategyWindow):
        target.end_time = max(target.end_time, other.end_time)
        t_n = target.meta.get("num_changes", 0)
        o_n = other.meta.get("num_changes", 0)
        total = t_n + o_n
        target.meta["num_changes"] = total

        keys_to_avg = [
            "te_score_0to1",
            "te_score_0to100",
            "alt_index",
            "param_entropy_bits",
            "entropy_norm",
            "simul_rate",
        ]
        for key in keys_to_avg:
            t_v = float(target.meta.get(key, 0.0))
            o_v = float(other.meta.get(key, 0.0))
            target.meta[key] = (t_v * t_n + o_v * o_n) / float(total) if total > 0 else 0.0

        t_ev = int(target.meta.get("simul_events", 0))
        o_ev = int(other.meta.get("simul_events", 0))
        target.meta["simul_events"] = t_ev + o_ev

    merged = merge_windows(
        windows,
        max_gap=cfg.merge_gap,
        group_key=group_key,
        combine=combine,
    )
    return merged


def aggregate_trial_error_scores(
    windows: List[StrategyWindow],
) -> Dict[int, float]:
    score_sum: Dict[int, float] = {}
    weight_sum: Dict[int, float] = {}

    for w in windows:
        if not w.users:
            continue

        for u in w.users:
            user_id = int(u)

            score = w.meta.get("te_score_0to100")
            if score is None:
                continue

            try:
                s = float(score)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(s):
                continue

            weight = w.meta.get("num_changes", 1.0)
            try:
                wgt = float(weight)
            except (TypeError, ValueError):
                wgt = 1.0
            if wgt <= 0:
                wgt = 1.0

            score_sum[user_id] = score_sum.get(user_id, 0.0) + s * wgt
            weight_sum[user_id] = weight_sum.get(user_id, 0.0) + wgt

    return {
        u: (score_sum[u] / weight_sum[u])
        for u in score_sum.keys()
        if weight_sum.get(u, 0.0) > 0.0
    }


# Public API
__all__ = ["TrialErrorConfig", "detect_trial_error", "aggregate_trial_error_scores"]


