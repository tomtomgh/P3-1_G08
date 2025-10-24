from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd
import math
from collections import Counter
from itertools import pairwise

@dataclass
class TrialErrorConfig:
    """
    Configuration for the Trial and Error analysis.
    - time_window_simul: seconds for counting 'simultaneous exploration'
    - backtrack_window: N previous steps per parameter to consider for backtracking
    """
    time_window_simul: float = 1.0
    backtrack_window: int = 5

class TrialErrorAnalysis:
    """
    Trial and Error analysis

    It computes per user:
    A) Switching / combination behavior:
    - alternation_index -> how often two consecutive changes target different parameters
    - avg_switch_interval -> average time gap between changes when switching parameters
    - param_entropy_bits -> variety of touched parameters
    - unique_params -> count of unique parameters a user changed
    - simul_explore_events, simul_explore_rate -> moments where two or more different params are changed
    within a time window

    B) Style of parameter step:
    - avg_abs_step -> magnitude of per-change adjustments (for each param, absolute differences of values are averaged,
    then averaged across parameters)
    - step_sd -> standard deviation of steps per parameter
    - max_abs_jump -> largest absolute jump between two consecutive steps

    C) Backtracking:
    - backtrack_rate_lastN -> how often a new value is a reused value of the same parameter
    - avg_revisit_gap_steps -> on average, how many changed occur before re-using a value
    """

    def __init__(self, changes_df: pd.DataFrame, config: Optional[TrialErrorConfig] = None) -> None:
        self.cfg = config or TrialErrorConfig()
        req = {"user", "time_sec", "param", "value"}
        missing = req - set(changes_df.columns)
        if missing:
            raise ValueError(f"Missing parameters: {missing}")

        df = changes_df.copy()
        df["param"] = df["param"].astype(str).str.lower().str.strip()
        df["time_sec"] = pd.to_numeric(df["time_sec"], errors="coerce")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["time_sec", "value"])
        self.df = df.sort_values(["user", "time_sec"]).reset_index(drop=True)

    # ----- HELPERS -------
    @staticmethod
    def _entropy(labels: List[str]) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return float(-sum(p * math.log(p + 1e-12, 2) for p in probs))

    @staticmethod
    def _alternation_index(params_sequence: List[str]) -> float:
        if len(params_sequence) < 2:
            return float("nan")
        switches = sum(1 for a, b in pairwise(params_sequence) if a != b)
        return switches / (len(params_sequence) - 1)

    @staticmethod
    def _avg_switch_interval(times: np.ndarray, params: np.ndarray) -> float:
        if len(times) < 2:
            return float("nan")
        intervals = [float(t2-t1) for (t1, p1), (t2, p2) in pairwise(zip(times, params)) if p1 != p2]
        return float(np.mean(intervals)) if intervals else float("nan")

    @staticmethod
    def _step_stats(values: np.ndarray) -> Tuple[float, float, float]:
        # Returns mean(|Δ|), std(Δ signed), max(|Δ|)
        if len(values) < 2:
            return (float("nan"), float("nan"), float("nan"))
        diffs = np.diff(values)
        return (float(np.mean(np.abs(diffs))), float(np.std(diffs)), float(np.max(np.abs(diffs))))

    def _backtrack_rate(self, values: List[float]) -> float:
        if len(values) < 2:
            return float("nan")
        w = int(self.cfg.backtrack_window)
        hits, total = 0, 0
        for i in range(1, len(values)):
            history = values[max(0, i-w): i]
            total += 1
            if values[i] in history and values[i] != values[i-1]:
                hits += 1
        return hits / total if total else float("nan")

    @staticmethod
    def _revisit_gap(values: List[float]) -> float:
        last_seen = {}
        gaps = []
        for i, v in enumerate(values):
            if v in last_seen:
                gaps.append(i - last_seen[v])
            last_seen[v] = i
        return float(np.mean(gaps)) if gaps else float("nan")

    @staticmethod
    def _fano_factor(inter_event: np.ndarray) -> float:
        if(len(inter_event) < 2):
            return float("nan")
        mu = float(np.mean(inter_event))
        if mu <= 0:
            return float("nan")
        var = float(np.var(inter_event, ddof=1)) if len (inter_event) > 1 else 0.0
        return var / mu

    def _simultaneous_exploration(self, user_df: pd.DataFrame) -> Tuple[int, float]:
        tw = float(self.cfg.time_window_simul)
        if user_df.empty:
            return 0, 0.0
        t = user_df["time_sec"].to_numpy()
        p = user_df["param"].to_numpy()
        n = 0
        for i in range(len(t)):
            touched = {p[i]}
            j = i + 1
            while j < len(t) and (t[j] - t[i]) <= tw:
                touched.add(p[j])
                j += 1
            if len(touched) >= 2:
                n += 1
        return n, n / len(t)

    # ------- public --------

    def compute_per_user_metrics(self) -> pd.DataFrame:
        rows = []
        for user, g in self.df.groupby("user"):
            times = g["time_sec"].to_numpy()
            params = g["param"].to_numpy()

            total_changes = len(g)
            duration = float(times[-1] - times[0]) if total_changes > 1 else float("nan")
            changes_per_min = (total_changes / duration * 60.0) if duration and duration > 0 else float("nan")

            inter_event = np.diff(times) if len(times) > 1 else np.array([])
            fano = self._fano_factor(inter_event)

            # A)
            param_entropy = self._entropy(list(params))
            alt_index = self._alternation_index(list(params))
            avg_switch_dt = self._avg_switch_interval(times, params)
            unique_params = int(len(set(params)))
            simul_n, simul_rate = self._simultaneous_exploration(g)

            # B)
            mean_steps, sd_steps, max_jumps, backtracks, revisits = [], [], [], [], []
            for p, gp in g.groupby("param"):
                mu, sd, mx = self._step_stats(gp["value"].to_numpy())
                mean_steps.append(mu)
                sd_steps.append(sd)
                max_jumps.append(mx)
                backtracks.append(self._backtrack_rate(list(gp["value"])))
                revisits.append(self._revisit_gap(list(gp["value"])))

            mean_abs_step = float(np.nanmean(mean_steps)) if mean_steps else float("nan")
            step_sd = float(np.nanmean(sd_steps)) if sd_steps else float("nan")
            max_abs_jump = float(np.nanmax(max_jumps)) if max_jumps else float("nan")
            backtrack_rate = float(np.nanmean(backtracks)) if backtracks else float("nan")
            avg_revisit_gap = float(np.nanmean(revisits)) if revisits else float("nan")

            rows.append({
                "user": user,
                "total_changes": total_changes,
                "duration_s": duration,
                "changes_per_min": changes_per_min,
                "fano_factor_interevent": fano,
                "param_entropy_bits": param_entropy,
                "alternation_index": alt_index,
                "avg_switch_interval_s": avg_switch_dt,
                "unique_params": unique_params,
                "simul_explore_events": simul_n,
                "simul_explore_rate": simul_rate,
                "avg_abs_step": mean_abs_step,
                "step_sd": step_sd,
                "max_abs_jump": max_abs_jump,
                "backtrack_rate_lastN": backtrack_rate,
                "avg_revisit_gap_steps": avg_revisit_gap,
            })
        return pd.DataFrame(rows).sort_values("user").reset_index(drop=True)


    def compute_transition_tables(self, stacked: bool = True):
        tables = {}
        for user, g in self.df.groupby("user"):
            params = g["param"].to_list()
            unique = sorted(set(params))
            M = np.zeros((len(unique), len(unique)), dtype=int)
            if len(params) >= 2:
                idx = {p: i  for i, p in enumerate(unique)}
                for a, b in pairwise(params):
                    M[idx[a], idx[b]] += 1
            tables[user] = pd.DataFrame(M, index=unique, columns=unique)

        if not stacked:
            return tables

        frames = []
        for user, tdf in tables.items():
            tmp = tdf.copy()
            tmp["from_param"] = tmp.index
            longf = tmp.melt(id_vars=["from_param"], var_name="to_param", value_name="count")
            longf.insert(0, "user", user)
            frames.append(longf)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["user", "from_param", "to_param", "count"])


    def composite_trial_error_score(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a trial and error score from:
        - Alternation_index (higher better)
        - param_entropy_bits (normalized by global max bits; higher better)
        - simul_explore_rate (higher better)
        - step_sd (higher better) -> mix of small and big moves
        - max_abs_jump (higher better) -> willingness for bold jumps
        - 1 / avg_switch_interval_s (lower interval is better)
        """

        df = metrics_df.copy()

        def minmax(s: pd.Series) -> pd.Series:
            if s.notna().sum() <= 1:
                return s * 0 + 0.0
            low, high = s.min(skipna=True), s.max(skipna=True)
            if pd.isna(low) or pd.isna(high) or high == low:
                return s * 0 + 0.0
            return (s - low) / (high - low)

        # Normalize entropy by max
        all_params = self.df["param"].unique()
        max_bits = math.log2(max(len(all_params), 1))
        df["param_entropy_norm"] = df["param_entropy_bits"] / max_bits if max_bits > 0 else 0.0

        inv_switch = 1.0 / df["avg_switch_interval_s"].replace({0.0:np.nan})
        inv_switch = inv_switch.replace([np.inf, -np.inf], np.nan)

        components = {
            "alternation": minmax(df["alternation_index"]),
            "entropy": df["param_entropy_norm"].clip(0, 1),
            "simul": minmax(df["simul_explore_rate"]),
            "step_sd": minmax(df["step_sd"]),
            "max_jump": minmax(df["max_abs_jump"]),
            "inv_switch": minmax(inv_switch),
        }

        weights = {
            "alternation": 0.24,
            "entropy": 0.24,
            "simul": 0.18,
            "step_sd": 0.14,
            "max_jump": 0.10,
            "inv_switch": 0.10,
        }

        score = sum(weights[k] * components[k].fillna(0.0) for k in weights)
        out = df.copy()
        out["trial_error_score_0to1"] = score.clip(0,1)
        return out

