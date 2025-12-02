from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _backtrack(row: pd.Series)->float:
    undo=row["undo_ratio"]; d1=row["config_dist_prev"]; d2=row["config_dist_prev2"]; ds2=row["delta_speed_prev2"]; ds=row["delta_speed_prev"]
    p=0.0
    if undo>0.2: p+=0.4
    if d1>0.1 and d2<0.1: p+=0.3
    if ds2<-0.1 and ds>0.1: p+=0.3
    return min(1.0,p)

def _undo(row: pd.Series)->float:
    u=row["undo_ratio"]; p=0.0
    if u>0.2: p+=0.6
    if u>0.4: p+=0.4
    return min(1.0,p)

def _loop(row: pd.Series)->float:
    rep=row["repeat_run_len"]; ds=abs(row["delta_speed_prev"]); ms=row["mean_speed"]
    p=0.0
    if rep>=3: p+=0.5
    if ds<0.1: p+=0.2
    if ms<=0: p+=0.3
    return min(1.0,p)

def get_error_backtracking_rules():
    return {
        "backtracking_recovery": StrategyRule("backtracking_recovery", _backtrack),
        # "undo_correction": StrategyRule("undo_correction", _undo),
        # "loop_stuck_state": StrategyRule("loop_stuck_state", _loop),
    }
