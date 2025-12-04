from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _rep(row: pd.Series)->float:
    rep=row["repeat_run_len"]; H=row["action_entropy"]; nt=row["num_action_types"]
    p=0.0
    if rep>=2: p+=0.5
    if H<0.4: p+=0.3
    if nt<=2: p+=0.2
    return min(1.0,p)

def _rep_improve(row: pd.Series)->float:
    base=_rep(row); ds=row["delta_speed_prev"]
    if ds>0.2: base+=0.3
    return min(1.0,base)

def _rep_noimprove(row: pd.Series)->float:
    base=_rep(row); ds=abs(row["delta_speed_prev"])
    if ds<0.1: base+=0.3
    return min(1.0,base)

def get_repetition_rules():
    return {
        "repetition_practice": StrategyRule("repetition_practice", _rep),
        # "trial_repetition_improvement": StrategyRule("trial_repetition_improvement", _rep_improve),
        # "trial_repetition_no_improvement": StrategyRule("trial_repetition_no_improvement", _rep_noimprove),
    }
