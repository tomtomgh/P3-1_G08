from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _help(row: pd.Series)->float:
    mp=row["mean_pause"]; dull=row["is_dull_speed"]; n=row["num_actions"]
    p=0.0
    if n>0 and mp>3.0: p+=0.5
    if dull and mp>5.0: p+=0.3
    return min(1.0,p)

def _inactivity(row: pd.Series)->float:
    n=row["num_actions"]; dur=row["segment_duration"]; dull=row["is_dull_speed"]
    p=0.0
    if n==0 and dur>5.0: p+=0.6
    if dull and dur>10.0: p+=0.4
    return min(1.0,p)

def _playful(row: pd.Series)->float:
    H=row["action_entropy"]; ms=row["mean_step_size"]; ds=row["delta_speed_prev"]
    p=0.0
    if H>0.7: p+=0.4
    if ms>0.5: p+=0.3
    if ds<=0: p+=0.3
    return min(1.0,p)

def get_behavioural_rules():
    return {
        # "help_seeking_pause": StrategyRule("help_seeking_pause", _help),
        "inactivity_wait": StrategyRule("inactivity_wait", _inactivity),
        # "playful_inefficient": StrategyRule("playful_inefficient", _playful),
    }
