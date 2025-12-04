from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _goal(row: pd.Series)->float:
    ds=row["delta_speed_prev"]; ms=row["mean_step_size"]; pos=row["segment_index_norm"]; n=row["num_actions"]
    p=0.0
    if ds>0.2: p+=0.4
    if ms<0.5: p+=0.3
    if pos>0.5: p+=0.2
    if n>=2: p+=0.1
    return min(1.0,p)

def _iterative(row: pd.Series)->float:
    ms=row["mean_step_size"]; vs=row["var_step_size"]; undo=row["undo_ratio"]; sp=row["single_param_cluster_ratio"]
    p=0.0
    if ms<0.4: p+=0.4
    if vs<0.1: p+=0.2
    if undo>0.1: p+=0.2
    if sp>0.3: p+=0.2
    return min(1.0,p)

def _incremental(row: pd.Series)->float:
    ms=row["mean_step_size"]; ds=row["delta_speed_prev"]; pos=row["segment_index_norm"]
    p=0.0
    if 0.1<ms<1.0: p+=0.4
    if ds>=0: p+=0.3
    if 0.3<pos<0.9: p+=0.3
    return min(1.0,p)

def get_tuning_rules():
    return {
        "goal_directed_tuning": StrategyRule("goal_directed_tuning", _goal),
        "iterative_finetuning": StrategyRule("iterative_finetuning", _iterative),
        "incremental_adjustment": StrategyRule("incremental_adjustment", _incremental),
    }
