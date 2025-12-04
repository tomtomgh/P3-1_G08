from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _has_param_activity(row: pd.Series, min_actions: int = 1, min_change: float = 0.0) -> bool:
    """Return True if the segment shows real parameter work."""
    actions = float(row.get("num_actions", 0))
    params = float(row.get("num_params_used", 0))
    change = float(row.get("config_dist_prev", 0.0))
    if actions < min_actions or params <= 0:
        return False
    return change >= min_change

def _goal(row: pd.Series)->float:
    if not _has_param_activity(row, min_actions=2, min_change=0.05):
        return 0.0
    ds=row.get("delta_speed_prev",0.0); ms=row.get("mean_step_size",0.0); pos=row.get("segment_index_norm",0.0)
    n=row.get("num_actions",0); cfg=row.get("config_dist_prev",0.0); params=row.get("num_params_used",0)
    p=0.0
    if ds>0.2: p+=0.35
    if 0.3<=ms<=1.5: p+=0.2
    if pos>0.5: p+=0.15
    if n>=3: p+=0.1
    if cfg>0.12: p+=0.1
    if params>=2: p+=0.1
    return min(1.0,p)

def _iterative(row: pd.Series)->float:
    if not _has_param_activity(row, min_actions=2):
        return 0.0
    ms=row.get("mean_step_size",0.0); vs=row.get("var_step_size",0.0); undo=row.get("undo_ratio",0.0)
    sp=row.get("single_param_cluster_ratio",0.0); cfg=row.get("config_dist_prev",0.0); params=row.get("num_params_used",0)
    p=0.0
    if ms<0.4: p+=0.3
    if vs<0.1: p+=0.2
    if undo>0.1: p+=0.2
    if sp>0.4: p+=0.15
    if cfg<0.15: p+=0.1
    if params==1: p+=0.05
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
