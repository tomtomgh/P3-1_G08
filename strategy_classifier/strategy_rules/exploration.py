from __future__ import annotations
import pandas as pd
from .base import StrategyRule

def _structured(row: pd.Series) -> float:
    H = row["action_entropy"]; n=row["num_action_types"]; dom=row["dominant_action_share"]; nparams=row["num_params_used"]
    p=0.0
    if H>0.7: p+=0.4
    if n>=3: p+=0.2
    if dom<0.6: p+=0.2
    if nparams>=2: p+=0.2
    return min(1.0,p)

def _random(row: pd.Series) -> float:
    H=row["action_entropy"]; zig=row["zigzag_ratio"]; vs=row["var_step_size"]; ds=abs(row["delta_speed_prev"])
    p=0.0
    if H>0.6: p+=0.3
    if zig>0.5: p+=0.4
    if vs>0.05: p+=0.2
    if ds<0.2: p+=0.1
    return min(1.0,p)

def _curiosity(row: pd.Series)->float:
    H=row["action_entropy"]; d1=row["config_dist_prev"]; d2=row["config_dist_prev2"]; pos=row["segment_index_norm"]
    p=0.0
    if H>0.6: p+=0.3
    if d1>0.1 or d2>0.1: p+=0.4
    if pos<0.5: p+=0.3
    return min(1.0,p)

def _sweep(row: pd.Series)->float:
    n=row["num_actions"]; zig=row["zigzag_ratio"]; H=row["action_entropy"]; sp=row["single_param_cluster_ratio"]
    p=0.0
    if n>=4: p+=0.2
    if zig<0.3: p+=0.3
    if 0.3<H<0.9: p+=0.3
    if sp>0.4: p+=0.2
    return min(1.0,p)

def get_exploration_rules():
    return {
        "structured_exploration": StrategyRule("structured_exploration", _structured),
        "random_trial_error": StrategyRule("random_trial_error", _random),
        # "curiosity_broad_exploration": StrategyRule("curiosity_broad_exploration", _curiosity),
        "systematic_parameter_sweep": StrategyRule("systematic_parameter_sweep", _sweep),
    }
