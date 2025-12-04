from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from .strategy_rules import get_all_strategy_rules
from .constants import ALL_STRATEGIES
from .features import DEFAULT_FEATURE_COLS

def apply_all_strategies(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    rules = get_all_strategy_rules()
    out = df.copy()
    for name, rule in rules.items():
        probs=[]; preds=[]
        for _, row in out.iterrows():
            p = rule.predict_prob(row)
            probs.append(p)
            preds.append(int(p>=0.5))
        out[f"{name}_prob"]=probs
        out[f"{name}_pred"]=preds
    return out

def train_ml_trees_from_rules(
    df: pd.DataFrame,
    strategy_names: List[str] | None = None,
    feature_cols: List[str] | None = None,
    max_depth: int = 4,
) -> Dict[str, DecisionTreeClassifier]:
    """Train DecisionTreeClassifier to imitate the rule-based labels for each strategy."""
    if df.empty:
        return {}
    if strategy_names is None:
        strategy_names = ALL_STRATEGIES
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    rules = get_all_strategy_rules()
    trees: Dict[str,DecisionTreeClassifier] = {}
    for name in strategy_names:
        if name not in rules:
            continue
        rule = rules[name]
        y = df.apply(lambda r: rule.predict(r), axis=1)
        if y.nunique()<2:
            continue
        X = df[feature_cols]
        clf = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=42)
        clf.fit(X, y)
        trees[name] = clf
    return trees

def apply_ml_trees(
    df: pd.DataFrame,
    trees: Dict[str, DecisionTreeClassifier],
    feature_cols: List[str] | None = None,
) -> pd.DataFrame:
    if df.empty or not trees:
        return df
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS
    out = df.copy()
    X = out[feature_cols]
    for name, clf in trees.items():
        probs = clf.predict_proba(X)
        pos_idx = int(np.where(clf.classes_==1)[0][0])
        p = probs[:,pos_idx]
        pred = (p>=0.5).astype(int)
        out[f"{name}_ml_prob"] = p
        out[f"{name}_ml_pred"] = pred
    return out
