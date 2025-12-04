from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import pandas as pd

@dataclass
class StrategyRule:
    name: str
    prob_fn: Callable[[pd.Series], float]
    def predict_prob(self, row: pd.Series) -> float:
        p = float(self.prob_fn(row))
        if p < 0: return 0.0
        if p > 1: return 1.0
        return p
    def predict(self, row: pd.Series, threshold: float=0.5) -> int:
        return int(self.predict_prob(row) >= threshold)
