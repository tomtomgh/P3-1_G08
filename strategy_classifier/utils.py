from __future__ import annotations
from typing import List
import pandas as pd

def print_strategy_summary(df: pd.DataFrame, strategy_names: List[str]) -> None:
    if df.empty:
        print("[summary] Empty DataFrame")
        return
    for name in strategy_names:
        col=f"{name}_pred"
        if col not in df.columns: 
            continue
        rate = df[col].mean()
        print(f"{name:28s}: {rate:.2f} of segments flagged")
