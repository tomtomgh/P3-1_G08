from __future__ import annotations
from typing import Dict, Any, List
import pandas as pd

def df_to_student_segment_json(
    df: pd.DataFrame,
    strategy_names: List[str] | None = None,
) -> Dict[str,Any]:
    res: Dict[str,Any]={}
    if df.empty:
        return res
    if strategy_names is None:
        strategy_names = sorted({c[:-5] for c in df.columns if c.endswith("_pred")})
    for (sid,uid), grp in df.groupby(["session_id","user_id"]):
        sk=str(sid); uk=str(uid)
        res.setdefault(sk,{})
        res[sk].setdefault(uk,[])
        g=grp.sort_values("seg_start")
        for _, row in g.iterrows():
            seg={
                "segment_id": int(row["segment_id"]),
                "start": float(row["seg_start"]),
                "end": float(row["seg_end"]),
                "trend": str(row["seg_trend"]),
                "strategies": {}
            }
            for name in strategy_names:
                pcol=f"{name}_prob"; dcol=f"{name}_pred"
                if pcol in row and dcol in row:
                    seg["strategies"][name]={"prob": float(row[pcol]), "pred": int(row[dcol])}
            res[sk][uk].append(seg)
    return res
