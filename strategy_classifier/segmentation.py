from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

@dataclass
class SpeedSegment:
    start: float
    end: float
    trend: str
    segment_id: int

    @property
    def duration(self) -> float:
        return self.end - self.start

def _load_speed_csv(path: Path | str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={"starttime": "start", "endtime": "end"})
    if not {"trend", "start", "end"}.issubset(df.columns):
        raise ValueError("Speed CSV must contain columns: trend,starttime,endtime")
    df = df.sort_values("start").reset_index(drop=True)
    return df

def _split_long_dull(df: pd.DataFrame, max_dur: float = 20.0, window: float = 10.0) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        trend = str(row["trend"])
        start = float(row["start"])
        end = float(row["end"])
        dur = end - start
        if trend.lower() != "dull" or dur <= max_dur:
            rows.append({"trend": trend, "start": start, "end": end})
            continue
        cur = start
        while cur < end:
            sub_end = min(cur + window, end)
            rows.append({"trend": trend, "start": cur, "end": sub_end})
            cur = sub_end
    return pd.DataFrame(rows).sort_values("start").reset_index(drop=True)

def build_segments_from_speed_csv(
    csv_path: Path | str,
    dull_max_duration: float = 20.0,
    dull_window: float = 10.0,
) -> List[SpeedSegment]:
    df = _load_speed_csv(csv_path)
    df = _split_long_dull(df, max_dur=dull_max_duration, window=dull_window)
    segs: List[SpeedSegment] = []
    for i, row in df.iterrows():
        segs.append(
            SpeedSegment(
                start=float(row["start"]),
                end=float(row["end"]),
                trend=str(row["trend"]),
                segment_id=i,
            )
        )
    return segs

def assign_events_to_segments(
    events: List[Dict[str, Any]],
    segments: List[SpeedSegment],
) -> Dict[int, List[Dict[str, Any]]]:
    seg_events: Dict[int, List[Dict[str, Any]]] = {s.segment_id: [] for s in segments}
    segs_sorted = sorted(segments, key=lambda s: s.start)
    j = 0
    n = len(segs_sorted)
    for ev in sorted(events, key=lambda e: e["time"]):
        t = ev["time"]
        while j < n and segs_sorted[j].end < t:
            j += 1
        if j >= n:
            break
        seg = segs_sorted[j]
        if seg.start <= t <= seg.end:
            seg_events[seg.segment_id].append(ev)
    return seg_events
