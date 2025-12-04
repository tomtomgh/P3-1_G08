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

def _normalize_trends_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Accept different column name variants and produce columns:
      - trend, starttime, endtime
    """
    cols_lower = {c.lower(): c for c in df.columns}
    def find(*names):
        for n in names:
            k = cols_lower.get(n.lower())
            if k:
                return k
        return None

    trend_col = find("trend", "state", "label")
    start_col = find("starttime", "start_time", "start", "begin")
    end_col = find("endtime", "end_time", "end", "stop")

    if not (trend_col and start_col and end_col):
        raise ValueError("Speed CSV must contain columns: trend,starttime,endtime (or compatible variants)")

    df2 = df.rename(columns={trend_col: "trend", start_col: "starttime", end_col: "endtime"})[["trend","starttime","endtime"]]
    # coerce numeric times
    df2["starttime"] = pd.to_numeric(df2["starttime"], errors="coerce")
    df2["endtime"] = pd.to_numeric(df2["endtime"], errors="coerce")
    return df2

def _load_speed_csv(csv_path: Path) -> pd.DataFrame:
    """
    Robust loader with debug prints.
    """
    print(f"[DEBUG] Loading trends CSV: {csv_path} (exists={csv_path.exists()} size={(csv_path.stat().st_size if csv_path.exists() else 'n/a')})")
    df = pd.read_csv(csv_path)
    print(f"[DEBUG] trends.csv head:\n{df.head().to_string(index=False)}")
    df_norm = _normalize_trends_df(df)
    print(f"[DEBUG] Normalized trends rows: {len(df_norm)}; sample:\n{df_norm.head().to_string(index=False)}")
    return df_norm

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
    # call _load_speed_csv to get normalized trends
    df = _load_speed_csv(csv_path)

    # debug: show counts per trend
    print(f"[DEBUG] trend value counts:\n{df['trend'].value_counts().to_string()}")

    # original segmentation logic follows — add an early debug to see produced segments
    segments: List[Any] = []
    for i, row in enumerate(df.itertuples(index=False)):
        # create a simple segment object with the attributes expected downstream
        seg = type("Seg", (), {})()
        seg.segment_id = i
        seg.start = float(getattr(row, "starttime"))
        seg.end = float(getattr(row, "endtime"))
        seg.trend = getattr(row, "trend")
        # add duration attribute required by features code
        seg.duration = seg.end - seg.start
        # skip invalid or zero-length segments
        if seg.end <= seg.start:
            print(f"[DEBUG] skipping invalid/zero-length trend segment: id={seg.segment_id} start={seg.start} end={seg.end} trend={seg.trend}")
            continue
        segments.append(seg)

    print(f"[DEBUG] build_segments_from_speed_csv created {len(segments)} segments")
    return segments

def _evt_ts(e: Dict[str, Any]) -> float:
    """
    Canonical event timestamp accessor. Prefer parser-standard keys.
    """
    return float(e.get("timestamp_sec") or e.get("timestamp") or e.get("time") or 0.0)

def assign_events_to_segments(events: List[Dict[str, Any]], segments: List[Any]) -> Dict[int, List[Dict[str, Any]]]:
    """
    Assign each event to the first segment whose [start,end) contains the event timestamp.
    Returns a map: segment_index -> list(events)
    """
    seg_map: Dict[int, List[Dict[str, Any]]] = {i: [] for i in range(len(segments))}

    # sort defensively using canonical timestamp
    for ev in sorted(events, key=_evt_ts):
        ts = _evt_ts(ev)
        for i, seg in enumerate(segments):
            seg_start = getattr(seg, "start", getattr(seg, "starttime", None))
            seg_end = getattr(seg, "end", getattr(seg, "endtime", None))
            if seg_start is None or seg_end is None:
                continue
            if seg_start <= ts < seg_end:
                seg_map[i].append(ev)
                break

    return seg_map
