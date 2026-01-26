import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from strategy_classifier.segmentation import build_segments_from_speed_csv

p = Path("segment_strategy_with_global_label.csv")
print("pred CSV exists:", p.exists(), "size=", p.stat().st_size if p.exists() else None)
df = pd.read_csv(p)
print("Pred cols:", df.columns.tolist())
# map common names
cols = {c.lower(): c for c in df.columns}
def col(*names):
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

startc = col("seg_start","starttime","start","seg_start_sec","start_time")
endc   = col("seg_end","endtime","end","seg_end_sec","end_time")
userc  = col("user_id","user","userid","uid")
predc  = col("pred_strategy","pred","strategy","label","global_label")

print("mapped:", startc, endc, userc, predc)
tmin,tmax = 155.0,160.0
if startc and endc:
    mask = (pd.to_numeric(df[startc],errors='coerce') <= tmax) & (pd.to_numeric(df[endc],errors='coerce') >= tmin)
    sel = df[mask].sort_values([startc])
    print("Predictions overlapping 155-160s (rows):", len(sel))
    if len(sel):
        cols_show = [c for c in (userc,startc,endc,predc) if c in df.columns]
        print(sel[cols_show].to_string(index=False))
else:
    print("No start/end columns found; showing head:")
    print(df.head().to_string(index=False))

# show segments used
segs = build_segments_from_speed_csv(Path("speed/trends.csv"), 20.0, 10.0)
print("Segments:", len(segs))
for s in segs:
    if s.start <= tmax and s.end >= tmin:
        print("SEG:", getattr(s,'segment_id',None), s.start, s.end, getattr(s,'trend',None))

# show normalized events in window using your runner's helper if available
try:
    from run_full_strategy_analysis import _normalize_events_for_pipeline
    from log_parser import parse_session
    ev = parse_session("./logs")
    nev = _normalize_events_for_pipeline(ev)
    print("Normalized events in 155-160s:")
    for e in nev:
        if e["timestamp_sec"] >= tmin and e["timestamp_sec"] <= tmax:
            print(e)
except Exception as ex:
    print("Could not print normalized events:", ex)