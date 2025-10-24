import pandas as pd
import numpy as np
from pathlib import Path

CSV_PATH = "parameter_changes_summary.csv" # input CSV file path
TIME_COL = "time_sec"            # numeric seconds from session start
USER_COL = "user"
PARAM_COL = "param"              
VALUE_COL = "value"              

# Dominance settings
WINDOW_S = 30.0                  # rolling window length (seconds)
HOP_S = 1.0                      # slide step (seconds)
DOMINANCE_THRESHOLD = 0.50       # ≥50% of changes within a window

# Inactivity settings
INACTIVITY_S = 5.0               # no event by user for ≥ 5 seconds to flag inactivity

# ---------- LOAD ----------
df = pd.read_csv(CSV_PATH)

assert TIME_COL in df.columns and USER_COL in df.columns, "Required columns missing." 
df = df[[TIME_COL, USER_COL, PARAM_COL, VALUE_COL]].copy()
df = df.sort_values(TIME_COL).reset_index(drop=True)

users = sorted(df[USER_COL].unique().tolist())
t_min = float(df[TIME_COL].min())
t_max = float(df[TIME_COL].max())

# ---------- INACTIVITY: build inactivity intervals per user ----------
inact_rows = []
for u in users:
    d = df[df[USER_COL] == u][[TIME_COL]].reset_index(drop=True)
    # add session boundaries
    times = d[TIME_COL].to_numpy()
    # Handle possible inactivity from session start
    if len(times) == 0:
        # user never acted: entire session is inactivity
        if t_max - t_min >= INACTIVITY_S:
            inact_rows.append({
                "user": u,
                "start_time": t_min,
                "end_time": t_max,
                "duration": t_max - t_min
            })
        continue

    # Check start gap
    if times[0] - t_min >= INACTIVITY_S:
        inact_rows.append({
            "user": u, "start_time": t_min, "end_time": times[0],
            "duration": times[0] - t_min
        })

    # Gaps between events
    gaps = np.diff(times)
    for i, gap in enumerate(gaps):
        if gap >= INACTIVITY_S:
            start = times[i]
            end = times[i+1]
            inact_rows.append({
                "user": u, "start_time": start, "end_time": end,
                "duration": end - start
            })

    # Check end gap (from last event to session end)
    if (t_max - times[-1]) >= INACTIVITY_S:
        inact_rows.append({
            "user": u, "start_time": times[-1], "end_time": t_max,
            "duration": t_max - times[-1]
        })

inactivity_events = pd.DataFrame(inact_rows).sort_values(["user", "start_time"]).reset_index(drop=True)

# ---------- DOMINANCE: rolling-window detection ----------
# Pre-index events by second to speed up counting
# (works well unless your logs are extremely large)
# Build hop grid
grid = np.arange(t_min, t_max - WINDOW_S + HOP_S, HOP_S)  # window = [t, t+WINDOW_S)

# Prepare an index to search quickly
# We’ll use cumulative counts per user over sorted times to get O(1) window counts.
df["_one"] = 1
cum = (
    df[[TIME_COL, USER_COL, "_one"]]
    .assign(idx=lambda x: np.arange(len(x)))
    .sort_values(TIME_COL)
)

# For each user, cumulative counts keyed by time
user_cums = {}
for u in users:
    dd = cum[cum[USER_COL] == u][[TIME_COL, "_one"]].copy()
    dd["cum"] = dd["_one"].cumsum()
    user_cums[u] = dd[["TIME_COL", "cum"]].reset_index(drop=True)

def count_in_window(u_df, t0, t1):
    # count events with t in [t0, t1)
    # Using searchsorted over the time column
    times = u_df["TIME_COL"].to_numpy()
    cum = u_df["cum"].to_numpy()
    # left index = first time >= t0
    li = np.searchsorted(times, t0, side="left")
    # right index = first time >= t1
    ri = np.searchsorted(times, t1, side="left")
    if ri == 0:
        return 0
    c_right = cum[ri-1]
    c_left = cum[li-1] if li > 0 else 0
    return int(c_right - c_left)

dom_rows = []
for t0 in grid:
    t1 = t0 + WINDOW_S
    counts = {u: count_in_window(user_cums[u], t0, t1) for u in users}
    total = sum(counts.values())
    if total == 0:
        continue
    props = {u: counts[u] / total for u in users}
    # find dominant users (could be >1 if tie ≥50%)
    doms = [u for u, p in props.items() if p >= DOMINANCE_THRESHOLD]
    if doms:
        dom_rows.append({
            "start_time": t0,
            "end_time": t1,
            "total_events": total,
            "dominant_users": ",".join(doms),
            **{f"count_{u}": counts[u] for u in users},
            **{f"prop_{u}": round(props[u], 4) for u in users},
        })

dominance_stream = pd.DataFrame(dom_rows).reset_index(drop=True)

# ---------- OPTIONAL: compress dominance into intervals where the same set stays dominant ----------
def compress_intervals(df_dom, key_col="dominant_users"):
    if df_dom.empty:
        return df_dom
    rows = []
    cur_key = None
    cur_start = None
    cur_end = None
    agg_counts = None
    agg_total = 0

    for _, r in df_dom.iterrows():
        key = r[key_col]
        if cur_key is None:
            cur_key = key
            cur_start = r["start_time"]
            cur_end = r["end_time"]
            agg_total = r["total_events"]
            agg_counts = {k: r[k] for k in df_dom.columns if k.startswith("count_")}
        elif key == cur_key and abs(r["start_time"] - cur_end) <= 1e-9:
            cur_end = r["end_time"]
            agg_total += r["total_events"]
            for k in agg_counts:
                agg_counts[k] += r[k]
        else:
            rows.append({
                "dominant_users": cur_key,
                "start_time": cur_start,
                "end_time": cur_end,
                "duration": cur_end - cur_start,
                "total_events": agg_total,
                **agg_counts
            })
            cur_key = key
            cur_start = r["start_time"]
            cur_end = r["end_time"]
            agg_total = r["total_events"]
            agg_counts = {k: r[k] for k in df_dom.columns if k.startswith("count_")}

    rows.append({
        "dominant_users": cur_key,
        "start_time": cur_start,
        "end_time": cur_end,
        "duration": cur_end - cur_start,
        "total_events": agg_total,
        **agg_counts
    })
    return pd.DataFrame(rows)

dominance_intervals = compress_intervals(dominance_stream)

# ---------- OVERALL SUMMARY ----------
overall_counts = df[USER_COL].value_counts().reindex(users, fill_value=0)
overall_props = (overall_counts / overall_counts.sum()).round(4) if overall_counts.sum() else overall_counts
overall_summary = pd.DataFrame({
    "user": users,
    "total_events": [overall_counts.get(u, 0) for u in users],
    "overall_prop": [overall_props.get(u, 0.0) for u in users]
})

# Inactivity totals (sum of inactive durations per user)
if not inactivity_events.empty:
    inactive_totals = (inactivity_events.groupby("user")["duration"]
                       .sum()
                       .reindex(users, fill_value=0)
                       .rename("inactive_seconds"))
    overall_summary = overall_summary.merge(inactive_totals, on="user", how="left")
else:
    overall_summary["inactive_seconds"] = 0.0

# ---------- OUTPUT ----------
out_dir = Path("detections_out")
out_dir.mkdir(exist_ok=True)

inactivity_events.to_csv(out_dir / "inactivity_events.csv", index=False)
dominance_stream.to_csv(out_dir / "dominance_stream_rolling.csv", index=False)
dominance_intervals.to_csv(out_dir / "dominance_intervals_compressed.csv", index=False)
overall_summary.to_csv(out_dir / "overall_summary.csv", index=False)

print("\n=== Files written to ./detections_out ===")
print("- inactivity_events.csv")
print("- dominance_stream_rolling.csv")
print("- dominance_intervals_compressed.csv")
print("- overall_summary.csv")

# Quick console preview
print("\n--- Overall summary ---")
print(overall_summary.to_string(index=False))

print("\n--- First few dominance intervals ---")
print(dominance_intervals.head(10).to_string(index=False))

print("\n--- First few inactivity events ---")
print(inactivity_events.head(10).to_string(index=False))
