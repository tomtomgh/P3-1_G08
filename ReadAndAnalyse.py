import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations

# ========== CONFIG ==========
LOG_PATTERN = "User*.log"         # all log files
TIME_RESOLUTION = 0.1             # seconds between samples in reconstructed timeline
WINDOW_FOR_SIMULTANEOUS = 1.0     # seconds for "simultaneous" parameter changes
OUTPUT_CSV = "parameter_changes_summary.csv"

# ========== STEP 1: PARSE LOG FILES ==========
pattern = re.compile(
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?(?:User\s*(?P<user_id>\d+))?\s*sets\s+(?P<param>[a-zA-Z ]+)\s+to\s+(?P<value>[-\d\.]+)',
    re.IGNORECASE
)

def time_to_seconds(t):
    h, m, s = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)

records = []

for file in sorted(glob.glob(LOG_PATTERN)):
    user_hint = re.search(r'User(\d+)', file)
    default_user = user_hint.group(1) if user_hint else None
    print(f"🔍 Reading {file} ...")
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue

            # Determine user (from log line or filename)
            user = m.group('user_id') or default_user

            # Clean up value string to remove stray periods or whitespace
            val_str = m.group('value').strip().rstrip('.')

            try:
                value = float(val_str)
            except ValueError:
                print(f"⚠️ Skipping invalid numeric value '{val_str}' in line: {line.strip()}")
                continue

            # Add the parsed record
            records.append({
                'time': m.group('time'),
                'user': int(user),
                'param': m.group('param').strip(),
                'value': value
            })

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("❌ No parameter change data found. Check your log files or regex.")

df['time_sec'] = df['time'].apply(time_to_seconds)
df = df.sort_values(by=['time_sec', 'user', 'param']).reset_index(drop=True)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Parsed {len(df)} parameter changes from {len(df['user'].unique())} users → saved to {OUTPUT_CSV}")

# ========== STEP 2: BUILD TIME SERIES ==========
users = sorted(df['user'].unique())
params = sorted(df['param'].unique())
t_min, t_max = df['time_sec'].min(), df['time_sec'].max()
timeline = np.arange(t_min, t_max, TIME_RESOLUTION)

series = {param: {u: np.full_like(timeline, np.nan, dtype=float) for u in users} for param in params}

for param in params:
    for u in users:
        user_data = df[(df['user'] == u) & (df['param'] == param)].sort_values('time_sec')
        val = np.nan
        idx = 0
        for i, t in enumerate(timeline):
            while idx < len(user_data) and t >= user_data.iloc[idx]['time_sec']:
                val = user_data.iloc[idx]['value']
                idx += 1
            series[param][u][i] = val

# ========== STEP 3: SYNCHRONY METRICS ==========
def cross_corr(a, b):
    mask = ~np.isnan(a) & ~np.isnan(b)
    if np.sum(mask) < 5:  # not enough overlap
        return np.nan
    a, b = a[mask], b[mask]
    a -= np.mean(a)
    b -= np.mean(b)
    corr = np.correlate(a, b, mode='full') / (np.std(a) * np.std(b) * len(a))
    return np.nanmax(corr)

synchrony_summary = []

for param in params:
    print(f"\n🔍 Analysing parameter: {param}")
    # Cross-correlation matrix
    corrs = []
    for u1, u2 in combinations(users, 2):
        corr_val = cross_corr(series[param][u1], series[param][u2])
        corrs.append(corr_val)
        synchrony_summary.append({'param': param, 'user_pair': f"{u1}-{u2}", 'cross_corr': corr_val})
        print(f"   Users {u1} & {u2}: correlation = {corr_val:.3f}" if corr_val is not np.nan else f"   Users {u1} & {u2}: no data")

# ========== STEP 4: TEMPORAL CLUSTERING (who changes together) ==========
print("\n🕒 Checking simultaneous parameter changes...")
for param in params:
    changes = df[df['param'] == param]
    for t in np.arange(changes['time_sec'].min(), changes['time_sec'].max(), WINDOW_FOR_SIMULTANEOUS):
        simultaneous = changes[(changes['time_sec'] >= t) & (changes['time_sec'] < t + WINDOW_FOR_SIMULTANEOUS)]
        if len(simultaneous['user'].unique()) > 1:
            users_involved = list(simultaneous['user'].unique())
            print(f"   Around {t:.1f}s: {len(users_involved)} users changed {param} ({users_involved})")

# ========== STEP 5: CONVERGENCE (final value similarity) ==========
df_end = df.sort_values('time_sec').groupby(['user', 'param']).tail(1)
spread = df_end.groupby('param')['value'].std().round(3)
print("\n📉 Final value spread (lower = more synchronised):")
print(spread)

# ========== STEP 6: VISUALISATION ==========
for param in params:
    plt.figure(figsize=(10,5))
    for u in users:
        plt.plot(timeline, series[param][u], label=f"User {u}")
    plt.title(f"{param.capitalize()} over Time")
    plt.xlabel("Time (s)")
    plt.ylabel(param)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ========== STEP 7: SAVE SUMMARY ==========
pd.DataFrame(synchrony_summary).to_csv("synchrony_summary.csv", index=False)
print("\n✅ Synchrony analysis complete! Results saved to synchrony_summary.csv")
