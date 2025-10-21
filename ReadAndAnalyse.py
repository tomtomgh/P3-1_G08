import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
from matplotlib.widgets import Slider, Button

# ========== CONFIG ==========
LOG_PATTERN = "User*.log"
TIME_RESOLUTION = 0.1
WINDOW_FOR_SIMULTANEOUS = 1.0
OUTPUT_CSV = "parameter_changes_summary.csv"
GRAPH_SAVE_DIR = "/Users/tomdaugherty/Documents/GitHub/P3-1_G08/graphs"
os.makedirs(GRAPH_SAVE_DIR, exist_ok=True)

# ========== STEP 1: PARSE LOG FILES ==========
pattern = re.compile(
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?'
    r'(?:User\s*(?P<user_id>\d+))?\s*sets\s+'
    r'(?P<param>[a-zA-Z ]+?)\s+to\s+(?P<value>[-+]?\d*\.?\d+)',
    re.IGNORECASE
)

def time_to_seconds(t):
    h, m, s = t.split(':')
    return int(h)*3600 + int(m)*60 + float(s)

records = []
for file in sorted(glob.glob(LOG_PATTERN)):
    default_user = re.search(r'User(\d+)', file).group(1)
    print(f"🔍 Reading {file} ...")
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            val_str = m.group('value').strip().rstrip('.')
            try:
                value = float(val_str)
            except ValueError:
                continue
            user = m.group('user_id') or default_user
            records.append({
                'time': m.group('time'),
                'user': int(user),
                'param': m.group('param').strip().lower(),
                'value': value
            })

df = pd.DataFrame(records)
if df.empty:
    raise ValueError("❌ No parameter change data found.")
df['time_sec'] = df['time'].apply(time_to_seconds)
df = df.sort_values(by=['time_sec','user','param']).reset_index(drop=True)
df.to_csv(OUTPUT_CSV,index=False)
print(f"\n✅ Parsed {len(df)} parameter changes from {len(df['user'].unique())} users → saved to {OUTPUT_CSV}")
print("📋 Parameters detected:", df['param'].unique())

# ========== STEP 2: BUILD TIME SERIES ==========
users = sorted(df['user'].unique())
params = sorted(df['param'].unique())
t_min, t_max = df['time_sec'].min(), df['time_sec'].max()
timeline = np.arange(t_min, t_max, TIME_RESOLUTION)
series = {p:{u:np.full_like(timeline, np.nan, dtype=float) for u in users} for p in params}

for p in params:
    for u in users:
        d = df[(df['user']==u)&(df['param']==p)].sort_values('time_sec')
        val = np.nan; idx = 0
        for i,t in enumerate(timeline):
            while idx < len(d) and t >= d.iloc[idx]['time_sec']:
                val = d.iloc[idx]['value']; idx += 1
            series[p][u][i] = val

# ========== STEP 3: VISUALISATION WITH SCROLLING ==========
visible_count = 2  # how many graphs to show at once
total_params = len(params)

fig, axes = plt.subplots(visible_count, 1, figsize=(10, 6), sharex=True)
plt.subplots_adjust(right=0.85, bottom=0.15)

# Function to plot the visible subset
def draw_visible(start_index):
    for ax in axes:
        ax.clear()
    subset = params[start_index:start_index + visible_count]
    for ax, p in zip(axes, subset):
        for u in users:
            ax.plot(timeline, series[p][u], label=f"User {u}")
        ax.set_ylabel(p)
        ax.set_title(p.capitalize())
        ax.grid(True)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc='upper right')
    for ax in axes:
        ax.set_xlim(t_min, t_min + 50)
    fig.canvas.draw_idle()

# Initial draw
current_top_param = [0]
draw_visible(current_top_param[0])

# Time slider (shared across all visible plots)
ax_slider_time = plt.axes([0.9, 0.15, 0.03, 0.7])
slider_time = Slider(ax_slider_time, 'Time', t_min, t_max-50, valinit=t_min, orientation='vertical')

def update_time(val):
    start = slider_time.val
    for ax in axes:
        ax.set_xlim(start, start + 50)
    fig.canvas.draw_idle()

slider_time.on_changed(update_time)

# Parameter scroll slider (to scroll up/down between graphs)
ax_slider_param = plt.axes([0.4, 0.05, 0.3, 0.03])
slider_param = Slider(ax_slider_param, 'Scroll', 0, max(0, total_params - visible_count), 
                      valinit=0, valstep=1)

def update_param(val):
    current_top_param[0] = int(slider_param.val)
    draw_visible(current_top_param[0])
slider_param.on_changed(update_param)

# Button to toggle full/zoom view
ax_button = plt.axes([0.88, 0.05, 0.08, 0.05])
button = Button(ax_button, 'Full View', color='lightgray', hovercolor='0.85')
full_view = [False]

def toggle_full(event):
    full_view[0] = not full_view[0]
    if full_view[0]:
        for ax in axes: ax.set_xlim(t_min, t_max)
        button.label.set_text("Zoom View")
    else:
        start = slider_time.val
        for ax in axes: ax.set_xlim(start, start + 50)
        button.label.set_text("Full View")
    fig.canvas.draw_idle()
button.on_clicked(toggle_full)

plt.tight_layout(rect=[0,0.08,0.85,1])
plt.show()

# ========== STEP 4: SAVE IMAGES ==========
# Save combined and per-parameter
combined_path = os.path.join(GRAPH_SAVE_DIR, "scrollable_combined_view.png")
fig.savefig(combined_path, dpi=200)
print(f"💾 Saved scrollable combined view → {combined_path}")

for p in params:
    fig_single, ax = plt.subplots(figsize=(8,5))
    for u in users:
        ax.plot(timeline, series[p][u], label=f"User {u}")
    ax.set_title(f"{p.capitalize()} over Time")
    ax.set_xlabel("Time (s)"); ax.set_ylabel(p)
    ax.legend(); ax.grid(True)
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(GRAPH_SAVE_DIR, f"{p}_plot.png"), dpi=200)
    plt.close(fig_single)
print(f"🖼️ Individual parameter plots saved to {GRAPH_SAVE_DIR}")
