import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
from matplotlib.widgets import Slider, Button, CheckButtons

# ========== CONFIG ==========
LOG_PATTERN = "User*.log"
TIME_RESOLUTION = 0.1
WINDOW_FOR_SIMULTANEOUS = 1.0
OUTPUT_CSV = "parameter_changes_summary.csv"
GRAPH_SAVE_DIR = "./graphs"
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

# ========== STEP 3: VISUALISATION WITH SCROLLING AND FILTERS ==========
visible_count = 2  # how many graphs to show at once
total_params = len(params)

# Create figure with room for checkboxes
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(visible_count, 2, width_ratios=[3, 1], 
                     left=0.08, right=0.85, bottom=0.15, top=0.95,
                     wspace=0.3, hspace=0.3)

# Create axes for plots
axes = [fig.add_subplot(gs[i, 0]) for i in range(visible_count)]

# Selection state
selected_users = {u: True for u in users}
selected_params = {p: True for p in params}
current_top_param = [0]

# Function to plot the visible subset
def draw_visible():
    # Get selected items
    active_users = [u for u in users if selected_users[u]]
    active_params = [p for p in params if selected_params[p]]
    
    if not active_users or not active_params:
        for ax in axes:
            ax.clear()
            ax.text(0.5, 0.5, 'No data selected', ha='center', va='center', transform=ax.transAxes)
        fig.canvas.draw_idle()
        return
    
    for ax in axes:
        ax.clear()
    
    # Get visible parameters based on scroll position
    visible_params = [p for p in active_params if selected_params[p]]
    start_idx = min(current_top_param[0], max(0, len(visible_params) - visible_count))
    current_top_param[0] = start_idx
    subset = visible_params[start_idx:start_idx + visible_count]
    
    for ax, p in zip(axes, subset):
        for u in active_users:
            ax.plot(timeline, series[p][u], label=f"User {u}", linewidth=2)
        ax.set_ylabel(p, fontsize=10)
        ax.set_title(p.capitalize(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)
    
    if subset:
        axes[-1].set_xlabel("Time (s)", fontsize=10)
    
    for ax in axes:
        ax.set_xlim(t_min, t_min + 50)
    
    fig.canvas.draw_idle()

# User selection checkboxes
ax_user_check = fig.add_subplot(gs[0, 1])
ax_user_check.set_title("Select Users", fontsize=10, fontweight='bold')
user_labels = [f"User {u}" for u in users]
user_check = CheckButtons(ax_user_check, user_labels, [True] * len(users))

def user_toggle(label):
    user_id = int(label.split()[-1])
    selected_users[user_id] = not selected_users[user_id]
    draw_visible()

user_check.on_clicked(user_toggle)

# Parameter selection checkboxes
ax_param_check = fig.add_subplot(gs[1, 1])
ax_param_check.set_title("Select Parameters", fontsize=10, fontweight='bold')
param_labels = [p.capitalize() for p in params]
# Adjust to show scrollable list if too many parameters
max_visible_params = min(len(params), 10)
display_params = params[:max_visible_params]
display_labels = [p.capitalize() for p in display_params]
param_check = CheckButtons(ax_param_check, display_labels, [True] * len(display_params))

def param_toggle(label):
    param_name = label.lower()
    selected_params[param_name] = not selected_params[param_name]
    draw_visible()

param_check.on_clicked(param_toggle)

# Initial draw
draw_visible()

# Time slider (shared across all visible plots)
ax_slider_time = plt.axes([0.88, 0.15, 0.02, 0.7])
slider_time = Slider(ax_slider_time, 'Time', t_min, t_max-50, valinit=t_min, orientation='vertical')

def update_time(val):
    start = slider_time.val
    for ax in axes:
        ax.set_xlim(start, start + 50)
    fig.canvas.draw_idle()

slider_time.on_changed(update_time)

# Parameter scroll slider (to scroll up/down between graphs)
ax_slider_param = plt.axes([0.3, 0.05, 0.3, 0.03])
slider_param = Slider(ax_slider_param, 'Scroll Params', 0, max(0, len([p for p in params if selected_params[p]]) - visible_count), 
                      valinit=0, valstep=1)

def update_param_scroll(val):
    current_top_param[0] = int(slider_param.val)
    draw_visible()

slider_param.on_changed(update_param_scroll)

# Button to toggle full/zoom view
ax_button = plt.axes([0.92, 0.05, 0.06, 0.05])
button = Button(ax_button, 'Full View', color='lightgray', hovercolor='0.85')
full_view = [False]

def toggle_full(event):
    full_view[0] = not full_view[0]
    if full_view[0]:
        for ax in axes: ax.set_xlim(t_min, t_max)
        button.label.set_text("Zoom")
    else:
        start = slider_time.val
        for ax in axes: ax.set_xlim(start, start + 50)
        button.label.set_text("Full View")
    fig.canvas.draw_idle()

button.on_clicked(toggle_full)

plt.show()

# ========== STEP 4: SAVE IMAGES ==========
# Save images with current selections
print("\n💾 Saving visualizations...")

# Save the interactive view
combined_path = os.path.join(GRAPH_SAVE_DIR, "interactive_view.png")
fig.savefig(combined_path, dpi=200, bbox_inches='tight')
print(f"💾 Saved interactive view → {combined_path}")

# Save individual parameter plots for all users
for p in params:
    fig_single, ax = plt.subplots(figsize=(10, 6))
    for u in users:
        ax.plot(timeline, series[p][u], label=f"User {u}", linewidth=2)
    ax.set_title(f"{p.capitalize()} over Time", fontsize=14, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel(p, fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig_single.tight_layout()
    fig_single.savefig(os.path.join(GRAPH_SAVE_DIR, f"{p}_plot.png"), dpi=200)
    plt.close(fig_single)

print(f"🖼️ Individual parameter plots saved to {GRAPH_SAVE_DIR}")
