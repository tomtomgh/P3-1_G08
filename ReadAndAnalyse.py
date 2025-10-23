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

# Fighting detection parameters
FIGHTING_TIME_WINDOW = 5.0  # Time window to look for rapid changes (seconds)
FIGHTING_MIN_CHANGES = 4    # Minimum number of changes to consider as fighting
FIGHTING_MIN_USERS = 2      # Minimum number of different users involved

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

# Track which user changed each parameter at each time (for shared parameters like frequency)
user_at_time = {p:np.full_like(timeline, -1, dtype=int) for p in params}
# Merged series for shared parameters (all users' changes combined)
merged_series = {p:np.full_like(timeline, np.nan, dtype=float) for p in params}

for p in params:
    # Get all changes for this parameter across all users, sorted by time
    all_changes = df[df['param']==p].sort_values('time_sec')
    
    for u in users:
        d = df[(df['user']==u)&(df['param']==p)].sort_values('time_sec')
        val = np.nan; idx = 0
        for i,t in enumerate(timeline):
            while idx < len(d) and t >= d.iloc[idx]['time_sec']:
                val = d.iloc[idx]['value']; idx += 1
            series[p][u][i] = val
    
    # Build merged series and track which user made each change (for shared parameters)
    val = np.nan; idx = 0; last_user = -1
    for i,t in enumerate(timeline):
        while idx < len(all_changes) and t >= all_changes.iloc[idx]['time_sec']:
            val = all_changes.iloc[idx]['value']
            last_user = all_changes.iloc[idx]['user']
            idx += 1
        merged_series[p][i] = val
        user_at_time[p][i] = last_user if not np.isnan(val) else -1

# ========== STEP 2.5: DETECT FIGHTING ON SHARED PARAMETERS ==========
def detect_fighting(param_name, time_window=FIGHTING_TIME_WINDOW, 
                    min_changes=FIGHTING_MIN_CHANGES, min_users=FIGHTING_MIN_USERS):
    """
    Detect 'fighting' behavior where multiple users rapidly change a shared parameter.
    
    Returns a list of fighting intervals: [(start_time, end_time, user_list, change_count), ...]
    """
    # Get all changes for this parameter
    param_changes = df[df['param'] == param_name].sort_values('time_sec').copy()
    
    if len(param_changes) < min_changes:
        return []
    
    fighting_intervals = []
    i = 0
    
    while i < len(param_changes):
        # Look ahead within the time window
        window_end_time = param_changes.iloc[i]['time_sec'] + time_window
        j = i
        
        # Find all changes within this time window
        while j < len(param_changes) and param_changes.iloc[j]['time_sec'] <= window_end_time:
            j += 1
        
        # Analyze this window
        window_changes = param_changes.iloc[i:j]
        num_changes = len(window_changes)
        unique_users = window_changes['user'].unique()
        num_users = len(unique_users)
        
        # Check if this qualifies as fighting
        if num_changes >= min_changes and num_users >= min_users:
            # Calculate "fighting intensity" - how many times users alternate
            user_switches = 0
            for k in range(1, len(window_changes)):
                if window_changes.iloc[k]['user'] != window_changes.iloc[k-1]['user']:
                    user_switches += 1
            
            # Only mark as fighting if there are actual back-and-forth changes
            if user_switches >= min_users:
                start_time = window_changes.iloc[0]['time_sec']
                end_time = window_changes.iloc[-1]['time_sec']
                
                fighting_intervals.append({
                    'start_time': start_time,
                    'end_time': end_time,
                    'users': list(unique_users),
                    'num_changes': num_changes,
                    'user_switches': user_switches
                })
                
                # Skip ahead to avoid overlapping intervals
                i = j
                continue
        
        i += 1
    
    # Merge overlapping or close intervals
    if not fighting_intervals:
        return []
    
    merged = []
    current = fighting_intervals[0].copy()
    
    for interval in fighting_intervals[1:]:
        # If intervals are close (within 2 seconds), merge them
        if interval['start_time'] - current['end_time'] <= 2.0:
            current['end_time'] = interval['end_time']
            current['users'] = list(set(current['users'] + interval['users']))
            current['num_changes'] += interval['num_changes']
            current['user_switches'] += interval['user_switches']
        else:
            merged.append(current)
            current = interval.copy()
    
    merged.append(current)
    return merged

# Detect fighting for frequency parameter
fighting_zones = {}
if 'frequency' in params:
    fighting_zones['frequency'] = detect_fighting('frequency')
    if fighting_zones['frequency']:
        print(f"\n⚔️  FIGHTING DETECTED on 'frequency' parameter!")
        for i, zone in enumerate(fighting_zones['frequency'], 1):
            print(f"   Zone {i}: {zone['start_time']:.1f}s - {zone['end_time']:.1f}s")
            print(f"           Users involved: {zone['users']}, Changes: {zone['num_changes']}, Switches: {zone['user_switches']}")

# ========== STEP 3: VISUALISATION WITH SCROLLING AND FILTERS ==========
visible_count = 2  # how many graphs to show at once
total_params = len(params)

# Create figure with room for checkboxes and bottom sliders
fig = plt.figure(figsize=(14, 8))
gs = fig.add_gridspec(visible_count, 2, width_ratios=[3, 1], 
                     left=0.08, right=0.85, bottom=0.18, top=0.95,
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
    
    # Define colors for each user
    user_colors = plt.cm.tab10(np.linspace(0, 1, len(users)))
    color_map = {u: user_colors[i] for i, u in enumerate(users)}
    
    for ax, p in zip(axes, subset):
        # Check if this is a shared parameter - just check by name
        is_shared = p in ['frequency']  # Add more shared parameters here if needed
        
        if is_shared:
            # For shared parameters, draw segments colored by which user made the change
            # Use the merged series that combines all users' changes
            merged_data = merged_series[p]
            
            # Plot segments with different colors based on who changed it
            i = 0
            added_labels = set()
            while i < len(timeline):
                if np.isnan(merged_data[i]):
                    i += 1
                    continue
                
                # Find the extent of this segment with the same user
                current_user = user_at_time[p][i]
                j = i
                while j < len(timeline) and user_at_time[p][j] == current_user and not np.isnan(merged_data[j]):
                    j += 1
                
                # Plot this segment only if the user is selected
                if current_user in active_users and current_user >= 0:
                    label = f"User {current_user}" if current_user not in added_labels else ""
                    ax.plot(timeline[i:j], merged_data[i:j], 
                           color=color_map[current_user], 
                           linewidth=2.5,
                           label=label)
                    added_labels.add(current_user)
                
                i = j
            
            # Remove duplicate labels in legend
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=9)
            ax.set_title(f"{p.capitalize()} (Shared Parameter)", fontsize=11, fontweight='bold')
            
            # Highlight fighting zones if they exist for this parameter
            if p in fighting_zones and fighting_zones[p]:
                from matplotlib.patches import Rectangle, Ellipse
                for zone in fighting_zones[p]:
                    # Calculate the center and dimensions of the fighting zone
                    center_time = (zone['start_time'] + zone['end_time']) / 2
                    time_span = zone['end_time'] - zone['start_time']
                    
                    # Get y-axis limits to size the highlight appropriately
                    y_data = merged_data[~np.isnan(merged_data)]
                    if len(y_data) > 0:
                        # Find values in the fighting zone
                        mask = (timeline >= zone['start_time']) & (timeline <= zone['end_time'])
                        zone_values = merged_data[mask]
                        zone_values = zone_values[~np.isnan(zone_values)]
                        
                        if len(zone_values) > 0:
                            y_min, y_max = zone_values.min(), zone_values.max()
                            y_center = (y_min + y_max) / 2
                            y_span = max(y_max - y_min, (y_data.max() - y_data.min()) * 0.15)
                            
                            # Draw an ellipse around the fighting zone
                            ellipse = Ellipse((center_time, y_center), 
                                            width=time_span * 1.3, 
                                            height=y_span * 1.5,
                                            fill=False, 
                                            edgecolor='red', 
                                            linewidth=2.5, 
                                            linestyle='--',
                                            alpha=0.7,
                                            zorder=10)
                            ax.add_patch(ellipse)
                            
                            # Add annotation
                            ax.annotate('⚔️ Fighting!', 
                                      xy=(center_time, y_max + y_span * 0.3),
                                      fontsize=9, 
                                      color='red', 
                                      fontweight='bold',
                                      ha='center',
                                      bbox=dict(boxstyle='round,pad=0.3', 
                                              facecolor='yellow', 
                                              alpha=0.7, 
                                              edgecolor='red'))
        else:
            # For user-specific parameters, draw separate lines for each user
            for u in active_users:
                ax.plot(timeline, series[p][u], 
                       color=color_map[u],
                       label=f"User {u}", 
                       linewidth=2)
            ax.legend(loc='upper right', fontsize=9)
            ax.set_title(p.capitalize(), fontsize=11, fontweight='bold')
        
        ax.set_ylabel(p, fontsize=10)
        ax.grid(True, alpha=0.3)
    
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

# Time slider (horizontal at the bottom, shared across all visible plots)
ax_slider_time = plt.axes([0.08, 0.08, 0.6, 0.03])
slider_time = Slider(ax_slider_time, 'Time Window', t_min, t_max-50, valinit=t_min, orientation='horizontal')

def update_time(val):
    start = slider_time.val
    for ax in axes:
        ax.set_xlim(start, start + 50)
    fig.canvas.draw_idle()

slider_time.on_changed(update_time)

# Parameter scroll slider (to scroll up/down between graphs)
ax_slider_param = plt.axes([0.08, 0.03, 0.3, 0.02])
slider_param = Slider(ax_slider_param, 'Scroll Params', 0, max(0, len([p for p in params if selected_params[p]]) - visible_count), 
                      valinit=0, valstep=1)

def update_param_scroll(val):
    current_top_param[0] = int(slider_param.val)
    draw_visible()

slider_param.on_changed(update_param_scroll)

# Button to toggle full/zoom view
ax_button = plt.axes([0.70, 0.08, 0.08, 0.03])
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
