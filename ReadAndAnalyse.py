import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
import os
from matplotlib.widgets import Slider, Button, CheckButtons
from matplotlib.widgets import RadioButtons

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

# ========== STEP 2.6: ANALYZE USER DOMINANCE AND ACTIVITY ==========
def analyze_user_activity():
    """
    Analyze user dominance, activity, and involvement in the system.
    
    Returns statistics including:
    - Total changes per user
    - Changes per parameter per user
    - Time in control (for shared parameters like frequency)
    - Activity rate (changes per minute)
    - Inactivity periods
    """
    total_duration = t_max - t_min
    
    user_stats = {}
    
    for user in users:
        user_df = df[df['user'] == user]
        
        # Basic activity metrics
        total_changes = len(user_df)
        changes_per_minute = (total_changes / total_duration) * 60 if total_duration > 0 else 0
        
        # Changes per parameter
        param_changes = {}
        for param in params:
            param_changes[param] = len(user_df[user_df['param'] == param])
        
        # Time in control for shared parameters (e.g., frequency)
        time_in_control = {}
        for param in params:
            if param in ['frequency']:  # Shared parameters
                # Count how many timeline points this user "owns"
                control_points = np.sum(user_at_time[param] == user)
                time_controlled = control_points * TIME_RESOLUTION
                time_in_control[param] = time_controlled
        
        # Calculate inactivity periods (gaps between changes)
        if len(user_df) > 1:
            sorted_times = user_df['time_sec'].sort_values().values
            gaps = np.diff(sorted_times)
            max_gap = gaps.max() if len(gaps) > 0 else 0
            avg_gap = gaps.mean() if len(gaps) > 0 else 0
            num_long_gaps = np.sum(gaps > 10)  # Gaps longer than 10 seconds
        else:
            max_gap = 0
            avg_gap = 0
            num_long_gaps = 0
        
        # First and last action times
        if len(user_df) > 0:
            first_action = user_df['time_sec'].min()
            last_action = user_df['time_sec'].max()
            active_duration = last_action - first_action
        else:
            first_action = 0
            last_action = 0
            active_duration = 0
        
        user_stats[user] = {
            'total_changes': total_changes,
            'changes_per_minute': changes_per_minute,
            'param_changes': param_changes,
            'time_in_control': time_in_control,
            'max_inactivity_gap': max_gap,
            'avg_gap_between_changes': avg_gap,
            'long_inactivity_periods': num_long_gaps,
            'first_action_time': first_action,
            'last_action_time': last_action,
            'active_duration': active_duration,
            'activity_percentage': (active_duration / total_duration * 100) if total_duration > 0 else 0
        }
    
    return user_stats, total_duration

# Calculate and display user statistics
print("\n" + "="*80)
print("📊 USER ACTIVITY & DOMINANCE STATISTICS")
print("="*80)

user_stats, total_duration = analyze_user_activity()

# ========== STEP 2.7: CAREFULNESS (INCREMENTAL vs BIG JUMPS) ==========
def analyze_carefulness(user_id, df, params):
    """
    Analyze how 'careful' a user's parameter changes are.
    Returns a dict of per-parameter carefulness metrics.
    """
    carefulness = {}
    for param in params:
        param_df = df[(df['user'] == user_id) & (df['param'] == param)].sort_values('time_sec')
        if len(param_df) < 2:
            carefulness[param] = {
                'avg_step': np.nan,
                'max_step': np.nan,
                'careful_ratio': np.nan,
                'behavior_type': 'Insufficient data'
            }
            continue

        values = param_df['value'].values
        diffs = np.abs(np.diff(values))
        if np.all(diffs == 0):
            behavior_type = 'Careful (no variation)'
            avg_step = max_step = careful_ratio = 0
        else:
            avg_step = np.mean(diffs)
            max_step = np.max(diffs)

            # "Small change" threshold = 10% of observed range for that parameter (fallback 1 if range=0)
            param_range = np.ptp(values)
            threshold = 0.1 * (param_range if param_range > 0 else 1)
            small_steps = np.sum(diffs <= threshold)
            careful_ratio = small_steps / len(diffs) * 100

            behavior_type = 'Careful' if careful_ratio > 70 else 'Reckless'

        carefulness[param] = {
            'avg_step': avg_step,
            'max_step': max_step,
            'careful_ratio': careful_ratio,
            'behavior_type': behavior_type
        }

    return carefulness

print("\n🤖 CONTROL CAREFULNESS ANALYSIS")
print("="*80)
for u in sorted(users):
    cstats = analyze_carefulness(u, df, params)
    user_stats[u]['carefulness'] = cstats
    for p in sorted(params):
        c = cstats[p]
        if np.isnan(c['avg_step']):
            print(f"User {u:<2} | {p:<12} | Insufficient data")
        else:
            print(f"User {u:<2} | {p:<12} | Avg Δ={c['avg_step']:.3f} | Max Δ={c['max_step']:.3f} | "
                  f"Careful Steps={c['careful_ratio']:.1f}% | Behavior={c['behavior_type']}")

# Overall summary
print(f"\n📅 Total Session Duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
print(f"📋 Parameters Tracked: {', '.join([p.capitalize() for p in params])}")

# Per-user statistics
for user in sorted(users):
    stats = user_stats[user]
    print(f"\n{'='*80}")
    print(f"👤 USER {user} - Activity Report")
    print(f"{'='*80}")
    
    # Basic activity
    print(f"  📈 Total Changes Made: {stats['total_changes']}")
    print(f"  ⚡ Activity Rate: {stats['changes_per_minute']:.2f} changes/minute")
    print(f"  ⏱️  Active Duration: {stats['active_duration']:.1f}s ({stats['active_duration']/60:.1f} min)")
    print(f"  📊 Activity Percentage: {stats['activity_percentage']:.1f}%")
    
    # Parameter-specific activity
    print(f"\n  📋 Changes by Parameter:")
    for param in sorted(params):
        count = stats['param_changes'][param]
        percentage = (count / stats['total_changes'] * 100) if stats['total_changes'] > 0 else 0
        print(f"     • {param.capitalize()}: {count} changes ({percentage:.1f}%)")
    
    # Dominance on shared parameters
    if stats['time_in_control']:
        print(f"\n  👑 Dominance on Shared Parameters:")
        for param, time_controlled in stats['time_in_control'].items():
            percentage = (time_controlled / total_duration * 100) if total_duration > 0 else 0
            print(f"     • {param.capitalize()}: {time_controlled:.1f}s ({percentage:.1f}% of total time)")
    
    # Inactivity analysis
    print(f"\n  💤 Inactivity Analysis:")
    print(f"     • Average gap between changes: {stats['avg_gap_between_changes']:.2f}s")
    print(f"     • Longest inactivity period: {stats['max_inactivity_gap']:.1f}s")
    print(f"     • Number of long gaps (>10s): {stats['long_inactivity_periods']}")
    
    # Timing
    print(f"\n  ⏰ Session Timing:")
    print(f"     • First action at: {stats['first_action_time']:.1f}s")
    print(f"     • Last action at: {stats['last_action_time']:.1f}s")

    # Carefulness per parameter
    print(f"\n  🤖 Control Carefulness:")
    for param in sorted(params):
        c = stats['carefulness'][param]
        if np.isnan(c['avg_step']):
            print(f"     • {param.capitalize()}: Insufficient data")
        else:
            print(f"     • {param.capitalize()}: Avg Δ={c['avg_step']:.3f}, Max Δ={c['max_step']:.3f}, "
                  f"Careful Steps={c['careful_ratio']:.1f}%, Behavior={c['behavior_type']}")

# Comparative analysis
print(f"\n{'='*80}")
print(f"🏆 COMPARATIVE ANALYSIS")
print(f"{'='*80}")

# Most active user
most_active_user = max(users, key=lambda u: user_stats[u]['total_changes'])
print(f"  🥇 Most Active User: User {most_active_user} ({user_stats[most_active_user]['total_changes']} changes)")

# Most dominant on shared parameters
if 'frequency' in params:
    freq_dominance = {u: user_stats[u]['time_in_control'].get('frequency', 0) for u in users}
    most_dominant = max(freq_dominance, key=freq_dominance.get)
    print(f"  👑 Most Dominant (Frequency): User {most_dominant} ({freq_dominance[most_dominant]:.1f}s, {freq_dominance[most_dominant]/total_duration*100:.1f}%)")

# Most responsive (highest changes per minute)
most_responsive = max(users, key=lambda u: user_stats[u]['changes_per_minute'])
print(f"  ⚡ Most Responsive: User {most_responsive} ({user_stats[most_responsive]['changes_per_minute']:.2f} changes/min)")

# Calculate engagement distribution
total_changes_all = sum(user_stats[u]['total_changes'] for u in users)
print(f"\n  📊 Engagement Distribution:")
for user in sorted(users):
    percentage = (user_stats[user]['total_changes'] / total_changes_all * 100) if total_changes_all > 0 else 0
    bar = '█' * int(percentage / 2)
    print(f"     User {user}: {bar} {percentage:.1f}%")

# Careful vs Reckless summary
print(f"\n  🤖 Carefulness Summary:")
for user in sorted(users):
    behaviors = [v['behavior_type'] for v in user_stats[user]['carefulness'].values()
                 if v['behavior_type'] not in ['Insufficient data']]
    if not behaviors:
        print(f"     User {user}: Insufficient data")
        continue
    careful_count = sum(b == 'Careful' or 'no variation' in b for b in behaviors)
    ratio = careful_count / len(behaviors) * 100
    label = "🟢 Careful" if ratio >= 70 else "🔴 Reckless"
    print(f"     User {user}: {label} ({ratio:.1f}% careful changes)")

print(f"\n{'='*80}\n")

# ========== STEP 3: VISUALISATION WITH SCROLLING AND FILTERS ==========
visible_count = 2  # how many graphs to show at once
total_params = len(params)

# Create figure with room for checkboxes, stats panel, and bottom sliders
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(visible_count + 1, 2, width_ratios=[3, 1], 
                     height_ratios=[1] * visible_count + [0.8],
                     left=0.08, right=0.88, bottom=0.15, top=0.95,
                     wspace=0.3, hspace=0.4)

# Create axes for plots
axes = [fig.add_subplot(gs[i, 0]) for i in range(visible_count)]

# Create stats display panel at the bottom
ax_stats = fig.add_subplot(gs[visible_count, :])
ax_stats.axis('off')

# Selection state
selected_users = {u: True for u in users}
selected_params = {p: True for p in params}
current_top_param = [0]
full_view = [False]  # Initialize full_view state before draw_visible()
time_window_start = [t_min]  # Initialize time window start position

# Function to update stats display
def update_stats_display():
    ax_stats.clear()
    ax_stats.axis('off')
    
    # Create stats text
    stats_text = "USER STATISTICS\n" + "─" * 100 + "\n\n"
    
    for user in sorted(users):
        stats = user_stats[user]
        
        # Create a compact summary for each user
        stats_text += f"User {user}: "
        stats_text += f"{stats['total_changes']} changes  |  "
        stats_text += f"{stats['changes_per_minute']:.1f} chg/min  |  "
        
        # Dominance on frequency
        if 'frequency' in stats['time_in_control']:
            freq_pct = (stats['time_in_control']['frequency'] / total_duration * 100)
            stats_text += f"Freq dominance: {freq_pct:.1f}%  |  "
        
        stats_text += f"Max gap: {stats['max_inactivity_gap']:.1f}s"
        stats_text += "\n"
    
    # Add comparative rankings
    stats_text += "\n" + "─" * 100 + "\n"
    most_active = max(users, key=lambda u: user_stats[u]['total_changes'])
    stats_text += f"[*] Most Active: User {most_active}  |  "
    
    if 'frequency' in params:
        freq_dom = {u: user_stats[u]['time_in_control'].get('frequency', 0) for u in users}
        most_dom = max(freq_dom, key=freq_dom.get)
        stats_text += f"[#] Most Dominant: User {most_dom}  |  "
    
    most_resp = max(users, key=lambda u: user_stats[u]['changes_per_minute'])
    stats_text += f"[!] Most Responsive: User {most_resp}"
    
    # Display engagement bar chart
    stats_text += "\n\nEngagement Distribution:\n"
    total_changes_all = sum(user_stats[u]['total_changes'] for u in users)
    for user in sorted(users):
        pct = (user_stats[user]['total_changes'] / total_changes_all * 100) if total_changes_all > 0 else 0
        bar_length = int(pct / 2)
        bar = '█' * bar_length + '░' * (50 - bar_length)
        stats_text += f"  User {user}: {bar} {pct:.1f}%\n"
    
    ax_stats.text(0.02, 0.98, stats_text, 
                 transform=ax_stats.transAxes,
                 verticalalignment='top',
                 fontfamily='monospace',
                 fontsize=8,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

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
                            ax.annotate('FIGHTING!', 
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
    
    # Apply zoom state without changing it
    if full_view[0]:
        for ax in axes:
            ax.set_xlim(t_min, t_max)
    else:
        # Use time_window_start which is always available
        start = time_window_start[0]
        for ax in axes:
            ax.set_xlim(start, start + 50)
    
    # Update stats display
    update_stats_display()
    
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
ax_slider_time = plt.axes([0.08, 0.05, 0.55, 0.02])
slider_time = Slider(ax_slider_time, 'Time Window', t_min, t_max-50, valinit=t_min, orientation='horizontal')

def update_time(val):
    if not full_view[0]:  # Only update if not in full view mode
        time_window_start[0] = slider_time.val
        start = time_window_start[0]
        for ax in axes:
            ax.set_xlim(start, start + 50)
        fig.canvas.draw_idle()

slider_time.on_changed(update_time)

# Parameter scroll slider (to scroll up/down between graphs)
ax_slider_param = plt.axes([0.08, 0.02, 0.3, 0.015])
slider_param = Slider(ax_slider_param, 'Scroll Params', 0, max(0, len([p for p in params if selected_params[p]]) - visible_count), 
                      valinit=0, valstep=1)

def update_param_scroll(val):
    current_top_param[0] = int(slider_param.val)
    draw_visible()

slider_param.on_changed(update_param_scroll)

# Button to toggle full/zoom view (moved to avoid overlap)
ax_button = plt.axes([0.65, 0.05, 0.10, 0.02])
button = Button(ax_button, 'Full View', color='lightgray', hovercolor='0.85')

def toggle_full(event):
    full_view[0] = not full_view[0]
    if full_view[0]:
        for ax in axes: 
            ax.set_xlim(t_min, t_max)
        button.label.set_text("Zoom View")
        # Disable and grey out the time slider
        slider_time.set_active(False)
        slider_time.poly.set_facecolor('0.85')  # Grey color
        slider_time.poly.set_alpha(0.5)
    else:
        start = time_window_start[0]
        for ax in axes: 
            ax.set_xlim(start, start + 50)
        button.label.set_text("Full View")
        # Re-enable the time slider
        slider_time.set_active(True)
        slider_time.poly.set_facecolor('lightblue')  # Active color
        slider_time.poly.set_alpha(1.0)
    fig.canvas.draw_idle()

button.on_clicked(toggle_full)

# ========== STEP 3.5: PAGE NAVIGATION (Graphs <-> Carefulness Dashboard) ==========
current_page = [0]  # 0 = Graphs, 1 = Carefulness Dashboard
radio_objs = {'ax': None, 'widget': None}  # track the radio button so we can remove it later

# dedicated axes for carefulness page
ax_care = fig.add_axes([0.08, 0.15, 0.8, 0.75])
ax_care.axis('off')

def draw_carefulness(uid=None):
    """Draw the carefulness report (all users or a single one)."""
    ax_care.clear()
    ax_care.axis('off')
    text = "🤖 CONTROL CAREFULNESS ANALYSIS\n" + "─" * 90 + "\n\n"
    users_to_show = [uid] if uid is not None else sorted(users)
    for user in users_to_show:
        stats = user_stats[user]
        text += f"👤 User {user}\n"
        for param, c in stats['carefulness'].items():
            if np.isnan(c['avg_step']):
                text += f"   • {param.capitalize()}: Insufficient data\n"
            else:
                color = "🟢" if c['behavior_type'] == "Careful" else "🔴"
                text += (f"   • {param.capitalize():<15} "
                         f"Avg Δ={c['avg_step']:.3f} | Max Δ={c['max_step']:.3f} | "
                         f"Careful Steps={c['careful_ratio']:.1f}% | {color} {c['behavior_type']}\n")
        text += "\n"

    text += "─" * 90 + "\nSUMMARY\n"
    for user in sorted(users):
        behaviors = [v['behavior_type'] for v in user_stats[user]['carefulness'].values()
                     if v['behavior_type'] not in ['Insufficient data']]
        if not behaviors:
            text += f"   User {user}: Insufficient data\n"
            continue
        careful_count = sum(b == 'Careful' or 'no variation' in b for b in behaviors)
        ratio = careful_count / len(behaviors) * 100
        label = "🟢 Careful" if ratio >= 70 else "🔴 Reckless"
        text += f"   User {user}: {label} ({ratio:.1f}% careful changes)\n"

    ax_care.text(0.02, 0.98, text,
                 transform=ax_care.transAxes,
                 verticalalignment='top',
                 fontfamily='monospace',
                 fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='ivory', alpha=0.8))
    fig.canvas.draw_idle()

def show_page(page_index):
    """Switch between main graphs and carefulness dashboard."""
    current_page[0] = page_index

    if page_index == 0:
        # === Show graph view ===
        for ax in axes:
            ax.set_visible(True)
        for ctl in [ax_user_check, ax_param_check, ax_stats]:
            ctl.set_visible(True)
        ax_care.set_visible(False)

        # remove radio if it exists
        if radio_objs['ax'] is not None:
            radio_objs['ax'].remove()
            radio_objs['ax'] = None
            radio_objs['widget'] = None

    elif page_index == 1:
        # === Show carefulness view ===
        for ax in axes:
            ax.set_visible(False)
        for ctl in [ax_user_check, ax_param_check, ax_stats]:
            ctl.set_visible(False)
        ax_care.set_visible(True)

        draw_carefulness()  # draw all users by default

        # --- create fresh radio selector every time ---
        radio_objs['ax'] = plt.axes([0.90, 0.45, 0.08, 0.2])
        labels = ['All'] + [f"User {u}" for u in users]
        radio_objs['widget'] = RadioButtons(radio_objs['ax'], labels)

        def on_user_select(label):
            if label == 'All':
                draw_carefulness()
            else:
                uid = int(label.split()[-1])
                draw_carefulness(uid)

        radio_objs['widget'].on_clicked(on_user_select)

    fig.canvas.draw_idle()

# Navigation buttons
ax_prev = plt.axes([0.77, 0.02, 0.10, 0.03])
btn_prev = Button(ax_prev, '⬅ Back', color='lightgray', hovercolor='0.85')

ax_next = plt.axes([0.89, 0.02, 0.10, 0.03])
btn_next = Button(ax_next, '➡ Next Page', color='lightgray', hovercolor='0.85')

def go_next(event):
    if current_page[0] == 0:
        show_page(1)
        btn_next.label.set_text('⬅ Graphs')
    else:
        show_page(0)
        btn_next.label.set_text('➡ Next Page')

def go_prev(event):
    show_page(0)
    btn_next.label.set_text('➡ Next Page')

btn_next.on_clicked(go_next)
btn_prev.on_clicked(go_prev)

# start on main graphs page
show_page(0)


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
