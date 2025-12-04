#!/usr/bin/env python3
# --------------------------------------------------------------
# Beautiful Strategy Timeline Plot
# Speed Curve + Speed Trends + Strategy Timelines (per user)
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from bisect import bisect_right
from matplotlib.widgets import Slider, Button
import time

from strategy_classifier.constants import ALL_STRATEGIES, PARAMS
from log_parser import parse_session

SHARED_PARAMS = {"frequency"}


# --------------------------------------------------------------
# Loaders
# --------------------------------------------------------------
def load_predictions():
    return pd.read_csv("segment_strategy_predictions.csv")


def load_speed_trends():
    return pd.read_csv(Path("speed/trends.csv"))


def load_speed_series():
    path = Path("speed/speed.csv")
    if not path.exists():
        raise FileNotFoundError("speed/speed.csv not found!")
    return pd.read_csv(path)


def load_parameter_events():
    """
    Parse raw logs to gather parameter changes for all users.
    """
    log_dir = Path("logs")
    log_paths = sorted(log_dir.glob("*.log"))
    if not log_paths:
        raise FileNotFoundError("No .log files found inside logs/ for parameter data")
    events = parse_session(log_paths, session_id="plot_view")
    if not events:
        raise RuntimeError("No parameter events parsed from logs")
    df = pd.DataFrame(events)
    df = df[df["param"].isin(PARAMS)].copy()
    df.rename(columns={"time": "time_sec"}, inplace=True)
    df["user"] = df["user"].astype(int)
    return df.sort_values("time_sec").reset_index(drop=True)


# --------------------------------------------------------------
# Parameter timelines
# --------------------------------------------------------------
def build_parameter_timelines(df, shared_params=None):
    """
    Build lookup of parameter values over time for each user and shared params.
    """
    shared_params = set(shared_params or [])
    per_user = {}
    for user, df_user in df.groupby("user"):
        param_map = {}
        for param in PARAMS:
            if param in shared_params:
                continue
            df_param = df_user[df_user["param"] == param].sort_values("time_sec")
            if df_param.empty:
                continue
            param_map[param] = {
                "times": df_param["time_sec"].tolist(),
                "values": df_param["value"].tolist(),
            }
        if param_map:
            per_user[int(user)] = param_map
    shared = {}
    for param in shared_params:
        df_param = df[df["param"] == param].sort_values("time_sec")
        if df_param.empty:
            continue
        shared[param] = {
            "times": df_param["time_sec"].tolist(),
            "values": df_param["value"].tolist(),
            "users": df_param["user"].tolist(),
        }
    return per_user, shared


def _lookup_timeline_value(timeline, current_time, default_value=0.0):
    """
    Return the latest value from a timeline, clamped to the range of data.
    """
    if not timeline:
        return default_value
    times = timeline["times"]
    values = timeline["values"]
    if not times:
        return default_value
    idx = bisect_right(times, current_time) - 1
    if idx < 0:
        return values[0]
    if idx >= len(values):
        return values[-1]
    return values[idx]


def get_param_snapshot(user, current_time, user_timelines, shared_timelines, default_value=0.0):
    """
    Return a dict of param -> latest value for a user at the given time.
    Shared parameters are resolved globally.
    """
    snapshot = {}
    user_timeline = user_timelines.get(user, {})
    for param in PARAMS:
        if param in shared_timelines:
            timeline = shared_timelines.get(param)
            snapshot[param] = _lookup_timeline_value(timeline, current_time, default_value)
        else:
            timeline = user_timeline.get(param)
            snapshot[param] = _lookup_timeline_value(timeline, current_time, default_value)
    return snapshot


def format_param_label(user, snapshot):
    """
    Format multi-line label showing user and latest parameter values.
    """
    lines = [f"User {user}"]
    for param in PARAMS:
        pretty = param.title()
        value = snapshot.get(param, 0.0)
        lines.append(f"{pretty}: {value:.1f}")
    return "\n".join(lines)


# --------------------------------------------------------------
# Colors for strategies
# --------------------------------------------------------------
def build_strategy_colors():
    import matplotlib.colors as mcolors
    base = (
        list(mcolors.TABLEAU_COLORS.values()) +
        list(mcolors.XKCD_COLORS.values())
    )
    return {s: base[i % len(base)] for i, s in enumerate(ALL_STRATEGIES)}


# --------------------------------------------------------------
# Pick strategy with highest probability for a segment row
# --------------------------------------------------------------
def dominant_strategy(row):
    best_name = None
    best_prob = -1
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in row and row[col] > best_prob:
            best_prob = row[col]
            best_name = strat
    return best_name, best_prob


# --------------------------------------------------------------
# Main plotter
# --------------------------------------------------------------
def plot_timeline():
    df = load_predictions()
    df_trends = load_speed_trends()
    df_speed = load_speed_series()
    param_events = load_parameter_events()
    user_param_timelines, shared_param_timelines = build_parameter_timelines(
        param_events,
        shared_params=SHARED_PARAMS,
    )
    colors = build_strategy_colors()

    df = df.sort_values(["user_id", "seg_start"]).reset_index(drop=True)
    users = sorted(df["user_id"].unique())
    if not users:
        raise RuntimeError("No users found in predictions to plot.")

    user_index = {user: idx for idx, user in enumerate(users)}
    freq_timeline = shared_param_timelines.get("frequency", {})
    freq_events = []
    if freq_timeline:
        times = freq_timeline.get("times", [])
        actors = freq_timeline.get("users", [])
        for t, actor in zip(times, actors):
            if actor in users:
                freq_events.append((t, actor))
    freq_events.sort(key=lambda x: x[0])

    # Determine timeline extents (include parameter events for padding)
    t_candidates = [
        df_trends["starttime"].min(),
        df_speed["timestamp_sec"].min(),
    ]
    t_max_candidates = [
        df_trends["endtime"].max(),
        df_speed["timestamp_sec"].max(),
    ]
    if not param_events.empty:
        t_candidates.append(param_events["time_sec"].min())
        t_max_candidates.append(param_events["time_sec"].max())
    t_min = min(t_candidates)
    t_max = max(t_max_candidates)

    # ----------------------------------------------------------
    # Build figure
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        len(users) + 2,
        1,
        figsize=(18, 2.4 * (len(users) + 2)),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.8] + [1.0] * len(users)},
    )

    # ----------------------------------------------------------
    # 1) SPEED PANEL
    # ----------------------------------------------------------
    ax_speed = axes[0]
    ax_speed.set_title("Robot Speed Timeline", fontsize=16, pad=10)

    # Trend shading
    for _, row in df_trends.iterrows():
        trend_color = (
            "lightgray" if row["trend"] == "dull" else
            "lightcoral" if row["trend"] == "decreasing" else
            "lightgreen"
        )
        ax_speed.axvspan(row["starttime"], row["endtime"], alpha=0.25, color=trend_color)

    # Speed curve
    ax_speed.plot(
        df_speed["timestamp_sec"],
        df_speed["speed_px/s"],
        color="blue",
        linewidth=2,
        label="speed_px/s"
    )
    ax_speed.set_ylabel("Speed (px/s)")
    ax_speed.grid(alpha=0.3)

    # Keep only speed curve legend here
    ax_speed.legend(loc="upper right", fontsize=10)

    # ----------------------------------------------------------
    # 2) FREQUENCY EVENTS PANEL
    # ----------------------------------------------------------
    ax_freq = axes[1]
    ax_freq.set_title("Frequency Changes (who adjusted frequency)", fontsize=14, pad=6)
    if freq_events:
        times = [evt[0] for evt in freq_events]
        freq_y = [user_index[evt[1]] for evt in freq_events]
        freq_colors = [
            plt.cm.tab10(user_index[evt[1]] % 10) for evt in freq_events
        ]
        ax_freq.scatter(
            times,
            freq_y,
            c=freq_colors,
            s=30,
            alpha=0.9,
            edgecolors="black",
            linewidths=0.4,
        )
    ax_freq.set_yticks(list(range(len(users))))
    ax_freq.set_yticklabels([f"User {u}" for u in users])
    ax_freq.set_ylim(-0.5, len(users) - 0.5)
    ax_freq.set_ylabel("Frequency\nchanges", rotation=0, labelpad=34, fontsize=10)
    ax_freq.grid(axis="x", alpha=0.2)

    # ----------------------------------------------------------
    # 3) USER PANELS
    # ----------------------------------------------------------
    used_strategies = set()
    user_segments = {}  # Store segments for each user for later lookup

    for i, user in enumerate(users):
        ax = axes[i + 2]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=13, pad=4)

        df_u = df[df["user_id"] == user]
        user_segments[user] = []  # Store segments for this user

        for _, row in df_u.iterrows():
            s, e = row["seg_start"], row["seg_end"]
            strat, _ = dominant_strategy(row)

            used_strategies.add(strat)
            user_segments[user].append((s, e, strat))

            ax.axvspan(
                s, e,
                color=colors[strat],
                alpha=0.8
            )

        ax.set_yticks([])
        snapshot = get_param_snapshot(
            user,
            t_min,
            user_param_timelines,
            shared_param_timelines,
        )
        ax.set_ylabel(
            format_param_label(user, snapshot),
            rotation=0,
            labelpad=38,
            fontsize=9,
        )
        ax.grid(axis='x', alpha=0.2)

    # ----------------------------------------------------------
    # GLOBAL TIME AXIS
    # ----------------------------------------------------------
    axes[-1].set_xlabel("Time (seconds)", fontsize=14)
    axes[-1].set_xlim(t_min, t_max)

    # ----------------------------------------------------------
    # STRATEGY LEGEND (ONLY USED STRATEGIES)
    # ----------------------------------------------------------
    legend_handles = []
    legend_labels = []

    for strat in sorted(used_strategies):
        patch = plt.Line2D([0], [0], color=colors[strat], linewidth=12)
        legend_handles.append(patch)
        legend_labels.append(strat)

    fig.legend(
        legend_handles,
        legend_labels,
        title="Strategies Used",
        loc="upper left",
        bbox_to_anchor=(0.82, 0.8),
        frameon=True,
        fontsize=10
    )

    # ----------------------------------------------------------
    # INTERACTIVE TIMELINE SLIDER
    # ----------------------------------------------------------
    fig.subplots_adjust(left=0.08, right=0.8, top=0.97, bottom=0.13, hspace=0.45)

    # Add time slider axis
    slider_ax = plt.axes([0.12, 0.08, 0.6, 0.025])
    time_slider = Slider(
        slider_ax,
        'Time',
        t_min,
        t_max,
        valinit=t_min,
        valstep=(t_max - t_min) / 1000 if t_max > t_min else 1.0,
        color='lightblue'
    )

    # Add zoom slider axis
    zoom_ax = plt.axes([0.12, 0.045, 0.6, 0.025])
    total_duration = t_max - t_min if t_max > t_min else 1.0
    zoom_slider = Slider(
        zoom_ax,
        'Zoom',
        1.0,  # Min zoom: show full timeline
        20.0,  # Max zoom: 20x zoomed in
        valinit=1.0,
        valstep=0.5,
        color='lightcoral'
    )

    # Draw vertical line on all axes at slider position
    vertical_lines = []
    for ax in axes:
        line = ax.axvline(t_min, color='black', linewidth=2, linestyle='-', alpha=0.8)
        vertical_lines.append(line)

    # Add timestamp text display
    time_text = fig.text(0.84, 0.08, '', fontsize=11, fontweight='bold',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Zoom level text display
    zoom_text = fig.text(0.84, 0.05, '', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # Autoplay state
    autoplay_state = {'playing': False, 'timer_id': None, 'last_real_time': None}

    def update_view():
        """Update the view window based on current time and zoom level."""
        current_time = time_slider.val
        zoom_level = zoom_slider.val

        # Calculate visible window width based on zoom
        window_width = total_duration / zoom_level

        # Center the window around current time
        window_start = current_time - window_width / 2
        window_end = current_time + window_width / 2

        # Clamp to valid range
        if window_start < t_min:
            window_start = t_min
            window_end = min(t_min + window_width, t_max)
        elif window_end > t_max:
            window_end = t_max
            window_start = max(t_max - window_width, t_min)

        # Update all axes
        for ax in axes:
            ax.set_xlim(window_start, window_end)

        # Update zoom level display
        zoom_text.set_text(f'Zoom: {zoom_level:.1f}x')

        fig.canvas.draw_idle()

    def get_current_strategy(user, current_time):
        """Get the strategy for a user at the current time."""
        if user not in user_segments:
            return "No Strategy"

        for start, end, strategy in user_segments[user]:
            if start <= current_time <= end:
                return strategy

        return "No Strategy"

    def update_timeline(val):
        """Update vertical line position and timestamp when slider moves."""
        current_time = time_slider.val

        # Update all vertical lines
        for line in vertical_lines:
            line.set_xdata([current_time, current_time])

        # Format timestamp as HH:MM:SS
        hours = int(current_time // 3600)
        minutes = int((current_time % 3600) // 60)
        seconds = int(current_time % 60)
        milliseconds = int((current_time % 1) * 1000)

        time_text.set_text(f'{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}')

        # Update titles and parameter labels with current strategy for each user
        for i, user in enumerate(users):
            ax = axes[i + 2]
            current_strategy = get_current_strategy(user, current_time)
            ax.set_title(f"User {user}: {current_strategy}", fontsize=13, pad=4)
            snapshot = get_param_snapshot(
                user,
                current_time,
                user_param_timelines,
                shared_param_timelines,
            )
            ax.set_ylabel(
                format_param_label(user, snapshot),
                rotation=0,
                labelpad=38,
                fontsize=9,
            )

        # Update view to follow current time with current zoom
        update_view()

    def update_zoom(val):
        """Update zoom level and adjust view."""
        update_view()

    def autoplay_step():
        """Advance timeline by real-time seconds elapsed."""
        if not autoplay_state['playing']:
            return

        current_real_time = time.time()

        # Calculate elapsed real time since last update
        if autoplay_state['last_real_time'] is not None:
            elapsed = current_real_time - autoplay_state['last_real_time']
        else:
            elapsed = 0

        autoplay_state['last_real_time'] = current_real_time

        # Advance timeline by elapsed seconds
        new_time = time_slider.val + elapsed

        if new_time >= t_max:
            # Reached end, stop autoplay
            autoplay_state['playing'] = False
            play_button.label.set_text('▶ Play')
            time_slider.set_val(t_max)
            autoplay_state['last_real_time'] = None
            return

        time_slider.set_val(new_time)

        # Schedule next update (approximately 30 FPS for smooth animation)
        autoplay_state['timer_id'] = fig.canvas.new_timer(interval=33)
        autoplay_state['timer_id'].add_callback(autoplay_step)
        autoplay_state['timer_id'].start()

    def toggle_autoplay(event):
        """Toggle autoplay on/off."""
        if autoplay_state['playing']:
            # Stop autoplay
            autoplay_state['playing'] = False
            play_button.label.set_text('▶ Play')
            if autoplay_state['timer_id']:
                autoplay_state['timer_id'].stop()
            autoplay_state['last_real_time'] = None
        else:
            # Start autoplay
            autoplay_state['playing'] = True
            play_button.label.set_text('⏸ Pause')
            autoplay_state['last_real_time'] = time.time()
            autoplay_step()

    # Add Play/Pause button
    button_ax = plt.axes([0.02, 0.07, 0.08, 0.035])
    play_button = Button(button_ax, '▶ Play', color='lightgreen', hovercolor='green')
    play_button.on_clicked(toggle_autoplay)

    # Connect sliders to update functions
    time_slider.on_changed(update_timeline)
    zoom_slider.on_changed(update_zoom)

    # Initialize the display
    update_timeline(t_min)

    plt.show()

# --------------------------------------------------------------
# Run as script
# --------------------------------------------------------------
if __name__ == "__main__":
    plot_timeline()
