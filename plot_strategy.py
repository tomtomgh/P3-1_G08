#!/usr/bin/env python3
# --------------------------------------------------------------
# Beautiful Strategy Timeline Plot
# Speed Curve + Speed Trends + Strategy Timelines (per user)
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.widgets import Slider, Button
import time

from strategy_classifier.constants import ALL_STRATEGIES


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
    colors = build_strategy_colors()

    df = df.sort_values(["user_id", "seg_start"]).reset_index(drop=True)
    users = sorted(df["user_id"].unique())

    # Determine timeline extents
    t_min = min(df_trends["starttime"].min(), df_speed["timestamp_sec"].min())
    t_max = max(df_trends["endtime"].max(), df_speed["timestamp_sec"].max())

    # ----------------------------------------------------------
    # Build figure
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        len(users) + 1,
        1,
        figsize=(20, 3 * (len(users) + 1)),
        sharex=True
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
    # 2) USER PANELS
    # ----------------------------------------------------------
    used_strategies = set()

    for i, user in enumerate(users):
        ax = axes[i + 1]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=14, pad=6)

        df_u = df[df["user_id"] == user]

        for _, row in df_u.iterrows():
            s, e = row["seg_start"], row["seg_end"]
            strat, _ = dominant_strategy(row)

            used_strategies.add(strat)

            ax.axvspan(
                s, e,
                color=colors[strat],
                alpha=0.8
            )

        ax.set_yticks([])
        ax.set_ylabel(f"User {user}", rotation=0, labelpad=30)
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
        bbox_to_anchor=(0.83, 0.75),
        frameon=True,
        fontsize=11
    )

    # ----------------------------------------------------------
    # INTERACTIVE TIMELINE SLIDER
    # ----------------------------------------------------------
    # Create space for slider and buttons at the bottom
    plt.tight_layout(rect=(0, 0.08, 0.82, 1))
    
    # Add slider axis at bottom
    slider_ax = plt.axes([0.15, 0.04, 0.65, 0.02])
    time_slider = Slider(
        slider_ax,
        'Time',
        t_min,
        t_max,
        valinit=t_min,
        valstep=(t_max - t_min) / 1000,  # Smooth sliding
        color='lightblue'
    )
    
    # Draw vertical line on all axes at slider position
    vertical_lines = []
    for ax in axes:
        line = ax.axvline(t_min, color='black', linewidth=2, linestyle='-', alpha=0.8)
        vertical_lines.append(line)
    
    # Add timestamp text display
    time_text = fig.text(0.85, 0.045, '', fontsize=12, fontweight='bold', 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Autoplay state
    autoplay_state = {'playing': False, 'timer_id': None, 'last_real_time': None}
    
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
        
        fig.canvas.draw_idle()
    
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
    button_ax = plt.axes([0.02, 0.04, 0.08, 0.03])
    play_button = Button(button_ax, '▶ Play', color='lightgreen', hovercolor='green')
    play_button.on_clicked(toggle_autoplay)
    
    # Connect slider to update function
    time_slider.on_changed(update_timeline)
    
    # Initialize the display
    update_timeline(t_min)
    
    plt.show()

# --------------------------------------------------------------
# Run as script
# --------------------------------------------------------------
if __name__ == "__main__":
    plot_timeline()
