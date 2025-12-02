#!/usr/bin/env python3
# --------------------------------------------------------------
# Beautiful Strategy Timeline Plot
# Speed Curve + Speed Trends + Strategy Timelines (per user)
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

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

    plt.tight_layout(rect=(0, 0, 0.82, 1))
    plt.show()

# --------------------------------------------------------------
# Run as script
# --------------------------------------------------------------
if __name__ == "__main__":
    plot_timeline()
