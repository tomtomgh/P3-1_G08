#!/usr/bin/env python3
# --------------------------------------------------------------
# plot_strategy_timeline.py
# VISUALISE SPEED AND STRATEGY TIMELINES
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Strategy constants
from strategy_classifier.constants import ALL_STRATEGIES


# --------------------------------------------------------------
# Load outputs from run_all.py
# --------------------------------------------------------------
def load_predictions():
    df = pd.read_csv("segment_strategy_predictions.csv")
    return df


# --------------------------------------------------------------
# Load speed/trends.csv (for speed segments & speed curves)
# --------------------------------------------------------------
def load_speed_trends():
    speed_csv = Path("speed/trends.csv")
    return pd.read_csv(speed_csv)


# --------------------------------------------------------------
# Build color map for strategies
# --------------------------------------------------------------
def build_strategy_colors():
    import matplotlib.colors as mcolors
    base_colors = list(mcolors.TABLEAU_COLORS.values()) + \
                  list(mcolors.CSS4_COLORS.values())

    colors = {}
    for i, strat in enumerate(ALL_STRATEGIES):
        colors[strat] = base_colors[i % len(base_colors)]
    return colors


# --------------------------------------------------------------
# Pick the highest-probability strategy for each segment row
# --------------------------------------------------------------
def dominant_strategy(row):
    best_strat = None
    best_prob = -1

    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in row:
            if row[col] > best_prob:
                best_prob = row[col]
                best_strat = strat

    return best_strat, best_prob


# --------------------------------------------------------------
# Main visualization function
# --------------------------------------------------------------
def plot_timeline():
    df = load_predictions()
    df_speed = load_speed_trends()
    strat_colors = build_strategy_colors()

    # Ensure correct sorting
    df = df.sort_values(["user_id", "seg_start"]).reset_index(drop=True)

    users = sorted(df["user_id"].unique())
    n_users = len(users)

    # Get time range from speed CSV
    t_min = df_speed["starttime"].min()
    t_max = df_speed["endtime"].max()

    # ----------------------------------------------------------
    # MULTI-PANEL FIGURE
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        n_users + 1,
        1,
        figsize=(18, 3 * (n_users + 1)),
        sharex=True
    )

    # ----------------------------------------------------------
    # TOP PANEL: PLOT SPEED TREND INTERVALS
    # ----------------------------------------------------------
    ax_speed = axes[0]
    ax_speed.set_title("Speed Trend Timeline", fontsize=14)

    for _, row in df_speed.iterrows():
        ax_speed.axvspan(
            row["starttime"],
            row["endtime"],
            alpha=0.25,
            label=row["trend"],
            color="gray" if row["trend"] == "dull" else
                  "lightcoral" if row["trend"] == "decreasing" else
                  "lightgreen"
        )

    ax_speed.set_ylabel("Speed state")
    ax_speed.set_xlim(t_min, t_max)

    # avoid duplicate legends
    handles, labels = ax_speed.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax_speed.legend(by_label.values(), by_label.keys(), loc="upper right")

    # ----------------------------------------------------------
    # USER PANELS
    # ----------------------------------------------------------
    for idx, user in enumerate(users):
        ax = axes[idx + 1]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=13)

        df_u = df[df["user_id"] == user]

        for _, row in df_u.iterrows():
            start = row["seg_start"]
            end = row["seg_end"]
            strat, prob = dominant_strategy(row)
            color = strat_colors[strat]

            ax.axvspan(
                start,
                end,
                alpha=0.7,
                color=color,
                label=strat
            )
            # Optionally label the strategy in the block
            ax.text(
                (start + end) / 2,
                0.5,
                strat.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=8,
                color="black"
            )

        ax.set_yticks([])
        ax.set_ylabel(f"User {user}")

        # One legend per user panel (condensed)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=7)

    plt.xlabel("Time (seconds)")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# Run as script
# --------------------------------------------------------------
if __name__ == "__main__":
    plot_timeline()
