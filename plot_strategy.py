#!/usr/bin/env python3
# --------------------------------------------------------------
# plot_strategy_timeline.py
# VISUALISE SPEED + SPEED TRENDS + USER STRATEGY TIMELINES
# WITH STRATEGY LEGEND ON RIGHT SIDE
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from strategy_classifier.constants import ALL_STRATEGIES


def load_predictions():
    return pd.read_csv("segment_strategy_predictions.csv")


def load_speed_trends():
    return pd.read_csv(Path("speed/trends.csv"))


def load_speed_series():
    path = Path("speed/speed.csv")
    if not path.exists():
        raise FileNotFoundError("speed/speed.csv not found!")
    return pd.read_csv(path)


def build_strategy_colors():
    import matplotlib.colors as mcolors
    base = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    return {s: base[i % len(base)] for i, s in enumerate(ALL_STRATEGIES)}


def dominant_strategy(row):
    best_name = None
    best_prob = -1
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in row and row[col] > best_prob:
            best_prob = row[col]
            best_name = strat
    return best_name, best_prob


def plot_timeline():
    df = load_predictions()
    df_trends = load_speed_trends()
    df_speed = load_speed_series()
    strat_colors = build_strategy_colors()

    df = df.sort_values(["user_id", "seg_start"]).reset_index(drop=True)
    users = sorted(df["user_id"].unique())

    t_min = min(df_trends["starttime"].min(), df_speed["timestamp_sec"].min())
    t_max = max(df_trends["endtime"].max(), df_speed["timestamp_sec"].max())

    fig, axes = plt.subplots(len(users) + 1, 1, figsize=(22, 3 * (len(users) + 1)), sharex=True)

    # ----------------------------------------------------------
    # TOP PANEL: SPEED + TRENDS
    # ----------------------------------------------------------
    ax_speed = axes[0]
    ax_speed.set_title("Robot Speed Timeline (Trends + Speed)", fontsize=14)

    for _, row in df_trends.iterrows():
        c = "gray" if row["trend"] == "dull" else "lightcoral" if row["trend"] == "decreasing" else "lightgreen"
        ax_speed.axvspan(row["starttime"], row["endtime"], alpha=0.25, color=c)

    ax_speed.plot(
        df_speed["timestamp_sec"],
        df_speed["speed_px/s"],
        color="blue",
        linewidth=2,
        label="speed_px/s"
    )

    # KEEP the speed legend
    ax_speed.legend(loc="upper right")
    ax_speed.set_ylabel("Speed\n(px/s)")
    ax_speed.set_xlim(t_min, t_max)
    ax_speed.grid(alpha=0.3)

    # ----------------------------------------------------------
    # USER PANELS
    # ----------------------------------------------------------
    for i, user in enumerate(users):
        ax = axes[i + 1]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=13)

        df_u = df[df["user_id"] == user]

        for _, row in df_u.iterrows():
            s, e = row["seg_start"], row["seg_end"]
            strat, _ = dominant_strategy(row)

            ax.axvspan(s, e, color=strat_colors[strat], alpha=0.7)
            ax.text((s + e) / 2, 0.5, strat.replace("_", "\n"),
                    ha='center', va='center', fontsize=8, color='black')

        ax.set_yticks([])
        ax.set_ylabel(f"User {user}")

    plt.xlabel("Time (seconds)")
    plt.tight_layout(rect=(0, 0, 0.82, 1))  # leave space on RIGHT

    # ----------------------------------------------------------
    # STRATEGY LEGEND ON THE RIGHT SIDE
    # ----------------------------------------------------------
    handles = []
    labels = []

    for strat, color in strat_colors.items():
        patch = plt.Line2D([0], [0], color=color, linewidth=12)
        handles.append(patch)
        labels.append(strat)

    fig.legend(
        handles,
        labels,
        title="Strategies",
        loc="upper left",
        bbox_to_anchor=(0.84, 0.75),  # right-hand side placement
        fontsize=9,
        frameon=True,
        ncol=1,
    )

    plt.show()


if __name__ == "__main__":
    plot_timeline()
