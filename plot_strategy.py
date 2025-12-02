#!/usr/bin/env python3
# --------------------------------------------------------------
# plot_strategy_timeline.py
# VISUALISE SPEED + SPEED TRENDS + USER STRATEGY TIMELINES
# WITH SEGMENT TIME LABELS
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
    df = pd.read_csv(Path("speed/speed.csv"))
    return df


def build_strategy_colors():
    import matplotlib.colors as mcolors
    base = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    return {strat: base[i % len(base)] for i, strat in enumerate(ALL_STRATEGIES)}


def dominant_strategy(row):
    best = None
    best_p = -1
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in row and row[col] > best_p:
            best_p = row[col]
            best = strat
    return best, best_p


def fmt_time(sec):
    """Format seconds into mm:ss or hh:mm:ss depending on size."""
    if sec < 3600:
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"
    else:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = int(sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"


def plot_timeline():
    df = load_predictions()
    df_trends = load_speed_trends()
    df_speed = load_speed_series()
    strat_colors = build_strategy_colors()

    df = df.sort_values(["user_id", "seg_start"])
    users = sorted(df["user_id"].unique())

    # Determine time range
    t_min = min(df_trends["starttime"].min(), df_speed["timestamp_sec"].min())
    t_max = max(df_trends["endtime"].max(), df_speed["timestamp_sec"].max())

    # Multi-panel figure
    fig, axes = plt.subplots(
        len(users) + 1,
        1,
        figsize=(20, 3 * (len(users) + 1)),
        sharex=True
    )

    # ----------------------------------------------------------
    # Top panel: speed trend + speed line
    # ----------------------------------------------------------
    ax_speed = axes[0]
    ax_speed.set_title("Robot Speed Timeline", fontsize=14)

    # Trend blocks
    for _, row in df_trends.iterrows():
        color = "gray" if row["trend"] == "dull" else \
                "lightcoral" if row["trend"] == "decreasing" else "lightgreen"
        ax_speed.axvspan(row["starttime"], row["endtime"], alpha=0.25, color=color)

    # Speed curve
    ax_speed.plot(
        df_speed["timestamp_sec"],
        df_speed["speed_px/s"],
        color="blue",
        linewidth=2,
        label="speed_px/s"
    )

    ax_speed.set_ylabel("Speed\n(px/s)")
    ax_speed.legend(loc="upper right")
    ax_speed.grid(alpha=0.25)

    # ----------------------------------------------------------
    # USER STRATEGY PANELS
    # ----------------------------------------------------------
    for idx, user in enumerate(users):
        ax = axes[idx + 1]
        ax.set_title(f"User {user} Strategy Timeline")

        df_u = df[df["user_id"] == user]

        for _, row in df_u.iterrows():
            s = row["seg_start"]
            e = row["seg_end"]

            strat, _ = dominant_strategy(row)
            color = strat_colors[strat]

            # Colored block
            ax.axvspan(s, e, color=color, alpha=0.7)

            # Strategy text (middle)
            ax.text(
                (s + e) / 2,
                0.65,
                strat.replace("_", "\n"),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

            # Time text below block
            ax.text(
                (s + e) / 2,
                0.30,
                f"{fmt_time(s)} → {fmt_time(e)}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

        ax.set_yticks([])
        ax.set_ylabel(f"User {user}")

    # ----------------------------------------------------------
    # Format X-axis with readable time
    # ----------------------------------------------------------
    plt.xlabel("Time (seconds)")

    # show minute:second ticks
    ticks = []
    labels = []
    step = max(1, int((t_max - t_min) // 12))  # ~12 ticks
    for sec in range(int(t_min), int(t_max) + 1, step):
        ticks.append(sec)
        labels.append(fmt_time(sec))

    plt.xticks(ticks, labels, rotation=45)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_timeline()
