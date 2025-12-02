#!/usr/bin/env python3
# --------------------------------------------------------------
# plot_strategy_timeline.py
# VISUALISE SPEED + SPEED TRENDS + USER STRATEGY TIMELINES
# WITH SEGMENT TIME LABELS
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
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


def blend_with_white(color, blend=0.65):
    """Return a lighter version of the provided color."""
    try:
        r, g, b = mcolors.to_rgb(color)
    except ValueError:
        r, g, b = mcolors.to_rgb("#999999")
    blend = max(0.0, min(1.0, blend))
    return (1 - blend) * r + blend, (1 - blend) * g + blend, (1 - blend) * b + blend


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


class StrategyTimelineUI:
    """Interactive matplotlib UI to scrub through strategies in time."""

    def __init__(self, df, df_trends, df_speed, strat_colors):
        self.df = df.sort_values(["user_id", "seg_start"]).copy()
        dom = self.df.apply(
            lambda row: pd.Series(dominant_strategy(row), index=["dominant_strategy", "dominant_prob"]),
            axis=1,
        )
        self.df[["dominant_strategy", "dominant_prob"]] = dom
        self.df["dominant_strategy"] = self.df["dominant_strategy"].fillna("unknown")

        self.df_trends = df_trends
        self.df_speed = df_speed
        self.strat_colors = strat_colors.copy()
        self.strat_colors.setdefault("unknown", "#999999")

        self.users = sorted(self.df["user_id"].unique())
        self.user_segments = self._build_user_segments()

        self.t_min = float(min(self.df_trends["starttime"].min(), self.df_speed["timestamp_sec"].min()))
        self.t_max = float(max(self.df_trends["endtime"].max(), self.df_speed["timestamp_sec"].max()))

        self.current_time = self.t_min
        self.playing = False
        self.play_step = 0.5  # seconds per timer tick

        self.fig_main = None
        self.fig_info = None
        self.ax_speed = None
        self.speed_time_line = None
        self.user_axes = {}
        self.user_info_axes = {}
        self.user_info_texts = {}
        self.user_info_timeline_axes = {}
        self.user_info_time_lines = {}
        self.user_time_lines = {}
        self.slider = None
        self.play_buttons = []
        self.timer = None
        self.time_text_main = None
        self.time_text_info = None

        self._build_ui()
        self._update_time(self.current_time)

    def _build_user_segments(self):
        segments = {}
        for user, df_user in self.df.groupby("user_id"):
            entries = []
            for _, row in df_user.iterrows():
                entries.append(
                    {
                        "start": float(row["seg_start"]),
                        "end": float(row["seg_end"]),
                        "strategy": row["dominant_strategy"],
                    }
                )
            segments[user] = entries
        return segments

    def _build_ui(self):
        num_rows = len(self.users) + 1
        self.fig_main = plt.figure(figsize=(18, 3.1 * num_rows))
        self.fig_main.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.16, hspace=0.55)
        grid_main = self.fig_main.add_gridspec(num_rows, 1, hspace=0.45)

        self.time_text_main = self.fig_main.text(0.07, 0.955, "", fontsize=13, fontweight="bold")

        # Build speed axis (top row)
        self.ax_speed = self.fig_main.add_subplot(grid_main[0, 0])
        self.ax_speed.set_title("Global Speed Timeline", fontsize=14)
        for _, row in self.df_trends.iterrows():
            color = (
                "gray"
                if row["trend"] == "dull"
                else "lightcoral"
                if row["trend"] == "decreasing"
                else "lightgreen"
            )
            self.ax_speed.axvspan(row["starttime"], row["endtime"], alpha=0.25, color=color)

        self.ax_speed.plot(
            self.df_speed["timestamp_sec"],
            self.df_speed["speed_px/s"],
            color="steelblue",
            linewidth=2,
            label="speed_px/s",
        )
        self.ax_speed.set_ylabel("Speed (px/s)")
        self.ax_speed.set_xlim(self.t_min, self.t_max)
        self.ax_speed.grid(alpha=0.25)
        self.ax_speed.legend(loc="upper right")
        self.speed_time_line = self.ax_speed.axvline(self.t_min, color="black", linestyle="--", linewidth=1.5)

        # Build user axes
        for idx, user in enumerate(self.users):
            ax = self.fig_main.add_subplot(grid_main[idx + 1, 0], sharex=self.ax_speed)
            ax.set_title(f"User {user} Strategy Timeline", loc="left", fontsize=12)
            for seg in self.user_segments[user]:
                color = self.strat_colors.get(seg["strategy"], "#cccccc")
                ax.axvspan(seg["start"], seg["end"], color=color, alpha=0.7)
                center = (seg["start"] + seg["end"]) / 2
                ax.text(
                    center,
                    0.55,
                    seg["strategy"].replace("_", "\n"),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )
                ax.text(
                    center,
                    0.16,
                    f"{fmt_time(seg['start'])} → {fmt_time(seg['end'])}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
            ax.set_ylim(0, 1)
            ax.set_yticks([])
            ax.set_ylabel(f"User {user}")
            time_line = ax.axvline(self.t_min, color="black", linestyle="--", linewidth=1.1)
            self.user_time_lines[user] = time_line
            self.user_axes[user] = ax

        self.fig_info = plt.figure(figsize=(7.5, 2.3 * len(self.users) + 1.7))
        self.fig_info.subplots_adjust(left=0.07, right=0.95, top=0.82, bottom=0.18, hspace=0.4)
        info_grid = self.fig_info.add_gridspec(len(self.users), 1, hspace=0.35)
        self.fig_info.suptitle("User Strategy Status", fontsize=14, fontweight="bold")
        self.time_text_info = self.fig_info.text(0.07, 0.86, "", fontsize=11, fontweight="bold")

        legend_ax = self.fig_info.add_axes([0.08, 0.02, 0.85, 0.12])
        legend_ax.axis("off")
        legend_ax.set_facecolor("#f7f7f7")
        legend_ax.text(0.01, 0.75, "Strategy Legend", fontsize=10, fontweight="bold", transform=legend_ax.transAxes)
        legend_entries = sorted(self.strat_colors.items())
        per_row = 4
        row_height = 0.25
        for idx, (strat, color) in enumerate(legend_entries):
            row = idx // per_row
            col = idx % per_row
            x = 0.02 + col * 0.23
            y = 0.45 - row * row_height
            legend_ax.add_patch(Rectangle((x, y), 0.035, 0.18, facecolor=color, edgecolor="#555555"))
            legend_ax.text(x + 0.045, y + 0.02, strat, fontsize=8, va="bottom")

        for idx, user in enumerate(self.users):
            info_ax = self.fig_info.add_subplot(info_grid[idx, 0])
            info_ax.axis("off")
            info_ax.set_facecolor(blend_with_white("#222222", 0.9))
            info_ax.text(0.02, 0.82, f"User {user}", fontsize=11, fontweight="bold", transform=info_ax.transAxes)
            prev_text = info_ax.text(0.04, 0.57, "", transform=info_ax.transAxes, fontsize=9)
            curr_text = info_ax.text(0.04, 0.34, "", transform=info_ax.transAxes, fontsize=11, fontweight="bold")
            next_text = info_ax.text(0.04, 0.11, "", transform=info_ax.transAxes, fontsize=9)
            self.user_info_texts[user] = {"prev": prev_text, "curr": curr_text, "next": next_text}
            self.user_info_axes[user] = info_ax
            mini_ax = info_ax.inset_axes([0.58, 0.2, 0.38, 0.55])
            mini_ax.set_xlim(self.t_min, self.t_max)
            mini_ax.set_ylim(0, 1)
            mini_ax.set_xticks([])
            mini_ax.set_yticks([])
            mini_ax.set_facecolor("#fefefe")
            for spine in mini_ax.spines.values():
                spine.set_color("#999999")
            for seg in self.user_segments[user]:
                color = self.strat_colors.get(seg["strategy"], "#cccccc")
                mini_ax.axvspan(seg["start"], seg["end"], color=color, alpha=0.9)
            line = mini_ax.axvline(self.t_min, color="#222222", linestyle="--", linewidth=1.1)
            self.user_info_timeline_axes[user] = mini_ax
            self.user_info_time_lines[user] = line

        slider_ax = self.fig_main.add_axes([0.12, 0.055, 0.65, 0.035])
        self.slider = Slider(
            slider_ax,
            "Global Time (s)",
            self.t_min,
            self.t_max,
            valinit=self.t_min,
            valfmt="%0.1f",
        )
        self.slider.on_changed(self._on_slider_change)

        button_ax = self.fig_main.add_axes([0.8, 0.05, 0.12, 0.045])
        play_button_main = Button(button_ax, "Play", hovercolor="#dddddd")
        play_button_main.on_clicked(self._toggle_play)
        self.play_buttons.append(play_button_main)

        button_info_ax = self.fig_info.add_axes([0.62, 0.92, 0.3, 0.06])
        play_button_info = Button(button_info_ax, "Play", hovercolor="#dddddd")
        play_button_info.on_clicked(self._toggle_play)
        self.play_buttons.append(play_button_info)

        self.timer = self.fig_main.canvas.new_timer(interval=200)
        self.timer.add_callback(self._tick)

        ticks, labels = self._build_xticks()
        self.ax_speed.set_xticks(ticks)
        self.ax_speed.set_xticklabels(labels, rotation=30)

    def _build_xticks(self):
        ticks = []
        labels = []
        span = max(1, int(self.t_max - self.t_min))
        step = max(1, span // 12)
        for sec in range(int(self.t_min), int(self.t_max) + 1, step):
            ticks.append(sec)
            labels.append(fmt_time(sec))
        return ticks, labels

    def _set_play_button_labels(self, text):
        for btn in self.play_buttons:
            btn.label.set_text(text)
            if btn.ax.figure:
                btn.ax.figure.canvas.draw_idle()

    def _on_slider_change(self, value):
        self._update_time(float(value), from_slider=True)

    def _toggle_play(self, _event=None):
        self.playing = not self.playing
        if self.playing:
            self._set_play_button_labels("Pause")
            self.timer.start()
        else:
            self._set_play_button_labels("Play")
            self.timer.stop()
        self.fig_main.canvas.draw_idle()
        if self.fig_info:
            self.fig_info.canvas.draw_idle()

    def _tick(self):
        if not self.playing:
            return
        new_time = self.current_time + self.play_step
        if new_time >= self.t_max:
            new_time = self.t_max
            self.playing = False
            self.timer.stop()
            self._set_play_button_labels("Play")
        self._update_time(new_time)

    def _update_time(self, value, from_slider=False):
        self.current_time = max(self.t_min, min(self.t_max, value))
        self.speed_time_line.set_xdata([self.current_time, self.current_time])
        for user, line in self.user_time_lines.items():
            line.set_xdata([self.current_time, self.current_time])
        for user, line in self.user_info_time_lines.items():
            line.set_xdata([self.current_time, self.current_time])
        time_label = f"Current Time: {fmt_time(self.current_time)}"
        if self.time_text_main is not None:
            self.time_text_main.set_text(time_label)
        if self.time_text_info is not None:
            self.time_text_info.set_text(time_label)
        self._update_info_panels()
        if not from_slider and self.slider is not None:
            self.slider.set_val(self.current_time)
        self.fig_main.canvas.draw_idle()
        if self.fig_info:
            self.fig_info.canvas.draw_idle()

    def _update_info_panels(self):
        for user in self.users:
            prev_seg, curr_seg, next_seg = self._describe_user_time(user, self.current_time)
            info_texts = self.user_info_texts[user]
            info_texts["prev"].set_text(self._format_line("Prev", prev_seg))
            info_texts["prev"].set_color(self._strategy_color(prev_seg))
            info_texts["curr"].set_text(self._format_line("Now", curr_seg))
            info_texts["curr"].set_color(self._strategy_color(curr_seg))
            info_texts["next"].set_text(self._format_line("Next", next_seg))
            info_texts["next"].set_color(self._strategy_color(next_seg))
            color = "#f0f0f0"
            if curr_seg:
                base = self.strat_colors.get(curr_seg["strategy"], "#cccccc")
                color = blend_with_white(base, 0.75)
            self.user_info_axes[user].set_facecolor(color)

    def _describe_user_time(self, user, time_value):
        segments = self.user_segments.get(user, [])
        prev_seg = None
        curr_seg = None
        next_seg = None
        for seg in segments:
            if seg["end"] <= time_value:
                prev_seg = seg
                continue
            if seg["start"] <= time_value < seg["end"]:
                curr_seg = seg
                continue
            if seg["start"] > time_value:
                next_seg = seg
                break
        if curr_seg is None:
            if segments and time_value < segments[0]["start"]:
                next_seg = segments[0]
            elif segments and time_value >= segments[-1]["end"]:
                prev_seg = segments[-1]
        return prev_seg, curr_seg, next_seg

    def _strategy_color(self, seg):
        if not seg:
            return "#333333"
        return self.strat_colors.get(seg["strategy"], "#333333")

    def _format_line(self, label, seg):
        if not seg:
            return f"{label}: —"
        return f"{label}: {seg['strategy']} ({fmt_time(seg['start'])} → {fmt_time(seg['end'])})"

    def show(self):
        plt.show()


def plot_timeline():
    df = load_predictions()
    df_trends = load_speed_trends()
    df_speed = load_speed_series()
    strat_colors = build_strategy_colors()

    viewer = StrategyTimelineUI(df, df_trends, df_speed, strat_colors)
    viewer.show()


if __name__ == "__main__":
    plot_timeline()
