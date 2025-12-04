#!/usr/bin/env python3
# --------------------------------------------------------------
# plot_strategy_timeline.py
# VISUALISE SPEED + SPEED TRENDS + USER STRATEGY TIMELINES
# WITH SEGMENT TIME LABELS
# --------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
<<<<<<< Updated upstream
import matplotlib.colors as mcolors
=======
from pathlib import Path
from bisect import bisect_right
>>>>>>> Stashed changes
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from pathlib import Path

from strategy_classifier.constants import ALL_STRATEGIES, PARAMS
from log_parser import parse_session


def load_predictions():
    return pd.read_csv("segment_strategy_predictions.csv")


def load_speed_trends():
    return pd.read_csv(Path("speed/trends.csv"))


def load_speed_series():
    df = pd.read_csv(Path("speed/speed.csv"))
    return df


<<<<<<< Updated upstream
=======
# --------------------------------------------------------------
# Parameter timelines
# --------------------------------------------------------------
SHARED_PARAMS = {"frequency"}


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
    df = df.sort_values("time_sec").reset_index(drop=True)
    return df


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
>>>>>>> Stashed changes
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
        self.view_span_main = max(1.0, min(60.0, self.t_max - self.t_min))

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
        self.ax_speed.set_xlim(self.t_min, min(self.t_max, self.t_min + self.view_span_main))
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
        self._update_view_limits()
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

    def _update_view_limits(self):
        if self.ax_speed is None or self.view_span_main <= 0:
            return
        half_span = self.view_span_main / 2.0
        left = self.current_time - half_span
        right = self.current_time + half_span
        total_span = self.t_max - self.t_min
        if total_span <= self.view_span_main:
            left = self.t_min
            right = self.t_max
        else:
            if left < self.t_min:
                right += self.t_min - left
                left = self.t_min
            if right > self.t_max:
                left -= right - self.t_max
                right = self.t_max
        self.ax_speed.set_xlim(left, right)

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
<<<<<<< Updated upstream
    strat_colors = build_strategy_colors()

    viewer = StrategyTimelineUI(df, df_trends, df_speed, strat_colors)
    viewer.show()
=======
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
        for t, user in zip(freq_timeline.get("times", []), freq_timeline.get("users", [])):
            if user in users:
                freq_events.append((t, user))
    freq_events.sort(key=lambda x: x[0])
>>>>>>> Stashed changes


<<<<<<< Updated upstream
=======
    # ----------------------------------------------------------
    # Build figure
    # ----------------------------------------------------------
    fig, axes = plt.subplots(
        len(users) + 2,
        1,
        figsize=(20, 3 * (len(users) + 2)),
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
    ax_freq.set_ylabel("Frequency\nchanges", rotation=0, labelpad=45)
    ax_freq.grid(axis="x", alpha=0.2)

    # ----------------------------------------------------------
    # 3) USER PANELS
    # ----------------------------------------------------------
    used_strategies = set()
    user_segments = {}  # Store segments for each user for later lookup

    for i, user in enumerate(users):
        ax = axes[i + 2]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=14, pad=6)

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
            labelpad=55,
            fontsize=10,
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
        bbox_to_anchor=(0.83, 0.75),
        frameon=True,
        fontsize=11
    )

    # ----------------------------------------------------------
    # INTERACTIVE TIMELINE SLIDER
    # ----------------------------------------------------------
    # Create space for slider and buttons at the bottom
    plt.tight_layout(rect=(0, 0.12, 0.82, 1))
    
    # Add time slider axis
    slider_ax = plt.axes([0.15, 0.07, 0.65, 0.02])
    time_slider = Slider(
        slider_ax,
        'Time',
        t_min,
        t_max,
        valinit=t_min,
        valstep=(t_max - t_min) / 1000,  # Smooth sliding
        color='lightblue'
    )
    
    # Add zoom slider axis
    zoom_ax = plt.axes([0.15, 0.04, 0.65, 0.02])
    total_duration = t_max - t_min
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
    time_text = fig.text(0.85, 0.075, '', fontsize=12, fontweight='bold', 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Zoom level text display
    zoom_text = fig.text(0.85, 0.045, '', fontsize=10,
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
            ax.set_title(f"User {user}: {current_strategy}", fontsize=14, pad=6)
            snapshot = get_param_snapshot(
                user,
                current_time,
                user_param_timelines,
                shared_param_timelines,
            )
            ax.set_ylabel(
                format_param_label(user, snapshot),
                rotation=0,
                labelpad=55,
                fontsize=10,
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
    button_ax = plt.axes([0.02, 0.07, 0.08, 0.03])
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
>>>>>>> Stashed changes
if __name__ == "__main__":
    plot_timeline()
