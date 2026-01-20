#!/usr/bin/env python3
# --------------------------------------------------------------
# Beautiful Strategy Timeline Plot
# Speed Curve + Speed Trends + Strategy Timelines (per user)
# --------------------------------------------------------------

import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"  # avoid missing glyph warnings for control icons
import matplotlib.colors as mcolors
import numpy as np
from pathlib import Path
from matplotlib.widgets import Slider, Button
import time
import json
from log_parser import parse_session  # already used in your runner

from strategy_classifier.constants import ALL_STRATEGIES

import re


# --------------------------------------------------------------
# Loaders
# --------------------------------------------------------------
def dominant_strategy(row):
    best_name = None
    best_prob = -1.0
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col in row and pd.notna(row[col]):
            try:
                p = float(row[col])
            except Exception:
                continue
            if p > best_prob:
                best_prob = p
                best_name = strat
    # If no valid prob found or best_prob <= 0, treat as "No Strategy"
    if best_name is None or best_prob <= 0.0:
        return "No Strategy", 0.0
    return best_name, best_prob


def _normalize_predictions_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure canonical columns exist:
      - user_id, seg_start, seg_end, num_actions, has_events, pred_strategy
    Also ensure per-strategy prob columns exist (set to 0.0 if missing).
    """
    df = df.copy()

    # canonical mapping helpers (case-insensitive)
    col_map = {}
    cols_lower = {c.lower(): c for c in df.columns}

    def find(*names):
        for n in names:
            k = cols_lower.get(n.lower())
            if k:
                return k
        return None

    # map user -> user_id
    u = find("user_id", "user", "userid", "uid")
    if u and u != "user_id":
        col_map[u] = "user_id"

    s = find("seg_start", "start", "segment_start", "starttime")
    if s and s != "seg_start":
        col_map[s] = "seg_start"

    e = find("seg_end", "end", "segment_end", "endtime")
    if e and e != "seg_end":
        col_map[e] = "seg_end"

    na = find("num_actions", "actions", "n_actions")
    if na and na != "num_actions":
        col_map[na] = "num_actions"

    he = find("has_events", "has_events_flag", "hasEvents")
    if he and he != "has_events":
        col_map[he] = "has_events"

    ps = find("pred_strategy", "pred", "prediction", "pred_label")
    if ps and ps != "pred_strategy":
        col_map[ps] = "pred_strategy"

    if col_map:
        df = df.rename(columns=col_map)

    # Ensure required columns exist with safe defaults
    if "user_id" not in df.columns:
        df["user_id"] = None
    if "seg_start" not in df.columns:
        df["seg_start"] = 0.0
    if "seg_end" not in df.columns:
        df["seg_end"] = 0.0
    if "num_actions" not in df.columns:
        df["num_actions"] = 0
    if "has_events" not in df.columns:
        # prefer boolean-like 0/1
        df["has_events"] = (df["num_actions"].astype(int) > 0).astype(int)
    if "pred_strategy" not in df.columns:
        # derive from *_prob if possible; else default "No Strategy"
        prob_cols = [c for c in df.columns if c.endswith("_prob")]
        if prob_cols:
            df["pred_strategy"] = df.apply(lambda r: dominant_strategy(r)[0], axis=1)
        else:
            df["pred_strategy"] = "No Strategy"

    # Ensure strategy prob cols exist
    for strat in ALL_STRATEGIES:
        col = f"{strat}_prob"
        if col not in df.columns:
            df[col] = 0.0

    # coerce types
    df["seg_start"] = pd.to_numeric(df["seg_start"], errors="coerce").fillna(0.0)
    df["seg_end"] = pd.to_numeric(df["seg_end"], errors="coerce").fillna(0.0)
    df["num_actions"] = pd.to_numeric(df["num_actions"], errors="coerce").fillna(0).astype(int)
    df["has_events"] = pd.to_numeric(df["has_events"], errors="coerce").fillna((df["num_actions"]>0).astype(int)).astype(int)

    return df

def load_predictions():
    """
    Robust loader: choose the largest non-empty CSV matching segment_strategy*.csv
    (or *with_global_label*.csv) and skip files that pandas cannot parse.
    Falls back to JSON if no usable CSV found.
    """
    script_dir = Path(__file__).parent

    # 1) prefer any "*with_global_label*.csv"
    candidates = []
    for d in (script_dir, Path(".")):
        candidates.extend(list(d.glob("*with_global_label*.csv")))

    # 2) fallback: any segment_strategy*.csv
    if not candidates:
        for d in (script_dir, Path(".")):
            candidates.extend(list(d.glob("segment_strategy*.csv")))

    # filter non-empty files and sort by size desc
    candidates = [p for p in candidates if p.exists() and p.stat().st_size > 0]
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)

    for chosen in candidates:
        try:
            df = pd.read_csv(chosen)
            print(f"[INFO] Loaded predictions from CSV: {chosen} (size={chosen.stat().st_size})")
            df = _normalize_predictions_df(df)
            # --- ensure a single label column 'global_label' exists for plotting ---
            if "global_label" not in df.columns:
                cols = list(df.columns)
                ml_cols = [c for c in cols if c.endswith("_ml_pred")]
                rule_cols = [c for c in cols if c.endswith("_pred") and not c.endswith("_ml_pred")]
                pred_cols = ml_cols if ml_cols else rule_cols

                if pred_cols:
                    def _row_label(r):
                        for c in pred_cols:
                            v = r.get(c)
                            try:
                                if int(v) == 1:
                                    # turn column name into readable label
                                    return re.sub(r'(_ml_pred|_pred)$', '', c).replace('_', ' ')
                            except Exception:
                                continue
                        return "No Strategy"
                    df["global_label"] = df.apply(_row_label, axis=1)
                else:
                    df["global_label"] = "No Strategy"

            print(f"[DEBUG] Using 'global_label' for plotting (sample): {df['global_label'].value_counts().to_dict()}")
            return df
        except pd.errors.EmptyDataError:
            print(f"[WARN] CSV {chosen} appears empty / has no header — skipping.")
            continue
        except (UnicodeDecodeError, ValueError) as ex:
            print(f"[WARN] Failed to parse CSV {chosen}: {ex} — skipping.")
            continue

    # fallback: try JSON export (script dir then cwd)
    j_candidates = [script_dir / "segment_strategy_predictions.json", Path("segment_strategy_predictions.json")]
    for pj in j_candidates:
        if pj.exists() and pj.stat().st_size > 0:
            print(f"[INFO] CSV missing/parseable — loading JSON fallback: {pj}")
            data = json.load(pj.open("r", encoding="utf-8"))
            df = pd.DataFrame(data)
            df = _normalize_predictions_df(df)
            return df

    raise FileNotFoundError(
        "No usable segment_strategy predictions file found. "
        "Searched CSVs and JSON fallbacks in script dir and cwd."
    )


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
    base = (
        list(mcolors.TABLEAU_COLORS.values()) +
        list(mcolors.XKCD_COLORS.values())
    )
    return {s: base[i % len(base)] for i, s in enumerate(ALL_STRATEGIES)}


def _resolve_strategy_color(label, colors):
    """
    Map a (possibly spaced) strategy label to a known palette color.
    Falls back to a neutral gray if it cannot be resolved.
    """
    if label is None:
        return "#BBBBBB"

    normalized = str(label).strip().lower().replace(" ", "_")
    for strat, color in colors.items():
        if strat.lower() == normalized:
            return color
        if strat.lower().replace("_", " ") == normalized.replace("_", " "):
            return color
    return "#BBBBBB"


def summarize_strategy_usage(df, label_column=None, min_actions=1):
    """
    Aggregate the number of segments per user/strategy.

    Returns (summary_df, label_column_used):
        summary_df columns -> user_id, strategy_label, count, total_segments, percent
    """
    df = df.copy()
    if label_column is None:
        if "pred_strategy" in df.columns:
            label_column = "pred_strategy"
        elif "global_label" in df.columns:
            label_column = "global_label"
        else:
            label_column = "pred_strategy"

    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found in dataframe.")

    df["num_actions"] = pd.to_numeric(df.get("num_actions", 0), errors="coerce").fillna(0).astype(int)
    if "has_events" in df.columns:
        df = df[df["has_events"].astype(int) > 0]
    if min_actions > 1:
        df = df[df["num_actions"] >= int(min_actions)]

    df = df[pd.notna(df["user_id"])]
    if df.empty:
        raise ValueError("No rows available after filtering by actions/events.")

    summary = (
        df.groupby(["user_id", label_column])
        .size()
        .reset_index(name="count")
    )
    summary["total_segments"] = summary.groupby("user_id")["count"].transform("sum")
    summary["percent"] = summary["count"] / summary["total_segments"]
    summary = summary.sort_values(["user_id", "count"], ascending=[True, False]).reset_index(drop=True)
    return summary, label_column


def plot_strategy_usage_page(summary_df, label_column, output_path, colors=None, cols=2, show=False):
    """
    Create a multi-panel bar chart with per-user strategy counts.
    """
    if colors is None:
        colors = build_strategy_colors()

    users = summary_df["user_id"].unique()
    if len(users) == 0:
        raise ValueError("Summary dataframe has no users to plot.")

    cols = max(1, min(cols, len(users)))
    rows = math.ceil(len(users) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4), squeeze=False)
    axes = axes.flatten()

    for idx, user in enumerate(users):
        ax = axes[idx]
        subset = summary_df[summary_df["user_id"] == user]
        subset = subset.sort_values("count", ascending=False)
        positions = np.arange(len(subset))
        bar_colors = [_resolve_strategy_color(label, colors) for label in subset[label_column]]
        bars = ax.bar(positions, subset["count"], color=bar_colors, edgecolor="#333333", alpha=0.9)
        ax.set_xticks(positions)
        ax.set_xticklabels(subset[label_column], rotation=35, ha="right")
        ax.set_ylabel("Segments")
        ax.set_title(f"User {user}", fontsize=12)
        for bar, count in zip(bars, subset["count"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05,
                f"{int(count)}",
                ha="center",
                va="bottom",
                fontsize=9
            )
        ax.grid(axis="y", alpha=0.2, linestyle="--")

    for j in range(len(users), len(axes)):
        axes[j].axis("off")

    fig.suptitle("Strategy Usage per User", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_strategy_usage_report(
    df,
    csv_path="graphs/strategy_usage_summary.csv",
    figure_path="graphs/strategy_usage_report.png",
    label_column=None,
    min_actions=1,
    show_plot=False,
):
    """
    Build the per-user summary CSV and the visualization page.
    """
    summary_df, label_column = summarize_strategy_usage(df, label_column=label_column, min_actions=min_actions)
    summary_df = summary_df.rename(columns={label_column: "strategy"})

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote strategy usage summary to {csv_path}")

    colors = build_strategy_colors()
    plot_strategy_usage_page(
        summary_df,
        "strategy",
        figure_path,
        colors=colors,
        show=show_plot
    )
    print(f"[INFO] Saved per-user strategy usage figure to {figure_path}")

    return summary_df


def plot_strategy_usage_radar(
    summary_df,
    figure_path,
    strategies=None,
    value_col="percent",
    show=False,
    cols=2,
):
    """
    Render a radar (spider) chart per user comparing strategy usage.
    """
    if "strategy" not in summary_df.columns:
        raise ValueError("summary_df must contain a 'strategy' column.")
    if value_col not in summary_df.columns:
        raise ValueError(f"summary_df must contain the '{value_col}' column.")

    users = summary_df["user_id"].unique()
    if len(users) == 0:
        raise ValueError("Summary dataframe has no users to plot.")

    if strategies is None:
        observed = list(summary_df["strategy"].unique())
        strategies = [s for s in ALL_STRATEGIES if s in observed]
        if not strategies:
            strategies = observed

    num_strats = len(strategies)
    if num_strats < 3:
        raise ValueError("Radar chart needs at least 3 strategies to display.")

    angles = np.linspace(0, 2 * np.pi, num_strats, endpoint=False).tolist()
    angles += angles[:1]

    cols = max(1, min(cols, len(users)))
    rows = math.ceil(len(users) / cols)
    fig, axes = plt.subplots(rows, cols, subplot_kw=dict(polar=True), figsize=(cols * 5, rows * 4), squeeze=False)
    axes = axes.flatten()

    scale = 100.0 if value_col == "percent" else 1.0
    max_value = (summary_df[value_col].max() or 1.0) * scale
    max_value = max(1e-6, max_value)

    for idx, user in enumerate(users):
        ax = axes[idx]
        user_df = summary_df[summary_df["user_id"] == user]
        values = []
        for strat in strategies:
            row = user_df[user_df["strategy"] == strat]
            val = row[value_col].iloc[0] if not row.empty else 0.0
            val *= scale
            values.append(val)
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=f"User {user}")
        ax.fill(angles, values, alpha=0.25)
        ax.set_title(f"User {user}", fontsize=12, pad=12)
        ax.set_xticks(np.linspace(0, 2 * np.pi, num_strats, endpoint=False))
        ax.set_xticklabels([s.replace("_", " ") for s in strategies], fontsize=9)
        ax.set_ylim(0, max_value)
        ax.set_yticks(np.linspace(0, max_value, 4))
        if value_col == "percent":
            ax.set_yticklabels([f"{v:.0f}%" for v in np.linspace(0, max_value, 4)], fontsize=8)
        else:
            ax.set_yticklabels([f"{int(round(v))}" for v in np.linspace(0, max_value, 4)], fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(len(users), len(axes)):
        axes[j].axis("off")

    ylabel = "Percent of segments" if value_col == "percent" else "Segment count"
    fig.suptitle(f"Strategy Usage Radar ({ylabel})", fontsize=16, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    figure_path = Path(figure_path)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)


def generate_strategy_usage_radar(
    df,
    figure_path="graphs/strategy_usage_radar.png",
    label_column=None,
    min_actions=1,
    value_col="percent",
    strategies=None,
    show_plot=False,
):
    """
    Prepare a radar visualization for per-user strategy usage.
    """
    summary_df, label_column = summarize_strategy_usage(df, label_column=label_column, min_actions=min_actions)
    summary_df = summary_df.rename(columns={label_column: "strategy"})
    plot_strategy_usage_radar(
        summary_df,
        figure_path,
        strategies=strategies,
        value_col=value_col,
        show=show_plot,
    )
    print(f"[INFO] Saved strategy usage radar chart to {figure_path}")
    return summary_df


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
    n_rows = len(users) + 1
    fig, axes = plt.subplots(nrows=n_rows, sharex=True, figsize=(12, 2 * n_rows))

    # Ensure axes is iterable and 1-D so axes[...] indexing always works
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]
    else:
        # flatten any 2D axes array to 1D list
        axes = list(np.array(axes).reshape(-1))

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
    user_segments = {}  # Store segments for each user for later lookup

    # ensure we only plot predictions for users who had activity in that segment
    if 'has_events' in df.columns:
        df_plot = df[df['has_events'].astype(bool) == True].copy()
    else:
        # fallback: require at least 1 action
        df_plot = df[pd.to_numeric(df.get('num_actions',0), errors='coerce').fillna(0) > 0].copy()

    # --- require minimum activity per user/segment before plotting ---
    df['num_actions'] = pd.to_numeric(df.get('num_actions', 0), errors='coerce').fillna(0).astype(int)

    # change threshold as you prefer (1 -> show single-action segments; 2 -> require >=2 actions)
    MIN_ACTIONS_TO_PLOT = 2
    df_plot = df[df['num_actions'] >= MIN_ACTIONS_TO_PLOT].copy()

    # fallback: if filtering removes everything, keep original df
    if df_plot.shape[0] == 0:
        df_plot = df.copy()

    # --- optional: merge adjacent tiny-gap segments for the same user+label ---
    # This avoids flicker when segments split by tiny speed noise
    MERGE_GAP_THRESHOLD = 0.5  # seconds
    def _merge_small_gaps(df_in):
        out_rows = []
        for (uid), g in df_in.groupby('user_id'):
            g = g.sort_values('seg_start').reset_index(drop=True)
            if g.empty:
                continue
            cur = g.iloc[0].to_dict()
            for i in range(1, len(g)):
                row = g.iloc[i].to_dict()
                # if same label and small gap, merge
                if (cur.get('global_label') == row.get('global_label')) and ((row['seg_start'] - cur['seg_end']) <= MERGE_GAP_THRESHOLD):
                    cur['seg_end'] = max(cur['seg_end'], row['seg_end'])
                    cur['segment_duration'] = cur['seg_end'] - cur['seg_start']
                    # aggregate action counts
                    cur['num_actions'] = int(cur.get('num_actions',0)) + int(row.get('num_actions',0))
                else:
                    out_rows.append(cur)
                    cur = row
            out_rows.append(cur)
        if len(out_rows) == 0:
            return df_in
        return pd.DataFrame(out_rows)

    df_plot = _merge_small_gaps(df_plot)

    # DEBUG: print counts so you can verify filtering
    print(f"[DEBUG] Plotting predictions: total_rows={len(df)} -> plotted_rows={len(df_plot)}; min_actions={MIN_ACTIONS_TO_PLOT}")

    for i, user in enumerate(users):
        ax = axes[i + 1]
        ax.set_title(f"User {user} Strategy Timeline", fontsize=14, pad=6)

        df_u = df_plot[df_plot["user_id"] == user]
        user_segments[user] = []  # Store segments for this user

        for _, row in df_u.iterrows():
            s, e = row["seg_start"], row["seg_end"]
            # guard against NaN times
            if pd.isna(s) or pd.isna(e):
                continue

            strat, _ = dominant_strategy(row)

            # record used strategies (skip No Strategy)
            if strat != "No Strategy":
                used_strategies.add(strat)
            user_segments[user].append((s, e, strat))

            # safe color lookup, fallback to neutral gray for unknown/no strategy
            color = colors.get(strat, "#CCCCCC")
            ax.axvspan(
                s, e,
                color=color,
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
        # safe: skip if color missing
        c = colors.get(strat, "#CCCCCC")
        patch = plt.Line2D([0], [0], color=c, linewidth=12)
        legend_handles.append(patch)
        legend_labels.append(strat)

    if legend_handles:
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
        
        # Update titles with current strategy for each user
        for i, user in enumerate(users):
            ax = axes[i + 1]
            current_strategy = get_current_strategy(user, current_time)
            ax.set_title(f"User {user}: {current_strategy}", fontsize=14, pad=6)
        
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

def _annotate_segment_boundaries(ax, preds_df, segments, view_start=None, view_end=None):
    """
    Draw thin vertical lines at each segment boundary (start/end) and label
    overlapping prediction rows. Helpful to debug off-by-one / rounding in plot.
    """
    try:
        for seg in segments:
            s = getattr(seg, "start", None)
            e = getattr(seg, "end", None)
            if s is None or e is None:
                continue
            # only draw if within view or if no view limits given
            if view_start is not None and view_end is not None:
                if e < view_start or s > view_end:
                    continue
            ax.axvline(s, color="tab:gray", linestyle="--", linewidth=0.7, alpha=0.7)
            ax.axvline(e, color="tab:gray", linestyle="--", linewidth=0.7, alpha=0.7)
            # annotate every few boundaries to avoid clutter
            if (s % 30) < 1.0:
                ax.text(s, ax.get_ylim()[1]*0.98, f"{s:.2f}s", fontsize=7, ha="center", va="top", color="gray")
    except Exception:
        pass

def _print_pred_rows_for_window(preds_df, view_start, view_end):
    """Print a compact table of prediction rows overlapping the view window."""
    if preds_df is None or view_start is None or view_end is None:
        return
    # try common column names
    cols = {c.lower(): c for c in preds_df.columns}
    def _col(*names):
        for n in names:
            if n.lower() in cols:
                return cols[n.lower()]
        return None
    startc = _col("seg_start","starttime","start")
    endc   = _col("seg_end","endtime","end")
    userc  = _col("user_id","user","userid")
    predc  = _col("global_label","pred","strategy","label")
    if not startc or not endc:
        print("[DEBUG] preds_df missing start/end cols")
        return
    import pandas as pd
    s = pd.to_numeric(preds_df[startc], errors="coerce")
    e = pd.to_numeric(preds_df[endc], errors="coerce")
    mask = (s <= view_end) & (e >= view_start)
    sel = preds_df.loc[mask, [c for c in (userc, startc, endc, predc) if c and c in preds_df.columns]]
    if len(sel):
        print(f"[DEBUG] Predictions overlapping {view_start}-{view_end}s:")
        print(sel.to_string(index=False))
    else:
        print(f"[DEBUG] No predictions overlap {view_start}-{view_end}s")

# Call these right after the plot axes are prepared and before plt.show()
# Example insertion point (inside plot_timeline or equivalent):
# view_start, view_end should be the current x-axis limits or the time window you inspect

def _legacy_debug_snippet(log_dir="logs", t_start=155.0, t_end=205.0):
    """
    Preserves the verbose diagnostics that used to run on import.
    Trigger with --debug-snippet if you still need that workflow.
    """
    from collections import Counter
    import os
    import sys
    try:
        from strategy_classifier.segmentation import build_segments_from_speed_csv
    except Exception:
        build_segments_from_speed_csv = None

    session_path = Path(log_dir)
    print(f"[DEBUG] Inspecting logs in {session_path.resolve()}")
    try:
        if session_path.is_dir():
            log_paths = list(session_path.glob("User*.log"))
        else:
            log_paths = [session_path]
        try:
            events = parse_session(log_paths, session_id="default")
        except TypeError:
            events = parse_session(log_paths)
    except Exception as exc:
        print(f"[WARN] Failed to parse logs: {exc}")
        events = []

    def event_ts(e):
        return e.get("timestamp_sec") or e.get("timestamp") or e.get("time") or 0.0

    evs_window = [e for e in events if t_start <= event_ts(e) < t_end]
    print(f"Total raw events in window {t_start}-{t_end}: {len(evs_window)}")
    for e in evs_window:
        print(json.dumps({
            "user_id": e.get("user_id"),
            "ts": event_ts(e),
            "type": e.get("type") or e.get("event") or e.get("action"),
            "payload": {k: e.get(k) for k in ("param","value","description") if k in e}
        }))

    cnt = Counter(e.get("user_id") for e in evs_window)
    print("per-user counts:", cnt)

    print("cwd:", os.getcwd())
    for p in Path(".").glob("segment_strategy*.csv"):
        print(p.name, p.stat().st_size)

    if build_segments_from_speed_csv:
        try:
            s = Path("speed/trends.csv")
            segs = build_segments_from_speed_csv(s, 20.0, 10.0)
            print("segments:", len(segs))
            for seg in segs:
                if seg.start <= 158 and seg.end >= 156:
                    print("SEG:", getattr(seg, 'segment_id', None), seg.start, seg.end,
                          getattr(seg, 'trend', None), "dur=", getattr(seg, 'duration', None))
        except Exception as exc:
            print(f"[WARN] Segment debug failed: {exc}")

    try:
        p = Path("segment_strategy_with_global_label.csv")
        if not p.exists():
            print(f"[WARN] Missing {p}")
            return

        df = pd.read_csv(p)
        print("FILE:", p, "rows:", len(df))
        print("COLUMNS:", df.columns.tolist())

        candidates = {c.lower(): c for c in df.columns}

        def col(*names):
            for n in names:
                if n.lower() in candidates:
                    return candidates[n.lower()]
            return None

        startc = col("seg_start", "starttime", "start", "seg_start_sec")
        endc = col("seg_end", "endtime", "end", "seg_end_sec")
        userc = col("user_id", "user", "userid", "uid")
        predc = col("pred_strategy", "pred", "strategy", "label")
        print("mapped:", startc, endc, userc, predc)

        if not (startc and endc):
            print("No start/end columns found; head preview:")
            print(df.head().to_string(index=False))
            return

        tmin, tmax = 156.0, 158.0
        mask = (pd.to_numeric(df[startc], errors="coerce") <= tmax) & \
               (pd.to_numeric(df[endc], errors="coerce") >= tmin)
        sel = df[mask].sort_values([userc or startc, startc])
        print("ROWS overlapping 156-158s:", len(sel))
        if len(sel) > 0:
            cols = [c for c in (userc, startc, endc, predc) if c in sel.columns]
            print(sel[cols].to_string(index=False))
        else:
            print("No predictions overlap that window.")
    except Exception as exc:
        print(f"[WARN] CSV inspection failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Strategy timeline and reporting utilities.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the per-user strategy usage report instead of the interactive timeline."
    )
    parser.add_argument("--report-csv", default="graphs/strategy_usage_summary.csv", help="CSV path for the summary.")
    parser.add_argument("--report-figure", default="graphs/strategy_usage_report.png",
                        help="Image path for the per-user plots.")
    parser.add_argument("--report-min-actions", type=int, default=1,
                        help="Minimum actions required for a segment to count in the report.")
    parser.add_argument("--label-column", default=None,
                        help="Override the label column used for aggregation (default: global_label or pred_strategy).")
    parser.add_argument("--show-report", action="store_true", help="Display the generated report figure interactively.")
    parser.add_argument("--radar", action="store_true", help="Generate radar chart(s) of per-user strategy usage.")
    parser.add_argument("--radar-figure", default="graphs/strategy_usage_radar.png",
                        help="Image path for the radar visualization.")
    parser.add_argument("--radar-min-actions", type=int, default=1,
                        help="Minimum actions required for a segment to count in the radar chart.")
    parser.add_argument("--radar-value", choices=("percent", "count"), default="percent",
                        help="Metric plotted on the radar chart.")
    parser.add_argument("--show-radar", action="store_true", help="Display the radar chart interactively.")
    parser.add_argument("--debug-snippet", action="store_true",
                        help="Run the legacy verbose debugging snippet after the main task.")
    parser.add_argument("--debug-log-dir", default="logs", help="Log directory used by the debug snippet.")
    parser.add_argument("--debug-start", type=float, default=155.0, help="Start time for debug snippet window.")
    parser.add_argument("--debug-end", type=float, default=205.0, help="End time for debug snippet window.")

    args = parser.parse_args()

    df = None
    if args.report or args.radar:
        df = load_predictions()

    if args.report:
        generate_strategy_usage_report(
            df,
            csv_path=args.report_csv,
            figure_path=args.report_figure,
            label_column=args.label_column,
            min_actions=args.report_min_actions,
            show_plot=args.show_report,
        )

    if args.radar:
        generate_strategy_usage_radar(
            df,
            figure_path=args.radar_figure,
            label_column=args.label_column,
            min_actions=args.radar_min_actions,
            value_col=args.radar_value,
            show_plot=args.show_radar,
        )

    if not args.report and not args.radar:
        plot_timeline()

    if args.debug_snippet:
        _legacy_debug_snippet(log_dir=args.debug_log_dir, t_start=args.debug_start, t_end=args.debug_end)


if __name__ == "__main__":
    main()
