#!/usr/bin/env python3
# --------------------------------------------------------------
# Beautiful Strategy Timeline Plot
# Speed Curve + Speed Trends + Strategy Timelines (per user)
# --------------------------------------------------------------

import argparse
import math
import subprocess
import sys
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"  # avoid missing glyph warnings for control icons
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import numpy as np
from pathlib import Path
from matplotlib.widgets import Slider, Button, RadioButtons
import time
import json
import matplotlib.image as mpimg
from log_parser import parse_session  # already used in your runner

from strategy_classifier.constants import ALL_STRATEGIES

import re

DEFAULT_REPORT_CSV = "graphs/strategy_usage_summary.csv"
DEFAULT_REPORT_FIGURE = "graphs/strategy_usage_report.png"
DEFAULT_RADAR_FIGURE = "graphs/strategy_usage_radar.png"
DASHBOARD_WINDOWS = []

# --------------------------------------------------------------
# Translations (EN / NL)
# --------------------------------------------------------------
TRANSLATIONS = {
    'EN': {
        'title': 'Robot Speed Timeline',
        'speed_ylabel': 'Speed (px/s)',
        'time_xlabel': 'Time (seconds)',
        'user_title': 'User {user}: {strategy}',
        'user_label': 'User {user}',
        'strategies_legend': 'Strategies Used',
        'no_strategy': 'No Strategy',
        'play': '▶ Play',
        'pause': '⏸ Pause',
        'time_slider': 'Time',
        'zoom_slider': 'Zoom',
        'zoom_display': 'Zoom: {level:.1f}x',
        'dashboard_button': 'Open Dashboard',
        'dashboard_title': 'Strategy Dashboard',
        'views_title': 'Views',
        'bar_chart': 'Bar Chart',
        'radar_chart': 'Radar Chart',
        'insights': 'Insights',
        'per_user_counts': 'Per-User Strategy Counts',
        'strategy_radar': 'Strategy Usage Radar',
        'player_roles': 'Player Roles & Insights',
    # Strategy translations
    'goal_directed_tuning': 'goal_directed_tuning',
        'incremental_adjustment': 'incremental_adjustment',
        'iterative_finetuning': 'iterative_finetuning',
        'random_trial_error': 'random_trial_error',
        'structured_exploration': 'structured_exploration',
        'systematic_parameter_sweep': 'systematic_parameter_sweep',
    },
    'NL': {
        'title': 'Robot Snelheid Tijdlijn',
        'speed_ylabel': 'Snelheid (px/s)',
        'time_xlabel': 'Tijd (seconden)',
        'user_title': 'Gebruiker {user}: {strategy}',
        'user_label': 'Gebruiker {user}',
        'strategies_legend': 'Gebruikte Strategieën',
        'no_strategy': 'Geen Strategie',
        'play': '▶ Afspelen',
        'pause': '⏸ Pauzeren',
        'time_slider': 'Tijd',
        'zoom_slider': 'Zoom',
        'zoom_display': 'Zoom: {level:.1f}x',
        'dashboard_button': 'Dashboard openen',
        'dashboard_title': 'Strategie Dashboard',
        'views_title': 'Weergaven',
        'bar_chart': 'Staafdiagram',
        'radar_chart': 'Radardiagram',
        'insights': 'Inzichten',
        'per_user_counts': 'Strategie-aantallen per gebruiker',
        'strategy_radar': 'Strategiegebruik Radar',
        'player_roles': 'Spelerrollen & inzichten',
        # Strategy translations
        'goal_directed_tuning': 'doelgerichte_afstemming',
        'incremental_adjustment': 'incrementele_aanpassing',
        'iterative_finetuning': 'iteratieve_fijnafstemming',
        'random_trial_error': 'willekeurige_proef_fout',
        'structured_exploration': 'gestructureerde_verkenning',
        'systematic_parameter_sweep': 'systematische_parameter_sweep',
    }
}

# --------------------------------------------------------------
# Insights translations (EN / NL)
# --------------------------------------------------------------
INSIGHTS_I18N = {
    "EN": {
        "insights_title": "Student Strategy Feedback (Teacher View)",
        "insights_subtitle": "Based on the detected strategies in this session. Use as a starting point for constructive feedback.",
        "user_header": "User {user}",
        "most_used": "Most used",
        "second_most": "Second most",
        "least_used": "Least used",
        "not_enough_data": "(not enough data)",
        "observed": "What we observed",
        "strength": "Strength",
        "growth_less_practiced": "Growth opportunity (less practiced)",
        "why_it_helps": "Why it helps",
        "growth_opportunity": "Growth opportunity",
        "try_next_lesson": "Try next lesson:",
        "scroll_hint": "Scroll to see more students (mouse wheel or slider).",
        "segments_word": "segments",
        "no_insights": "No insights available.",
    },
    "NL": {
        "insights_title": "Feedback op Strategie (Docentweergave)",
        "insights_subtitle": "Gebaseerd op de gedetecteerde strategieën in deze sessie. Gebruik dit als startpunt voor constructieve feedback.",
        "user_header": "Gebruiker {user}",
        "most_used": "Meest gebruikt",
        "second_most": "Op één na meest",
        "least_used": "Minst gebruikt",
        "not_enough_data": "(onvoldoende data)",
        "observed": "Wat we zagen",
        "strength": "Sterkte",
        "growth_less_practiced": "Groeikans (minder geoefend)",
        "why_it_helps": "Waarom dit helpt",
        "growth_opportunity": "Groeikans",
        "try_next_lesson": "Probeer in de volgende les:",
        "scroll_hint": "Scroll om meer leerlingen te zien (muiswiel of slider).",
        "segments_word": "segmenten",
        "no_insights": "Geen inzichten beschikbaar.",
    },
}

def it(key, **kwargs):
    """Insights translation helper."""
    lang = current_language["lang"]
    text = INSIGHTS_I18N.get(lang, INSIGHTS_I18N["EN"]).get(key, key)
    return text.format(**kwargs) if kwargs else text

# Nicely formatted strategy display names per language (for Insights)
STRATEGY_DISPLAY = {
    "EN": {
        "structured_exploration": "Structured exploration",
        "random_trial_error": "Random trial-and-error",
        "systematic_parameter_sweep": "Systematic parameter sweep",
        "goal_directed_tuning": "Goal-directed tuning",
        "iterative_finetuning": "Iterative fine-tuning",
        "incremental_adjustment": "Incremental adjustment",
        "inactivity_wait": "Inactivity",
        "No Strategy": "No clear strategy",
    },
    "NL": {
        "structured_exploration": "Gestructureerde verkenning",
        "random_trial_error": "Willekeurige proef & fout",
        "systematic_parameter_sweep": "Systematische parameter-sweep",
        "goal_directed_tuning": "Doelgerichte afstemming",
        "iterative_finetuning": "Iteratieve fijnafstemming",
        "incremental_adjustment": "Incrementele aanpassing",
        "inactivity_wait": "Inactiviteit",
        "No Strategy": "Geen duidelijke strategie",
    },
}

def _strategy_display(strategy: str) -> str:
    lang = current_language["lang"]
    return STRATEGY_DISPLAY.get(lang, STRATEGY_DISPLAY["EN"]).get(
        strategy, str(strategy).replace("_", " ").title()
    )


# Strategy definitions for hover tooltips
STRATEGY_DEFINITIONS = {
    'EN': {
        'goal_directed_tuning': 'Goal-Directed Tuning: Sample definition - adjusting parameters with a specific target outcome in mind.',
        'incremental_adjustment': 'Incremental Adjustment: Sample definition - making small, step-by-step changes to parameters.',
        'iterative_finetuning': 'Iterative Finetuning: Sample definition - repeatedly refining parameters based on feedback.',
        'random_trial_error': 'Random Trial & Error: Sample definition - exploring parameter space through random experimentation.',
        'structured_exploration': 'Structured Exploration: Sample definition - systematically exploring different parameter combinations.',
        'systematic_parameter_sweep': 'Systematic Parameter Sweep: Sample definition - methodically testing all parameter values in a range.',
        'No Strategy': 'No Strategy: No specific tuning strategy detected in this segment.',
    },
    'NL': {
        'goal_directed_tuning': 'Doelgerichte Afstemming: Voorbeelddefinitie - parameters aanpassen met een specifiek doelresultaat in gedachten.',
        'incremental_adjustment': 'Incrementele Aanpassing: Voorbeelddefinitie - kleine, stapsgewijze wijzigingen aan parameters.',
        'iterative_finetuning': 'Iteratieve Fijnafstemming: Voorbeelddefinitie - herhaaldelijk verfijnen van parameters op basis van feedback.',
        'random_trial_error': 'Willekeurige Proef & Fout: Voorbeelddefinitie - parameterruimte verkennen door willekeurige experimenten.',
        'structured_exploration': 'Gestructureerde Verkenning: Voorbeelddefinitie - systematisch verkennen van verschillende parametercombinaties.',
        'systematic_parameter_sweep': 'Systematische Parameter Sweep: Voorbeelddefinitie - methodisch testen van alle parameterwaarden in een bereik.',
        'No Strategy': 'Geen Strategie: Geen specifieke afstemmingsstrategie gedetecteerd in dit segment.',
    }
}

# Current language state
current_language = {'lang': 'EN'}

def t(key, **kwargs):
    """Get translated string for current language."""
    lang = current_language['lang']
    text = TRANSLATIONS.get(lang, TRANSLATIONS['EN']).get(key, key)
    if kwargs:
        return text.format(**kwargs)
    return text

def translate_strategy(strategy):
    """Translate a strategy name to current language."""
    lang = current_language['lang']
    # Convert strategy name to lookup key (replace spaces with underscores)
    key = strategy.replace(' ', '_').lower()
    translations = TRANSLATIONS.get(lang, TRANSLATIONS['EN'])
    return translations.get(key, strategy)

def get_strategy_definition(strategy):
    """Get the definition for a strategy in current language."""
    lang = current_language['lang']
    definitions = STRATEGY_DEFINITIONS.get(lang, STRATEGY_DEFINITIONS['EN'])
    # Try exact match first, then lowercase with underscores
    if strategy in definitions:
        return definitions[strategy]
    key = strategy.replace(' ', '_').lower()
    return definitions.get(key, f"{strategy}: No definition available.")


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
    """Map a label (with spaces) to a color from ALL_STRATEGIES palette."""
    if label is None:
        return "#BBBBBB"
    normalized = str(label).strip().lower().replace(" ", "_")
    for strat, color in colors.items():
        key = strat.lower()
        if key == normalized or key.replace("_", " ") == normalized.replace("_", " "):
            return color
    return "#BBBBBB"


def summarize_strategy_usage(df, label_column=None, min_actions=1):
    """
    Aggregate number of segments per user/strategy.
    Returns (summary_df, label_column_used).
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
        raise ValueError(f"Label column '{label_column}' not found.")

    df["num_actions"] = pd.to_numeric(df.get("num_actions", 0), errors="coerce").fillna(0).astype(int)
    if "has_events" in df.columns:
        df = df[df["has_events"].astype(int) > 0]
    if min_actions > 1:
        df = df[df["num_actions"] >= int(min_actions)]

    df = df[pd.notna(df["user_id"])]
    if df.empty:
        raise ValueError("No rows available after filtering.")

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
    """Render multi-panel bar charts summarizing strategy counts per user."""
    if colors is None:
        colors = build_strategy_colors()

    users = summary_df["user_id"].unique()
    if len(users) == 0:
        raise ValueError("Summary dataframe has no users.")

    cols = max(1, min(cols, len(users)))
    rows = math.ceil(len(users) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 4), squeeze=False)
    axes = axes.flatten()

    for idx, user in enumerate(users):
        ax = axes[idx]
        subset = summary_df[summary_df["user_id"] == user].sort_values("count", ascending=False)
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
                fontsize=9,
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
    """Build the per-user summary CSV and bar-chart visualization."""
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
        show=show_plot,
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
    """Render radar (spider) charts summarizing strategy usage per user."""
    if "strategy" not in summary_df.columns:
        raise ValueError("summary_df must contain a 'strategy' column.")
    if value_col not in summary_df.columns:
        raise ValueError(f"summary_df must contain '{value_col}'.")

    users = summary_df["user_id"].unique()
    if len(users) == 0:
        raise ValueError("Summary dataframe has no users.")

    if strategies is None:
        observed = list(summary_df["strategy"].unique())
        strategies = [s for s in ALL_STRATEGIES if s in observed]
        if not strategies:
            strategies = observed

    num_strats = len(strategies)
    if num_strats < 3:
        raise ValueError("Radar chart needs at least 3 strategies.")

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
    """Prepare radar visualization for per-user strategy usage."""
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


STRATEGY_ROLE_MAP = {
    "goal_directed_tuning": "Leader",
    "iterative_finetuning": "Leader",
    "systematic_parameter_sweep": "Explorer",
    "structured_exploration": "Explorer",
    "random_trial_error": "Explorer",
    "incremental_adjustment": "Follower",
    "repetition_practice": "Follower",
    "backtracking_recovery": "Support",
    "inactivity_wait": "Observer",
}

ROLE_DESCRIPTIONS = {
    "Leader": "Optimizes toward a clear goal and guides the team toward refined solutions.",
    "Explorer": "Covers new parameter space broadly to discover possibilities.",
    "Follower": "Builds on existing ideas with cautious adjustments and repetitions.",
    "Support": "Keeps the team on track by undoing mistakes or recovering prior states.",
    "Observer": "Pauses or monitors rather than acting, often waiting for others.",
}


def derive_player_roles(summary_df):
    """Return list of {user_id, strategy, role, percent, count} records."""
    roles = []
    for user, group in summary_df.groupby("user_id"):
        if group.empty:
            continue
        top = group.sort_values("percent", ascending=False).iloc[0]
        strategy = top["strategy"]
        role = STRATEGY_ROLE_MAP.get(strategy, "Explorer")
        roles.append({
            "user_id": user,
            "strategy": strategy,
            "role": role,
            "percent": float(top["percent"]),
            "count": int(top["count"]),
        })
    return roles


def _display_image(ax, image_path, title):
    ax.clear()
    p = Path(image_path)
    if not p.exists():
        ax.text(0.5, 0.5, f"Image not found:\n{p}", ha="center", va="center", fontsize=12)
        ax.axis("off")
        return
    img = mpimg.imread(p)
    ax.imshow(img)
    ax.set_title(title, fontsize=14)
    ax.axis("off")


STRATEGY_FRIENDLY = {
    "structured_exploration": "Structured exploration",
    "random_trial_error": "Random trial-and-error",
    "systematic_parameter_sweep": "Systematic parameter sweep",
    "goal_directed_tuning": "Goal-directed tuning",
    "iterative_finetuning": "Iterative fine-tuning",
    "incremental_adjustment": "Incremental adjustment",
    "inactivity_wait": "Inactivity",
    "No Strategy": "No clear strategy",
}
STRATEGY_FEEDBACK = {
    "EN": {
        "structured_exploration": {
            "observation": "They try different settings in an organized way across multiple controls.",
            "strength": "Good for discovering options and learning what the robot can do.",
            "growth": "Sometimes ideas stay broad; they may learn faster by isolating one change at a time.",
            "next_steps": [
                "Pick one setting to change while keeping the rest the same.",
                "After each test, ask: “What changed and what happened?”",
            ],
        },
        "random_trial_error": {
            "observation": "They change many things quickly, sometimes reversing direction.",
            "strength": "Shows curiosity and willingness to experiment.",
            "growth": "Learning can be slower if too many settings change at once.",
            "next_steps": [
                "Try “one change per run” for 3 runs to see cause and effect.",
                "Before pressing start, ask them to predict what will happen.",
            ],
        },
        "systematic_parameter_sweep": {
            "observation": "They focus on one control and vary it step-by-step while keeping others steady.",
            "strength": "Excellent for understanding exactly what one setting does.",
            "growth": "They may miss interactions between settings if they never combine changes.",
            "next_steps": [
                "After a sweep, choose the best value and test it with one other setting.",
                "Keep a simple note: value tried → outcome.",
            ],
        },
        "goal_directed_tuning": {
            "observation": "They make targeted changes toward a clear goal (e.g., faster, straighter, smoother).",
            "strength": "Efficient when close to a working solution.",
            "growth": "If the goal isn’t clear, changes can become guessy or inconsistent.",
            "next_steps": [
                "Ask them to say the goal out loud before changing anything.",
                "Ask for a short reason: “Why this change?”",
            ],
        },
        "iterative_finetuning": {
            "observation": "They make very small adjustments and often correct or undo earlier changes.",
            "strength": "Careful refinement and precision near a solution.",
            "growth": "Progress can stall if they never try a bigger step to explore.",
            "next_steps": [
                "If stuck, try one bigger change once, then return to small steps.",
                "Ask: “Why did you undo that change?”",
            ],
        },
        "incremental_adjustment": {
            "observation": "They make moderate, careful changes that usually avoid big drops in performance.",
            "strength": "Stable progress and fewer “bad runs.”",
            "growth": "They might learn faster by briefly testing extremes to see boundaries.",
            "next_steps": [
                "Try one quick “high vs low” test to learn faster, then choose a middle value.",
                "Ask them to write one sentence: change → expected effect.",
            ],
        },
        "inactivity_wait": {
            "observation": "They pause for a while without changing settings.",
            "strength": "They may be thinking or planning before acting.",
            "growth": "If they’re unsure, they may need a clear next step to restart.",
            "next_steps": [
                "Ask: “What will you try next, and why?”",
                "Use a routine: goal → one change → test → explain.",
            ],
        },
        "No Strategy": {
            "observation": "There is not enough consistent behavior to label a clear strategy here.",
            "strength": "This can happen when students are exploring controls or still learning the task.",
            "growth": "More structure can help their actions become purposeful and easier to reflect on.",
            "next_steps": [
                "Clarify the goal and what each control changes.",
                "Use: goal → one change → test → explain.",
            ],
        },
    },
    "NL": {
        "structured_exploration": {
            "observation": "Ze proberen verschillende instellingen op een georganiseerde manier uit over meerdere knoppen/parameters.",
            "strength": "Goed om opties te ontdekken en te leren wat de robot kan.",
            "growth": "Soms blijft het te breed; sneller leren lukt vaak door één wijziging per keer te isoleren.",
            "next_steps": [
                "Kies één instelling om te veranderen en houd de rest hetzelfde.",
                "Vraag na elke test: “Wat veranderde je en wat gebeurde er?”",
            ],
        },
        "random_trial_error": {
            "observation": "Ze veranderen veel dingen snel achter elkaar en draaien soms weer terug.",
            "strength": "Toont nieuwsgierigheid en bereidheid om te experimenteren.",
            "growth": "Leren kan trager zijn als er te veel instellingen tegelijk veranderen.",
            "next_steps": [
                "Probeer 3 runs lang: “één wijziging per run” om oorzaak-gevolg te zien.",
                "Vraag vóór start: “Wat denk je dat er gaat gebeuren?”",
            ],
        },
        "systematic_parameter_sweep": {
            "observation": "Ze focussen op één instelling en variëren die stap-voor-stap terwijl de rest gelijk blijft.",
            "strength": "Uitstekend om precies te begrijpen wat één instelling doet.",
            "growth": "Ze kunnen interacties tussen instellingen missen als ze veranderingen nooit combineren.",
            "next_steps": [
                "Na een sweep: kies de beste waarde en test die met één extra instelling.",
                "Houd een simpel lijstje bij: waarde → uitkomst.",
            ],
        },
        "goal_directed_tuning": {
            "observation": "Ze doen gerichte aanpassingen richting een duidelijk doel (bv. sneller, rechter, smoother).",
            "strength": "Efficiënt als je dicht bij een werkende oplossing zit.",
            "growth": "Als het doel niet helder is, worden veranderingen sneller ‘gokjes’ of inconsistent.",
            "next_steps": [
                "Laat ze het doel hardop noemen vóórdat ze iets veranderen.",
                "Vraag om een korte reden: “Waarom deze verandering?”",
            ],
        },
        "iterative_finetuning": {
            "observation": "Ze doen hele kleine aanpassingen en corrigeren/undo’en vaak eerdere veranderingen.",
            "strength": "Zorgvuldige verfijning en precisie vlak bij een oplossing.",
            "growth": "Voortgang kan stagneren als ze nooit één keer een grotere stap proberen om te verkennen.",
            "next_steps": [
                "Als je vastzit: probeer één keer een grotere wijziging en ga daarna terug naar kleine stapjes.",
                "Vraag: “Waarom heb je die verandering teruggedraaid?”",
            ],
        },
        "incremental_adjustment": {
            "observation": "Ze doen gematigde, voorzichtige wijzigingen en vermijden vaak grote prestatiedalingen.",
            "strength": "Stabiele vooruitgang en minder ‘slechte runs’.",
            "growth": "Sneller leren kan door kort extremen te testen om grenzen te zien.",
            "next_steps": [
                "Doe één snelle ‘hoog vs laag’ test om sneller te leren, kies daarna een middenwaarde.",
                "Laat ze één zin opschrijven: wijziging → verwacht effect.",
            ],
        },
        "inactivity_wait": {
            "observation": "Ze pauzeren een tijdje zonder instellingen te veranderen.",
            "strength": "Ze denken mogelijk na of plannen eerst.",
            "growth": "Als ze onzeker zijn, helpt een duidelijke volgende stap om weer op gang te komen.",
            "next_steps": [
                "Vraag: “Wat ga je hierna proberen, en waarom?”",
                "Gebruik een routine: doel → één wijziging → test → leg uit.",
            ],
        },
        "No Strategy": {
            "observation": "Er is niet genoeg consistent gedrag om hier een duidelijke strategie te labelen.",
            "strength": "Dit kan gebeuren als leerlingen nog aan het oriënteren zijn of de taak nog leren.",
            "growth": "Meer structuur helpt om acties doelgerichter te maken en beter te kunnen reflecteren.",
            "next_steps": [
                "Maak het doel en effect van elke instelling duidelijk.",
                "Gebruik: doel → één wijziging → test → leg uit.",
            ],
        },
    },
}


def _feedback_for(strategy: str):
    """Safe access to STRATEGY_FEEDBACK with fallback, per language."""
    lang = current_language["lang"]
    fb = STRATEGY_FEEDBACK.get(lang, STRATEGY_FEEDBACK["EN"])
    if strategy in fb:
        return fb[strategy]
    return fb["No Strategy"]


STRATEGY_WHAT_IT_LOOKS_LIKE = {
    "structured_exploration": "Tries different settings in a fairly organized way across multiple controls.",
    "random_trial_error": "Changes many things quickly, often reversing direction, with little consistent improvement.",
    "systematic_parameter_sweep": "Focuses on one control and varies it step-by-step while keeping others steady.",
    "goal_directed_tuning": "Makes targeted changes toward a clear goal and improves faster when close to success.",
    "iterative_finetuning": "Makes very small adjustments and often corrects/undoes earlier changes.",
    "incremental_adjustment": "Makes moderate, careful changes that usually avoid performance drops.",
    "inactivity_wait": "Few or no changes for a while (thinking, waiting, or uncertainty).",
    "No Strategy": "Not enough consistent behavior in this time window to label a clear strategy.",
}

STRATEGY_COACHING_TIPS = {
    "structured_exploration": [
        "Turn exploration into mini-experiments: keep everything the same except one setting.",
        "Limit to 3 planned tests, then reflect on what changed.",
    ],
    "random_trial_error": [
        "Ask them to pause after each run: “What changed, and what did it do?”",
        "Encourage one-change-at-a-time testing to learn cause-and-effect.",
    ],
    "systematic_parameter_sweep": [
        "Ask: “What did you learn about this one setting?” before changing another.",
        "Keep a simple notes table: setting → value tried → outcome.",
    ],
    "goal_directed_tuning": [
        "Have them say the goal out loud before changing anything (“We’re optimizing for…”).",
        "Ask for a short prediction before each change (“I think this will…”).",
    ],
    "iterative_finetuning": [
        "If progress stalls, try one deliberate larger test once, then return to small steps.",
        "Ask them to explain why they reverted/undid a change.",
    ],
    "incremental_adjustment": [
        "Encourage a hypothesis: “I changed X because I expected Y.”",
        "If improvement is slow, suggest a short sweep on one parameter to learn faster.",
    ],
    "inactivity_wait": [
        "Check if they are planning: ask them to explain what they will try next.",
        "If unsure, give a starting routine: goal → one change → test → explain.",
    ],
    "No Strategy": [
        "Clarify the task and controls; confusion can hide real strategy.",
        "Use a guided routine: goal → one change → test → explain.",
    ],
}

def _friendly(strategy: str) -> str:
    return _strategy_display(strategy)



def _build_insight_lines(top1, top2, bottom):
    def fmt_row(label_key, row):
        if row is None:
            return f"{it(label_key)}: {it('not_enough_data')}"
        return (
            f"{it(label_key)}: {_friendly(row['strategy'])} — "
            f"{float(row['percent'])*100:.1f}% ({int(row['count'])} {it('segments_word')})"
        )

    s1 = top1["strategy"]
    sb = bottom["strategy"] if bottom is not None else None

    f1 = _feedback_for(s1)
    lines = []

    lines.append(fmt_row("most_used", top1))
    lines.append(fmt_row("second_most", top2))
    lines.append(fmt_row("least_used", bottom))
    lines.append("")

    lines.append(f"{it('observed')}: {f1['observation']}")
    lines.append(f"{it('strength')}: {f1['strength']}")

    if sb is not None and sb != s1:
        fb = _feedback_for(sb)
        lines.append(f"{it('growth_less_practiced')}: {_friendly(sb)}")
        lines.append(f"{it('why_it_helps')}: {fb['strength']}")
        tips = fb["next_steps"][:2]
    else:
        lines.append(f"{it('growth_opportunity')}: {f1['growth']}")
        tips = f1["next_steps"][:2]

    lines.append(it("try_next_lesson"))
    for tip in tips:
        lines.append(f"• {tip}")

    return lines



def _pick_top2_bottom_used(group_df: pd.DataFrame):
    """
    group_df columns: strategy, count, percent
    Returns (top1_row, top2_row_or_None, bottom_row_or_None)
    Bottom is least-used among strategies with count>0.
    """
    g = group_df.copy()
    g = g[pd.to_numeric(g["count"], errors="coerce").fillna(0).astype(int) > 0]
    if g.empty:
        return None, None, None

    g = g.sort_values(["percent", "count"], ascending=False)
    top1 = g.iloc[0]
    top2 = g.iloc[1] if len(g) > 1 else None

    bottom = None
    if len(g) > 1:
        bottom = g.sort_values(["percent", "count"], ascending=True).iloc[0]

    return top1, top2, bottom

def _profile_text(top1, top2, bottom):
    """
    Build a teacher-friendly profile narrative + 2–3 coaching tips.
    """
    # Primary narrative driven by top1
    s1 = top1["strategy"]
    narrative = STRATEGY_WHAT_IT_LOOKS_LIKE.get(s1, "")

    # Tips: mostly from top1, plus one "growth idea" from bottom if different
    tips = list(STRATEGY_COACHING_TIPS.get(s1, []))[:2]

    if bottom is not None:
        sb = bottom["strategy"]
        if sb != s1:
            tips.append(f"Growth idea: practice more '{_friendly(sb)}' moments (small, controlled tests).")

    tips = tips[:3]
    return narrative, tips

def _display_insights(ax, summary_df):
    """
    Scrollable teacher-facing insights view.
    Uses a vertical slider + mouse wheel scrolling.
    """
    fig = ax.figure
    ax.clear()
    ax.axis("off")

    if summary_df is None or not hasattr(summary_df, "empty") or summary_df.empty:
        ax.text(0.5, 0.5, it("no_insights"), ha="center", va="center", fontsize=12)
        return

    needed = {"user_id", "strategy", "count", "percent"}
    if not needed.issubset(set(summary_df.columns)):
        ax.text(0.5, 0.5, f"Insights need columns: {sorted(needed)}", ha="center", va="center", fontsize=11)
        return

    # Build ALL text lines first (for all users)
    lines = []
    lines.append(it("insights_title"))
    lines.append(it("insights_subtitle"))

    lines.append("")  # spacer

    for user_id, group in summary_df.groupby("user_id"):
        top1, top2, bottom = _pick_top2_bottom_used(group)
        if top1 is None:
            continue

        lines.append(it("user_header", user=user_id))
        lines.append("-" * 40)
        lines.extend(_build_insight_lines(top1, top2, bottom))
        lines.append("")  # spacer between students

    # ----- Rendering parameters -----
    # How many lines fit in the panel:
    max_visible = 26  # adjust if you want more/less text density
    total = len(lines)
    scroll_max = max(0, total - max_visible)

    # Keep scroll state on the figure so it persists when toggling views/language
    if not hasattr(fig, "_insights_scroll"):
        fig._insights_scroll = {"offset": 0, "slider_ax": None, "slider": None, "cid_scroll": None}

    state = fig._insights_scroll
    state.setdefault("render_fn", None)

    state["offset"] = min(max(0, state["offset"]), scroll_max)

    # Create / reuse a vertical slider at the right side of the display panel
    # We place it relative to the figure, so it stays aligned.
    # If it already exists from a previous Insights render, reuse it.
    if state["slider_ax"] is None or state["slider"] is None:
        slider_ax = fig.add_axes([0.94, 0.12, 0.02, 0.76])  # x,y,w,h in fig coords
        slider = Slider(
            slider_ax,
            "",
            0,
            max(1, scroll_max),
            valinit=state["offset"],
            valstep=1,
            orientation="vertical",
        )
        state["slider_ax"] = slider_ax
        state["slider"] = slider

        def _on_slider(val):
            state["offset"] = int(val)
            if state.get("render_fn"):
                state["render_fn"]()


        slider.on_changed(_on_slider)
    else:
        # Update slider bounds if number of lines changed
        slider = state["slider"]
        slider.valmin = 0
        slider.valmax = max(1, scroll_max)
        slider.ax.set_ylim(slider.valmin, slider.valmax)
        slider.set_val(state["offset"])

    # Mouse wheel scroll support
    if state.get("cid_scroll") is None:
        def _on_scroll(event):
            if event.inaxes != ax:
                return

            step = 3
            if event.button == "up":
                state["offset"] = max(0, state["offset"] - step)
            elif event.button == "down":
                state["offset"] = min(scroll_max, state["offset"] + step)

            # sync slider without breaking
            try:
                state["slider"].set_val(state["offset"])
            except Exception:
                pass

            if state.get("render_fn"):
                state["render_fn"]()

        state["cid_scroll"] = fig.canvas.mpl_connect("scroll_event", _on_scroll)

    def _render():
        state["render_fn"] = _render

        ax.clear()
        ax.axis("off")

        # Title block (fixed)
        ax.text(0.0, 0.98, lines[0], fontsize=14, fontweight="bold", transform=ax.transAxes)
        ax.text(0.0, 0.93, lines[1], fontsize=9, color="dimgray", transform=ax.transAxes)

        # Visible window into the lines (skip first 3 "header" lines in scrolling)
        body = lines[3:]
        offset = state["offset"]
        view = body[offset: offset + max_visible]

        y = 0.88
        line_h = 0.032  # tighter for more content

        for L in view:
            if L.startswith("User ") or L.startswith("Gebruiker "):
                ax.text(0.0, y, L, fontsize=11.5, fontweight="bold", transform=ax.transAxes)
            elif L.startswith("-" * 5):
                ax.text(0.0, y, L, fontsize=9, color="gray", transform=ax.transAxes)
            elif L.startswith(it("try_next_lesson")):
                ax.text(0.02, y, L, fontsize=9.5, fontweight="bold", transform=ax.transAxes)
            elif L.startswith("• "):
                ax.text(0.04, y, L, fontsize=9.2, color="dimgray", transform=ax.transAxes)
            elif L.strip() == "":
                # spacer line
                pass
            else:
                ax.text(0.02, y, L, fontsize=9.2, transform=ax.transAxes)

            y -= line_h
            if y < 0.06:
                break

        # Hint
        if scroll_max > 0:
            ax.text(0.0, 0.02, it("scroll_hint"), fontsize=9, color="dimgray", transform=ax.transAxes)


        fig.canvas.draw_idle()

    _render()




def show_strategy_dashboard(summary_df, bar_figure_path, radar_figure_path, show=True, block=True):
    """
    Display a second 'page' with menu controls to view bar chart, radar chart, or insights.
    Includes a language toggle button (EN/NL) just like the main window.
    """
    roles = derive_player_roles(summary_df)
    fig = plt.figure(figsize=(12, 6))
    DASHBOARD_WINDOWS.append(fig)

    def _on_close(_event):
        if fig in DASHBOARD_WINDOWS:
            DASHBOARD_WINDOWS.remove(fig)

    fig.canvas.mpl_connect("close_event", _on_close)

    # --- layout axes ---
    menu_ax = plt.axes([0.02, 0.25, 0.15, 0.45])
    display_ax = plt.axes([0.25, 0.1, 0.7, 0.8])

    # --- translated option labels ---
    def _option_labels():
        return [t('bar_chart'), t('radar_chart'), t('insights')]

    options = _option_labels()
    radio = RadioButtons(menu_ax, options)

    def _which_view(label):
        labels = _option_labels()
        if label == labels[0]:
            return "bar"
        if label == labels[1]:
            return "radar"
        return "insights"

    dashboard_state = {"view": "bar"}  # remember last view across language toggles

    def update_display(label_or_view):
        # accept either "bar"/"radar"/"insights" or a radio label
        view = label_or_view
        if view not in ("bar", "radar", "insights"):
            view = _which_view(label_or_view)

        dashboard_state["view"] = view

        if view == "bar":
            _display_image(display_ax, bar_figure_path, t('per_user_counts'))
        elif view == "radar":
            _display_image(display_ax, radar_figure_path, t('strategy_radar'))
        else:

            _display_insights(display_ax, summary_df)


        fig.canvas.draw_idle()

    # initial titles
    fig.suptitle(t('dashboard_title'), fontsize=16)
    menu_ax.set_title(t('views_title'), fontsize=11)

    radio.on_clicked(update_display)
    update_display("bar")  # default view

    # ----------------------------------------------------------
    # LANGUAGE TOGGLE BUTTON (add to dashboard)
    # ----------------------------------------------------------
    lang_button_ax = plt.axes([0.02, 0.02, 0.04, 0.04])
    lang_button = Button(lang_button_ax, current_language['lang'], color='#4e409f', hovercolor='#64748b')
    lang_button.label.set_color('white')
    lang_button.label.set_fontweight('bold')
    lang_button.label.set_fontsize(11)

    def refresh_dashboard_text():
        # Update suptitle + menu title
        fig.suptitle(t('dashboard_title'), fontsize=16)
        menu_ax.set_title(t('views_title'), fontsize=11)

        # Update radio button labels (RadioButtons doesn't have a "set_labels" API,
        # easiest is: rewrite the Text objects in-place)
        new_labels = _option_labels()
        for txt, new in zip(radio.labels, new_labels):
            txt.set_text(new)

        # Re-render current view title in the display panel
        update_display(dashboard_state["view"])
        fig.canvas.draw_idle()

    def toggle_language_dashboard(_event):
        current_language['lang'] = 'NL' if current_language['lang'] == 'EN' else 'EN'
        lang_button.label.set_text(current_language['lang'])
        refresh_dashboard_text()

    lang_button.on_clicked(toggle_language_dashboard)

    if show:
        plt.show(block=block)
        if not block:
            plt.pause(0.05)
    else:
        plt.close(fig)


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
    ax_speed.set_title(t('title'), fontsize=16, pad=10)

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
    ax_speed.set_ylabel(t('speed_ylabel'))
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

    # Store all strategy patches for hover detection
    strategy_patches = []  # List of (patch, strategy_name, user, start, end)

    for i, user in enumerate(users):
        ax = axes[i + 1]
        ax.set_title(t('user_title', user=user, strategy=t('no_strategy')), fontsize=14, pad=6)

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
            patch = ax.axvspan(
                s, e,
                color=color,
                alpha=0.8,
                picker=True  # Enable picking for hover detection
            )
            # Store patch info for hover detection
            strategy_patches.append((patch, strat, user, s, e, ax))

        ax.set_yticks([])
        ax.set_ylabel(t('user_label', user=user), rotation=0, labelpad=30)
        ax.grid(axis='x', alpha=0.2)

    # ----------------------------------------------------------
    # GLOBAL TIME AXIS
    # ----------------------------------------------------------
    axes[-1].set_xlabel(t('time_xlabel'), fontsize=14)
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
        legend_labels.append(translate_strategy(strat))

    # Store legend reference for updating
    legend_ref = {'legend': None, 'strategies': sorted(used_strategies), 'expanded_strategy': None}
    if legend_handles:
        legend_ref['legend'] = fig.legend(
            legend_handles,
            legend_labels,
            title=t('strategies_legend'),
            loc="upper left",
            bbox_to_anchor=(0.83, 0.75),
            frameon=True,
            fontsize=11
        )

    # ----------------------------------------------------------
    # HOVER TOOLTIP FOR STRATEGY DEFINITION
    # ----------------------------------------------------------
    # Create a text annotation for showing strategy definitions on hover (follows mouse)
    # Store in a dict so we can update reference without nonlocal issues
    hover_annotation_ref = {
        'annot': axes[1].annotate(
            '', 
            xy=(0, 0),
            xytext=(15, 15),
            textcoords='offset points',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95),
            visible=False,
            zorder=1000
        )
    }
    hover_state = {'current_strategy': None, 'current_ax': None}

    def update_legend_with_expansion(hovered_strategy=None, mouse_x=None, mouse_y=None, ax=None):
        """Update legend, expanding the hovered strategy with its definition."""
        if legend_ref['legend'] is not None:
            legend_ref['legend'].remove()
        
        legend_handles = []
        legend_labels = []
        
        for strat in legend_ref['strategies']:
            c = colors.get(strat, "#CCCCCC")
            patch = plt.Line2D([0], [0], color=c, linewidth=12)
            legend_handles.append(patch)
            
            translated = translate_strategy(strat)
            if hovered_strategy and strat == hovered_strategy:
                # Add arrow indicator for expanded strategy
                legend_labels.append(f"► {translated}")
            else:
                legend_labels.append(translated)
        
        if legend_handles:
            legend_ref['legend'] = fig.legend(
                legend_handles,
                legend_labels,
                title=t('strategies_legend'),
                loc="upper left",
                bbox_to_anchor=(0.83, 0.75),
                frameon=True,
                fontsize=11
            )
        
        # Show/hide definition tooltip next to mouse
        if hovered_strategy and mouse_x is not None and mouse_y is not None and ax is not None:
            definition = get_strategy_definition(hovered_strategy)
            # Wrap text manually for better display
            wrapped = '\n'.join([definition[i:i+40] for i in range(0, len(definition), 40)])
            
            # Move annotation to the correct axes if needed
            if hover_state['current_ax'] != ax:
                hover_annotation_ref['annot'].remove()
                hover_state['current_ax'] = ax
                # Recreate annotation on the new axes
                hover_annotation_ref['annot'] = ax.annotate(
                    wrapped,
                    xy=(mouse_x, mouse_y),
                    xytext=(15, 15),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95),
                    visible=True,
                    zorder=1000
                )
            else:
                # Update annotation position and text
                hover_annotation_ref['annot'].xy = (mouse_x, mouse_y)
                hover_annotation_ref['annot'].set_text(wrapped)
                hover_annotation_ref['annot'].set_visible(True)
        else:
            hover_annotation_ref['annot'].set_visible(False)
        
        legend_ref['expanded_strategy'] = hovered_strategy
        fig.canvas.draw_idle()

    def on_hover(event):
        """Handle mouse motion for hover detection on strategy bars."""
        if event.inaxes is None:
            if hover_state['current_strategy'] is not None:
                hover_state['current_strategy'] = None
                update_legend_with_expansion(None)
            return
        
        # Check if mouse is over any strategy patch
        found_strategy = None
        found_ax = None
        mouse_x, mouse_y = event.xdata, event.ydata
        
        for patch, strat, user, start, end, ax in strategy_patches:
            if event.inaxes == ax:
                # Check if x position is within the patch bounds
                if start <= event.xdata <= end:
                    found_strategy = strat
                    found_ax = ax
                    break
        
        # Always update position when hovering on a strategy bar
        if found_strategy is not None:
            # Strategy changed - update legend and recreate annotation
            if found_strategy != hover_state['current_strategy'] or hover_state['current_ax'] != found_ax:
                hover_state['current_strategy'] = found_strategy
                update_legend_with_expansion(found_strategy, mouse_x, mouse_y, found_ax)
            else:
                # Same strategy, same axes - just update position
                # Remove old annotation and recreate at new position for smooth following
                hover_annotation_ref['annot'].remove()
                definition = get_strategy_definition(found_strategy)
                wrapped = '\n'.join([definition[i:i+40] for i in range(0, len(definition), 40)])
                hover_annotation_ref['annot'] = found_ax.annotate(
                    wrapped,
                    xy=(mouse_x, mouse_y),
                    xytext=(15, 15),
                    textcoords='offset points',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray', alpha=0.95),
                    visible=True,
                    zorder=1000
                )
                fig.canvas.draw_idle()
        elif hover_state['current_strategy'] is not None:
            # Mouse moved off a bar
            hover_state['current_strategy'] = None
            update_legend_with_expansion(None)

    # Connect hover event
    fig.canvas.mpl_connect('motion_notify_event', on_hover)

    # ----------------------------------------------------------
    # INTERACTIVE TIMELINE SLIDER
    # ----------------------------------------------------------
    # Create space for slider and buttons at the bottom
    plt.tight_layout(rect=(0, 0.12, 0.82, 1))
    
    # Add time slider axis
    slider_ax = plt.axes([0.15, 0.07, 0.65, 0.02])
    time_slider = Slider(
        slider_ax,
        t('time_slider'),
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
        t('zoom_slider'),
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
        zoom_text.set_text(t('zoom_display', level=zoom_level))
        
        fig.canvas.draw_idle()
    
    def get_current_strategy(user, current_time):
        """Get the strategy for a user at the current time."""
        if user not in user_segments:
            return t('no_strategy')
        
        for start, end, strategy in user_segments[user]:
            if start <= current_time <= end:
                return translate_strategy(strategy)
        
        return t('no_strategy')
    
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
            ax.set_title(t('user_title', user=user, strategy=current_strategy), fontsize=14, pad=6)
        
        # Update view to follow current time with current zoom
        update_view()
    
    def update_zoom(val):
        """Update zoom level and adjust view."""
        update_view()
    
    # Autoplay state
    autoplay_state = {'playing': False, 'last_real_time': None, 'anim': None}
    
    def animation_frame(frame):
        """Animation frame function for autoplay."""
        if not autoplay_state['playing']:
            return vertical_lines
        
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
            play_button.label.set_text(t('play'))
            time_slider.set_val(t_max)
            autoplay_state['last_real_time'] = None
            return vertical_lines
        
        time_slider.set_val(new_time)
        return vertical_lines
    
    def toggle_autoplay(event):
        """Toggle autoplay on/off."""
        if autoplay_state['playing']:
            # Stop autoplay
            autoplay_state['playing'] = False
            play_button.label.set_text(t('play'))
            autoplay_state['last_real_time'] = None
        else:
            # Start autoplay
            autoplay_state['playing'] = True
            play_button.label.set_text(t('pause'))
            autoplay_state['last_real_time'] = time.time()
    
    # Create animation for autoplay (runs continuously but only advances when playing)
    anim = FuncAnimation(fig, animation_frame, interval=33, blit=False, cache_frame_data=False)
    
    def refresh_all_text():
        """Refresh all translatable text elements after language change."""
        # Update main title
        axes[0].set_title(t('title'), fontsize=16, pad=10)
        axes[0].set_ylabel(t('speed_ylabel'))
        
        # Update time axis label
        axes[-1].set_xlabel(t('time_xlabel'), fontsize=14)
        
        # Update user panel labels and titles
        current_time = time_slider.val
        for i, user in enumerate(users):
            ax = axes[i + 1]
            current_strategy = get_current_strategy(user, current_time)
            ax.set_title(t('user_title', user=user, strategy=current_strategy), fontsize=14, pad=6)
            ax.set_ylabel(t('user_label', user=user), rotation=0, labelpad=30)
        
        # Update slider labels
        slider_ax.set_xlabel(t('time_slider'))
        zoom_ax.set_xlabel(t('zoom_slider'))
        
        # Update zoom text
        zoom_text.set_text(t('zoom_display', level=zoom_slider.val))
        
        # Update play button
        if autoplay_state['playing']:
            play_button.label.set_text(t('pause'))
        else:
            play_button.label.set_text(t('play'))
        
        # Update legend
        if legend_ref['legend'] is not None:
            legend_ref['legend'].remove()
        
        legend_handles = []
        legend_labels = []
        for strat in legend_ref['strategies']:
            c = colors.get(strat, "#CCCCCC")
            patch = plt.Line2D([0], [0], color=c, linewidth=12)
            legend_handles.append(patch)
            legend_labels.append(translate_strategy(strat))
        
        if legend_handles:
            legend_ref['legend'] = fig.legend(
                legend_handles,
                legend_labels,
                title=t('strategies_legend'),
                loc="upper left",
                bbox_to_anchor=(0.83, 0.75),
                frameon=True,
                fontsize=11
            )
        
        fig.canvas.draw_idle()
    
    def toggle_language(event):
        """Toggle between EN and NL languages (NS app style)."""
        if current_language['lang'] == 'EN':
            current_language['lang'] = 'NL'
            lang_button.label.set_text('NL')
            lang_button.color = '#4e409f'  # NS blue
        else:
            current_language['lang'] = 'EN'
            lang_button.label.set_text('EN')
            lang_button.color = '#4e409f'  # NS yellow
        
        refresh_all_text()
    
    # Add Play/Pause button
    button_ax = plt.axes([0.02, 0.07, 0.08, 0.03])
    play_button = Button(button_ax, t('play'), color='#4e409f', hovercolor='#64748b')
    play_button.label.set_color('white')
    play_button.label.set_fontweight('bold')
    play_button.on_clicked(toggle_autoplay)

    def open_dashboard(event):
        script_path = Path(__file__).resolve()
        python_exe = sys.executable or "python3"
        cmd = [
            python_exe,
            str(script_path),
            "--dashboard",
            "--show-dashboard",
        ]
        try:
            subprocess.Popen(cmd, cwd=str(script_path.parent))
        except Exception as exc:
            print(f"[WARN] Failed to launch dashboard process: {exc}")

    nav_button_ax = plt.axes([0.82, 0.94, 0.12, 0.035])
    nav_button = Button(nav_button_ax, t('dashboard_button'), color='#0d9488', hovercolor='#14b8a6')
    nav_button.label.set_color('white')
    nav_button.label.set_fontweight('bold')
    nav_button.on_clicked(open_dashboard)
    
    # ----------------------------------------------------------
    # LANGUAGE TOGGLE BUTTON (NS app style)
    # ----------------------------------------------------------
    lang_button_ax = plt.axes([0.02, 0.02, 0.04, 0.03])
    lang_button = Button(lang_button_ax, 'EN', color='#4e409f', hovercolor='#64748b')
    lang_button.label.set_color('white')
    lang_button.label.set_fontweight('bold')
    lang_button.label.set_fontsize(11)
    lang_button.on_clicked(toggle_language)
    
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
    Legacy verbose diagnostics that previously ran unconditionally.
    Trigger with --debug-snippet if you still need that workflow.
    """
    from collections import Counter
    import os
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
    parser = argparse.ArgumentParser(description="Strategy timeline, reporting, and radar tools.")
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate the per-user strategy usage report instead of the interactive timeline."
    )
    parser.add_argument("--report-csv", default="graphs/strategy_usage_summary.csv", help="CSV path for the summary.")
    parser.add_argument("--report-figure", default="graphs/strategy_usage_report.png",
                        help="Image path for the per-user bar plots.")
    parser.add_argument("--report-min-actions", type=int, default=1,
                        help="Minimum actions required for a segment to count in the report.")
    parser.add_argument("--label-column", default=None,
                        help="Override the label column used for aggregation (default: pred_strategy/global_label).")
    parser.add_argument("--show-report", action="store_true", help="Display the generated report figure interactively.")
    parser.add_argument("--radar", action="store_true", help="Generate radar chart(s) of per-user strategy usage.")
    parser.add_argument("--radar-figure", default="graphs/strategy_usage_radar.png",
                        help="Image path for the radar visualization.")
    parser.add_argument("--radar-min-actions", type=int, default=1,
                        help="Minimum actions required for radar aggregation.")
    parser.add_argument("--radar-value", choices=("percent", "count"), default="percent",
                        help="Metric plotted on the radar chart.")
    parser.add_argument("--show-radar", action="store_true", help="Display the radar chart interactively.")
    parser.add_argument("--dashboard", action="store_true",
                        help="Open a second page with menu-driven charts and player insights.")
    parser.add_argument("--show-dashboard", action="store_true",
                        help="Display the dashboard interactively when --dashboard is used.")
    parser.add_argument("--debug-snippet", action="store_true",
                        help="Run the legacy verbose debugging snippet after the main task.")
    parser.add_argument("--debug-log-dir", default="logs", help="Log directory used by the debug snippet.")
    parser.add_argument("--debug-start", type=float, default=155.0, help="Start time for debug snippet window.")
    parser.add_argument("--debug-end", type=float, default=205.0, help="End time for debug snippet window.")
    parser.add_argument("--language", choices=list(TRANSLATIONS.keys()), default="EN",
                        help="Interface language for the timeline plot.")

    args = parser.parse_args()
    current_language["lang"] = args.language

    df = None
    summary_df = None
    if args.report or args.radar or args.dashboard:
        df = load_predictions()

    if args.report or args.dashboard:
        summary_df = generate_strategy_usage_report(
            df,
            csv_path=args.report_csv,
            figure_path=args.report_figure,
            label_column=args.label_column,
            min_actions=args.report_min_actions,
            show_plot=args.show_report,
        )

    if args.radar or args.dashboard:
        radar_summary = generate_strategy_usage_radar(
            df,
            figure_path=args.radar_figure,
            label_column=args.label_column,
            min_actions=args.radar_min_actions,
            value_col=args.radar_value,
            show_plot=args.show_radar,
        )
        if summary_df is None:
            summary_df = radar_summary

    if args.dashboard:
        if summary_df is None:
            min_actions = min(args.report_min_actions, args.radar_min_actions)
            summary_df_raw, used_label = summarize_strategy_usage(
                df,
                label_column=args.label_column,
                min_actions=min_actions,
            )
            summary_df = summary_df_raw.rename(columns={used_label: "strategy"})
        show_strategy_dashboard(summary_df, args.report_figure, args.radar_figure, show=args.show_dashboard)

    if not args.report and not args.radar and not args.dashboard:
        plot_timeline()

    if args.debug_snippet:
        _legacy_debug_snippet(log_dir=args.debug_log_dir, t_start=args.debug_start, t_end=args.debug_end)


if __name__ == "__main__":
    main()
