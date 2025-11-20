from typing import List, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import re

STRAT_COL_FIGHT = "#D32F2F"
STRAT_COL_CARE = "#2E7D32"
STRAT_COL_RECKLESS = "#1565C0"
STRAT_COL_TRIAL = "#EF6C00"
LANE_BG = "#f6f6f6"
LANE_EDGE = "#cccccc"
LANE_ALPHA_BG = 0.9
LANE_HEIGHT = 0.06

def draw_fighting_overlays(ax, param_name: str, fighting_windows, merged_arr=None):
    """Draw fighting strategy windows as colored axvspan rectangles."""
    if not fighting_windows:
        return
    for w in fighting_windows:
        if getattr(w, "param", None) is not None and param_name is not None and w.param != param_name:
            continue
        start = float(w.start_time)
        end = float(w.end_time)
        ax.axvspan(start, end,
                   facecolor=STRAT_COL_FIGHT, alpha=0.35,
                   edgecolor="k", linewidth=0.4, zorder=2)

def draw_carefulness_overlays(ax, param_name: str, care_windows):
    """Draw carefulness windows for careful vs reckless metadata."""
    if not care_windows:
        return
    for w in care_windows:
        if getattr(w, "param", None) is not None and param_name is not None and w.param != param_name:
            continue
        start = float(w.start_time)
        end = float(w.end_time)
        behavior = (w.meta.get("behavior_type") or "").lower() if getattr(w, "meta", None) else ""
        if "reckless" in behavior:
            col = STRAT_COL_RECKLESS
            alpha = 0.30
        else:
            col = STRAT_COL_CARE
            alpha = 0.35
        ax.axvspan(start, end,
                   facecolor=col, alpha=alpha,
                   edgecolor="k", linewidth=0.35, zorder=2)

def draw_trial_error_overlays(ax, param_name: str, te_windows):
    """Draw trial & error high windows as colored axvspan rectangles."""
    if not te_windows:
        return
    for w in te_windows:
        tag = (getattr(w, "tag", "") or "").lower()
        if "trial_error_high" not in tag and "trial_error:high" not in tag:
            continue
        if getattr(w, "param", None) is not None and param_name is not None and w.param != param_name:
            continue
        start = float(w.start_time)
        end = float(w.end_time)
        ax.axvspan(start, end,
                   facecolor=STRAT_COL_TRIAL, alpha=0.30,
                   edgecolor="k", linewidth=0.3, zorder=2)

def draw_strategy_lanes(ax, param_name: str,
                        fighting_windows, care_windows, te_windows):
    """Draw non-overlapping horizontal lanes at the bottom of the axis."""
    lane_specs = []
    if fighting_windows:
        lane_specs.append(("fighting", fighting_windows))
    if care_windows:
        lane_specs.append(("carefulness", care_windows))
    high_te = [w for w in (te_windows or []) if "trial_error_high" in str(getattr(w, "tag", "")).lower()
               or "trial_error:high" in str(getattr(w, "tag", "")).lower()]
    if high_te:
        lane_specs.append(("trial_error", high_te))

    if not lane_specs:
        return

    total_lane_height = LANE_HEIGHT * len(lane_specs)
    if total_lane_height > 0.30:
        scale = 0.30 / total_lane_height
    else:
        scale = 1.0

    for i, (kind, windows) in enumerate(lane_specs):
        ymin = i * LANE_HEIGHT * scale
        ymax = (i + 1) * LANE_HEIGHT * scale
        ax.axhspan(ymin, ymax, facecolor=LANE_BG, alpha=LANE_ALPHA_BG,
                   edgecolor=LANE_EDGE, linewidth=0.4, zorder=1,
                   transform=ax.get_yaxis_transform())

    for i, (kind, windows) in enumerate(lane_specs):
        ymin = i * LANE_HEIGHT * scale
        ymax = (i + 1) * LANE_HEIGHT * scale
        for w in windows:
            if getattr(w, "param", None) is not None and param_name is not None and w.param != param_name:
                continue
            start = float(w.start_time)
            end = float(w.end_time)

            if kind == "fighting":
                color = STRAT_COL_FIGHT
                alpha = 0.65
            elif kind == "carefulness":
                behavior = (w.meta.get("behavior_type") or "").lower() if getattr(w, "meta", None) else ""
                if "reckless" in behavior:
                    color = STRAT_COL_RECKLESS
                    alpha = 0.55
                else:
                    color = STRAT_COL_CARE
                    alpha = 0.65
            elif kind == "trial_error":
                color = STRAT_COL_TRIAL
                alpha = 0.60
            else:
                color = "gray"
                alpha = 0.5

            ax.axvspan(start, end,
                       ymin=ymin, ymax=ymax,
                       facecolor=color, alpha=alpha,
                       edgecolor="k", linewidth=0.35, zorder=3)

def _is_high_trial_error(w) -> bool:
    tag = (getattr(w, "tag", "") or "").lower()
    return bool(re.search(r"\btrial[_:]error[:_]high\b", tag))

def _merge_time_spans(windows, max_gap: float = 0.0):
    """Merge overlapping or touching intervals."""
    spans = sorted(
        [(float(getattr(w, "start_time", 0.0)), float(getattr(w, "end_time", 0.0))) for w in (windows or [])],
        key=lambda x: x[0],
    )
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1] + max_gap:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return merged

def draw_strategy_bands(ax, param_name: str,
                        fighting_windows, care_windows, te_windows):
    """Partition entire plot height into equal horizontal bands."""
    fighting_ws = [w for w in (fighting_windows or [])
                   if (getattr(w, "param", None) in (None, param_name))]
    care_ws_all = [w for w in (care_windows or [])
                   if (getattr(w, "param", None) in (None, param_name))]
    careful_ws = []
    reckless_ws = []
    for w in care_ws_all:
        behavior = (w.meta.get("behavior_type") or "").lower() if getattr(w, "meta", None) else ""
        if "reckless" in behavior:
            reckless_ws.append(w)
        else:
            careful_ws.append(w)
    te_high_ws = [
        w for w in (te_windows or [])
        if (getattr(w, "param", None) in (None, param_name)) and _is_high_trial_error(w)
    ]

    bands = []
    if fighting_ws:
        bands.append(("Fighting", fighting_ws, STRAT_COL_FIGHT))
    if careful_ws:
        bands.append(("Careful", careful_ws, STRAT_COL_CARE))
    if reckless_ws:
        bands.append(("Reckless", reckless_ws, STRAT_COL_RECKLESS))
    if te_high_ws:
        bands.append(("Trial & Error (High)", te_high_ws, STRAT_COL_TRIAL))

    if not bands:
        return

    n = len(bands)
    for i, (label, windows, color) in enumerate(bands):
        ymin = i / n
        ymax = (i + 1) / n
        ax.axhspan(ymin, ymax,
                   facecolor=LANE_BG, alpha=LANE_ALPHA_BG,
                   edgecolor=LANE_EDGE, linewidth=0.4, zorder=1,
                   transform=ax.get_yaxis_transform())

    for i, (label, windows, color) in enumerate(bands):
        ymin = i / n
        ymax = (i + 1) / n

        if label.lower().startswith("trial"):
            for s, e in _merge_time_spans(windows, max_gap=0.0):
                ax.axvspan(s, e,
                           ymin=ymin, ymax=ymax,
                           facecolor=color, alpha=0.9,
                           edgecolor="k", linewidth=0.35, zorder=2)
        else:
            for w in windows:
                start = float(w.start_time)
                end = float(w.end_time)
                ax.axvspan(start, end,
                           ymin=ymin, ymax=ymax,
                           facecolor=color, alpha=0.65,
                           edgecolor="k", linewidth=0.35, zorder=2)

def draw_strategy_legend(ax_legend, fighting_windows, care_windows, te_windows):
    """Legend synchronized with band allocation."""
    ax_legend.clear()
    ax_legend.set_facecolor("#ececec")
    ax_legend.patch.set_edgecolor("#cfcfcf")
    ax_legend.patch.set_linewidth(0.8)
    ax_legend.axis("off")

    handles = []
    has_fight = any(fighting_windows)
    care_ws_all = care_windows or []
    has_careful = any(
        ("reckless" not in ((w.meta or {}).get("behavior_type") or "").lower())
        for w in care_ws_all
    )
    has_reckless = any(
        ("reckless" in ((w.meta or {}).get("behavior_type") or "").lower())
        for w in care_ws_all
    )
    has_te_high = any(
        ("trial_error_high" in str(getattr(w, "tag", "")).lower() or
         "trial_error:high" in str(getattr(w, "tag", "")).lower())
        for w in (te_windows or [])
    )
    if has_fight:
        handles.append(Patch(facecolor=STRAT_COL_FIGHT, edgecolor="k", label="Fighting", alpha=0.9))
    if has_careful:
        handles.append(Patch(facecolor=STRAT_COL_CARE, edgecolor="k", label="Careful", alpha=0.9))
    if has_reckless:
        handles.append(Patch(facecolor=STRAT_COL_RECKLESS, edgecolor="k", label="Reckless", alpha=0.9))
    if has_te_high:
        handles.append(Patch(facecolor=STRAT_COL_TRIAL, edgecolor="k", label="Trial & Error (High)", alpha=0.9))

    if handles:
        ax_legend.legend(handles=handles, loc="center", frameon=False, fontsize=9)