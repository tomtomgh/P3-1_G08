from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt


def plot_shared_data(ax, timeline, merged_data_for_param, owner_arr, active_users, color_map):
    i = 0
    added_labels = set()
    while i < len(timeline):
        if __import__("numpy").isnan(merged_data_for_param[i]):
            i += 1
            continue
        current_owner = owner_arr[i]
        j = i
        while (
            j < len(timeline)
            and owner_arr[j] == current_owner
            and not __import__("numpy").isnan(merged_data_for_param[j])
        ):
            j += 1
        if current_owner in active_users and current_owner >= 0:
            label = f"User {current_owner}" if current_owner not in added_labels else ""
            ax.plot(
                timeline[i:j],
                merged_data_for_param[i:j],
                color=color_map[current_owner],
                linewidth=2.5,
                label=label,
            )
            added_labels.add(current_owner)
        i = j
    if added_labels:
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=9)


def plot_per_user_data(ax, timeline, series_data: Dict[int, List[float]], active_users, color_map):
    for u in sorted(active_users):
        ax.plot(
            timeline,
            series_data[u],
            color=color_map[u],
            linewidth=2.0,
            label=f"User {u}",
        )
    ax.legend(loc="upper right", fontsize=9)


def update_x_limits(full_view, time_window_start, axes, t_min, t_max):
    if full_view[0]:
        for ax in axes:
            ax.set_xlim(t_min, t_max)
    else:
        start = time_window_start[0]
        end = start + 50.0
        for ax in axes:
            ax.set_xlim(start, end)


def draw_carefulness_page(ax_care, carefulness_summary, users, fig=None):
    from .StatsPanel import draw_carefulness_page as _draw_carefulness_page
    return _draw_carefulness_page(ax_care, carefulness_summary, users, fig)