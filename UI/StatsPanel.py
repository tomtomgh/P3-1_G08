import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from Analysis import UserActivityStats

USER_COLORS = {
    0: "#1f77b4",  # Blue
    1: "#ff7f0e",  # Orange
    2: "#2ca02c",  # Green
    3: "#d62728",  # Red
}

def update_stats_display(
    ax: plt.Axes,
    user_stats: Dict[int, UserActivityStats],
    total_duration: float,
    trial_scores: Optional[Dict[int, float]] = None,
    carefulness_summary: Optional[Dict[int, Dict[str, Dict[str, float | str]]]] = None,
) -> None:
    ax.clear()
    ax.axis("off")

    # Left panel: textual statistics (draw directly on ax)
    stats_text = ""
    stats_text += "║ USER STATISTICS\n"
    for user, stats in user_stats.items():
        stats_text += f"║ \u25aa User {user}:\n"
        stats_text += f"   ║ Changes: {stats.total_changes} | Rate: {stats.changes_per_minute:.1f} chg/min\n"
        if "frequency" in stats.time_in_control and total_duration > 0:
            freq_t = stats.time_in_control["frequency"]
            freq_pct = (freq_t / total_duration) * 100.0
            stats_text += f"   ║ Frequency Dominance: {freq_pct:.1f}%\n"
        if trial_scores and user in trial_scores:
            stats_text += f"   ║ Trial & Error Score: {trial_scores[user]:.0f}%\n"
        stats_text += f"   ║ Max Inactivity Gap: {stats.max_inactivity_gap:.1f}s\n\n"

    stats_text += "║ HIGHLIGHTS\n"
    if user_stats:
        users = list(user_stats.keys())
        most_active = max(users, key=lambda u: user_stats[u].total_changes)
        most_resp = max(users, key=lambda u: user_stats[u].changes_per_minute)
        most_freq = max(users, key=lambda u: user_stats[u].time_in_control.get("frequency", 0.0))
        stats_text += f"║ Most Active: User {most_active}\n"
        stats_text += f"║ Most Responsive: User {most_resp}\n"
        stats_text += f"║ Most Dominant (Frequency): User {most_freq}\n"
        if trial_scores:
            top_te = max(trial_scores, key=trial_scores.get)
            stats_text += f"║ Most Trial & Error: User {top_te} ({trial_scores[top_te]:.0f}%)\n"

    ax.text(
        0.02, 0.98, stats_text, va="top", ha="left",
        fontfamily="monospace", fontsize=8,
        bbox=dict(boxstyle="round", facecolor="#ffffe0", edgecolor="gray"),
        transform=ax.transAxes
    )

    # Right panel: compact bars drawn as inset_axes
    def draw_bars(inset_ax, title, data, labelsuffix=""):
        inset_ax.set_title(title, fontsize=8, loc="left")
        y_pos = np.arange(len(data))
        bars = inset_ax.barh(y_pos, [d[1] for d in data], color=[USER_COLORS[u] for u, _ in data], edgecolor='black')
        inset_ax.set_xlim(0, 100)
        inset_ax.set_yticks(y_pos)
        inset_ax.set_yticklabels([f"User {u}" for u, _ in data], fontsize=7)
        inset_ax.invert_yaxis()
        inset_ax.set_xticks([])
        for i, (user, val) in enumerate(data):
            inset_ax.text(102, i, f"{val:.1f}{labelsuffix}", va="center", fontsize=7)
        inset_ax.tick_params(left=False, labelleft=True)

    # Inset positions (right side of ax)
    if user_stats:
        total_changes = sum(s.total_changes for s in user_stats.values())
        engage_data = [(u, (s.total_changes / total_changes) * 100 if total_changes else 0.0) for u, s in user_stats.items()]
        draw_bars(ax.inset_axes([0.53, 0.72, 0.42, 0.25]), "Engagement Distribution", engage_data, "%")

    if trial_scores:
        trial_data = [(u, trial_scores[u]) for u in user_stats]
        draw_bars(ax.inset_axes([0.53, 0.42, 0.42, 0.25]), "Trial & Error Scores", trial_data, "%")

    if carefulness_summary:
        care_data = []
        for u in user_stats:
            behaviors = carefulness_summary.get(u, {})
            vals = [v for v in behaviors.values() if v.get("behavior_type") not in ("Insufficient data",)]
            if vals:
                careful = sum((v.get("behavior_type") == "Careful") or ("no variation" in v.get("behavior_type", "").lower()) for v in vals)
                pct = (careful / len(vals)) * 100
                care_data.append((u, pct))
        if care_data:
            draw_bars(ax.inset_axes([0.53, 0.12, 0.42, 0.25]), "Carefulness Summary", care_data, "% careful")
