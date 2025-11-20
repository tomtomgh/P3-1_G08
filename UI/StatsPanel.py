from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from Analysis import UserActivityStats

def update_stats_display(ax: plt.Axes,
                         user_stats: Dict[int, UserActivityStats],
                         total_duration: float,
                         trial_scores: Optional[Dict[int, float]] = None,
                         carefulness_summary: Optional[Dict[int, Dict[str, Dict[str, float | str]]]] = None
                         ) -> None:
    ax.clear()
    ax.axis("off")

    # Clean up any previous inset (kept for safety, even though we no longer create one)
    fig = ax.figure
    for child in list(fig.axes):
        if hasattr(child, "get_gid") and child.get_gid() == "carefulness_bar_plot":
            try:
                child.remove()
            except Exception:
                pass

    stats_text = "USER STATISTICS\n" + "─" * 100 + "\n\n"

    users = sorted(user_stats.keys())
    for user in users:
        st = user_stats[user]
        stats_text += f"User {user}: "
        stats_text += f"{st.total_changes} changes  |  "
        stats_text += f"{st.changes_per_minute:.1f} chg/min  |  "

        if "frequency" in st.time_in_control and total_duration > 0:
            freq_t = st.time_in_control["frequency"]
            freq_pct = (freq_t / total_duration) * 100.0
            stats_text += f"Freq dominance: {freq_pct:.1f}%  |  "

        if trial_scores is not None and user in trial_scores:
            te = trial_scores[user]
            if isinstance(te, (int, float)) and np.isfinite(te):
                stats_text += f"Trial & Error score: {te:.0f}%  | "

        stats_text += f"Max gap: {st.max_inactivity_gap:.1f}s\n"

    stats_text += "\n" + "─" * 100 + "\n"

    if users:
        most_active = max(users, key=lambda u: user_stats[u].total_changes)
        stats_text += f"[*] Most Active: User {most_active}  |  "

        have_freq = any("frequency" in user_stats[u].time_in_control for u in users)
        if have_freq and total_duration > 0:
            freq_dom = {u: user_stats[u].time_in_control.get("frequency", 0.0) for u in users}
            most_dom = max(freq_dom, key=freq_dom.get)
            stats_text += f"[#] Most Dominant (Frequency): User {most_dom}  |  "

        most_resp = max(users, key=lambda u: user_stats[u].changes_per_minute)
        stats_text += f"[!] Most Responsive: User {most_resp}"

        if trial_scores:
            te_user = max(users, key=lambda u: trial_scores.get(u, float("-inf")))
            best = trial_scores.get(te_user, float("nan"))
            if isinstance(best, (int, float)) and np.isfinite(best):
                stats_text += f"  | Most Trial & Error: User {te_user} ({best:.0f}%)"

    stats_text += "\n\nEngagement Distribution:\n"
    total_changes_all = sum(user_stats[u].total_changes for u in users) if users else 0
    for user in users:
        st = user_stats[user]
        pct = (st.total_changes / total_changes_all) * 100.0 if total_changes_all > 0 else 0.0
        bar_length = int(pct / 2.0)
        bar = "█" * bar_length + "░" * (50 - bar_length)
        stats_text += f"  User {user}: {bar} {pct:.1f}%\n"

    if trial_scores:
        stats_text += "\nTrial & Error Scores:\n"
        for user in users:
            s = trial_scores.get(user, np.nan)
            if not isinstance(s, (int, float)) or np.isnan(s):
                bar = "░" * 50
                stats_text += f"  User {user}: {bar} n/a\n"
                continue
            score = float(np.clip(s, 0.0, 100.0))
            bar_length = int(score / 2.0)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            stats_text += f"  User {user}: {bar} {score:.0f}%\n"

    # Carefulness summary as ASCII bars (same style as above sections)
    if carefulness_summary:
        stats_text += "\nCarefulness Summary:\n"
        for u in users:
            user_stats_dict = carefulness_summary.get(u, {})
            behaviors = [
                (v.get("behavior_type") or "")
                for v in user_stats_dict.values()
                if v.get("behavior_type") not in ("Insufficient data",)
            ]
            if not behaviors:
                bar = "░" * 50
                stats_text += f"  User {u}: {bar} n/a\n"
                continue

            careful_count = sum((b == "Careful") or ("no variation" in b.lower()) for b in behaviors)
            ratio = (careful_count / len(behaviors)) * 100.0
            bar_length = int(ratio / 2.0)
            bar = "█" * bar_length + "░" * (50 - bar_length)
            stats_text += f"  User {u}: {bar} {ratio:.1f}% careful\n"

    # Render full text once (after assembling all sections)
    ax.text(
        0.02, 0.98, stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=8,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

# NEW: move carefulness page (text) here to live with other text stats
def draw_carefulness_page(ax_care: plt.Axes,
                          carefulness_summary: Optional[Dict[int, Dict[str, Dict[str, float | str]]]],
                          users,
                          fig: Optional[plt.Figure] = None) -> None:
    ax_care.clear()
    ax_care.axis("off")
    if carefulness_summary is None:
        ax_care.text(0.5, 0.5, "No carefulness summary provided.", ha="center", va="center", transform=ax_care.transAxes)
        if fig is not None:
            fig.canvas.draw_idle()
        return

    text = "CONTROL CAREFULNESS ANALYSIS\n" + "─" * 90 + "\n\n"
    users_to_show = sorted(users)
    for user in users_to_show:
        user_stats_dict = carefulness_summary.get(user, {})
        text += f"User {user}\n"
        for param_name, c in sorted(user_stats_dict.items()):
            avg_step = c.get("avg_step", float("nan"))
            max_step = c.get("max_step", float("nan"))
            careful_ratio = c.get("careful_ratio", float("nan"))
            behavior = c.get("behavior_type", "")
            if isinstance(avg_step, float) and np.isnan(avg_step):
                text += f"  • {param_name.capitalize():<12} No data\n"
            else:
                text += (
                    f"  • {param_name.capitalize():<12}"
                    f" Avg Δ={avg_step:.3f} | Max Δ={max_step:.3f} | "
                    f"Careful Steps={careful_ratio:.1f}% | "
                    f"{behavior}\n"
                )
        text += "\n"

    text += "─" * 90 + "\nSUMMARY\n"
    for user in sorted(users):
        user_stats_dict = carefulness_summary.get(user, {})
        behaviors = [
            v.get("behavior_type", "")
            for v in user_stats_dict.values()
            if v.get("behavior_type") not in ("Insufficient data",)
        ]
        if not behaviors:
            continue
        careful_count = sum((b == "Careful") or ("no variation" in b.lower()) for b in behaviors)
        ratio = careful_count / len(behaviors) * 100.0
        label = "Careful" if ratio >= 70.0 else "Reckless"
        text += f"  User {user}: {label:<8} ({ratio:.1f}% careful changes)\n"

    ax_care.text(
        0.02, 0.98, text,
        transform=ax_care.transAxes,
        verticalalignment="top",
        fontfamily="monospace",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="ivory", alpha=0.9),
    )
    if fig is not None:
        fig.canvas.draw_idle()