from __future__ import annotations

from typing import Dict, List, Mapping, Optional


import numpy as np
import matplotlib.pyplot as plt

from Analysis import TimeSeriesResult, UserActivityStats
from Strategies.Utils import StrategyWindow

from .Plotting import plot_shared_data, plot_per_user_data, update_x_limits
from .Overlays import draw_strategy_legend
from .Overlays import draw_strategy_bands
from .Widgets import (
    create_user_widgets,
    create_time_sliders, create_navigation_buttons,
    deactivate_checkbuttons,
)
from .StatsPanel import update_stats_display


def show_dashboard(
    ts: TimeSeriesResult,
    user_stats: dict[int, UserActivityStats],
    total_duration: float,
    strategy_windows: dict[str, list[StrategyWindow]] | None = None,
    trial_scores: dict[int, float] | None = None,
    carefulness_summary: dict[int, dict[str, dict[str, float | str]]] | None = None,
) -> None:
    # ------------------------------------------------------------------
    # Unpack time series
    # ------------------------------------------------------------------
    timeline = ts.timeline
    series = ts.series
    merged_series = ts.merged_series
    user_at_time = ts.user_at_time
    users = ts.users
    params = ts.params
    t_min = ts.t_min
    t_max = ts.t_max
    dt = ts.time_resolution

    shared_params = set(merged_series.keys())

    
    USER_CHECK_POS = [0.08, 0.14, 0.84, 0.08]
    CARE_PANEL_POS = [0.08, 0.25, 0.80, 0.65]
    SLIDER_TIME_POS = [0.08, 0.09, 0.84, 0.02]
    BUTTON_FULL_POS = [0.65, 0.05, 0.10, 0.02]
    NAV_PREV_POS = [0.77, 0.02, 0.10, 0.03]
    NAV_NEXT_POS = [0.89, 0.02, 0.10, 0.03]
    WINDOW_SECONDS = 50.0

    # Helper for safe-calling GUI helpers without repeating try/except
    def _safe(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None

    # Precompute a consistent color map once
    user_colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(users))))
    color_map = {u: user_colors[i % len(user_colors)] for i, u in enumerate(users)}

    # PRINTS: Overview (mimic old script’s header and core info)
    print("\n" + "=" * 80)
    print("📊 DASHBOARD OVERVIEW")
    print("=" * 80)
    print(f"Users: {len(users)}  |  Parameters: {len(params)}  |  Samples: {len(timeline)}")
    print(f"Time range: {t_min:.1f}s → {t_max:.1f}s  (Δ={t_max - t_min:.1f}s)  |  Δt={dt:.3f}s")
    if params:
        print("📋 Parameters detected:", ", ".join(p for p in params))

    # PRINTS: Per-user compact activity preview
    try:
        for u in sorted(users):
            st = user_stats.get(u)
            if not st:
                continue
            freq_pct = 0.0
            if isinstance(getattr(st, "time_in_control", {}), dict):
                freq_pct = (st.time_in_control.get("frequency", 0.0) / total_duration * 100.0) if total_duration > 0 else 0.0
            print(
                f"User {u:<2} | {st.total_changes} changes | {st.changes_per_minute:.2f} chg/min | "
                f"Freq dominance: {freq_pct:.1f}% | Max gap: {st.max_inactivity_gap:.1f}s"
            )
    except Exception:
        pass

    # PRINTS: Engagement distribution as ASCII bars
    try:
        print("\n  📊 Engagement Distribution:")
        total_changes_all = sum(getattr(user_stats[u], "total_changes", 0) for u in user_stats)
        for u in sorted(users):
            tc = getattr(user_stats.get(u, None), "total_changes", 0)
            pct = (tc / total_changes_all * 100.0) if total_changes_all > 0 else 0.0
            bar = "█" * int(pct / 2)
            print(f"     User {u}: {bar} {pct:.1f}%")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Figure layout
    # ------------------------------------------------------------------
    visible_count = min(4, len(params))  # how many graphs to show at once

    # smaller default figure so the GUI pops up smaller on screen
    fig = plt.figure(figsize=(12, 9))
    try:
        mgr = plt.get_current_fig_manager()
        try:
            # Qt backend
            mgr.window.setGeometry(50, 50, 1000, 600)
        except Exception:
            try:
                # TkAgg backend
                mgr.window.wm_geometry("1000x600+50+50")
            except Exception:
                pass
    except Exception:
        pass

    if visible_count == 4:
        height_ratios = [1, 1, 1, 1, 0.5]  # 4 plots + stats row
    else:
        height_ratios = [1] * visible_count + [0.8]

    gs = fig.add_gridspec(
        visible_count + 1,
        2,
        width_ratios=[4, 1],  # left column wider than right
        height_ratios=height_ratios,
        left=0.08,
        right=0.95,
        bottom=0.15,
        top=0.95,
        wspace=0.0,
        hspace=0.6,  # more vertical spacing between rows
    )

    # Main signal axes (single column) – all plots same width
    axes = [fig.add_subplot(gs[i, 0]) for i in range(visible_count)]

    # Stats panel at the bottom (spans both columns, only used on page 2)
    ax_stats = fig.add_subplot(gs[visible_count, :])
    ax_stats.axis("off")

    # Right column legend area (used by draw_strategy_legend)
    ax_strat_legend = fig.add_subplot(gs[0:visible_count, 1])
    ax_strat_legend.axis("off")
    ax_strat_legend.set_visible(True)

    # Right column: user selection box in row 0, right column (free-positioned)
    ax_user_check = fig.add_axes(USER_CHECK_POS)

    # Carefulness panel axis (hidden by default; referenced in show_page)
    ax_care = fig.add_axes(CARE_PANEL_POS)
    ax_care.axis("off")
    ax_care.set_visible(False)

    # ------------------------------------------------------------------
    # GUI state
    # ------------------------------------------------------------------
    selected_users: dict[int, bool] = {u: True for u in users}
    selected_params: dict[str, bool] = {p: True for p in params}
    current_top_param = [0]  # index into active params
    full_view = [False]
    time_window_start = [t_min]

    current_page = [-1]  # 0 = Graphs, 1 = Carefulness dashboard; start at -1 so first show_page runs
    ui_state = {
        "user_check": None,
        "radio_ax": None,
        "radio_widget": None,
    }
    nav_busy = [False]  # prevent duplicate navigation handling

    # ------------------------------------------------------------------
    # Strategy overlays (input)
    # ------------------------------------------------------------------
    fighting_windows = (strategy_windows.get("fighting", []) if strategy_windows is not None else [])
    care_windows = (strategy_windows.get("carefulness", []) if strategy_windows is not None else [])
    te_windows = (strategy_windows.get("trial_error", []) if strategy_windows is not None else [])

    # PRINTS: Strategy windows summary (similar to old “detected” sections)
    try:
        def _span(w):
            return float(getattr(w, "start_time", 0.0)), float(getattr(w, "end_time", 0.0))
        def _merge_spans(spans, max_gap=0.0):
            spans = sorted(spans, key=lambda x: x[0])
            merged = []
            for s, e in spans:
                if not merged or s > merged[-1][1] + max_gap:
                    merged.append([s, e])
                else:
                    merged[-1][1] = max(merged[-1][1], e)
            return merged
        def _dur(spans):
            return sum(e - s for s, e in spans)

        # Fighting summary
        if fighting_windows:
            spans = [_span(w) for w in fighting_windows]
            merged = _merge_spans(spans, max_gap=0.0)
            print(f"\n⚔️  FIGHTING windows: {len(fighting_windows)}  | merged intervals: {len(merged)}  | total duration: {_dur(merged):.1f}s")
            # Optional per-param breakdown if available
            per_param = {}
            for w in fighting_windows:
                p = getattr(w, "param", None)
                per_param.setdefault(p, 0)
                per_param[p] += 1
            for p, c in sorted(per_param.items(), key=lambda x: (str(x[0]), x[1])):
                print(f"   • {str(p).capitalize() if p else 'n/a'}: {c} windows")

        # Carefulness windows summary
        if care_windows:
            print(f"\n🟢 Carefulness windows: {len(care_windows)}")
            care_count = sum(1 for w in care_windows if "reckless" not in str(getattr(w, "tag", "")).lower())
            reckless_count = len(care_windows) - care_count
            print(f"   • Careful: {care_count}   |   Reckless: {reckless_count}")

        # Trial & Error (High) summary
        if te_windows:
            def _is_high_te(w):
                tag = (getattr(w, "tag", "") or "").lower()
                return ("trial_error_high" in tag) or ("trial_error:high" in tag)
            te_high = [w for w in te_windows if _is_high_te(w)]
            spans = [_span(w) for w in te_high]
            merged = _merge_spans(spans, max_gap=0.0)
            print(f"\n🟧 Trial & Error (High) windows: {len(te_high)}  | merged intervals: {len(merged)}  | total duration: {_dur(merged):.1f}s")
    except Exception:
        pass

    # PRINTS: Trial & Error scores per user
    try:
        if trial_scores:
            print("\nTrial & Error Scores:")
            for u in sorted(users):
                s = trial_scores.get(u, None)
                if s is None:
                    print(f"  User {u}: n/a")
                else:
                    print(f"  User {u}: {float(s):.0f}%")
    except Exception:
        pass

    # PRINTS: Detailed per-user activity report 
    try:
        for u in sorted(users):
            st = user_stats.get(u)
            if not st:
                continue
            print(f"\n{'='*80}")
            print(f"👤 USER {u} - Activity Report")
            print(f"{'='*80}")
            print(f"  📈 Total Changes Made: {st.total_changes}")
            print(f"  ⚡ Activity Rate: {st.changes_per_minute:.2f} changes/minute")
            active_duration = getattr(st, "active_duration", float("nan"))
            print(f"  ⏱️  Active Duration: {active_duration:.1f}s ({active_duration/60.0 if active_duration==active_duration else 0.0:.1f} min)")
            ap = getattr(st, "activity_percentage", float("nan"))
            if ap == ap:
                print(f"  📊 Activity Percentage: {ap:.1f}%")
            param_changes = getattr(st, "param_changes", None)
            if isinstance(param_changes, dict):
                print(f"\n  📋 Changes by Parameter:")
                total_c = st.total_changes if st.total_changes else 1
                for p in sorted(param_changes.keys(), key=str):
                    cnt = param_changes[p]
                    pct = (cnt / total_c * 100.0) if total_c > 0 else 0.0
                    print(f"     • {str(p).capitalize()}: {cnt} changes ({pct:.1f}%)")
            if isinstance(getattr(st, "time_in_control", {}), dict) and total_duration > 0:
                tic = st.time_in_control
                if tic:
                    print(f"\n  👑 Dominance on Shared Parameters:")
                    for p, tctrl in tic.items():
                        pct = (tctrl / total_duration * 100.0) if total_duration > 0 else 0.0
                        print(f"     • {str(p).capitalize()}: {tctrl:.1f}s ({pct:.1f}% of total time)")
            print(f"\n  💤 Inactivity Analysis:")
            print(f"     • Average gap between changes: {getattr(st, 'avg_gap_between_changes', 0.0):.2f}s")
            print(f"     • Longest inactivity period: {getattr(st, 'max_inactivity_gap', 0.0):.1f}s")
            print(f"     • Number of long gaps (>10s): {getattr(st, 'long_inactivity_periods', 0)}")
            print(f"\n  ⏰ Session Timing:")
            print(f"     • First action at: {getattr(st, 'first_action_time', 0.0):.1f}s")
            print(f"     • Last action at: {getattr(st, 'last_action_time', 0.0):.1f}s")
            if carefulness_summary is not None:
                print(f"\n  🤖 Control Carefulness:")
                ustats = carefulness_summary.get(u, {})
                for p in sorted(ustats.keys(), key=str):
                    c = ustats[p]
                    avg_step = c.get("avg_step", float("nan"))
                    if isinstance(avg_step, float) and np.isnan(avg_step):
                        print(f"     • {str(p).capitalize()}: Insufficient data")
                    else:
                        print(
                            f"     • {str(p).capitalize()}: Avg Δ={c.get('avg_step', 0.0):.3f}, "
                            f"Max Δ={c.get('max_step', 0.0):.3f}, "
                            f"Careful Steps={c.get('careful_ratio', 0.0):.1f}%, "
                            f"Behavior={c.get('behavior_type', '')}"
                        )
    except Exception:
        pass

    # PRINTS: Comparative analysis 
    try:
        print(f"\n{'='*80}")
        print(f"🏆 COMPARATIVE ANALYSIS")
        print(f"{'='*80}")
        if users:
            most_active_user = max(users, key=lambda u: getattr(user_stats[u], "total_changes", 0))
            print(f"  🥇 Most Active User: User {most_active_user} ({user_stats[most_active_user].total_changes} changes)")
            if any("frequency" in getattr(user_stats[u], "time_in_control", {}) for u in users):
                freq_dom = {u: getattr(user_stats[u], "time_in_control", {}).get("frequency", 0.0) for u in users}
                most_dominant = max(freq_dom, key=freq_dom.get)
                pct = (freq_dom[most_dominant] / total_duration * 100.0) if total_duration > 0 else 0.0
                print(f"  👑 Most Dominant (Frequency): User {most_dominant} ({freq_dom[most_dominant]:.1f}s, {pct:.1f}%)")
            most_responsive = max(users, key=lambda u: getattr(user_stats[u], "changes_per_minute", 0.0))
            print(f"  ⚡ Most Responsive: User {most_responsive} ({user_stats[most_responsive].changes_per_minute:.2f} changes/min)")
        print(f"\n{'='*80}\n")
    except Exception:
        pass

    # ------------------------------------------------------------------
    # Main plotting routine
    # ------------------------------------------------------------------
    def draw_visible() -> None:
        active_users = [u for u in users if selected_users[u]]
        active_params = [p for p in params if selected_params.get(p, False)]

        if not active_users or not active_params:
            for ax in axes:
                ax.clear()
                ax.text(0.5, 0.5, "No data selected", ha="center", va="center", transform=ax.transAxes)
            if current_page[0] == 1:
                
                update_stats_display(ax_stats, user_stats, total_duration, trial_scores, carefulness_summary)
            _resize_stats_panel(current_page[0] == 1)
            fig.canvas.draw_idle()
            return

        for ax in axes:
            ax.clear()

        visible_params = active_params
        max_start = max(0, len(visible_params) - visible_count)
        start_idx = min(current_top_param[0], max_start)
        current_top_param[0] = start_idx
        subset = visible_params[start_idx : start_idx + visible_count]

        # color_map is precomputed above 
        for ax, p in zip(axes, subset):
            is_shared = p in shared_params
            if is_shared:
                merged_data_for_param = merged_series[p]
                owner_arr = user_at_time[p]
                plot_shared_data(ax, timeline, merged_data_for_param, owner_arr, active_users, color_map)
                ax.set_title(f"{p.capitalize()} (Shared Parameter)", fontsize=11, fontweight="bold")
            else:
                plot_per_user_data(ax, timeline, series[p], active_users, color_map)
                ax.set_title(p.capitalize(), fontsize=11, fontweight="bold")

            ax.set_ylabel(p, fontsize=10)
            ax.grid(True, alpha=0.3)

        if subset:
            axes[-1].set_xlabel("Time (s)", fontsize=10)

        # X-axis windowing
        update_x_limits(full_view, time_window_start, axes, t_min, t_max)

        # draw full-height partitioned bands per axis
        for ax, p in zip(axes, subset):
            try:
                draw_strategy_bands(ax, p, fighting_windows, care_windows, te_windows)
            except Exception:
                pass

        try:
            draw_strategy_legend(ax_strat_legend, fighting_windows, care_windows, te_windows)
        except Exception:
            pass

        # Update stats panel content only on page 1
        if current_page[0] == 1:
            update_stats_display(ax_stats, user_stats, total_duration, trial_scores, carefulness_summary)

        # Resize stats depending on page
        _resize_stats_panel(current_page[0] == 1)

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # User & parameter selection widgets (created after draw_visible exists)
    # ------------------------------------------------------------------
    

    ax_user_check.set_title("Select Users", fontsize=10, fontweight="bold")
    user_check = create_user_widgets(ax_user_check, users, selected_users, draw_visible)
    ui_state["user_check"] = user_check
    # set the container background AFTER widget creation
    ax_user_check.set_facecolor("#2d1010")
    ax_user_check.patch.set_edgecolor("#cfcfcf")
    ax_user_check.patch.set_linewidth(0.8)

    # ------------------------------------------------------------------
    # Time slider
    # ------------------------------------------------------------------
    ax_slider_time = plt.axes(SLIDER_TIME_POS)

    def on_time_update(val: float) -> None:
        if not full_view[0]:
            time_window_start[0] = val
            start = time_window_start[0]
            end = start + WINDOW_SECONDS
            for ax in axes:
                ax.set_xlim(start, end)
            fig.canvas.draw_idle()


    slider_time = create_time_sliders(
        ax_slider_time,
        t_min,
        t_max,
        time_window_start,
        on_time_update,
    )

    # ------------------------------------------------------------------
    # Full / zoom view toggle
    # ------------------------------------------------------------------
    ax_button_full = plt.axes(BUTTON_FULL_POS)

    def toggle_full(event) -> None:
        full_view[0] = not full_view[0]
        if full_view[0]:
            for ax in axes:
                ax.set_xlim(t_min, t_max)
            slider_time.set_active(False)
            try:
                slider_time.poly.set_facecolor("0.85")
                slider_time.poly.set_alpha(0.5)
            except Exception:
                pass
        else:
            start = time_window_start[0]
            end = start + WINDOW_SECONDS
            for ax in axes:
                ax.set_xlim(start, end)
            slider_time.set_active(True)
            try:
                slider_time.poly.set_facecolor("lightblue")
                slider_time.poly.set_alpha(1.0)
            except Exception:
                pass
        fig.canvas.draw_idle()

    btn_full = plt.Button(ax_button_full, "Full View", color="lightgray", hovercolor="0.85")
    btn_full.on_clicked(toggle_full)

    # ------------------------------------------------------------------
    # Carefulness dashboard (page 2)
    # ------------------------------------------------------------------
    def _print_carefulness_summary_to_console() -> None:
        if carefulness_summary is None:
            print("No carefulness summary provided.")
            return
        text = []
        text.append("CONTROL CAREFULNESS ANALYSIS (console)")
        text.append("─" * 90)
        for user in sorted(users):
            user_stats_dict = carefulness_summary.get(user, {})
            text.append(f"User {user}")
            for param_name, c in sorted(user_stats_dict.items()):
                avg_step = c.get("avg_step", float("nan"))
                max_step = c.get("max_step", float("nan"))
                careful_ratio = c.get("careful_ratio", float("nan"))
                behavior = c.get("behavior_type", "")
                if isinstance(avg_step, float) and np.isnan(avg_step):
                    text.append(f"  • {param_name.capitalize():<12} No data")
                else:
                    text.append(
                        f"  • {param_name.capitalize():<12}"
                        f" Avg Δ={avg_step:.3f} | Max Δ={max_step:.3f} | "
                        f"Careful Steps={careful_ratio:.1f}% | "
                        f"{behavior}"
                    )
            # summary per user
            behaviors = [
                v.get("behavior_type", "")
                for v in user_stats_dict.values()
                if v.get("behavior_type") not in ("Insufficient data",)
            ]
            if behaviors:
                careful_count = sum((b == "Careful") or ("no variation" in b.lower()) for b in behaviors)
                ratio = careful_count / len(behaviors) * 100.0
                label = "Careful" if ratio >= 70.0 else "Reckless"
                text.append(f"  -> Summary: {label} ({ratio:.1f}% careful changes)")
            text.append("")

        # Add final SUMMARY block (matches the page-2 panel)
        text.append("─" * 90 + "\nSUMMARY")
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
            text.append(f"  User {user}: {label:<8} ({ratio:.1f}% careful changes)")

        print("\n".join(text))

    def show_page(page_index: int) -> None:
        if page_index == current_page[0]:
            return

        if page_index == 0:
            print("[UI] Switched to: Main Graphs")
        elif page_index == 1:
            print("[UI] Switched to: User Statistics (console carefulness summary enabled)")

        current_page[0] = page_index

        if page_index == 0:
            ax_strat_legend.set_visible(True)
            for ax in axes:
                ax.set_visible(True)

            # Rebuild the user widgets fresh 
            deactivate_checkbuttons(ui_state.get("user_check") or [], ax_user_check)
            ax_user_check.set_visible(True)
            ax_button_full.set_visible(True)
            ax_slider_time.set_visible(True)
            ax_user_check.set_title("Select Users", fontsize=10, fontweight="bold")
            ui_state["user_check"] = create_user_widgets(ax_user_check, users, selected_users, draw_visible)
            ax_user_check.set_facecolor("#2d1010")
            ax_user_check.patch.set_edgecolor("#cfcfcf")
            ax_user_check.patch.set_linewidth(0.8)

            ax_stats.set_visible(False)
            _resize_stats_panel(False)
            ax_care.set_visible(False)

            if ui_state["radio_ax"] is not None:
                try:
                    ui_state["radio_ax"].remove()
                except Exception:
                    pass
                ui_state["radio_ax"] = None
                ui_state["radio_widget"] = None

        elif page_index == 1:
            for ax in axes:
                ax.set_visible(False)

            # Cleanly hide user widgets and disconnect events 
            deactivate_checkbuttons(ui_state.get("user_check") or [], ax_user_check)
            ax_strat_legend.set_visible(False)
            ax_care.set_visible(False)
            ax_button_full.set_visible(False)
            ax_slider_time.set_visible(False)

            ax_stats.set_visible(True)
            _resize_stats_panel(True)
            update_stats_display(ax_stats, user_stats, total_duration, trial_scores, carefulness_summary)

            if ui_state["radio_ax"] is not None:
                try:
                    ui_state["radio_ax"].remove()
                except Exception:
                    pass
                ui_state["radio_ax"] = None
                ui_state["radio_widget"] = None

            try:
                _print_carefulness_summary_to_console()
            except Exception:
                pass

        fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Navigation buttons
    # ------------------------------------------------------------------
    ax_prev = plt.axes(NAV_PREV_POS)
    ax_next = plt.axes(NAV_NEXT_POS)

    def go_prev(event) -> None:
        if getattr(event, "inaxes", None) is not ax_prev or nav_busy[0]:
            return
        nav_busy[0] = True
        try:
            show_page(0)
        finally:
            nav_busy[0] = False

    def go_next(event) -> None:
        if getattr(event, "inaxes", None) is not ax_next or nav_busy[0]:
            return
        nav_busy[0] = True
        try:
            if current_page[0] == 0:
                show_page(1)
            else:
                show_page(0)
        finally:
            nav_busy[0] = False


    btn_prev, btn_next = create_navigation_buttons(ax_prev, ax_next, go_prev, go_next)

    # ------------------------------------------------------------------
    # Save and manage stats panel sizing
    # ------------------------------------------------------------------
    stats_pos_default = list(ax_stats.get_position().bounds)  # [x0, y0, width, height]

    def _resize_stats_panel(fullscreen: bool) -> None:
        if fullscreen:
            # Fill most of the figure; avoid overlaps with margins
            ax_stats.set_position([0.08, 0.12, 0.84, 0.80])
            ax_stats.set_zorder(10)
        else:
            ax_stats.set_position(stats_pos_default)
            ax_stats.set_zorder(1)

    # ------------------------------------------------------------------
    # Initial draw
    # ------------------------------------------------------------------
    show_page(0)
    draw_visible()

    plt.show()