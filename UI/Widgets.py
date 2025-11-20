from matplotlib.widgets import Slider, Button, CheckButtons, RadioButtons
from typing import Dict, List, Callable, Any, Optional


def create_check_buttons(ax, labels: List[str], initial: List[bool], on_click: Callable[[str], None]) -> CheckButtons:
    check_buttons = CheckButtons(ax, labels, initial)
    check_buttons.on_clicked(on_click)
    return check_buttons


def create_slider(ax, label: str, val_min: float, val_max: float, val_init: float, on_change: Callable[[float], None], valstep: Optional[float] = None) -> Slider:
    slider = Slider(ax, label, val_min, val_max, valinit=val_init, orientation='horizontal', valstep=valstep)
    slider.on_changed(on_change)
    return slider


def create_button(ax, label: str, on_click: Callable[[Any], None]) -> Button:
    button = Button(ax, label, color='lightgray', hovercolor='0.85')
    button.on_clicked(on_click)
    return button


def create_radio_buttons(ax, labels: List[str], on_select: Callable[[str], None]) -> RadioButtons:
    radio_buttons = RadioButtons(ax, labels)
    radio_buttons.on_clicked(on_select)
    return radio_buttons


def create_user_widgets(ax_user_check, users: List[int], selected_users: Dict[int, bool], on_toggle: Callable[[], None]):
    """
    Create one CheckButtons control per user and lay them out horizontally.
    """
    fig = ax_user_check.figure
    bbox = ax_user_check.get_position()
    ax_user_check.clear()
    box_color = "#cb2626"
    ax_user_check.set_facecolor(box_color)
    ax_user_check.patch.set_edgecolor("#cfcfcf")
    ax_user_check.patch.set_linewidth(0.8)
    ax_user_check.axis("off")

    n = len(users)
    if n == 0:
        return []

    total_pad = 0.02
    inter_pad = 0.01
    available_width = bbox.width - total_pad - (n - 1) * inter_pad
    if available_width <= 0:
        cell_width = max(0.01, bbox.width / n)
        left_pad = bbox.x0
        spacing = 0.0
    else:
        cell_width = available_width / n
        left_pad = bbox.x0 + (total_pad / 2.0)
        spacing = inter_pad

    check_widgets = []

    for i, u in enumerate(users):
        x0 = left_pad + i * (cell_width + spacing)
        y0 = bbox.y0
        h = bbox.height
        sub_ax = fig.add_axes([x0, y0, cell_width, h])
        sub_ax.set_facecolor("none")
        sub_ax.axis("off")

        label = f"User {u}"
        init = [selected_users[u]]

        def make_handler(uid):
            def handler(lbl: str):
                selected_users[uid] = not selected_users[uid]
                on_toggle()
            return handler

        cb = create_check_buttons(sub_ax, [label], init, make_handler(u))
        check_widgets.append(cb)

    return check_widgets


def create_param_widgets(ax_param_check, params: List[str], selected_params: Dict[str, bool], on_toggle: Callable[[], None]):
    labels = [p.capitalize() for p in params]
    initial = [selected_params.get(p, True) for p in params]

    def handler(label: str):
        param_name = label.lower()
        selected_params[param_name] = not selected_params.get(param_name, True)
        on_toggle()

    return create_check_buttons(ax_param_check, labels, initial, handler)


def create_time_sliders(ax_time, t_min: float, t_max: float, time_window_start: List[float], on_time_update: Callable[[float], None]):
    slider_max = max(t_min, t_max - 50.0)
    slider_time = create_slider(ax_time, "Time Window", t_min, slider_max, time_window_start[0], on_time_update)
    return slider_time


def create_navigation_buttons(ax_prev, ax_next, on_prev: Callable[[Any], None], on_next: Callable[[Any], None]):
    btn_prev = create_button(ax_prev, "← Back", on_prev)
    btn_next = create_button(ax_next, "→ Next Page", on_next)
    return btn_prev, btn_next


def deactivate_checkbuttons(checkbuttons_list: List[CheckButtons], container_ax) -> None:
    """
    Disable event bindings and hide the CheckButtons axes and artists.
    """
    cbs = checkbuttons_list or []
    for cb in cbs:
        try:
            cb.disconnect_events()
        except Exception:
            pass
        try:
            cb.ax.set_visible(False)
        except Exception:
            pass
        for art in getattr(cb, "lines", []):
            try:
                art.set_visible(False)
            except Exception:
                pass
        for art in getattr(cb, "rectangles", []):
            try:
                art.set_visible(False)
            except Exception:
                pass
        for lab in getattr(cb, "labels", []):
            try:
                lab.set_visible(False)
            except Exception:
                pass

    try:
        container_ax.clear()
        container_ax.axis("off")
        container_ax.set_visible(False)
    except Exception:
        pass


def update_check_buttons(check_buttons: CheckButtons, selected: Dict[int, bool]) -> None:
    for label in check_buttons.labels:
        user_id = int(label.get_text().split()[-1])
        selected[user_id] = not selected[user_id]