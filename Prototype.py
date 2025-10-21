import os
import re
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# --- Tk / plotting ---
import tkinter as tk
from tkinter import filedialog
from tkinter.ttk import Combobox, Treeview, Scrollbar

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# =========================
# Parsing (tolerant, timed)
# =========================

# Time at start of line, e.g. "00:05:02.493"
TIME_RE = re.compile(
    r"^\s*(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.(?P<ms>\d+))?",
    re.IGNORECASE,
)

PARAM_ALIASES = {
    "frequency":   [r"freq(?:uency)?"],
    "amplitude":   [r"amp(?:litude)?"],
    "phase":       [r"phase"],
    "offset":      [r"offset", r"bias"],
    "duration":    [r"duration", r"length", r"time(?:span)?"],
    "period":      [r"period"],
    "sample_rate": [r"(?:sample[_\s-]?rate|sr|fs)"],
}
VALUE = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
SEP = r"[:=\-\s]+"

UNIT = r"(?:\s*(?:hz|deg|rad|s|ms|count))?"  # optional, harmless if absent

def _mk_patterns_for_alias(alias: str):
    """Return several regexes that can extract '<param> = 1.2' AND 'sets <param> to 1.2'."""
    return [
        # frequency = 0.3  | frequency: 0.3 | frequency 0.3
        re.compile(rf"(?i)\b{alias}\b{SEP}{VALUE}{UNIT}"),
        # 0.3 frequency   (rare, but cheap)
        re.compile(rf"(?i){VALUE}{UNIT}\s*\b{alias}\b"),
        # sets frequency to 0.3
        re.compile(rf"(?i)\bset(?:s|ting)?\s+{alias}\s+to\s*{VALUE}{UNIT}"),
        # user 1 changes frequency to 0.3 / adjust frequency to 0.3
        re.compile(rf"(?i)\b(?:change[sd]?|adjust(?:s|ed|ing)?)\s+{alias}\s+to\s*{VALUE}{UNIT}"),
        # ... frequency to 0.3
        re.compile(rf"(?i)\b{alias}\b.*?\bto\b\s*{VALUE}{UNIT}"),
    ]

EXTRACTION_PATTERNS = {
    k: sum((_mk_patterns_for_alias(alias) for alias in aliases), [])
    for k, aliases in PARAM_ALIASES.items()
}

GENERIC_PATTERNS = [
    # generic “<any alias> to <value>”
    re.compile(rf"(?i)\b({alias})\b.*?\bto\b\s*{VALUE}{UNIT}")
    for aliases in PARAM_ALIASES.values()
    for alias in aliases
] + [
    # generic “<any alias> : <value>”
    re.compile(rf"(?i)\b({alias})\b{SEP}{VALUE}{UNIT}")
    for aliases in PARAM_ALIASES.values()
    for alias in aliases
]


def _time_to_seconds(line: str, fallback_idx: int) -> float:
    m = TIME_RE.search(line)
    if not m:
        return float(fallback_idx)  # monotonic fallback
    h = int(m.group("h")); mnt = int(m.group("m")); s = int(m.group("s"))
    ms = m.group("ms")
    frac = 0.0 if not ms else float(f"0.{ms}")
    return h*3600 + mnt*60 + s + frac

@dataclass
class ParsedFile:
    # param -> list[(t_sec, value)]
    series: Dict[str, List[Tuple[float, float]]]
    # quick summary param -> {'last': float, 'mean': float}
    summary: Dict[str, Dict[str, float]]
    # short debug sample
    debug_lines: List[str]

def extract_params_from_text(text: str) -> ParsedFile:
    series: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    debug: List[str] = []
    line_idx = 0

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        t = _time_to_seconds(raw, line_idx)
        line_idx += 1

        matched_any = False

        # strong patterns per known param
        for key, pats in EXTRACTION_PATTERNS.items():
            for pat in pats:
                for m in pat.finditer(line):
                    try:
                        val = float(m.group(1))
                        series[key].append((t, val))
                        matched_any = True
                    except:  # noqa
                        pass

        # generic fallback
        if not matched_any:
            for gpat in GENERIC_PATTERNS:
                m = gpat.search(line)
                if m:
                    alias = m.group(1).lower()
                    try:
                        val = float(m.group(2))
                        for k, aliases in PARAM_ALIASES.items():
                            if any(re.fullmatch(a, alias, flags=re.IGNORECASE) for a in aliases):
                                series[k].append((t, val))
                                matched_any = True
                                break
                    except:  # noqa
                        pass

        if matched_any:
            if len([d for d in debug if d.startswith("✓")]) < 40:
                debug.append("✓ " + raw)
        else:
            if len([d for d in debug if d.startswith("∅")]) < 20:
                debug.append("∅ " + raw)

    # summaries
    summary: Dict[str, Dict[str, float]] = {}
    for k, pts in series.items():
        if not pts: 
            continue
        vals = [v for _, v in pts]
        summary[k] = {"last": vals[-1], "mean": sum(vals)/len(vals)}

    return ParsedFile(series=dict(series), summary=summary, debug_lines=debug)

def parse_folder(folder: str) -> Tuple[Dict[str, ParsedFile], str]:
    users = sorted([f for f in os.listdir(folder) if re.match(r"(?i)User\d+\.log$", f)])
    if not users:
        users = sorted([f for f in os.listdir(folder)
                        if f.lower().startswith("user") and f.lower().endswith(".log")])

    parsed: Dict[str, ParsedFile] = {}
    dbg_all = []

    for fname in users:
        path = os.path.join(folder, fname)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception as e:
            dbg_all.append(f"!! Could not read {fname}: {e}")
            continue
        pf = extract_params_from_text(text)
        parsed[fname] = pf
        dbg_all.append(f"# {fname}")
        dbg_all.extend(pf.debug_lines)

    return parsed, "\n".join(dbg_all) if dbg_all else "No logs parsed."


# =========================
# UI
# =========================

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Prototype: Per-User Parameter Viewer")
        self.folder = os.getcwd()
        self.parsed: Dict[str, ParsedFile] = {}

        # --- top bar ---
        top = tk.Frame(root); top.pack(fill=tk.X, padx=8, pady=6)
        self.folder_label = tk.Label(top, text=f"Folder: {self.folder}")
        self.folder_label.pack(side=tk.LEFT)
        tk.Button(top, text="Browse Folder", command=self.browse).pack(side=tk.RIGHT, padx=4)
        tk.Button(top, text="Load", command=self.load).pack(side=tk.RIGHT, padx=4)

        # --- left: user list ---
        left = tk.Frame(root); left.pack(side=tk.LEFT, fill=tk.Y, padx=(8,4), pady=6)
        tk.Label(left, text="Users").pack(anchor="w")
        self.user_list = tk.Listbox(left, height=18, width=24, exportselection=False)
        self.user_list.pack(fill=tk.Y, expand=False)
        self.user_list.bind("<<ListboxSelect>>", self.on_user_select)

        # --- right: controls + plots ---
        right = tk.Frame(root); right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(4,8), pady=6)

        ctrl = tk.Frame(right); ctrl.pack(fill=tk.X)
        tk.Label(ctrl, text="Parameter").pack(side=tk.LEFT)
        self.param_combo = Combobox(ctrl, state="readonly", width=24, values=[])
        self.param_combo.pack(side=tk.LEFT, padx=6)
        self.param_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())

        # Matplotlib canvas: 3 panels
        self.fig = plt.Figure(figsize=(9.5, 4.2), dpi=100)
        self.ax1 = self.fig.add_subplot(131)  # time series
        self.ax2 = self.fig.add_subplot(132)  # histogram
        self.ax3 = self.fig.add_subplot(133)  # delta/time
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=(6,0))

        # --- debug box ---
        dbg_frame = tk.Frame(right); dbg_frame.pack(fill=tk.BOTH, expand=False, pady=(6,0))
        tk.Label(dbg_frame, text="Debug (matched/ignored sample)").pack(anchor="w")
        self.debug = tk.Text(dbg_frame, height=10, wrap="word")
        self.debug.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb = Scrollbar(dbg_frame, command=self.debug.yview); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.debug.config(yscrollcommand=sb.set)

    # -------- actions --------
    def browse(self):
        folder = filedialog.askdirectory(initialdir=self.folder)
        if folder:
            self.folder = folder
            self.folder_label.config(text=f"Folder: {self.folder}")

    def load(self):
        self.user_list.delete(0, tk.END)
        self.param_combo.set("")
        self.param_combo["values"] = []
        self.debug.config(state=tk.NORMAL); self.debug.delete("1.0", tk.END)

        parsed, dbg = parse_folder(self.folder)
        self.parsed = parsed
        for fname in parsed.keys():
            self.user_list.insert(tk.END, fname)

        self.debug.insert(tk.END, dbg)
        self.debug.config(state=tk.DISABLED)
        self.clear_plots()
        if self.user_list.size():
            self.user_list.selection_set(0)
            self.on_user_select(None)

    def on_user_select(self, _event):
        sel = self.get_selected_user()
        if not sel: 
            return
        # populate param dropdown from that user's available series
        params = sorted(self.parsed[sel].series.keys())
        self.param_combo["values"] = params
        if params:
            self.param_combo.set(params[0])
        else:
            self.param_combo.set("")
        self.update_plots()

    def get_selected_user(self) -> Optional[str]:
        sel = self.user_list.curselection()
        if not sel:
            return None
        return self.user_list.get(sel[0])

    # -------- plotting --------
    def clear_plots(self, msg: str = "Load data, select user & parameter"):
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()
        self.ax1.text(0.5, 0.5, msg, ha="center", va="center")
        self.ax2.text(0.5, 0.5, msg, ha="center", va="center")
        self.ax3.text(0.5, 0.5, msg, ha="center", va="center")
        self.fig.tight_layout(); self.canvas.draw_idle()

    def update_plots(self):
        user = self.get_selected_user()
        param = self.param_combo.get()
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()

        if not user or not param or user not in self.parsed:
            self.clear_plots(); return

        series = self.parsed[user].series.get(param, [])
        if not series:
            self.clear_plots("No data for this parameter"); return

        t = [p[0] - series[0][0] for p in series]  # start at t=0 for readability
        v = [p[1] for p in series]

        # 1) Time series
        self.ax1.plot(t, v, marker="o", ms=2, lw=1)
        self.ax1.set_title(f"{param} over time")
        self.ax1.set_xlabel("time (s)"); self.ax1.set_ylabel(param)

        # 2) Histogram
        bins = min(20, max(5, int(math.sqrt(len(v)))))
        self.ax2.hist(v, bins=bins)
        self.ax2.set_title(f"{param} distribution")
        self.ax2.set_xlabel(param); self.ax2.set_ylabel("count")

        # 3) Change over time (Δvalue/Δt)
        if len(v) > 1:
            dv = []
            tt = []
            for i in range(1, len(v)):
                dt = max(1e-9, t[i] - t[i-1])
                dv.append((v[i] - v[i-1]) / dt)
                tt.append(0.5*(t[i] + t[i-1]))  # mid time
            self.ax3.plot(tt, dv, marker=".", lw=1)
            self.ax3.set_title(f"Δ{param}/Δt")
            self.ax3.set_xlabel("time (s)"); self.ax3.set_ylabel(f"d{param}/dt")
        else:
            self.ax3.text(0.5, 0.5, "Need ≥2 points", ha="center", va="center")

        self.fig.tight_layout()
        self.canvas.draw_idle()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
