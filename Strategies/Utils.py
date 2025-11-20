from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Callable, Iterable
import numpy as np

@dataclass
class StrategyWindow:
    """Generic strategy window."""
    start_time: float
    end_time: float
    users: List[int] = field(default_factory=list)
    param: str | None = None
    tag: str | None = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'start_time': self.start_time,
            'end_time': self.end_time,
            'users': self.users,
            'param': self.param,
            'tag': self.tag,
        }
        d.update(self.meta)
        return d

def sliding_time_windows(
    times: np.ndarray,
    window_size: float,
    start_idx: int = 0,
) -> Iterable[tuple[int, int]]:
    """Yield (i, j) indices covering window_size starting at times[i]."""
    n = len(times)
    i = start_idx
    while i < n:
        end_time = times[i] + window_size
        j = i
        while j < n and times[j] <= end_time:
            j += 1
        yield i, j
        i += 1

def merge_windows(
    windows: List[StrategyWindow],
    max_gap: float,
    group_key: Callable[[StrategyWindow], Any],
    combine: Callable[[StrategyWindow, StrategyWindow], None],
) -> List[StrategyWindow]:
    """Merge windows in same group_key within max_gap."""
    if not windows:
        return []
    windows_sorted = sorted(windows, key=lambda w: (group_key(w), w.start_time))
    merged: List[StrategyWindow] = []
    current = windows_sorted[0]
    for w in windows_sorted[1:]:
        same_group = group_key(w) == group_key(current)
        close_time = (w.start_time - current.end_time) <= max_gap
        if same_group and close_time:
            current.end_time = max(current.end_time, w.end_time)
            combine(current, w)
        else:
            merged.append(current)
            current = w
    merged.append(current)
    return merged


