# parser_fixed.py
# ------------------------------------------------------------
# Final working parser for 4-user logs where missing user IDs
# must be inferred from filename.
# ------------------------------------------------------------

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

# ============================================================
# REGEX FOR EVENTS
# ============================================================

EVENT_REGEX = re.compile(
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+\[Info\]\s+User(?:\s+(?P<user>\d))?\s+sets\s+'
    r'(?P<param>frequency|amplitude|offset|phase shift)\s+to\s+(?P<value>-?\d+(?:\.\d+)?)'
)


# ============================================================
# Extract user ID from filename
# ============================================================

def infer_user_from_filename(path: Path) -> Optional[int]:
    """
    Accepts filenames like:
      User0.log
      user1.LOG
      Player2.txt (if needed)
    Returns integer user ID or None.
    """
    m = re.search(r"([Uu]ser)(\d+)", path.name)
    if m:
        return int(m.group(2))
    return None


# ============================================================
# TIME CONVERSION
# ============================================================

def parse_time_to_seconds(tstr: str) -> float:
    h, m, s = tstr.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


# ============================================================
# PARSE A SINGLE LOG FILE
# ============================================================

def parse_log_file(path: Path, session_id: str) -> List[Dict[str, Any]]:
    """
    Parse a log file. If user ID is missing in the log line,
    infer it from the filename (e.g., User3.log → user=3).
    """
    events = []
    inferred_user = infer_user_from_filename(path)

    if inferred_user is None:
        print(f"[WARNING] Could not infer user from filename: {path.name}")

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            m = EVENT_REGEX.search(line)
            if not m:
                continue

            # Parse timestamp
            t = parse_time_to_seconds(m.group("time"))

            # Determine user
            user_str = m.group("user")
            if user_str is not None:
                user = int(user_str)
            else:
                user = inferred_user   # <-- FIXED BEHAVIOR

            param = m.group("param")
            raw_val = m.group("value").rstrip(".")
            value = float(raw_val)

            events.append({
                "session_id": session_id,
                "time": t,
                "user": user,
                "param": param,
                "value": value,
            })

    events.sort(key=lambda e: e["time"])
    return events


# ============================================================
# PARSE FULL SESSION
# ============================================================

def parse_session(log_paths: List[Path], session_id: str) -> List[Dict[str, Any]]:
    all_events = []
    for p in log_paths:
        all_events.extend(parse_log_file(p, session_id=session_id))
    all_events.sort(key=lambda e: e["time"])
    return all_events


# ============================================================
# TEST RUN
# ============================================================

if __name__ == "__main__":
    base = Path("logs")
    log_paths = list(base.glob("*.log"))

    if not log_paths:
        print("No log files found in ./logs")
        exit()

    events = parse_session(log_paths, session_id="session_1")

    print(f"Parsed {len(events)} events.")
    print("First 10 events:")
    for e in events[:10]:
        print(e)
