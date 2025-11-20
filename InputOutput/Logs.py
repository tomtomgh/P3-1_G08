# io/logs.py
from __future__ import annotations

import re
import glob
from pathlib import Path
from typing import List

import pandas as pd


def time_to_seconds(t: str) -> float:
    """Convert 'HH:MM:SS.sss' into seconds since midnight."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_user_logs(log_pattern: str = "User*.log") -> pd.DataFrame:

    # Load all user log files
    files = sorted(glob.glob(log_pattern))

    # Regex for parsing log lines
    pattern = re.compile(
        r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+).*?(?:User\s*(?P<user_id>\d+))?\s*sets\s+(?P<param>[a-zA-Z ]+)\s+to\s+(?P<value>-?\d+(?:\.\d+)?)',
        re.IGNORECASE
    )

    records: List[dict] = []

    # Step 3: Extract from all logs
    for file in files:
        user_hint = re.search(r'User(\d+)', file)
        default_user = user_hint.group(1) if user_hint else None

        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    user = m.group('user_id') or default_user
                    if user is None:
                        continue
                    records.append({
                        'time': m.group('time'),
                        'user': int(user),
                        'param': m.group('param').strip(),
                        'value': float(m.group('value'))
                    })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No parameter change data found in logs.")

    df["time_sec"] = df["time"].apply(time_to_seconds)
    df = df.sort_values(by=["time_sec", "user", "param"]).reset_index(drop=True)
    return df


def compute_parameter_change_segments(df: pd.DataFrame) -> pd.DataFrame:

    if not {"user", "param", "time", "value"}.issubset(df.columns):
        raise ValueError(
            "DataFrame must contain columns: 'user', 'param', 'time', 'value'"
        )

    results = []

    for (user, param), group in df.groupby(["user", "param"]):
        group = group.sort_values("time").reset_index(drop=True)

        for i in range(1, len(group)):
            prev = group.loc[i - 1]
            curr = group.loc[i]
            if prev["value"] != curr["value"]:
                results.append(
                    {
                        "user": user,
                        "param": param,
                        "start_time": prev["time"],
                        "end_time": curr["time"],
                        "from_value": prev["value"],
                        "to_value": curr["value"],
                    }
                )

    summary = pd.DataFrame(results)
    return summary

def export_parameter_changes(log_pattern: str = "User*.log", output_csv: str = "parameter_changes_summary.csv") -> pd.DataFrame:
    df = load_user_logs(log_pattern=log_pattern)
    summary = compute_parameter_change_segments(df)

    summary.to_csv(output_csv, index=False)

    print(f"Saved parameter change summary to {output_csv}")

    return summary


