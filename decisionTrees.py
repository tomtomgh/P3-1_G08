import re
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ============================================================
# 1. PARSING LOGS
# ============================================================

# Example log line:
# "00:05:02.4907287 [Info] User 1 sets frequency to 0.2."
EVENT_REGEX = re.compile(
    r'(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+\[Info\]\s+User(?:\s+(?P<user>\d))?\s+sets\s+'
        r'(?P<param>frequency|amplitude|offset|phase shift)\s+to\s+(?P<value>-?\d+(?:\.\d+)?)'

)


def parse_time_to_seconds(tstr: str) -> float:
    h, m, s = tstr.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_log_file(path: Path, session_id: str, fallback_user: int = None) -> List[Dict[str, Any]]:
    """
    Parses a single .log file into a list of events.
    Each event: {session_id, time, user, param, value}
    """
    events: List[Dict[str, Any]] = []

    with path.open('r', encoding='utf-8') as f:
        for line in f:
            m = EVENT_REGEX.search(line)
            if not m:
                continue

            t = parse_time_to_seconds(m.group('time'))
            user_str = m.group('user')
            if user_str is not None:
                user = int(user_str)
            else:
                user = fallback_user
            param = m.group('param')
            raw_val = m.group('value').rstrip('.')  # fail-safe
            value = float(raw_val)

            events.append(
                {
                    "session_id": session_id,
                    "time": t,
                    "user": user,
                    "param": param,
                    "value": value,
                }
            )

    events.sort(key=lambda e: e["time"])
    return events


def parse_session(log_paths: List[Path], session_id: str) -> List[Dict[str, Any]]:
    """
    Parses all log files for one session into a single, time-sorted event list.
    """
    all_events: List[Dict[str, Any]] = []
    for p in log_paths:
        fallback_user = None
        name = p.stem
        match = re.search(r"user(\d+)", name, re.IGNORECASE)
        if match:
            try:
                fallback_user = int(match.group(1))
            except ValueError:
                fallback_user = None
        all_events.extend(parse_log_file(p, session_id=session_id, fallback_user=fallback_user))
    all_events.sort(key=lambda e: e["time"])
    return all_events


def load_events_from_csv(csv_path: Path, session_id: str = "csv_session") -> List[Dict[str, Any]]:
    """
    Load parameter/velocity changes from a CSV file with columns such as:
    time, user, param, value, time_sec.
    """
    if not csv_path.exists():
        print(f"[csv loader] File not found: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    events: List[Dict[str, Any]] = []

    for _, row in df.iterrows():
        if "time_sec" in row and not pd.isna(row["time_sec"]):
            t = float(row["time_sec"])
        else:
            time_str = row.get("time")
            if isinstance(time_str, str):
                t = parse_time_to_seconds(time_str)
            else:
                continue

        user = row.get("user")
        user_id = int(user) if pd.notna(user) else None
        param = str(row.get("param", "")).strip()
        if not param:
            continue
        value = row.get("value")
        try:
            val = float(value)
        except (TypeError, ValueError):
            continue

        events.append(
            {
                "session_id": session_id,
                "time": t,
                "user": user_id,
                "param": param,
                "value": val,
            }
        )

    events.sort(key=lambda e: e["time"])
    print(f"[csv loader] Loaded {len(events)} events from {csv_path}")
    return events


# ============================================================
# 2. LEADERSHIP FEATURES (RULE-BASED)
# ============================================================

def frequency_metrics(events: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, float], int]:
    """
    Returns per-user frequency change counts and share.
    """
    freq_counts: Dict[int, int] = {}
    total_freq = 0

    for e in events:
        if e["param"] != "frequency":
            continue
        if e["user"] is None:
            continue
        u = e["user"]
        freq_counts[u] = freq_counts.get(u, 0) + 1
        total_freq += 1

    freq_share: Dict[int, float] = {}
    if total_freq > 0:
        for u, c in freq_counts.items():
            freq_share[u] = c / total_freq
    else:
        for u in freq_counts:
            freq_share[u] = 0.0

    return freq_counts, freq_share, total_freq


def compute_initiation_reaction(
    events: List[Dict[str, Any]], window_size: float = 2.0
) -> Tuple[Dict[int, int], Dict[int, int]]:
    """
    Groups all events (any param) into time windows of size `window_size`.
    For each cluster:
      - the first event with a user is the initiator
      - any later events with users are reactions
    """
    initiations: Dict[int, int] = {}
    reactions: Dict[int, int] = {}

    if not events:
        return initiations, reactions

    n = len(events)
    idx = 0

    while idx < n:
        base_time = events[idx]["time"]
        cluster = [events[idx]]
        j = idx + 1

        while j < n and events[j]["time"] - base_time <= window_size:
            cluster.append(events[j])
            j += 1

        initiator = None
        for ev in cluster:
            if ev["user"] is not None:
                initiator = ev["user"]
                break

        if initiator is not None:
            initiations[initiator] = initiations.get(initiator, 0) + 1
            first_seen = False
            for ev in cluster:
                if ev["user"] is None:
                    continue
                if not first_seen:
                    first_seen = True  # skip the initiator event
                    continue
                u = ev["user"]
                reactions[u] = reactions.get(u, 0) + 1

        idx = j

    return initiations, reactions


def compute_lead_fraction(
    initiations: Dict[int, int], reactions: Dict[int, int]
) -> Dict[int, float]:
    """
    lead_fraction[u] = initiations / (initiations + reactions)
    """
    lf: Dict[int, float] = {}
    all_users = set(initiations.keys()) | set(reactions.keys())
    for u in all_users:
        ini = initiations.get(u, 0)
        rea = reactions.get(u, 0)
        tot = ini + rea
        lf[u] = ini / tot if tot > 0 else 0.0
    return lf


def assign_leader_role(freq_share: Dict[int, float]) -> Dict[int, str]:
    """
    Simple rule-based leader assignment:
    - user with highest freq_share is 'leader' IF freq_share > 0.3
    - others are 'non-leader'
    """
    if not freq_share:
        return {}

    max_user = max(freq_share, key=freq_share.get)
    max_share = freq_share[max_user]

    roles: Dict[int, str] = {}
    for u, share in freq_share.items():
        if u == max_user and max_share > 0.3:
            roles[u] = "leader"
        else:
            roles[u] = "non-leader"
    return roles


# ============================================================
# 3. LEARNING STRATEGY FEATURES
# ============================================================

def player_events(events: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
    return [e for e in events if e["user"] == user_id]


def compute_action_diversity(player_evts: List[Dict[str, Any]]) -> float:
    """
    Shannon entropy of action distribution (param types changed).
    High entropy = diverse exploration.
    """
    if not player_evts:
        return 0.0
    
    param_counts = {}
    for e in player_evts:
        p = e["param"]
        param_counts[p] = param_counts.get(p, 0) + 1
    
    total = sum(param_counts.values())
    if total == 0:
        return 0.0
    
    # Calculate Shannon entropy
    entropy = 0.0
    for count in param_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * np.log2(p)
    
    # Normalize by max possible entropy based on number of unique params used
    # This gives more meaningful scores for small time windows
    num_unique_params = len(param_counts)
    
    if num_unique_params <= 1:
        return 0.0  # No diversity if only one parameter type
    
    max_entropy = np.log2(num_unique_params)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_repetition_score(player_evts: List[Dict[str, Any]], window: float = 5.0, debug: bool = False) -> Dict[str, Any]:
    """
    Detect repetitive patterns: same parameter changed multiple times in quick succession.
    Returns repetition ratio and count of repetitive sequences.
    
    A repetitive sequence is 3+ consecutive changes to the SAME parameter where
    each event is within 'window' seconds of the previous event.
    """
    if len(player_evts) < 3:  # Need at least 3 events to have repetition
        return {"repetition_ratio": 0.0, "repetitive_sequences": 0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    
    # Group consecutive same-parameter events
    sequences = []
    i = 0
    
    while i < len(player_evts):
        current_param = player_evts[i]["param"]
        sequence = [i]
        
        # Look ahead for same parameter with gaps <= window
        for j in range(i + 1, len(player_evts)):
            time_gap = player_evts[j]["time"] - player_evts[j-1]["time"]
            
            if player_evts[j]["param"] == current_param and time_gap <= window:
                sequence.append(j)
            else:
                break
        
        if len(sequence) >= 3:
            sequences.append(sequence)
        
        # Move to the next unprocessed event
        i = sequence[-1] + 1 if len(sequence) > 1 else i + 1
    
    # Calculate metrics
    events_in_sequences = set()
    for seq in sequences:
        events_in_sequences.update(seq)
    
    repetition_ratio = len(events_in_sequences) / len(player_evts)
    
    # Debug output
    if debug:
        user_id = player_evts[0].get("user", "?")
        print(f"\n[DEBUG] User {user_id} Repetition Analysis:")
        print(f"  Total events: {len(player_evts)}")
        print(f"  Events in repetitive sequences: {len(events_in_sequences)}")
        print(f"  Number of repetitive sequences (3+ events): {len(sequences)}")
        print(f"  Repetition ratio: {repetition_ratio:.3f}")
        
        # Show first few sequences
        if sequences:
            print(f"  First 3 sequences:")
            for idx, seq in enumerate(sequences[:3]):
                param = player_evts[seq[0]]["param"]
                start_time = player_evts[seq[0]]["time"]
                end_time = player_evts[seq[-1]]["time"]
                duration = end_time - start_time
                print(f"    Sequence {idx+1}: {len(seq)} events, param={param}, duration={duration:.2f}s")
    
    return {
        "repetition_ratio": repetition_ratio,
        "repetitive_sequences": len(sequences)
    }


def compute_speed_acceleration(player_evts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect if player speeds up toward the end (incremental tuning).
    Compare inter-action intervals in first half vs second half.
    """
    if len(player_evts) < 4:
        return {"speed_acceleration": 0.0, "avg_early_interval": 0.0, "avg_late_interval": 0.0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    mid = len(player_evts) // 2
    
    early_evts = player_evts[:mid]
    late_evts = player_evts[mid:]
    
    def avg_interval(evts):
        if len(evts) < 2:
            return 0.0
        intervals = [evts[i+1]["time"] - evts[i]["time"] for i in range(len(evts)-1)]
        return np.mean(intervals) if intervals else 0.0
    
    early_interval = avg_interval(early_evts)
    late_interval = avg_interval(late_evts)
    
    # Positive acceleration means faster (shorter intervals) later
    acceleration = (early_interval - late_interval) / (early_interval + 1e-6)
    
    return {
        "speed_acceleration": acceleration,
        "avg_early_interval": early_interval,
        "avg_late_interval": late_interval
    }


def compute_backtracking(player_evts: List[Dict[str, Any]], threshold: float = 10.0) -> Dict[str, Any]:
    """
    Detect value reversals: player changes a parameter then changes it back (or close to previous value).
    """
    if len(player_evts) < 3:
        return {"backtrack_count": 0, "backtrack_ratio": 0.0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    
    # Track last value for each parameter
    param_history = defaultdict(list)  # param -> [(time, value), ...]
    
    for e in player_evts:
        param_history[e["param"]].append((e["time"], e["value"]))
    
    backtrack_count = 0
    
    for param, history in param_history.items():
        if len(history) < 3:
            continue
        
        for i in range(2, len(history)):
            val_prev = history[i-2][1]
            val_mid = history[i-1][1]
            val_curr = history[i][1]
            
            # Check if current value is closer to i-2 than to i-1 (backtracking)
            if abs(val_curr - val_prev) < abs(val_mid - val_prev) * 0.5:
                backtrack_count += 1
    
    backtrack_ratio = backtrack_count / len(player_evts) if player_evts else 0.0
    
    return {
        "backtrack_count": backtrack_count,
        "backtrack_ratio": backtrack_ratio
    }


def compute_hesitation_pauses(player_evts: List[Dict[str, Any]], pause_threshold: float = 5.0) -> Dict[str, Any]:
    """
    Detect unusually long pauses between actions (hesitation before decisions).
    """
    if len(player_evts) < 2:
        return {"long_pause_count": 0, "pause_ratio": 0.0, "avg_pause": 0.0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    intervals = [player_evts[i+1]["time"] - player_evts[i]["time"] for i in range(len(player_evts)-1)]
    
    if not intervals:
        return {"long_pause_count": 0, "pause_ratio": 0.0, "avg_pause": 0.0}
    
    avg_interval = np.mean(intervals)
    long_pauses = [iv for iv in intervals if iv > pause_threshold]
    
    return {
        "long_pause_count": len(long_pauses),
        "pause_ratio": len(long_pauses) / len(intervals),
        "avg_pause": avg_interval
    }


def compute_inefficient_moves(player_evts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Detect oscillating or non-goal-directed behavior:
    - Rapid back-and-forth value changes (e.g., increase then decrease then increase)
    """
    if len(player_evts) < 3:
        return {"oscillation_count": 0, "oscillation_ratio": 0.0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    param_history = defaultdict(list)
    
    for e in player_evts:
        param_history[e["param"]].append(e["value"])
    
    oscillation_count = 0
    
    for param, values in param_history.items():
        if len(values) < 3:
            continue
        
        for i in range(2, len(values)):
            delta1 = values[i-1] - values[i-2]
            delta2 = values[i] - values[i-1]
            
            # Oscillation: direction changes (sign flip)
            if delta1 * delta2 < 0:  # opposite signs
                oscillation_count += 1
    
    oscillation_ratio = oscillation_count / len(player_evts) if player_evts else 0.0
    
    return {
        "oscillation_count": oscillation_count,
        "oscillation_ratio": oscillation_ratio
    }


def compute_iterative_tuning(player_evts: List[Dict[str, Any]], window: float = 3.0) -> Dict[str, Any]:
    """
    Detect systematic add-remove or cyclic patterns.
    Look for parameter changes that are repeatedly adjusted in small increments.
    """
    if len(player_evts) < 4:
        return {"small_adjustment_count": 0, "tuning_ratio": 0.0}
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    param_history = defaultdict(list)
    
    for e in player_evts:
        param_history[e["param"]].append((e["time"], e["value"]))
    
    small_adjustment_count = 0
    
    for param, history in param_history.items():
        if len(history) < 2:
            continue
        
        for i in range(1, len(history)):
            delta = abs(history[i][1] - history[i-1][1])
            time_diff = history[i][0] - history[i-1][0]
            
            # Small adjustments: small value changes in quick succession
            if delta < 0.5 and time_diff < window:  # tuning threshold
                small_adjustment_count += 1
    
    tuning_ratio = small_adjustment_count / len(player_evts) if player_evts else 0.0
    
    return {
        "small_adjustment_count": small_adjustment_count,
        "tuning_ratio": tuning_ratio
    }


def compute_value_entropy(player_evts: List[Dict[str, Any]]) -> float:
    """
    Compute entropy of value changes to detect random vs. purposeful behavior.
    High entropy = random/chaotic changes across different value ranges.
    """
    if len(player_evts) < 2:
        return 0.0
    
    player_evts = sorted(player_evts, key=lambda e: e["time"])
    
    # Bin values into ranges and compute entropy
    all_values = []
    for e in player_evts:
        all_values.append(e["value"])
    
    if not all_values:
        return 0.0
    
    # Create bins based on value distribution
    min_val = min(all_values)
    max_val = max(all_values)
    
    if max_val == min_val:
        return 0.0  # No variation = no entropy
    
    # Use 10 bins to discretize values
    num_bins = min(10, len(set(all_values)))
    bins = np.linspace(min_val, max_val, num_bins + 1)
    
    # Count occurrences in each bin
    bin_counts = np.zeros(num_bins)
    for val in all_values:
        bin_idx = min(int((val - min_val) / (max_val - min_val) * num_bins), num_bins - 1)
        bin_counts[bin_idx] += 1
    
    # Compute Shannon entropy
    total = len(all_values)
    entropy = 0.0
    for count in bin_counts:
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Normalize by max possible entropy
    max_entropy = np.log2(num_bins) if num_bins > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return normalized_entropy


def classify_learning_strategy(features: Dict[str, float]) -> str:
    """
    Rule-based classification of learning strategies based on extracted features.
    
    Strategies:
    1. Random/Unstructured: Very high entropy with chaotic oscillations
    2. Structured/Curiosity-driven Exploration: High diversity, balanced parameter usage
    3. Repetition/Practice: Low diversity, high repetition
    4. Incremental/Goal-Directed Tuning: Moderate tuning with steady progress
    5. Backtracking and Recovery: High backtrack ratio
    6. Pause/Hesitation: High pause ratio at decision points
    7. Playful/Inefficient Moves: High oscillation, inefficient patterns
    8. Iterative Trial Patterns: High tuning ratio, systematic adjustments
    """
    
    # Extract features
    diversity = features.get("action_diversity", 0.0)
    repetition = features.get("repetition_ratio", 0.0)
    backtrack = features.get("backtrack_ratio", 0.0)
    pause = features.get("pause_ratio", 0.0)
    oscillation = features.get("oscillation_ratio", 0.0)
    tuning = features.get("tuning_ratio", 0.0)
    entropy = features.get("value_entropy", 0.0)
    
    # Decision rules (priority order matters). Each branch returns one of the 8 target strategies.
    # The ordering favors more distinctive signals (entropy, pauses, backtracking) before general ones.

    # 1. Random/Unstructured: chaotic values with direction changes
    if entropy >= 0.85 and oscillation >= 0.25:
        return "Random_Unstructured"

    # 2. Pause/Hesitation: long pauses dominate the rhythm
    if pause >= 0.45:
        return "Pause_Hesitation"

    # 3. Backtracking and Recovery: frequent reversals toward prior values
    if backtrack >= 0.35:
        return "Backtracking_Recovery"

    # 4. Repetition/Practice: hammering the same parameter repeatedly
    if repetition >= 0.55 and diversity < 0.55:
        return "Repetition_Practice"

    # 5. Iterative Trial Patterns: sustained small adjustments
    if tuning >= 0.65:
        return "Iterative_Trial_Patterns"

    # 6. Incremental/Goal-Directed Tuning: moderate tuning, low chaos
    if tuning >= 0.35 and backtrack < 0.25 and entropy < 0.8:
        return "Incremental_Goal_Directed_Tuning"

    # 7. Playful/Inefficient Moves: frequent oscillations without strong tuning intent
    if oscillation >= 0.5:
        return "Playful_Inefficient"

    # 8. Structured/Curiosity-driven Exploration: diverse, balanced, not overly repetitive
    if diversity >= 0.6 and repetition < 0.4 and entropy <= 0.8:
        return "Structured_Curiosity_Driven"

    # Fallback: choose the closest fit between structured curiosity and random noise
    if entropy > 0.75:
        return "Random_Unstructured"
    return "Structured_Curiosity_Driven"


def compute_param_usage(player_evts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Basic usage stats: how many times each param is changed by this player.
    """
    param_counts: Dict[str, int] = {}
    for e in player_evts:
        p = e["param"]
        param_counts[p] = param_counts.get(p, 0) + 1

    total = sum(param_counts.values()) or 1  # avoid div-by-zero
    dominant_param = max(param_counts, key=param_counts.get) if param_counts else None
    dominant_share = param_counts[dominant_param] / total if dominant_param else 0.0

    return {
        "param_counts": param_counts,
        "dominant_param": dominant_param,
        "dominant_share": dominant_share,
        "total_param_changes": total,
    }


def compute_single_param_clusters(player_evts: List[Dict[str, Any]], window_size: float = 1.0) -> Dict[str, float]:
    """
    Cluster events by this player within `window_size` seconds.
    Measure how many clusters use only one param vs multiple.
    Used to detect VOTAT-like behavior (vary one parameter at a time).
    """
    if not player_evts:
        return {
            "num_clusters": 0,
            "single_param_clusters": 0,
            "multi_param_clusters": 0,
            "single_param_cluster_ratio": 0.0,
        }

    player_evts = sorted(player_evts, key=lambda e: e["time"])
    clusters: List[List[Dict[str, Any]]] = []
    current_cluster: List[Dict[str, Any]] = [player_evts[0]]

    for ev in player_evts[1:]:
        if ev["time"] - current_cluster[0]["time"] <= window_size:
            current_cluster.append(ev)
        else:
            clusters.append(current_cluster)
            current_cluster = [ev]
    clusters.append(current_cluster)

    single_param = 0
    multi_param = 0

    for cl in clusters:
        params = {ev["param"] for ev in cl}
        if len(params) == 1:
            single_param += 1
        else:
            multi_param += 1

    total_clusters = single_param + multi_param or 1
    ratio = single_param / total_clusters

    return {
        "num_clusters": total_clusters,
        "single_param_clusters": single_param,
        "multi_param_clusters": multi_param,
        "single_param_cluster_ratio": ratio,
    }


def classify_strategy_rule_based(
    param_usage: Dict[str, Any],
    cluster_stats: Dict[str, float],
    hotat_dominant_thresh: float = 0.8,
    votat_single_cluster_thresh: float = 0.8,
) -> str:
    """
    Simple rule-based:
      HOTAT-like  -> almost always same parameter
      VOTAT-like  -> uses multiple params, but mostly one per cluster
      Mixed       -> everything else
    """
    dominant_share = param_usage["dominant_share"]
    param_counts = param_usage["param_counts"]
    single_ratio = cluster_stats["single_param_cluster_ratio"]
    num_params_used = len(param_counts)

    if num_params_used == 0:
        return "inactive"

    if num_params_used == 1 or dominant_share >= hotat_dominant_thresh:
        return "HOTAT-like"

    if num_params_used > 1 and single_ratio >= votat_single_cluster_thresh:
        return "VOTAT-like"

    return "Mixed"


# ============================================================
# 4. COORDINATION METRICS (SESSION + PAIRWISE)
# ============================================================

def compute_coordination(events: List[Dict[str, Any]], window: float = 1.5) -> Dict[str, Any]:
    """
    Session-level coordination summary:
    - straight_coord_score: average coordination between opposite legs
    - diagonal_coord_score: average coordination between neighboring legs
    - coord_style: "straight", "diagonal", or "uncoordinated"
    """

    users = sorted({e["user"] for e in events if e["user"] is not None})
    if len(users) < 2:
        return {
            "straight_coord_score": 0.0,
            "diagonal_coord_score": 0.0,
            "coord_style": "uncoordinated",
        }

    raw_changes: Dict[int, List[Tuple[float, float]]] = defaultdict(list)

    for e in events:
        if e["user"] is None:
            continue
        if e["param"] not in ("frequency", "amplitude"):
            continue
        raw_changes[e["user"]].append((e["time"], e["value"]))

    step_changes: Dict[int, List[Tuple[float, int]]] = {}
    for u, lst in raw_changes.items():
        lst = sorted(lst, key=lambda x: x[0])
        if len(lst) < 2:
            continue
        steps = []
        for i in range(1, len(lst)):
            t_prev, v_prev = lst[i - 1]
            t_cur, v_cur = lst[i]
            delta = v_cur - v_prev
            sign = int(np.sign(delta))
            steps.append((t_cur, sign))
        if steps:
            step_changes[u] = steps

    if len(step_changes) < 2:
        return {
            "straight_coord_score": 0.0,
            "diagonal_coord_score": 0.0,
            "coord_style": "uncoordinated",
        }

    if len(users) == 4:
        straight_pairs = [(users[0], users[2]), (users[1], users[3])]
    else:
        straight_pairs = []

    diagonal_pairs = []
    for i in range(len(users)):
        diagonal_pairs.append((users[i], users[(i + 1) % len(users)]))

    def pair_coord(u1: int, u2: int) -> float:
        if u1 not in step_changes or u2 not in step_changes:
            return 0.5

        s1 = step_changes[u1]
        s2 = step_changes[u2]

        matches: List[int] = []
        j = 0
        for t1, sign1 in s1:
            while j < len(s2) and s2[j][0] < t1 - window:
                j += 1
            k = j
            while k < len(s2) and s2[k][0] <= t1 + window:
                t2, sign2 = s2[k]
                if sign1 == 0 or sign2 == 0:
                    k += 1
                    continue
                matches.append(1 if sign1 == sign2 else -1)
                k += 1

        if not matches:
            return 0.5

        mean_sign = float(np.mean(matches))  # -1..1
        return (mean_sign + 1.0) / 2.0      # map to 0..1

    straight_scores = [pair_coord(a, b) for a, b in straight_pairs] if straight_pairs else []
    diagonal_scores = [pair_coord(a, b) for a, b in diagonal_pairs] if diagonal_pairs else []

    straight_score = float(np.mean(straight_scores)) if straight_scores else 0.5
    diagonal_score = float(np.mean(diagonal_scores)) if diagonal_scores else 0.5

    if straight_score > diagonal_score + 0.05:
        coord_style = "straight"
    elif diagonal_score > straight_score + 0.05:
        coord_style = "diagonal"
    else:
        coord_style = "uncoordinated"

    return {
        "straight_coord_score": straight_score,
        "diagonal_coord_score": diagonal_score,
        "coord_style": coord_style,
    }


def compute_pairwise_coordination(events: List[Dict[str, Any]], window: float = 1.5) -> Dict[int, Dict[int, float]]:
    """
    Returns pairwise coordination scores:

    {
        userA: { userB: scoreAB, ... },
        userB: { userA: scoreBA, ... },
        ...
    }

    Score in [0,1]; > ~0.6 suggests coordinated behavior within the timeframe.
    """
    users = sorted({e["user"] for e in events if e["user"] is not None})
    if len(users) < 2:
        return {}

    raw_changes = defaultdict(list)
    for e in events:
        if e["user"] is None:
            continue
        if e["param"] not in ("frequency", "amplitude"):
            continue
        raw_changes[e["user"]].append((e["time"], e["value"]))

    step_changes = {}
    for u, lst in raw_changes.items():
        lst = sorted(lst, key=lambda x: x[0])
        if len(lst) < 2:
            continue
        steps = []
        for i in range(1, len(lst)):
            tprev, vprev = lst[i-1]
            tcur, vcur = lst[i]
            delta = vcur - vprev
            sign = int(np.sign(delta))
            steps.append((tcur, sign))
        step_changes[u] = steps

    def pair_coord(u1: int, u2: int) -> float:
        if u1 not in step_changes or u2 not in step_changes:
            return 0.5
        s1 = step_changes[u1]
        s2 = step_changes[u2]
        matches = []
        j = 0
        for t1, sgn1 in s1:
            while j < len(s2) and s2[j][0] < t1 - window:
                j += 1
            k = j
            while k < len(s2) and s2[k][0] <= t1 + window:
                t2, sgn2 = s2[k]
                if sgn1 != 0 and sgn2 != 0:
                    matches.append(1 if sgn1 == sgn2 else -1)
                k += 1
        if not matches:
            return 0.5
        mean = np.mean(matches)  # -1..1
        return (mean + 1) / 2.0  # 0..1

    result: Dict[int, Dict[int, float]] = {u: {} for u in users}
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            u1 = users[i]
            u2 = users[j]
            score = pair_coord(u1, u2)
            result[u1][u2] = score
            result[u2][u1] = score

    return result


# ============================================================
# 5. LABEL DIVERSITY HELPER
# ============================================================

def ensure_label_diversity(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Ensures there are at least 2 different strategy labels.
    If not, auto-creates a second class by splitting players using freq_share.
    """
    df = df.copy()
    unique_labels = df[label_col].unique()

    if len(unique_labels) > 1:
        return df  # already diverse enough

    print("[INFO] Only one strategy_full_label class found. Auto-generating second class.")

    median_freq = df["freq_share"].median()

    df[label_col] = df.apply(
        lambda row: f"{row[label_col]}_B" if row["freq_share"] > median_freq
        else f"{row[label_col]}_A",
        axis=1
    )

    return df


# ============================================================
# 6. BUILD PER-PLAYER FEATURE TABLE
# ============================================================

def filter_events_by_time_window(
    events: List[Dict[str, Any]], 
    start_time: str, 
    end_time: str
) -> List[Dict[str, Any]]:
    """
    Filter events to only include those within a specific time window.
    
    Args:
        events: List of event dictionaries
        start_time: Start time in format "HH:MM:SS" or "MM:SS"
        end_time: End time in format "HH:MM:SS" or "MM:SS"
    
    Returns:
        Filtered list of events within the time window
    """
    def parse_time_string(time_str: str) -> float:
        """Parse time string to seconds."""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        else:
            raise ValueError(f"Invalid time format: {time_str}. Use 'HH:MM:SS' or 'MM:SS'")
    
    start_seconds = parse_time_string(start_time)
    end_seconds = parse_time_string(end_time)
    
    filtered = [e for e in events if start_seconds <= e["time"] <= end_seconds]
    
    print(f"[Time Window] Time range: {start_seconds}s to {end_seconds}s")
    print(f"[Time Window] Filtered {len(filtered)} events from {start_time} to {end_time}")
    
    if filtered:
        users_in_window = set(e["user"] for e in filtered if e["user"] is not None)
        print(f"[Time Window] Users active: {sorted(users_in_window)}")
    
    return filtered


def analyze_time_window(
    events: List[Dict[str, Any]],
    start_time: str,
    end_time: str,
    session_id: str = None
) -> pd.DataFrame:
    """
    Analyze player strategies within a specific time window.
    
    Args:
        events: Full list of events
        start_time: Start time in format "HH:MM:SS" or "MM:SS"
        end_time: End time in format "HH:MM:SS" or "MM:SS"
        session_id: Optional session identifier for the window
    
    Returns:
        DataFrame with player features and strategy classifications for the time window
    """
    # Filter events to time window
    windowed_events = filter_events_by_time_window(events, start_time, end_time)
    
    if not windowed_events:
        print(f"[Warning] No events found in time window {start_time} to {end_time}")
        return pd.DataFrame()
    
    # Override session_id if provided
    if session_id:
        for e in windowed_events:
            e["session_id"] = session_id
    
    # Use existing build_player_features to analyze the windowed data
    df_window = build_player_features(windowed_events)
    
    # Add time window metadata
    df_window["time_window_start"] = start_time
    df_window["time_window_end"] = end_time
    
    return df_window


def build_player_features(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Returns DataFrame with one row per (session_id, user_id) containing:
      - leadership-related metrics (rule-based)
      - learning strategy features (diversity, repetition, speed, etc.)
      - coordination metrics (session + pairwise)
      - rule-based strategy labels
      - human-readable strategy analysis list & string
    """
    if not events:
        return pd.DataFrame()

    session_id = events[0]["session_id"]

    # Global leadership metrics
    freq_counts, freq_share, total_freq = frequency_metrics(events)
    initiations, reactions = compute_initiation_reaction(events)
    lead_frac = compute_lead_fraction(initiations, reactions)
    leader_roles = assign_leader_role(freq_share)

    # Coordination (session-level + pairwise)
    coord = compute_coordination(events)
    pairwise = compute_pairwise_coordination(events)

    users = sorted({e["user"] for e in events if e["user"] is not None})
    rows = []

    for u in users:
        p_evts = player_events(events, u)
        param_usage = compute_param_usage(p_evts)
        
        # NEW: Learning strategy features
        diversity = compute_action_diversity(p_evts)
        repetition = compute_repetition_score(p_evts, debug=True)  # Enable debug output
        backtrack = compute_backtracking(p_evts)
        pauses = compute_hesitation_pauses(p_evts)
        inefficient = compute_inefficient_moves(p_evts)
        tuning = compute_iterative_tuning(p_evts)
        value_entropy = compute_value_entropy(p_evts)
        
        # Combine features for classification
        strategy_features = {
            "action_diversity": diversity,
            "repetition_ratio": repetition["repetition_ratio"],
            "backtrack_ratio": backtrack["backtrack_ratio"],
            "pause_ratio": pauses["pause_ratio"],
            "oscillation_ratio": inefficient["oscillation_ratio"],
            "tuning_ratio": tuning["tuning_ratio"],
            "value_entropy": value_entropy,
        }
        
        # Classify learning strategy
        learning_strategy = classify_learning_strategy(strategy_features)

        # Who is this player coordinated with (pairwise score > threshold)?
        coord_partners = []
        for other, score in pairwise.get(u, {}).items():
            if score > 0.6:  # threshold for "coordinated"
                coord_partners.append(other)

        # Build human-readable analysis list
        analysis_list = []

        # Leadership
        role = leader_roles.get(u, "non-leader")
        analysis_list.append(f"Leadership: {role}")

        # Learning strategy
        analysis_list.append(f"Learning Strategy: {learning_strategy}")
        analysis_list.append(f"Action Diversity: {diversity:.2f}")
        analysis_list.append(f"Repetition Pattern: {repetition['repetition_ratio']:.2f}")
        
        # Parameter usage
        dom_param = param_usage["dominant_param"]
        analysis_list.append(f"Dominant parameter: {dom_param}")

        # Coordination style
        if coord_partners:
            analysis_list.append(f"Coordinated with players {coord_partners}")
        else:
            analysis_list.append("Uncoordinated with other players (pairwise)")

        strategy_analysis_string = "; ".join(analysis_list)

        row = {
            "session_id": session_id,
            "user_id": u,

            # leadership-related
            "freq_changes": freq_counts.get(u, 0),
            "freq_share": freq_share.get(u, 0.0),
            "initiations": initiations.get(u, 0),
            "reactions": reactions.get(u, 0),
            "lead_fraction": lead_frac.get(u, 0.0),
            "leader_role": role,

            # learning strategy features
            "action_diversity": diversity,
            "repetition_ratio": repetition["repetition_ratio"],
            "repetitive_sequences": repetition["repetitive_sequences"],
            "backtrack_ratio": backtrack["backtrack_ratio"],
            "backtrack_count": backtrack["backtrack_count"],
            "pause_ratio": pauses["pause_ratio"],
            "long_pause_count": pauses["long_pause_count"],
            "avg_pause": pauses["avg_pause"],
            "oscillation_ratio": inefficient["oscillation_ratio"],
            "oscillation_count": inefficient["oscillation_count"],
            "tuning_ratio": tuning["tuning_ratio"],
            "small_adjustment_count": tuning["small_adjustment_count"],
            "value_entropy": value_entropy,

            # per-player param behavior (legacy)
            "total_param_changes": param_usage["total_param_changes"],
            "num_params_used": len(param_usage["param_counts"]),
            "dominant_param_share": param_usage["dominant_share"],

            # coordination (session-level)
            "straight_coord_score": coord["straight_coord_score"],
            "diagonal_coord_score": coord["diagonal_coord_score"],
            "coord_style": coord["coord_style"],

            # pairwise coordination partners
            "coord_partners": coord_partners,

            # learning strategy label
            "learning_strategy": learning_strategy,

            # human-readable strategy analysis
            "strategy_analysis": analysis_list,
            "strategy_analysis_string": strategy_analysis_string,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 7. DECISION TREE FOR STRATEGY (WITH CONFIDENCE)
# ============================================================

def train_strategy_tree(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "learning_strategy",
    max_depth: int = 5,
    augment_with_prototypes: bool = True,
) -> Tuple[Any, pd.DataFrame]:
    """
    Trains a decision tree to classify learning strategies.
    Returns (classifier or None, test_results_df).

    If there is not enough data (e.g. only one class), classifier will be None.
    """
    data = df.dropna(subset=[label_col]).copy()
    if data.empty:
        print(f"[strategy tree] No data to train a tree on '{label_col}'.")
        return None, pd.DataFrame()

    base_class_count = data[label_col].nunique()
    original_rows = len(data)

    # ------------------------------------------------------------------
    # Prototype augmentation: create representative synthetic samples for
    # all 8 strategies so the plotted tree shows richer branching when
    # real data has limited class variety.
    # ------------------------------------------------------------------
    if augment_with_prototypes and base_class_count < 4:
        prototypes = [
            {
                "learning_strategy": "Random_Unstructured",
                "action_diversity": 0.7,
                "repetition_ratio": 0.1,
                "value_entropy": 0.95,
                "backtrack_ratio": 0.05,
                "pause_ratio": 0.05,
                "oscillation_ratio": 0.6,
                "tuning_ratio": 0.2,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Structured_Curiosity_Driven",
                "action_diversity": 0.85,
                "repetition_ratio": 0.1,
                "value_entropy": 0.55,
                "backtrack_ratio": 0.05,
                "pause_ratio": 0.05,
                "oscillation_ratio": 0.15,
                "tuning_ratio": 0.35,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Repetition_Practice",
                "action_diversity": 0.1,
                "repetition_ratio": 0.8,
                "value_entropy": 0.2,
                "backtrack_ratio": 0.05,
                "pause_ratio": 0.05,
                "oscillation_ratio": 0.05,
                "tuning_ratio": 0.15,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Incremental_Goal_Directed_Tuning",
                "action_diversity": 0.45,
                "repetition_ratio": 0.25,
                "value_entropy": 0.45,
                "backtrack_ratio": 0.1,
                "pause_ratio": 0.05,
                "oscillation_ratio": 0.2,
                "tuning_ratio": 0.5,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Backtracking_Recovery",
                "action_diversity": 0.35,
                "repetition_ratio": 0.35,
                "value_entropy": 0.4,
                "backtrack_ratio": 0.6,
                "pause_ratio": 0.1,
                "oscillation_ratio": 0.25,
                "tuning_ratio": 0.25,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Pause_Hesitation",
                "action_diversity": 0.25,
                "repetition_ratio": 0.2,
                "value_entropy": 0.3,
                "backtrack_ratio": 0.1,
                "pause_ratio": 0.7,
                "oscillation_ratio": 0.1,
                "tuning_ratio": 0.1,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Playful_Inefficient",
                "action_diversity": 0.4,
                "repetition_ratio": 0.2,
                "value_entropy": 0.6,
                "backtrack_ratio": 0.15,
                "pause_ratio": 0.1,
                "oscillation_ratio": 0.65,
                "tuning_ratio": 0.25,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
            {
                "learning_strategy": "Iterative_Trial_Patterns",
                "action_diversity": 0.3,
                "repetition_ratio": 0.25,
                "value_entropy": 0.35,
                "backtrack_ratio": 0.1,
                "pause_ratio": 0.1,
                "oscillation_ratio": 0.2,
                "tuning_ratio": 0.8,
                "freq_share": 0.25,
                "lead_fraction": 0.25,
            },
        ]

        proto_df = pd.DataFrame(prototypes)

        # Ensure all expected feature columns exist; fill missing with 0
        for col in feature_cols:
            if col not in proto_df.columns:
                proto_df[col] = 0.0

        data = pd.concat([data, proto_df], ignore_index=True)
        print(
            f"[strategy tree] Augmented with {len(proto_df)} prototype samples to enrich branching "
            f"(real samples: {original_rows}, classes: {base_class_count} -> {data[label_col].nunique()})."
        )

    if data[label_col].nunique() < 2:
        print(f"[strategy tree] Not enough class variety to train a tree on '{label_col}'.")
        print(f"[strategy tree] Found classes: {data[label_col].unique()}")
        print(f"[strategy tree] Class distribution:\n{data[label_col].value_counts()}")
        return None, pd.DataFrame()

    # Default weights = 1.0; prototypes (if present) get down-weighted
    data["_sample_weight"] = 1.0
    if augment_with_prototypes and base_class_count < 4 and len(data) > original_rows:
        data.loc[original_rows:, "_sample_weight"] = 0.3

    X = data[feature_cols]
    y = data[label_col]
    sample_weights = data["_sample_weight"]
    
    print(f"[strategy tree] Training on {len(data)} samples with {y.nunique()} unique classes")
    print(f"[strategy tree] Class distribution:\n{y.value_counts()}")

    class_counts = y.value_counts()
    min_class = class_counts.min()

    # If too few samples OR any class has <2 rows, don't stratify/split; train on all data
    if len(data) < 10 or min_class < 2:
        reason = (
            f"few samples ({len(data)})"
            if len(data) < 10
            else f"class imbalance (min class count={min_class})"
        )
        print(f"[strategy tree] {reason} - training on all data without test split.")
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            criterion="entropy",
            random_state=42,
            min_samples_split=2,
            min_samples_leaf=1
        )
        clf.fit(X, y, sample_weight=sample_weights.to_numpy())
        
        # Print tree structure info
        print(f"[strategy tree] Tree depth: {clf.get_depth()}")
        print(f"[strategy tree] Number of leaves: {clf.get_n_leaves()}")
        
        return clf, pd.DataFrame()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="entropy",
        random_state=42,
        min_samples_split=2,
        min_samples_leaf=1
    )

    # Align weights with the train split indices
    train_idx = X_train.index
    clf.fit(X_train, y_train, sample_weight=sample_weights.loc[train_idx].to_numpy())
    acc = clf.score(X_test, y_test)
    print(f"[strategy tree] Accuracy on test split: {acc:.3f}")
    print(f"[strategy tree] Tree depth: {clf.get_depth()}")
    print(f"[strategy tree] Number of leaves: {clf.get_n_leaves()}")

    test_results = X_test.copy()
    test_results[label_col] = y_test
    test_results[label_col + "_pred"] = clf.predict(X_test)

    return clf, test_results


def add_strategy_predictions(
    df: pd.DataFrame,
    clf: DecisionTreeClassifier,
    feature_cols: List[str],
    label_col: str = "learning_strategy",
) -> pd.DataFrame:
    """
    Adds prediction and confidence columns to df using the trained classifier.
    If clf is None, uses rule-based label with confidence = 1.0.
    """
    df = df.copy()

    if clf is None:
        df["strategy_tree_pred"] = df[label_col]
        df["strategy_tree_confidence"] = 1.0
        return df

    X = df[feature_cols]
    probs = clf.predict_proba(X)
    preds = clf.predict(X)
    max_probs = probs.max(axis=1)

    df["strategy_tree_pred"] = preds
    df["strategy_tree_confidence"] = max_probs

    return df


def plot_strategy_tree(
    clf: DecisionTreeClassifier,
    feature_names: List[str],
    class_names: List[str],
    title: str = "Strategy Decision Tree",
):
    if len(class_names) != len(clf.classes_):
        print("[plot tree] class_names mismatch, falling back to clf.classes_.")
        class_names = [str(c) for c in clf.classes_]

    plt.figure(figsize=(16, 9))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# 8. CONSOLE "UI" – PER-PLAYER BEHAVIOR REPORT & CSV LOADER
# ============================================================


def read_strategy_results(csv_path: Path) -> pd.DataFrame:
    """
    Load saved strategy results and rehydrate list-like columns so reports print cleanly.
    """
    if not csv_path.exists():
        print(f"[read csv] File not found: {csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Attempt to parse list-like columns that may have been stringified in CSV
    list_like_cols = ["coord_partners", "strategy_analysis"]
    for col in list_like_cols:
        if col in df.columns:
            df[col] = df[col].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith("[") else x
            )
    return df


def print_player_overview(df: pd.DataFrame):
    """
    Quick-glance overview page to help navigate the detailed terminal output.
    """
    if df.empty:
        print("[overview] No player data to display.")
        return

    print("=" * 80)
    print("PAGE 1/2: Strategy Overview (per player)")
    print("=" * 80)
    header = f"{'Idx':>3} | {'Session':<15} | {'Player':<6} | {'Strategy':<28} | {'Div':>6} | {'Entropy':>8} | {'Tuning':>7}"
    print(header)
    print("-" * len(header))

    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        print(
            f"{idx:>3} | {row['session_id']:<15} | {row['user_id']:<6} | "
            f"{row['learning_strategy']:<28} | {row['action_diversity']:>6.2f} | "
            f"{row['value_entropy']:>8.3f} | {row['tuning_ratio']:>7.2f}"
        )

    print("-" * len(header))
    print("Use the index to locate the matching detailed entry below (Page 2/2).")
    print()

def print_player_report(df: pd.DataFrame):
    """
    Simple console UI printing behavior analysis for each player.
    """
    if df.empty:
        print("[report] No player data to display.")
        return

    print("=" * 80)
    print("PAGE 2/2: Detailed Player Profiles")
    print("=" * 80)

    for _, row in df.iterrows():
        print("=" * 80)
        print(f"Session {row['session_id']} - Player {row['user_id']}")
        print("=" * 80)
        
        print("\n[LEARNING STRATEGY CLASSIFICATION]")
        print(f"  Detected Strategy:     {row['learning_strategy']}")
        print(f"  Tree Prediction:       {row.get('strategy_tree_pred', 'N/A')}")
        print(f"  Prediction Confidence: {row.get('strategy_tree_confidence', np.nan):.2f}")
        
        print("\n[BEHAVIORAL FEATURES]")
        print(f"  Action Diversity:      {row['action_diversity']:.3f}  (high = exploration)")
        print(f"  Repetition Ratio:      {row['repetition_ratio']:.3f}  (high = practice)")
        print(f"  Value Entropy:         {row['value_entropy']:.3f}  (high = random/chaotic)")
        print(f"  Backtrack Ratio:       {row['backtrack_ratio']:.3f}  (high = error correction)")
        print(f"  Pause Ratio:           {row['pause_ratio']:.3f}  (high = hesitation)")
        print(f"  Oscillation Ratio:     {row['oscillation_ratio']:.3f}  (high = playful/inefficient)")
        print(f"  Tuning Ratio:          {row['tuning_ratio']:.3f}  (high = iterative refinement)")
        
        print("\n[LEADERSHIP METRICS]")
        print(f"  Role:                  {row['leader_role']}")
        print(f"  Frequency changes:     {row['freq_changes']}")
        print(f"  Frequency share:       {row['freq_share']:.2f}")
        print(f"  Lead fraction:         {row['lead_fraction']:.2f}")

        print("\n[COORDINATION]")
        print(f"  Coord style (session): {row['coord_style']}")
        print(f"  Straight coord score:  {row['straight_coord_score']:.2f}")
        print(f"  Diagonal coord score:  {row['diagonal_coord_score']:.2f}")
        print(f"  Coordinated with:      {row['coord_partners']}")

        print("\n[SUMMARY]")
        print(f"  {row.get('strategy_analysis_string', '')}")
        print("=" * 80)
        print()


# ============================================================
# 9. MAIN (for CLI usage)
# ============================================================

if __name__ == "__main__":
    base = Path(".")
    log_paths = [
        base / "User0.log",
        base / "User1.log",
        base / "User2.log",
        base / "User3.log",
    ]

    session_id = "session_1"
    events = parse_session(log_paths, session_id=session_id)

    if not events:
        print("No events found. Check your log file paths.")
        exit(0)

    df_players = build_player_features(events)

    print("=== Per-player behavior report (rule-based classification) ===")
    print_player_overview(df_players)
    print_player_report(df_players)

    # Learning strategy decision-tree features
    strategy_feature_cols = [
        "action_diversity",
        "repetition_ratio",
        "value_entropy",
        "backtrack_ratio",
        "pause_ratio",
        "oscillation_ratio",
        "tuning_ratio",
        "freq_share",
        "lead_fraction",
    ]

    # Ensure we have at least 2 strategy classes
    df_players = ensure_label_diversity(df_players, "learning_strategy")

    clf, test_results = train_strategy_tree(
        df_players,
        feature_cols=strategy_feature_cols,
        label_col="learning_strategy",
        max_depth=5,
    )

    df_players = add_strategy_predictions(
        df_players,
        clf,
        feature_cols=strategy_feature_cols,
        label_col="learning_strategy",
    )

    print("\n=== Per-player behavior report (with ML predictions) ===")
    print_player_overview(df_players)
    print_player_report(df_players)

    # Export results
    output_path = Path("learning_strategies_analysis.csv")
    df_players.to_csv(output_path, index=False)
    print(f"\n[INFO] Results saved to {output_path}")

    # Demonstrate reading existing results back from CSV
    reloaded_df = read_strategy_results(output_path)
    if not reloaded_df.empty:
        print("\n=== Reloaded results from CSV ===")
        print_player_overview(reloaded_df)
        print_player_report(reloaded_df)

    # Example: Analyze a specific time window
    print("\n" + "=" * 80)
    print("EXAMPLE: Analyzing time window 00:05:00 to 00:06:00")
    print("=" * 80)
    df_window = analyze_time_window(
         events,
        start_time="00:06:00",
        end_time="00:07:00",
        session_id="session_1_window_6-7min"
    )
    
    if not df_window.empty:
        print("\n=== Time Window Analysis ===")
        print_player_overview(df_window)
        print_player_report(df_window)
        
        # Save time window results
        window_output = Path("time_window_6-7min_analysis.csv")
        df_window.to_csv(window_output, index=False)
        print(f"\n[INFO] Time window results saved to {window_output}")
    else:
        print("[INFO] No data in time window or insufficient events for analysis")

         # Analyze the same window directly from parameter_changes_summary.csv (includes velocity-friendly data)
    csv_path = Path("parameter_changes_summary.csv")
    if csv_path.exists():
        csv_events = load_events_from_csv(csv_path, session_id="parameter_changes_csv")
        csv_window_df = analyze_time_window(
            csv_events,
            start_time="00:06:00",
            end_time="00:07:00",
            session_id="parameter_changes_window_6-7min"
        )
        if not csv_window_df.empty:
            print("\n=== Parameter CSV Time Window Analysis (6-7 min) ===")
            print_player_overview(csv_window_df)
            print_player_report(csv_window_df)
            csv_window_output = Path("parameter_changes_6-7min_analysis.csv")
            csv_window_df.to_csv(csv_window_output, index=False)
            print(f"\n[INFO] Parameter CSV window results saved to {csv_window_output}")
        else:
            print("[INFO] CSV-based window had no events in the requested time range")
    else:
        print(f"[INFO] parameter_changes_summary.csv not found at {csv_path}")
    
    # Plot decision tree (at the end to avoid blocking)
    if clf is not None:
        class_names = sorted(df_players["learning_strategy"].unique())
        plot_strategy_tree(
            clf,
            feature_names=strategy_feature_cols,
            class_names=class_names,
            title="Learning Strategy Decision Tree",
        )
