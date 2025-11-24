import re
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


def parse_log_file(path: Path, session_id: str) -> List[Dict[str, Any]]:
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
            user = int(user_str) if user_str is not None else None
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
        all_events.extend(parse_log_file(p, session_id=session_id))
    all_events.sort(key=lambda e: e["time"])
    return all_events


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
    You can refine this rule later (e.g., requiring lead_fraction, etc.)
    """
    if not freq_share:
        return {}

    # find user with highest share
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
# 3. PARAM USAGE & HOTAT / VOTAT FEATURES
# ============================================================

def player_events(events: List[Dict[str, Any]], user_id: int) -> List[Dict[str, Any]]:
    return [e for e in events if e["user"] == user_id]


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
# 4. COORDINATION METRICS (STRAIGHT / DIAGONAL GAIT)
# ============================================================

def compute_coordination(events: List[Dict[str, Any]], window: float = 1.5) -> Dict[str, Any]:
    """
    Compute coordination scores between players, focusing on amplitude/frequency changes.

    - straight_coord_score: average coordination between opposite legs
    - diagonal_coord_score: average coordination between neighboring legs
    - coord_style: "straight", "diagonal", or "uncoordinated"
    """

    # Collect unique users
    users = sorted({e["user"] for e in events if e["user"] is not None})
    if len(users) < 2:
        return {
            "straight_coord_score": 0.0,
            "diagonal_coord_score": 0.0,
            "coord_style": "uncoordinated",
        }

    # Build per-player change sequences for frequency & amplitude
    raw_changes: Dict[int, List[Tuple[float, float]]] = defaultdict(list)  # user -> [(time, value)]

    for e in events:
        if e["user"] is None:
            continue
        if e["param"] not in ("frequency", "amplitude"):
            continue
        raw_changes[e["user"]].append((e["time"], e["value"]))

    # Convert to (time, delta_sign) sequences
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

    # Helpers to define pairs
    # For exactly 4 users, treat them as (0,1,2,3) or whatever their ids are.
    # Opposite pairs: user[0]-user[2], user[1]-user[3]
    # Neighbor (diagonal/adjacent) pairs: cyclic neighbors
    if len(users) == 4:
        straight_pairs = [(users[0], users[2]), (users[1], users[3])]
    else:
        straight_pairs = []  # if not 4 players, we can't define "opposites" reliably

    diagonal_pairs = []
    for i in range(len(users)):
        diagonal_pairs.append((users[i], users[(i + 1) % len(users)]))

    def pair_coord(u1: int, u2: int) -> float:
        """
        Returns a coordination score in [0, 1] for user pair (u1, u2).
        1 = always same direction, 0 = always opposite, ~0.5 = random/uncorrelated.
        """
        if u1 not in step_changes or u2 not in step_changes:
            return 0.5  # neutral

        s1 = step_changes[u1]
        s2 = step_changes[u2]

        matches: List[int] = []
        j = 0
        for t1, sign1 in s1:
            # advance s2 index to be within window
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
            return 0.5  # no evidence => neutral

        mean_sign = float(np.mean(matches))  # -1..1
        return (mean_sign + 1.0) / 2.0      # map to 0..1

    straight_scores = []
    for a, b in straight_pairs:
        straight_scores.append(pair_coord(a, b))

    diagonal_scores = []
    for a, b in diagonal_pairs:
        diagonal_scores.append(pair_coord(a, b))

    straight_score = float(np.mean(straight_scores)) if straight_scores else 0.5
    diagonal_score = float(np.mean(diagonal_scores)) if diagonal_scores else 0.5

    # Turn neutral 0.5 into "uncoordinated" if both are near it
    coord_style: str
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


# ============================================================
# 5. BUILD PER-PLAYER FEATURE TABLE
# ============================================================

def build_player_features(events: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Returns DataFrame with one row per (session_id, user_id) containing:
      - leadership-related metrics (rule-based)
      - HOTAT/VOTAT-related metrics
      - coordination metrics
      - rule-based strategy labels (simple + full: strategy + coord_style)
    """
    if not events:
        return pd.DataFrame()

    session_id = events[0]["session_id"]

    # Global leadership metrics
    freq_counts, freq_share, total_freq = frequency_metrics(events)
    initiations, reactions = compute_initiation_reaction(events)
    lead_frac = compute_lead_fraction(initiations, reactions)
    leader_roles = assign_leader_role(freq_share)

    # Coordination (session-level)
    coord = compute_coordination(events)

    users = sorted({e["user"] for e in events if e["user"] is not None})
    rows = []

    for u in users:
        p_evts = player_events(events, u)
        param_usage = compute_param_usage(p_evts)
        cluster_stats = compute_single_param_clusters(p_evts)

        simple_strat_label = classify_strategy_rule_based(param_usage, cluster_stats)
        strategy_full_label = f"{simple_strat_label}_{coord['coord_style']}"

        row = {
            "session_id": session_id,
            "user_id": u,

            # leadership-related
            "freq_changes": freq_counts.get(u, 0),
            "freq_share": freq_share.get(u, 0.0),
            "initiations": initiations.get(u, 0),
            "reactions": reactions.get(u, 0),
            "lead_fraction": lead_frac.get(u, 0.0),
            "leader_role": leader_roles.get(u, "non-leader"),

            # per-player param behavior
            "total_param_changes": param_usage["total_param_changes"],
            "num_params_used": len(param_usage["param_counts"]),
            "dominant_param_share": param_usage["dominant_share"],
            "single_param_cluster_ratio": cluster_stats["single_param_cluster_ratio"],

            # coordination (same for all players in session, but we copy it here)
            "straight_coord_score": coord["straight_coord_score"],
            "diagonal_coord_score": coord["diagonal_coord_score"],
            "coord_style": coord["coord_style"],

            # labels (rule-based)
            "strategy_rule_label": simple_strat_label,
            "strategy_full_label": strategy_full_label,
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# 6. DECISION TREE FOR STRATEGY (WITH CONFIDENCE)
# ============================================================

def train_strategy_tree(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "strategy_full_label",
    max_depth: int = 4,
) -> Tuple[Any, pd.DataFrame]:
    """
    Trains a decision tree to classify strategies (full label = HOTAT/VOTAT + coord style).
    Returns (classifier or None, test_results_df).

    If there is not enough data (e.g. only one class), classifier will be None.
    """
    data = df.dropna(subset=[label_col]).copy()
    if data.empty or data[label_col].nunique() < 2:
        print(f"[strategy tree] Not enough class variety to train a tree on '{label_col}'.")
        return None, pd.DataFrame()

    X = data[feature_cols]
    y = data[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        criterion="entropy",
        random_state=42
    )

    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    print(f"[strategy tree] Accuracy on test split: {acc:.3f}")

    test_results = X_test.copy()
    test_results[label_col] = y_test
    test_results[label_col + "_pred"] = clf.predict(X_test)

    return clf, test_results


def add_strategy_predictions(
    df: pd.DataFrame,
    clf: DecisionTreeClassifier,
    feature_cols: List[str],
    label_col: str = "strategy_full_label",
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
# 7. CONSOLE "UI" – PER-PLAYER BEHAVIOR REPORT
# ============================================================

def print_player_report(df: pd.DataFrame):
    """
    Simple console UI printing behavior analysis for each player.
    """
    for _, row in df.iterrows():
        print("=" * 60)
        print(f"Session {row['session_id']} - Player {row['user_id']}")
        print("- Leadership-related (rule-based)")
        print(f"  Role:                  {row['leader_role']}")
        print(f"  Frequency changes:     {row['freq_changes']}")
        print(f"  Frequency share:       {row['freq_share']:.2f}")
        print(f"  Initiations:           {row['initiations']}")
        print(f"  Reactions:             {row['reactions']}")
        print(f"  Lead fraction:         {row['lead_fraction']:.2f}")

        print("- Strategy-related (rule-based HOTAT / VOTAT)")
        print(f"  Total param changes:   {row['total_param_changes']}")
        print(f"  # parameters used:     {row['num_params_used']}")
        print(f"  Dominant param share:  {row['dominant_param_share']:.2f}")
        print(f"  Single-param clusters: {row['single_param_cluster_ratio']:.2f}")
        print(f"  Simple strategy label: {row['strategy_rule_label']}")

        print("- Coordination-related")
        print(f"  Coord style:           {row['coord_style']}")
        print(f"  Straight coord score:  {row['straight_coord_score']:.2f}")
        print(f"  Diagonal coord score:  {row['diagonal_coord_score']:.2f}")

        print("- Decision-tree strategy prediction")
        print(f"  Predicted strategy:    {row.get('strategy_tree_pred', 'N/A')}")
        print(f"  Prediction confidence: {row.get('strategy_tree_confidence', np.nan):.2f}")
        print("=" * 60)
        print()


# ============================================================
# 8. MAIN
# ============================================================

if __name__ == "__main__":
    # Adjust these paths to point to your log files
    base = Path(".")  # or Path("/Users/tomdaugherty/Documents/GitHub/P3-1_G08")
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

    print("=== Per-player behavior report (before decision-tree training) ===")
    print_player_report(df_players)

    # Strategy decision-tree
    strategy_feature_cols = [
        "freq_changes",
        "freq_share",
        "lead_fraction",
        "total_param_changes",
        "num_params_used",
        "dominant_param_share",
        "single_param_cluster_ratio",
        "straight_coord_score",
        "diagonal_coord_score",
    ]

    clf, test_results = train_strategy_tree(
        df_players,
        feature_cols=strategy_feature_cols,
        label_col="strategy_full_label",
        max_depth=4,
    )

    # Add predictions & confidences to df_players
    df_players = add_strategy_predictions(
        df_players,
        clf,
        feature_cols=strategy_feature_cols,
        label_col="strategy_full_label",
    )

    print("=== Per-player behavior report (with decision-tree strategies) ===")
    print_player_report(df_players)

    # Optionally visualize the tree if it exists
    if clf is not None:
        class_names = list(df_players["strategy_full_label"].unique())
        plot_strategy_tree(
            clf,
            feature_names=strategy_feature_cols,
            class_names=class_names,
            title="Strategy (HOTAT/VOTAT + coordination) Decision Tree",
        )
