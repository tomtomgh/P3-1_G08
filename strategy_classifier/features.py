from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

# import segmentation helpers used by this module
from .segmentation import assign_events_to_segments, SpeedSegment

# import constants used in feature computations
from strategy_classifier.constants import PARAMS

def build_feature_table_for_session(
    events: List[Dict[str, Any]],
    segments: List[SpeedSegment],
    session_id: str | None = None,
) -> pd.DataFrame:
    """
    Produce one row per (segment, user). Guarantees `num_actions` and `has_events`
    are present and that all features are computed from events filtered by user_id.
    This prevents partner-induced leakage.
    """
    rows = []
    # deduce users from events; fallback to 0-3 if none found
    users = sorted({e.get("user_id") for e in events_sorted if e.get("user_id") is not None})
    if not users:
        users = [0, 1, 2, 3]

    for seg_index, seg in enumerate(segments):
        seg_start = getattr(seg, "start", getattr(seg, "starttime", None))
        seg_end = getattr(seg, "end", getattr(seg, "endtime", None))
        seg_dur = float(seg_end - seg_start) if seg_start is not None and seg_end is not None else 0.0

        for user in users:
            # per-user events inside this segment
            evs = [
                e for e in events
                if e.get("user_id") == user
                and ("timestamp_sec" in e and seg_start <= e["timestamp_sec"] < seg_end
                     or "timestamp" in e and seg_start <= e["timestamp"] < seg_end)
            ]
            num_actions = len(evs)
            has_events = 1 if num_actions > 0 else 0
            action_rate = num_actions / seg_dur if seg_dur > 0 else 0.0

            # minimal safe stats (you can replace with richer computations)
            row = {
                "session_id": session_id,
                "user_id": user,
                "seg_index": seg_index,
                "seg_start": seg_start,
                "seg_end": seg_end,
                "segment_duration": seg_dur,
                "num_actions": num_actions,
                "has_events": has_events,
                "action_rate": action_rate,
            }

            # ensure all DEFAULT_FEATURE_COLS exist with safe defaults
            for col in DEFAULT_FEATURE_COLS:
                if col not in row:
                    row[col] = 0.0 if col not in ("has_events", "num_actions") else int(row.get(col, 0))

            rows.append(row)

    df = pd.DataFrame(rows)
    # keep deterministic ordering
    df = df.sort_values(["user_id", "seg_index"]).reset_index(drop=True)
    return df

@dataclass
class SegmentFeatures:
    session_id: str
    user_id: int
    segment_id: int
    seg_start: float
    seg_end: float
    seg_trend: str
    segment_duration: float
    segment_index_norm: float
    num_actions: int
    num_action_types: int
    action_entropy: float
    dominant_action_share: float
    mean_step_size: float
    var_step_size: float
    zigzag_ratio: float
    undo_ratio: float
    action_rate: float
    mean_pause: float
    config_dist_prev: float
    config_dist_prev2: float
    mean_speed: float
    delta_speed_prev: float
    delta_speed_prev2: float
    has_events: int
    is_dull_speed: int
    num_params_used: int
    dominant_param_share: float
    single_param_cluster_ratio: float
    repeat_run_len: int
    coord_activity_ratio: float
    coord_num_partners: int
    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

DEFAULT_FEATURE_COLS = [
    "segment_duration", "segment_index_norm",
    "num_actions", "num_action_types", "action_entropy",
    "dominant_action_share", "mean_step_size", "var_step_size",
    "zigzag_ratio", "undo_ratio", "action_rate", "mean_pause",
    "config_dist_prev", "config_dist_prev2", "mean_speed",
    "delta_speed_prev", "delta_speed_prev2", "has_events",
    "is_dull_speed", "num_params_used", "dominant_param_share",
    "single_param_cluster_ratio", "repeat_run_len",
    "coord_activity_ratio", "coord_num_partners",
]

def _safe_entropy(counts: Dict[str, int]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    probs = np.array(list(counts.values()), dtype=float) / float(total)
    probs = probs[probs > 0]
    return float(-(probs * np.log(probs)).sum())

def _compute_action_stats(events: List[Dict[str, Any]], init_state: Dict[str, float]):
    if not events:
        return (0,0,0.0,0.0,0.0,0.0,0.0,0.0,0,{},0.0)
    last_vals = init_state.copy()
    action_types: Dict[str,int] = {}
    param_counts: Dict[str,int] = {}
    deltas: List[float] = []
    dirs: List[int] = []
    undo_pairs = 0
    total_pairs = 0
    # cluster for VOTAT-ish
    cluster_window = 1.0
    clusters: List[List[Dict[str,Any]]] = []
    cur = [events[0]]
    base_t = events[0]["time"]
    for ev in events[1:]:
        if ev["time"] - base_t <= cluster_window:
            cur.append(ev)
        else:
            clusters.append(cur)
            cur = [ev]
            base_t = ev["time"]
    clusters.append(cur)
    for cluster in clusters:
        for idx, ev in enumerate(cluster):
            p = ev["param"]
            if p not in PARAMS:
                continue
            new = float(ev["value"])
            old = float(last_vals.get(p,0.0))
            delta = new - old
            dsign = 0 if delta == 0 else int(np.sign(delta))
            atype = f"{p}_{dsign}"
            action_types[atype] = action_types.get(atype,0)+1
            param_counts[p] = param_counts.get(p,0)+1
            deltas.append(abs(delta))
            dirs.append(dsign)
            if idx>0:
                prev_ev = cluster[idx-1]
                if prev_ev["param"] == p:
                    prev_val = float(prev_ev["value"])
                    prev_delta = prev_val - old
                    if prev_delta != 0:
                        ratio = delta/(prev_delta+1e-9)
                        if ratio < -0.8:
                            undo_pairs += 1
                    total_pairs += 1
            last_vals[p] = new
    num_actions = sum(action_types.values())
    num_types = len(action_types)
    ent = _safe_entropy(action_types)
    dom_share = max(action_types.values())/float(num_actions) if num_actions>0 else 0.0
    if deltas:
        m_step = float(np.mean(deltas))
        v_step = float(np.var(deltas))
    else:
        m_step = v_step = 0.0
    nz = [d for d in dirs if d!=0]
    flips = 0
    if len(nz)>=2:
        for i in range(1,len(nz)):
            if nz[i]!=nz[i-1]:
                flips+=1
        zig = flips/float(len(nz)-1)
    else:
        zig = 0.0
    undo_ratio = undo_pairs/float(total_pairs) if total_pairs>0 else 0.0
    num_params = len(param_counts)
    single = multi = 0
    for cl in clusters:
        ps = {ev["param"] for ev in cl}
        if len(ps)==1: single+=1
        elif len(ps)>1: multi+=1
    tot_cl = single+multi
    sp_ratio = single/float(tot_cl) if tot_cl>0 else 0.0
    return (num_actions,num_types,ent,dom_share,m_step,v_step,zig,undo_ratio,num_params,param_counts,sp_ratio)

def _time_features(events: List[Dict[str,Any]], seg: SpeedSegment):
    if not events:
        return 0.0, seg.duration
    times = sorted(ev["time"] for ev in events)
    dur = max(seg.duration,1e-6)
    rate = len(times)/dur
    if len(times)>=2:
        pauses = [t2-t1 for t1,t2 in zip(times[:-1],times[1:])]
        mp = float(np.mean(pauses))
    else:
        mp = dur
    return rate, mp

def _trend_to_speed(trend: str) -> float:
    t = str(trend).lower()
    if t=="increasing": return 1.0
    if t=="decreasing": return -1.0
    return 0.0

def _init_state()->Dict[str,float]:
    return {p:0.0 for p in PARAMS}

def _coord_features(seg_events: List[Dict[str,Any]], users: List[int], window: float=1.0):
    res = {u:(0.0,0) for u in users}
    if not seg_events: return res
    times_by_user = {u:[] for u in users}
    for ev in seg_events:
        u = ev.get("user")
        if u in times_by_user:
            times_by_user[u].append(ev["time"])
    for u in users:
        tu = sorted(times_by_user[u])
        if not tu: continue
        coinc = 0
        partners=set()
        for t in tu:
            for v in users:
                if v==u: continue
                for tv in times_by_user[v]:
                    if abs(tv-t)<=window:
                        coinc+=1
                        partners.add(v)
                        break
        ratio = coinc/float(len(tu)) if tu else 0.0
        res[u]=(ratio,len(partners))
    return res

def build_feature_table_for_session(
    events: List[Dict[str, Any]],
    segments: List[SpeedSegment],
    session_id: str,
) -> pd.DataFrame:
    if not events or not segments:
        return pd.DataFrame()

    # normalize / sort events by a canonical timestamp field
    def _evt_ts(e: Dict[str, Any]) -> float:
        # prefer parser-standard "timestamp_sec", fall back to other common names, default 0.0
        return float(e.get("timestamp_sec") or e.get("timestamp") or e.get("time") or 0.0)

    events_sorted = sorted(events, key=_evt_ts)

    seg_map = assign_events_to_segments(events_sorted, segments)
    users = sorted({e.get("user_id") for e in events_sorted if e.get("user_id") is not None})
    if not users:
        return pd.DataFrame()

    rows: List[Dict[str, Any]] = []
    prev_cfg = {u: np.zeros(len(PARAMS)) for u in users}
    prev2_cfg = {u: np.zeros(len(PARAMS)) for u in users}
    prev_sp = {u: 0.0 for u in users}
    prev2_sp = {u: 0.0 for u in users}
    rep_run = {u: 0 for u in users}
    state = {u: _init_state() for u in users}
    total_segs = len(segments)
    segs_sorted = sorted(segments, key=lambda s: s.start)

    for idx, seg in enumerate(segs_sorted):
        # normalize events for this segment so downstream code can use
        # ev["time"] and ev["user"] (compat with older code)
        seg_evts_raw = seg_map.get(seg.segment_id, [])
        seg_evts = []
        for ev in seg_evts_raw:
            ev2 = ev.copy()
            ev2["time"] = ev.get("timestamp_sec") or ev.get("timestamp") or ev.get("time") or ev.get("ts") or 0.0
            # prefer explicit user_id but fall back to legacy 'user'
            ev2["user"] = ev.get("user_id") if ev.get("user_id") is not None else ev.get("user")
            # ensure param/value live at top-level (payload variants)
            payload = ev.get("payload") or {}
            if ev2.get("param") is None:
                ev2["param"] = payload.get("param")
            if ev2.get("value") is None:
                ev2["value"] = payload.get("value")
            seg_evts.append(ev2)

        coord = _coord_features(seg_evts, users)

        by_user = {u: [] for u in users}
        for ev in seg_evts:
            u = ev.get("user")
            if u in by_user:
                by_user[u].append(ev)

        for u in users:
            u_evts = sorted(by_user[u], key=lambda e: e["time"])
            # ✅ Skip user if they made no valid parameter changes in this segment
            if not any(ev["param"] in PARAMS for ev in u_evts):
                continue

            for ev in u_evts:
                if ev["param"] in PARAMS:
                    state[u][ev["param"]] = float(ev["value"])

            (num_actions, num_types, ent, dom, ms, vs, zig, undo, num_params, param_counts, sp_ratio) = _compute_action_stats(u_evts, state[u])
            cfg_end = np.array([state[u].get(p, 0.0) for p in PARAMS], dtype=float)
            d_prev = float(np.linalg.norm(cfg_end - prev_cfg[u]))
            d_prev2 = float(np.linalg.norm(cfg_end - prev2_cfg[u]))

            if num_actions > 0 and d_prev < 0.05:
                rep_run[u] += 1
            elif num_actions > 0:
                rep_run[u] = 1

            rate, mp = _time_features(u_evts, seg)
            m_speed = _trend_to_speed(seg.trend)
            ds = m_speed - prev_sp[u]
            ds2 = m_speed - prev2_sp[u]
            seg_pos = idx / float(max(total_segs - 1, 1))
            is_dull = 1 if str(seg.trend).lower() == "dull" else 0
            dom_param_share = max(param_counts.values()) / float(num_actions) if num_actions > 0 and param_counts else 0.0
            coord_ratio, coord_partners = coord.get(u, (0.0, 0))

            f = SegmentFeatures(
                session_id=session_id,
                user_id=u,
                segment_id=seg.segment_id,
                seg_start=seg.start,
                seg_end=seg.end,
                seg_trend=seg.trend,
                segment_duration=seg.duration,
                segment_index_norm=seg_pos,
                num_actions=num_actions,
                num_action_types=num_types,
                action_entropy=ent,
                dominant_action_share=dom,
                mean_step_size=ms,
                var_step_size=vs,
                zigzag_ratio=zig,
                undo_ratio=undo,
                action_rate=rate,
                mean_pause=mp,
                config_dist_prev=d_prev,
                config_dist_prev2=d_prev2,
                mean_speed=m_speed,
                delta_speed_prev=ds,
                delta_speed_prev2=ds2,
                has_events=1 if num_actions > 0 else 0,
                is_dull_speed=is_dull,
                num_params_used=num_params,
                dominant_param_share=dom_param_share,
                single_param_cluster_ratio=sp_ratio,
                repeat_run_len=rep_run[u],
                coord_activity_ratio=coord_ratio,
                coord_num_partners=coord_partners,
            )
            rows.append(f.to_dict())

            prev2_cfg[u] = prev_cfg[u]
            prev_cfg[u] = cfg_end
            prev2_sp[u] = prev_sp[u]
            prev_sp[u] = m_speed

    return pd.DataFrame(rows)

