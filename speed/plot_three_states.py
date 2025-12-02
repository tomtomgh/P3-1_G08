#!/usr/bin/env python3
"""Plot only three states (dull/flat, increasing, decreasing) using a rolling-window average

Rules implemented:
 - For each sample i, compute avg_prev (the average of the previous N samples) and avg_curr (average of the last N samples including i).
 - If both windows are complete, compute delta = avg_curr - avg_prev.
 - If |delta| <= max(delta_abs, delta_rel * |avg_prev|) => 'dull' (no significant change).
 - If delta > threshold => 'increasing'. If delta < -threshold => 'decreasing'.
 - Short windows that contain very large spikes are ignored by spike-detection (local median & MAD) and spikes are replaced by interpolation.

This script is intentionally small and focuses only on the 3-state detection and plotting.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def detect_spikes_local(y, window=3, multiplier=6.0):
    """Return boolean mask where True indicates an outlier spike.

    Uses centered rolling median and MAD as robust detection.
    """
    n = len(y)
    if n == 0:
        return np.zeros(0, dtype=bool)
    s = pd.Series(y)
    k = max(1, int(window))
    med = s.rolling(window=k, center=True, min_periods=1).median().to_numpy()
    resid = y - med
    mad = np.median(np.abs(resid - np.median(resid)))
    if mad == 0:
        mad = np.std(resid) if np.std(resid) > 0 else 1e-6
    return np.abs(resid) > multiplier * mad


def interpolate_spikes(y, mask):
    y2 = y.astype(float).copy()
    if np.sum(~mask) < 2:
        # not enough non-spike points to interpolate, replace spikes with median
        m = np.nanmedian(y2[~mask]) if np.any(~mask) else 0.0
        y2[mask] = m
        return y2
    idx = np.arange(len(y2))
    y2[mask] = np.nan
    good = ~np.isnan(y2)
    return np.interp(idx, idx[good], y2[good])


def classify_three_states(t, y, window_size=200, delta_abs=None, delta_rel=0.02,
                          ignore_spikes=True, spike_window=3, spike_multiplier=6.0,
                          min_segment_duration_sec=5.0):
    """Return segments [{'start', 'end', 'label'}] labeling dull/up/down.

    window_size: number of samples in the rolling windows (prev and curr)
    delta_abs / delta_rel: thresholds
    ignore_spikes: whether to detect & remove spikes before classification
    """
    t = np.asarray(t)
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n == 0:
        return []

    # Step 1 - optionally remove/ignore spikes
    y_clean = y.copy()
    if ignore_spikes and spike_multiplier and spike_multiplier > 0:
        mask = detect_spikes_local(y_clean, window=spike_window, multiplier=spike_multiplier)
        if np.any(mask):
            y_clean = interpolate_spikes(y_clean, mask)

    # Step 2 - compute windowed averages
    w = int(window_size)
    if w < 1:
        raise ValueError('window_size must be >= 1')

    avg_curr = np.full(n, np.nan)
    avg_prev = np.full(n, np.nan)
    for i in range(n):
        start_curr = i - w + 1
        prev_start = i - 2*w + 1
        if start_curr >= 0:
            avg_curr[i] = np.mean(y_clean[start_curr:i+1])
        if prev_start >= 0:
            avg_prev[i] = np.mean(y_clean[prev_start:prev_start+w])

    # Step 3 - compute delta and thresholds
    delta = avg_curr - avg_prev
    thr_abs = np.zeros(n)
    thr_rel = np.zeros(n)
    if delta_abs is not None:
        thr_abs[:] = delta_abs
    if delta_rel is not None:
        base = np.abs(avg_prev)
        # fallback for NaNs
        baseline = np.nanmean(np.abs(y_clean)) if np.any(~np.isnan(y_clean)) else 1.0
        base[np.isnan(base)] = baseline
        thr_rel = delta_rel * base

    thr = np.maximum(thr_abs, thr_rel)

    labels = np.array(['dull'] * n, dtype=object)
    valid = ~np.isnan(delta)
    labels[valid & (delta > thr)] = 'increasing'
    labels[valid & (delta < -thr)] = 'decreasing'

    # group contiguous labels into segments
    segments = []
    start_idx = 0
    for i in range(1, n):
        if labels[i] != labels[start_idx]:
            segments.append({'start_idx': start_idx, 'end_idx': i-1, 'label': labels[start_idx]})
            start_idx = i
    segments.append({'start_idx': start_idx, 'end_idx': n-1, 'label': labels[start_idx]})

    # convert to times and merge short segments
    segs = []
    for s in segments:
        s0, s1 = s['start_idx'], s['end_idx']
        segs.append({'start': float(t[s0]), 'end': float(t[s1]), 'label': s['label']})

    # merge tiny segments
    out = []
    i = 0
    while i < len(segs):
        seg = segs[i].copy()
        dur = seg['end'] - seg['start']
        if dur >= min_segment_duration_sec or len(segs) == 1:
            out.append(seg)
            i += 1
            continue
        # merge into neighbor with larger duration
        left = out[-1] if out else None
        right = segs[i+1] if i+1 < len(segs) else None
        if left is None and right is not None:
            right['start'] = seg['start']
            i += 1
        elif right is None and left is not None:
            left['end'] = seg['end']
            i += 1
        elif left is not None and right is not None:
            left_d = left['end'] - left['start']
            right_d = right['end'] - right['start']
            if left_d >= right_d:
                left['end'] = seg['end']
            else:
                right['start'] = seg['start']
            i += 1
        else:
            out.append(seg)
            i += 1

    # Ensure adjacent segments with the same label are merged cleanly
    return merge_adjacent_same_label(out)


def merge_adjacent_same_label(segments, eps=1e-9):
    """Merge adjacent segments with identical labels into a single longer segment.

    segments: list of {'start','end','label'} (assumed sorted by start)
    eps: small tolerance for adjacency
    """
    if not segments:
        return []
    merged = [segments[0].copy()]
    for seg in segments[1:]:
        last = merged[-1]
        if seg['label'] == last['label'] and seg['start'] <= last['end'] + eps:
            # extend last segment
            last['end'] = max(last['end'], seg['end'])
        else:
            merged.append(seg.copy())
    return merged


def plot_three_states(input_csv='speed.csv', out='three_states.png', window_size=200, delta_abs=None, delta_rel=0.02,
                       ignore_spikes=True, spike_window_sec=0.2, spike_multiplier=6.0, min_segment_duration_sec=0.5,
                       show=True, summary_csv: str | None = None):
    p = Path(input_csv)
    if not p.exists():
        raise FileNotFoundError(input_csv)
    df = pd.read_csv(p)
    if 'timestamp_sec' not in df.columns:
        raise ValueError('timestamp_sec column required')
    # choose speed column (prefer speed_px/s)
    col_candidates = ['speed_px/s', 'speed', 'speed_px']
    col = next((c for c in col_candidates if c in df.columns), None)
    if col is None:
        # fallback to first numeric column except frame
        for c in df.select_dtypes(include=[float, int]).columns:
            if c != 'frame' and c != 'timestamp_sec':
                col = c
                break
    if col is None:
        raise ValueError('no suitable speed-like column found in CSV')

    t = df['timestamp_sec'].values
    y = df[col].values

    # convert spike window sec to samples (use median dt)
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    spike_window = max(1, int(round(spike_window_sec / dt)))

    segments = classify_three_states(t, y, window_size=window_size, delta_abs=delta_abs, delta_rel=delta_rel,
                                     ignore_spikes=ignore_spikes, spike_window=spike_window, spike_multiplier=spike_multiplier,
                                     min_segment_duration_sec=min_segment_duration_sec)

    # Colors
    colmap = {'dull': '#888888', 'increasing': '#18a03d', 'decreasing': '#d01919'}

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, y, color='silver', linewidth=1, label=col)

    for seg in segments:
        ax.axvspan(seg['start'], seg['end'], color=colmap.get(seg['label'], '#cccccc'), alpha=0.35)

    # build a legend manually
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=colmap[k], edgecolor='k', alpha=0.5, label=k.capitalize()) for k in ['dull','increasing','decreasing']]
    ax.legend(handles=legend_handles + [ax.lines[0]])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(col)
    ax.set_title('Three-states: dull / increasing / decreasing')
    plt.tight_layout()
    fig.savefig(out, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    # optionally save segments summary CSV (trend,starttime,endtime)
    if summary_csv:
        import csv
        with open(summary_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['trend', 'starttime', 'endtime'])
            for s in segments:
                # format times as floats with 3 decimals
                w.writerow([s['label'], f"{s['start']:.3f}", f"{s['end']:.3f}"])

    return out, segments


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file', default='speed.csv')
    p.add_argument('--out', default='three_states.png')
    p.add_argument('--window', type=int, default=200)
    p.add_argument('--delta-abs', type=float, default=None)
    p.add_argument('--delta-rel', type=float, default=0.02)
    p.add_argument('--no-ignore-spikes', dest='ignore_spikes', action='store_false')
    p.add_argument('--spike-multiplier', type=float, default=6.0)
    p.add_argument('--spike-window-sec', type=float, default=0.2)
    p.add_argument('--min-duration', type=float, default=5.0)
    # show the plot by default; provide --no-show to disable
    p.add_argument('--no-show', dest='show', action='store_false')
    p.add_argument('--summary-csv', default=None, help='Optional CSV path to write a summary of detected segments (trend,starttime,endtime)')
    args = p.parse_args()

    out, segs = plot_three_states(input_csv=args.file, out=args.out, window_size=args.window,
                                  delta_abs=args.delta_abs, delta_rel=args.delta_rel,
                                  ignore_spikes=args.ignore_spikes, spike_window_sec=args.spike_window_sec,
                                  spike_multiplier=args.spike_multiplier, min_segment_duration_sec=args.min_duration,
                                  show=args.show, summary_csv=args.summary_csv)
    print(f"Saved plot: {out}")
    print('Detected segments:')
    for s in segs:
        print(f"  {s['label']}: {s['start']:.3f}s - {s['end']:.3f}s")


if __name__ == '__main__':
    main()
