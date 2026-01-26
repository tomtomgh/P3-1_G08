#!/usr/bin/env python3
# --------------------------------------------------------------
# run_all.py
# FULL PIPELINE RUNNER FOR STRATEGY CLASSIFIER
# --------------------------------------------------------------

import json
import os
import sys
from pathlib import Path
import glob
import pandas as pd

# ---------------------------
# IMPORT YOUR PARSER
# ---------------------------
from log_parser import parse_session

# ---------------------------
# IMPORT STRATEGY CLASSIFIER
# ---------------------------
from strategy_classifier.pipeline import run_full_pipeline_for_session
from strategy_classifier.export import df_to_student_segment_json
from strategy_classifier.constants import ALL_STRATEGIES
from strategy_classifier.utils import print_strategy_summary
from strategy_classifier.trees import (
    train_ml_trees_from_rules,
    apply_ml_trees,
)
from strategy_classifier.features import DEFAULT_FEATURE_COLS
from strategy_classifier.prediction_utils import enforce_inactive_guard


def _find_speed_csv_candidate() -> Path | None:
    """
    Heuristic: pick a CSV in the repo that contains a 'speed' column (case-insensitive).
    If none found, return the largest CSV as a fallback.
    """
    script_dir = Path(".")
    csvs = sorted(script_dir.glob("*.csv"), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
    speed_like = {"speed", "speed_px/s", "speed_px_s", "speed_pxps", "speed_px_per_s", "speed_px"}
    for p in csvs:
        try:
            df_sample = pd.read_csv(p, nrows=5)
            cols = {c.lower() for c in df_sample.columns}
            if cols & speed_like:
                print(f"[DEBUG] Using speed CSV candidate (has speed-like column): {p}")
                return p
        except Exception:
            continue
    # fallback: largest csv bigger than small threshold
    for p in csvs:
        try:
            if p.stat().st_size > 1024:
                print(f"[DEBUG] No explicit speed CSV found — using largest CSV fallback: {p}")
                return p
        except Exception:
            continue
    return None


def _normalize_events_for_pipeline(raw_events: list) -> list:
    """
    Convert various event shapes to the canonical format expected by the classifier:
      - timestamp_sec (float)
      - user_id (int or None)
      - param (str) and value (float) when available
      - keep original fields as fallback
    """
    norm = []
    for e in raw_events:
        # timestamp: prefer known names, include 'ts' seen in your logs
        ts = e.get("timestamp_sec") or e.get("timestamp") or e.get("time") or e.get("ts") or e.get("t")
        try:
            ts = float(ts) if ts is not None else 0.0
        except Exception:
            ts = 0.0

        # user id
        user = e.get("user_id")
        if user is None:
            user = e.get("user") or e.get("userid") or e.get("uid")
        try:
            user = int(user) if user is not None else None
        except Exception:
            user = None

        # parameter/value: try top-level then payload style
        param = e.get("param")
        value = e.get("value")
        payload = e.get("payload") or {}
        if param is None and isinstance(payload, dict):
            param = payload.get("param")
        if value is None and isinstance(payload, dict):
            value = payload.get("value")
        try:
            value = float(value) if value is not None else None
        except Exception:
            value = None

        norm.append({
            "session_id": e.get("session_id"),
            "timestamp_sec": ts,
            "user_id": user,
            "param": param,
            "value": value,
            # keep original for debugging
            "_raw": e,
        })
    return norm


def main():

    base = Path(".")
    logs_folder = base / "logs"
    speed_folder = base / "speed"

    # ---------------------------------------------
    # 1. Locate log files
    # ---------------------------------------------
    log_paths = sorted(logs_folder.glob("*.log"))
    if len(log_paths) == 0:
        raise FileNotFoundError("❌ No .log files found in /logs folder")

    print(f"[INFO] Found {len(log_paths)} logs")

    # ---------------------------------------------
    # 2. Parse logs into event list
    # ---------------------------------------------
    session_id = "session_1"
    events = parse_session(log_paths, session_id=session_id)

    if not events:
        raise RuntimeError("❌ No events parsed from logs — check parser or logs")

    print(f"[INFO] Parsed {len(events)} events")

    # ---------------------------------------------
    # 3. Load trends.csv for segmentation
    # ---------------------------------------------
    speed_csv_default = speed_folder / "trends.csv"
    if speed_csv_default.exists():
        speed_csv = speed_csv_default
        print("[INFO] Using speed/trends.csv from speed folder")
    else:
        print("[INFO] speed/trends.csv not found — attempting auto-detect in working dir")
        speed_csv = _find_speed_csv_candidate()
        if speed_csv is None:
            raise FileNotFoundError("❌ No suitable speed CSV found (trends.csv missing and auto-detect failed)")
    print("[INFO] Loading speed CSV…")
    print(f"[DEBUG] Selected speed CSV: {speed_csv} (size={speed_csv.stat().st_size if speed_csv.exists() else 'n/a'})")

    # ---------------------------------------------
    # 4. Run full strategy pipeline
    # ---------------------------------------------

    print("[INFO] Normalizing parsed events for pipeline...")
    events_norm = _normalize_events_for_pipeline(events)
    print(f"[DEBUG] Events -> normalized: {len(events)} -> {len(events_norm)}; sample:")
    for x in events_norm[:6]:
        print(x)
    df_pred, segments = run_full_pipeline_for_session(
        events=events_norm,
        speed_csv_path=speed_csv,
        session_id=session_id,
        dull_max_duration=20.0,
        dull_window=10.0,
    )

    print(f"[INFO] Pipeline produced {len(df_pred)} per-player segments")

    # ---------------------------------------------
    # 5. Save RULE-BASED predictions
    # ---------------------------------------------
    df_pred.to_csv("segment_strategy_predictions.csv", index=False)
    print("✔ Saved: segment_strategy_predictions.csv")

    # Save JSON format
    result_json = df_to_student_segment_json(df_pred)
    with open("segment_strategy_predictions.json", "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    print("✔ Saved: segment_strategy_predictions.json")

    # ---------------------------------------------
    # 6. Display summary of strategies
    # ---------------------------------------------
    print("\n=== RULE-BASED STRATEGY SUMMARY ===")
    print_strategy_summary(df_pred, ALL_STRATEGIES)

    # ---------------------------------------------
    # 7. OPTIONAL: Train ML trees that imitate rules
    # ---------------------------------------------
    print("\n[INFO] Training ML trees to imitate rules…")

    trees = train_ml_trees_from_rules(
        df_pred,
        strategy_names=ALL_STRATEGIES,
        feature_cols=DEFAULT_FEATURE_COLS,
        max_depth=4,
    )

    print("[INFO] Applying ML trees to dataset…")

    df_with_ml = apply_ml_trees(
        df_pred,
        trees,
        feature_cols=DEFAULT_FEATURE_COLS,
    )

    # --- ENFORCE INACTIVITY GUARD (MUST RUN AFTER ALL PREDICTIONS) ---
    # Ensure ML / global tree predictions cannot overwrite inactivity
    df_with_ml = enforce_inactive_guard(df_with_ml)

    # DEBUG: list any inactive rows that still have a non-"No Strategy" prediction
    df_check = df_with_ml.copy()
    df_check["is_inactive"] = (df_check.get("num_actions", 0) == 0) | (df_check.get("has_events", 0) == 0)
    bad = df_check[df_check["is_inactive"] & (df_check.get("pred_strategy", df_check.get("pred", "")) != "No Strategy")]
    if not bad.empty:
        print("\n[WARN] Inactive rows still have a non-No Strategy prediction:")
        print(bad[["user_id", "seg_start", "seg_end", "num_actions", "has_events", "pred_strategy"]].to_string(index=False))
    else:
        print("\n[INFO] No inactive rows misclassified (post-guard).")

    # --- ENSURE final predictions are valid and save atomically ---
    out_primary = Path("segment_strategy_with_global_label.csv")
    out_backup = Path("segment_strategy_predictions_with_ml.csv")
    out_json = Path("segment_strategy_predictions.json")

    # If pipeline produced no segments / no predictions, do NOT overwrite existing outputs
    if df_with_ml is None or getattr(df_with_ml, "empty", True):
        print("[WARN] Final predictions are empty — aborting save to avoid overwriting existing results.")
        if out_primary.exists() and out_primary.stat().st_size > 128:
            print(f"[INFO] Leaving existing file intact: {out_primary} ({out_primary.stat().st_size} bytes)")
        else:
            print("[ERROR] No valid prediction output exists. Investigate upstream pipeline (segments/events).")
        sys.exit(1)

    # atomic write to primary file
    tmp = out_primary.with_suffix(".csv.tmp")
    df_with_ml.to_csv(tmp, index=False, encoding="utf-8")
    try:
        tmp.replace(out_primary)  # atomic-ish replace
    except Exception:
        tmp.rename(out_primary)

    # also write compatible filenames
    df_with_ml.to_csv(out_backup, index=False, encoding="utf-8")
    df_with_ml.to_csv("segment_strategy_predictions.csv", index=False, encoding="utf-8")
    df_with_ml.to_json(out_json, orient="records", force_ascii=False)

    print(f"✔ Saved: {out_primary} ({out_primary.stat().st_size} bytes)")

    print("\n🎉 DONE!")
    print("Generated files:")
    print("  - segment_strategy_predictions.csv")
    print("  - segment_strategy_predictions.json")
    print("  - segment_strategy_predictions_with_ml.csv")


if __name__ == "__main__":
    main()
