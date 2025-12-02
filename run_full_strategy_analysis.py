#!/usr/bin/env python3
# --------------------------------------------------------------
# run_all.py
# FULL PIPELINE RUNNER FOR STRATEGY CLASSIFIER
# --------------------------------------------------------------

import json
from pathlib import Path

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
    speed_csv = speed_folder / "trends.csv"
    if not speed_csv.exists():
        raise FileNotFoundError("❌ speed/trends.csv not found!")

    print("[INFO] Loading speed CSV…")

    # ---------------------------------------------
    # 4. Run full strategy pipeline
    # ---------------------------------------------
    df_pred, segments = run_full_pipeline_for_session(
        events=events,
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

    df_with_ml.to_csv("segment_strategy_predictions_with_ml.csv", index=False)
    print("✔ Saved: segment_strategy_predictions_with_ml.csv")

    print("\n🎉 DONE!")
    print("Generated files:")
    print("  - segment_strategy_predictions.csv")
    print("  - segment_strategy_predictions.json")
    print("  - segment_strategy_predictions_with_ml.csv")


if __name__ == "__main__":
    main()
