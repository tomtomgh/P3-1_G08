#!/usr/bin/env python3
"""
Ground Truth Label Format for Strategy Classifier Testing

This file defines the format for manually labeling GUI interaction data
with ground truth strategy labels. Use this format to create labeled test
data in the logdata/Labelled folder.

LABEL FILE FORMAT (JSON):
=========================

Create a file named "ground_truth_labels.json" in the same directory as your 
labeled log files. The format is:

[
    {
        "user_id": 0,
        "strategy": "goal_directed_tuning",
        "start_time": 38.5,
        "end_time": 90.0,
        "notes": "User is making small consistent amplitude adjustments"
    },
    {
        "user_id": 0,
        "strategy": "random_trial_error",
        "start_time": 90.0,
        "end_time": 120.0,
        "notes": "User appears to be trying random values"
    }
]

AVAILABLE STRATEGIES:
=====================

Exploration Strategies:
- structured_exploration: Systematic exploration of parameter space
- random_trial_error: Erratic parameter changes with large variance
- systematic_parameter_sweep: One parameter varied while others held constant

Tuning Strategies:
- goal_directed_tuning: Focused adjustments toward a specific goal
- iterative_finetuning: Repeated adjustments with occasional reversals
- incremental_adjustment: Small incremental changes to parameters

Repetition Strategies:
- repetition_practice: Repeating similar actions for practice
- trial_repetition_improvement: Repeating with improvement
- trial_repetition_no_improvement: Repeating without improvement

Error/Backtracking Strategies:
- backtracking_recovery: Going back to fix errors
- undo_correction: Using undo to correct mistakes
- loop_stuck_state: Stuck in a repetitive loop

Behavioural Strategies:
- help_seeking_pause: Pausing to seek help
- inactivity_wait: Waiting/inactive period
- playful_inefficient: Playing around inefficiently

CSV FORMAT (Alternative):
=========================

You can also use a CSV file named "ground_truth_labels.csv":

user_id,strategy,start_time,end_time,notes
0,goal_directed_tuning,38.5,90.0,Small consistent adjustments
0,random_trial_error,90.0,120.0,Trying random values


HOW TO CREATE LABELS:
=====================

1. Open the log file (e.g., logdata/Labelled/User0.log)
2. Identify time periods with distinct interaction patterns
3. Note the timestamps (HH:MM:SS.SSSSSSS format -> convert to seconds)
4. Match the pattern to one of the available strategies
5. Record in the ground_truth_labels.json file

EXAMPLE - Converting timestamps:
   00:01:30.5 -> 90.5 seconds
   00:05:00.0 -> 300.0 seconds
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Import available strategies
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from strategy_classifier.constants import ALL_STRATEGIES


def validate_ground_truth_labels(labels_path: Path) -> List[Dict[str, Any]]:
    """
    Load and validate ground truth labels from JSON or CSV file.
    
    Args:
        labels_path: Path to ground_truth_labels.json or .csv
    
    Returns:
        List of validated label dictionaries
    
    Raises:
        ValueError: If labels are invalid
    """
    labels_path = Path(labels_path)
    
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    
    # Load based on extension
    if labels_path.suffix.lower() == ".json":
        with open(labels_path, "r") as f:
            labels = json.load(f)
    elif labels_path.suffix.lower() == ".csv":
        df = pd.read_csv(labels_path)
        labels = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported label file format: {labels_path.suffix}")
    
    # Validate each label
    validated = []
    errors = []
    
    for i, label in enumerate(labels):
        # Check required fields
        required = ["user_id", "strategy", "start_time", "end_time"]
        missing = [f for f in required if f not in label]
        if missing:
            errors.append(f"Label {i}: Missing required fields: {missing}")
            continue
        
        # Validate strategy name
        if label["strategy"] not in ALL_STRATEGIES:
            errors.append(
                f"Label {i}: Unknown strategy '{label['strategy']}'. "
                f"Must be one of: {ALL_STRATEGIES}"
            )
            continue
        
        # Validate times
        try:
            start = float(label["start_time"])
            end = float(label["end_time"])
            if end <= start:
                errors.append(f"Label {i}: end_time ({end}) must be > start_time ({start})")
                continue
        except (TypeError, ValueError) as e:
            errors.append(f"Label {i}: Invalid time values: {e}")
            continue
        
        validated.append({
            "user_id": int(label["user_id"]),
            "strategy": label["strategy"],
            "start_time": start,
            "end_time": end,
            "notes": label.get("notes", ""),
        })
    
    if errors:
        error_msg = "\n".join(errors)
        raise ValueError(f"Label validation errors:\n{error_msg}")
    
    print(f"✅ Validated {len(validated)} ground truth labels")
    return validated


def timestamp_to_seconds(timestamp: str) -> float:
    """
    Convert HH:MM:SS.SSSSSSS format to seconds.
    
    Args:
        timestamp: Time string in format "HH:MM:SS.SSSSSSS"
    
    Returns:
        Time in seconds as float
    """
    parts = timestamp.split(":")
    if len(parts) != 3:
        raise ValueError(f"Invalid timestamp format: {timestamp}")
    
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    return hours * 3600 + minutes * 60 + seconds


def create_template_labels_file(
    output_path: Path,
    log_path: Path = None,
    user_id: int = 0,
) -> Path:
    """
    Create a template ground truth labels file.
    If log_path is provided, it will extract timestamps to help with labeling.
    
    Args:
        output_path: Where to save the template
        log_path: Optional path to a log file to extract timestamps
        user_id: User ID for the labels
    
    Returns:
        Path to created template file
    """
    output_path = Path(output_path)
    
    template = [
        {
            "user_id": user_id,
            "strategy": "goal_directed_tuning",
            "start_time": 0.0,
            "end_time": 60.0,
            "notes": "REPLACE: Describe why this is goal_directed_tuning"
        },
        {
            "user_id": user_id,
            "strategy": "random_trial_error",
            "start_time": 60.0,
            "end_time": 120.0,
            "notes": "REPLACE: Describe why this is random_trial_error"
        },
    ]
    
    # If log file provided, extract first and last timestamps
    if log_path and Path(log_path).exists():
        import re
        timestamps = []
        with open(log_path, "r") as f:
            for line in f:
                match = re.match(r'(\d{2}:\d{2}:\d{2}\.\d+)', line)
                if match:
                    try:
                        timestamps.append(timestamp_to_seconds(match.group(1)))
                    except:
                        pass
        
        if timestamps:
            min_t = min(timestamps)
            max_t = max(timestamps)
            mid_t = (min_t + max_t) / 2
            
            template[0]["start_time"] = round(min_t, 2)
            template[0]["end_time"] = round(mid_t, 2)
            template[1]["start_time"] = round(mid_t, 2)
            template[1]["end_time"] = round(max_t, 2)
            template[0]["notes"] = f"Log spans {min_t:.1f}s to {max_t:.1f}s. EDIT these labels!"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(template, f, indent=2)
    
    print(f"📝 Created template labels file: {output_path}")
    print(f"   Available strategies: {', '.join(ALL_STRATEGIES)}")
    
    return output_path


def print_available_strategies():
    """Print all available strategies with descriptions."""
    from strategy_classifier.constants import (
        EXPLORATION_STRATEGIES,
        TUNING_STRATEGIES,
        REPETITION_STRATEGIES,
        ERROR_BACKTRACKING_STRATEGIES,
        BEHAVIOURAL_STRATEGIES,
    )
    
    print("\n" + "=" * 60)
    print("AVAILABLE STRATEGIES FOR LABELING")
    print("=" * 60)
    
    categories = [
        ("Exploration Strategies", EXPLORATION_STRATEGIES),
        ("Tuning Strategies", TUNING_STRATEGIES),
        ("Repetition Strategies", REPETITION_STRATEGIES),
        ("Error/Backtracking Strategies", ERROR_BACKTRACKING_STRATEGIES),
        ("Behavioural Strategies", BEHAVIOURAL_STRATEGIES),
    ]
    
    for category_name, strategies in categories:
        print(f"\n{category_name}:")
        print("-" * 40)
        for s in strategies:
            print(f"  • {s}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Ground Truth Labels Utility")
    parser.add_argument("--create-template", type=Path, 
                        help="Create a template labels file at this path")
    parser.add_argument("--validate", type=Path,
                        help="Validate a labels file")
    parser.add_argument("--log-file", type=Path,
                        help="Log file to extract timestamps from (for template creation)")
    parser.add_argument("--list-strategies", action="store_true",
                        help="List all available strategies")
    parser.add_argument("--user-id", type=int, default=0,
                        help="User ID for template labels")
    
    args = parser.parse_args()
    
    if args.list_strategies:
        print_available_strategies()
    elif args.create_template:
        create_template_labels_file(
            output_path=args.create_template,
            log_path=args.log_file,
            user_id=args.user_id,
        )
    elif args.validate:
        try:
            labels = validate_ground_truth_labels(args.validate)
            print(f"\n✅ Valid! Found {len(labels)} labels:")
            for label in labels:
                print(f"   [{label['start_time']:.1f}s - {label['end_time']:.1f}s] "
                      f"User {label['user_id']}: {label['strategy']}")
        except (FileNotFoundError, ValueError) as e:
            print(f"\n❌ Validation failed: {e}")
            sys.exit(1)
    else:
        print_available_strategies()
        print("\nUsage examples:")
        print("  python ground_truth_format.py --list-strategies")
        print("  python ground_truth_format.py --create-template labels.json --log-file User0.log")
        print("  python ground_truth_format.py --validate labels.json")
