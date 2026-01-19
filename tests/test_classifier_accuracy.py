#!/usr/bin/env python3
"""
Strategy Classifier Accuracy Testing Framework

This module provides tools to:
1. Generate synthetic labeled log data that mimics specific strategy patterns
2. Run the classifier on the labeled data
3. Compare predictions with ground truth labels
4. Calculate accuracy metrics (Accuracy, F1, Precision, Recall, Confusion Matrix)

Usage:
    python tests/test_classifier_accuracy.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

from log_parser import parse_session
from strategy_classifier.pipeline import run_full_pipeline_for_session
from strategy_classifier.constants import ALL_STRATEGIES

# ============================================================
# STRATEGY PATTERNS - Define interaction patterns for each strategy
# ============================================================

STRATEGY_PATTERNS = {
    "goal_directed_tuning": {
        "description": "Focused adjustments toward a specific goal with small consistent steps",
        "pattern": {
            "params": ["amplitude"],  # Focus on one param
            "step_sizes": [0.5, 1.0, 0.8, 0.7],  # Small consistent steps
            "direction": "monotonic",  # Same direction adjustments
            "timing": "steady",  # Consistent timing
            "duration_sec": 10,
        }
    },
    "incremental_adjustment": {
        "description": "Small incremental changes to parameters",
        "pattern": {
            "params": ["frequency", "amplitude"],
            "step_sizes": [0.1, 0.2, 0.15, 0.1],  # Very small steps
            "direction": "progressive",
            "timing": "regular",
            "duration_sec": 15,
        }
    },
    "iterative_finetuning": {
        "description": "Repeated adjustments with occasional reversals for fine-tuning",
        "pattern": {
            "params": ["amplitude"],
            "step_sizes": [0.3, -0.1, 0.2, -0.05, 0.1],  # Small with undos
            "direction": "oscillating_small",
            "timing": "clustered",
            "duration_sec": 12,
        }
    },
    "random_trial_error": {
        "description": "Erratic parameter changes with large variance",
        "pattern": {
            "params": ["frequency", "amplitude", "offset"],
            "step_sizes": [5.0, -8.0, 3.0, -10.0, 7.0],  # Large random steps
            "direction": "zigzag",
            "timing": "irregular",
            "duration_sec": 20,
        }
    },
    "structured_exploration": {
        "description": "Systematic exploration of parameter space",
        "pattern": {
            "params": ["frequency", "amplitude", "offset"],
            "step_sizes": [2.0, 2.0, 2.0, 2.0],  # Consistent exploratory steps
            "direction": "cycling",
            "timing": "regular",
            "duration_sec": 25,
        }
    },
    "systematic_parameter_sweep": {
        "description": "One parameter varied while others held constant",
        "pattern": {
            "params": ["amplitude"],  # Single param focus
            "step_sizes": [5.0, 5.0, 5.0, 5.0, 5.0],  # Systematic sweep
            "direction": "monotonic",
            "timing": "regular",
            "duration_sec": 15,
        }
    },
}

# ============================================================
# LOG DATA GENERATOR
# ============================================================

class LabeledLogGenerator:
    """Generate synthetic log data with known strategy labels."""
    
    def __init__(self, user_id: int = 0, session_id: str = "test_session"):
        self.user_id = user_id
        self.session_id = session_id
        self.current_time = 0.0
        self.current_values = {
            "frequency": 0.5,
            "amplitude": 50.0,
            "offset": 100.0,
            "phase shift": 0.0,
        }
    
    def _format_timestamp(self, seconds: float) -> str:
        """Convert seconds to log timestamp format HH:MM:SS.SSSSSSS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{secs:010.7f}"
    
    def generate_strategy_segment(
        self,
        strategy: str,
        start_time: float,
        duration: float = None,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Generate log lines for a specific strategy pattern.
        
        Returns:
            Tuple of (log_lines, label_info)
        """
        if strategy not in STRATEGY_PATTERNS:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        pattern = STRATEGY_PATTERNS[strategy]["pattern"]
        duration = duration or pattern.get("duration_sec", 15)
        
        log_lines = []
        events = []
        self.current_time = start_time
        
        params = pattern["params"]
        steps = pattern["step_sizes"]
        direction = pattern["direction"]
        timing = pattern["timing"]
        
        # Generate events based on pattern
        num_events = max(4, int(duration / 2))  # At least 4 events
        
        for i in range(num_events):
            param = params[i % len(params)]
            step = steps[i % len(steps)]
            
            # Apply direction logic
            if direction == "zigzag":
                step = step * (1 if np.random.random() > 0.5 else -1)
            elif direction == "monotonic":
                step = abs(step)
            elif direction == "oscillating_small":
                if i % 3 == 2:  # Every 3rd is a small reversal
                    step = -abs(step) * 0.3
            
            # Calculate timing
            if timing == "regular":
                time_delta = duration / num_events
            elif timing == "irregular":
                time_delta = np.random.uniform(0.5, 3.0)
            elif timing == "clustered":
                time_delta = np.random.uniform(0.1, 0.5) if i % 3 != 0 else np.random.uniform(1.5, 3.0)
            else:  # steady
                time_delta = duration / num_events
            
            self.current_time += time_delta
            
            # Update value
            new_value = self.current_values[param] + step
            # Clamp to reasonable ranges
            if param == "frequency":
                new_value = max(0.0, min(2.0, new_value))
            elif param == "amplitude":
                new_value = max(0.0, min(100.0, new_value))
            elif param == "offset":
                new_value = max(0, min(360, new_value))
            elif param == "phase shift":
                new_value = max(-180, min(180, new_value))
            
            self.current_values[param] = new_value
            
            # Create log line
            timestamp = self._format_timestamp(self.current_time)
            log_line = f"{timestamp} [Info] User {self.user_id} sets {param} to {new_value:.1f}."
            log_lines.append(log_line)
            
            events.append({
                "timestamp_sec": self.current_time,
                "param": param,
                "value": new_value,
            })
        
        label_info = {
            "user_id": self.user_id,
            "strategy": strategy,
            "start_time": start_time,
            "end_time": self.current_time,
            "num_events": len(events),
            "events": events,
        }
        
        return log_lines, label_info


def generate_labeled_test_logs(
    output_dir: Path,
    strategies_sequence: List[Tuple[str, float]],
    user_id: int = 0,
) -> Tuple[Path, List[Dict[str, Any]]]:
    """
    Generate a complete labeled log file with multiple strategy segments.
    
    Args:
        output_dir: Directory to write the log file
        strategies_sequence: List of (strategy_name, duration) tuples
        user_id: User ID for the log file
    
    Returns:
        Tuple of (log_file_path, ground_truth_labels)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    generator = LabeledLogGenerator(user_id=user_id)
    all_log_lines = []
    ground_truth = []
    
    # Add initial control message
    all_log_lines.append("00:00:00.0000000 [Info] A is in control.")
    
    current_time = 1.0  # Start after control message
    
    for strategy, duration in strategies_sequence:
        log_lines, label_info = generator.generate_strategy_segment(
            strategy=strategy,
            start_time=current_time,
            duration=duration,
        )
        all_log_lines.extend(log_lines)
        ground_truth.append(label_info)
        current_time = label_info["end_time"] + 1.0  # Small gap between segments
    
    # Write log file
    log_path = output_dir / f"User{user_id}.log"
    with open(log_path, "w") as f:
        f.write("\n".join(all_log_lines))
    
    # Write ground truth labels
    labels_path = output_dir / "ground_truth_labels.json"
    with open(labels_path, "w") as f:
        json.dump(ground_truth, f, indent=2, default=str)
    
    print(f"[INFO] Generated {len(all_log_lines)} log lines to {log_path}")
    print(f"[INFO] Ground truth labels saved to {labels_path}")
    
    return log_path, ground_truth


# ============================================================
# SPEED TRENDS GENERATOR (for segmentation)
# ============================================================

def generate_speed_trends_for_labels(
    ground_truth: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    """
    Generate speed/trends.csv that aligns with the labeled segments.
    This ensures the classifier segments match our labeled time windows.
    """
    rows = []
    
    for i, label in enumerate(ground_truth):
        rows.append({
            "trend": "rising" if i % 2 == 0 else "falling",  # Alternate trends
            "starttime": label["start_time"],
            "endtime": label["end_time"],
        })
    
    df = pd.DataFrame(rows)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[INFO] Generated speed trends to {output_path}")
    return output_path


# ============================================================
# EVALUATION METRICS
# ============================================================

def evaluate_classifier(
    predictions_df: pd.DataFrame,
    ground_truth: List[Dict[str, Any]],
    strategy_col: str = "pred_strategy",
) -> Dict[str, Any]:
    """
    Compare classifier predictions against ground truth labels.
    
    Returns dictionary with:
        - overall_accuracy
        - per_strategy_metrics
        - confusion_matrix
        - classification_report
    """
    results = {}
    
    # Build prediction mapping: for each ground truth segment, find best matching prediction
    y_true = []
    y_pred = []
    
    for label in ground_truth:
        user_id = label["user_id"]
        start = label["start_time"]
        end = label["end_time"]
        true_strategy = label["strategy"]
        
        # Find predictions overlapping this time window for this user
        mask = (
            (predictions_df["user_id"] == user_id) &
            (predictions_df["seg_start"] < end) &
            (predictions_df["seg_end"] > start)
        )
        matching_preds = predictions_df[mask]
        
        if len(matching_preds) == 0:
            # No prediction for this segment
            y_true.append(true_strategy)
            y_pred.append("no_prediction")
            continue
        
        # Get the dominant predicted strategy for this window
        # Use the strategy with highest probability sum across matching segments
        if strategy_col in matching_preds.columns:
            pred_strategy = matching_preds[strategy_col].mode()
            if len(pred_strategy) > 0:
                y_true.append(true_strategy)
                y_pred.append(pred_strategy.iloc[0])
            else:
                y_true.append(true_strategy)
                y_pred.append("no_strategy")
        else:
            # Fall back to finding highest probability strategy
            best_strategy = None
            best_prob = -1
            for strat in ALL_STRATEGIES:
                prob_col = f"{strat}_prob"
                if prob_col in matching_preds.columns:
                    avg_prob = matching_preds[prob_col].mean()
                    if avg_prob > best_prob:
                        best_prob = avg_prob
                        best_strategy = strat
            
            y_true.append(true_strategy)
            y_pred.append(best_strategy or "no_strategy")
    
    # Calculate metrics
    unique_labels = sorted(set(y_true + y_pred))
    
    results["overall_accuracy"] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    results["per_strategy_metrics"] = {}
    for strategy in set(y_true):
        y_true_binary = [1 if y == strategy else 0 for y in y_true]
        y_pred_binary = [1 if y == strategy else 0 for y in y_pred]
        
        if sum(y_true_binary) > 0:  # Only if strategy appears in ground truth
            results["per_strategy_metrics"][strategy] = {
                "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
                "support": sum(y_true_binary),
            }
    
    # Confusion matrix
    results["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
    results["confusion_matrix_labels"] = unique_labels
    
    # Classification report
    results["classification_report"] = classification_report(
        y_true, y_pred,
        labels=unique_labels,
        zero_division=0,
    )
    
    # Macro and weighted averages
    results["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)
    results["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    results["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    results["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    
    # Raw predictions for debugging
    results["y_true"] = y_true
    results["y_pred"] = y_pred
    
    return results


def print_evaluation_report(results: Dict[str, Any]) -> None:
    """Print a formatted evaluation report."""
    print("\n" + "=" * 70)
    print("STRATEGY CLASSIFIER EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\n📊 Overall Accuracy: {results['overall_accuracy']:.2%}")
    print(f"📈 Macro F1-Score:   {results['macro_f1']:.2%}")
    print(f"📈 Weighted F1-Score: {results['weighted_f1']:.2%}")
    print(f"📈 Macro Precision:  {results['macro_precision']:.2%}")
    print(f"📈 Macro Recall:     {results['macro_recall']:.2%}")
    
    print("\n" + "-" * 70)
    print("Per-Strategy Metrics:")
    print("-" * 70)
    print(f"{'Strategy':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 70)
    
    for strategy, metrics in sorted(results["per_strategy_metrics"].items()):
        print(f"{strategy:<35} {metrics['precision']:>10.2%} {metrics['recall']:>10.2%} "
              f"{metrics['f1']:>10.2%} {metrics['support']:>10}")
    
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    print(results["classification_report"])
    
    print("\n" + "-" * 70)
    print("Confusion Matrix:")
    print("-" * 70)
    labels = results["confusion_matrix_labels"]
    cm = np.array(results["confusion_matrix"])
    
    # Print header
    print(f"{'True \\ Pred':<20}", end="")
    for label in labels:
        print(f"{label[:15]:<16}", end="")
    print()
    
    # Print rows
    for i, true_label in enumerate(labels):
        print(f"{true_label[:20]:<20}", end="")
        for j in range(len(labels)):
            print(f"{cm[i][j]:<16}", end="")
        print()
    
    print("\n" + "=" * 70)


# ============================================================
# MAIN TEST RUNNER
# ============================================================

def run_classifier_test(
    test_name: str = "default",
    strategies_sequence: List[Tuple[str, float]] = None,
    output_base_dir: Path = None,
) -> Dict[str, Any]:
    """
    Run a complete classifier accuracy test.
    
    Args:
        test_name: Name for this test run
        strategies_sequence: List of (strategy_name, duration) to test
        output_base_dir: Base directory for test outputs
    
    Returns:
        Evaluation results dictionary
    """
    if output_base_dir is None:
        output_base_dir = Path(__file__).parent.parent / "logs" / "Labelled" / "test_data"
    
    if strategies_sequence is None:
        # Default test: one segment of each available strategy
        strategies_sequence = [
            ("goal_directed_tuning", 12),
            ("incremental_adjustment", 15),
            ("iterative_finetuning", 12),
            ("random_trial_error", 20),
            ("structured_exploration", 25),
            ("systematic_parameter_sweep", 15),
        ]
    
    print(f"\n🧪 Running Classifier Test: {test_name}")
    print(f"   Testing {len(strategies_sequence)} strategy segments")
    
    test_dir = output_base_dir / test_name
    
    # Generate test data
    log_path, ground_truth = generate_labeled_test_logs(
        output_dir=test_dir,
        strategies_sequence=strategies_sequence,
        user_id=0,
    )
    
    # Generate speed trends
    trends_path = generate_speed_trends_for_labels(
        ground_truth=ground_truth,
        output_path=test_dir / "speed" / "trends.csv",
    )
    
    # Parse the generated log
    events = parse_session([log_path], session_id="test_session")
    print(f"[INFO] Parsed {len(events)} events from generated log")
    
    # Run classifier pipeline
    predictions_df, segments = run_full_pipeline_for_session(
        events=events,
        speed_csv_path=trends_path,
        session_id="test_session",
    )
    
    # Derive predicted strategy for each row (highest prob strategy)
    def get_dominant_strategy(row):
        best_strat = None
        best_prob = -1
        for strat in ALL_STRATEGIES:
            prob_col = f"{strat}_prob"
            if prob_col in row.index and row[prob_col] > best_prob:
                best_prob = row[prob_col]
                best_strat = strat
        return best_strat if best_strat else "no_strategy"
    
    predictions_df["pred_strategy"] = predictions_df.apply(get_dominant_strategy, axis=1)
    
    # Save predictions
    pred_path = test_dir / "predictions.csv"
    predictions_df.to_csv(pred_path, index=False)
    print(f"[INFO] Predictions saved to {pred_path}")
    
    # Evaluate
    results = evaluate_classifier(predictions_df, ground_truth)
    
    # Print report
    print_evaluation_report(results)
    
    # Save results
    results_path = test_dir / "evaluation_results.json"
    # Convert numpy arrays to lists for JSON serialization
    results_serializable = {
        k: v.tolist() if isinstance(v, np.ndarray) else v
        for k, v in results.items()
    }
    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2, default=str)
    print(f"[INFO] Results saved to {results_path}")
    
    return results


# ============================================================
# CLI ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Strategy Classifier Accuracy")
    parser.add_argument("--test-name", default="default_test", help="Name for this test run")
    parser.add_argument("--strategies", nargs="+", 
                        help="Strategies to test (space-separated)")
    parser.add_argument("--output-dir", type=Path, 
                        help="Output directory for test data")
    
    args = parser.parse_args()
    
    strategies_sequence = None
    if args.strategies:
        # Each strategy gets 15 seconds by default
        strategies_sequence = [(s, 15) for s in args.strategies]
    
    results = run_classifier_test(
        test_name=args.test_name,
        strategies_sequence=strategies_sequence,
        output_base_dir=args.output_dir,
    )
    
    # Exit with non-zero if accuracy is too low
    if results["overall_accuracy"] < 0.5:
        print("\n⚠️  Warning: Overall accuracy below 50%!")
        sys.exit(1)
    else:
        print("\n✅ Test completed successfully!")
        sys.exit(0)
