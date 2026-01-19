#!/usr/bin/env python3
"""
Strategy Classifier Evaluation Runner

This script evaluates the strategy classifier against labeled ground truth data.
It can work with:
1. Manually labeled data (logs/Labelled/ with ground_truth_labels.json)
2. Synthetically generated test data

Usage:
    # Evaluate against manually labeled data
    python evaluate_classifier.py --labeled-dir logs/Labelled
    
    # Run synthetic test
    python evaluate_classifier.py --synthetic
    
    # Run both
    python evaluate_classifier.py --all
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from tests.ground_truth_format import validate_ground_truth_labels


class ClassifierEvaluator:
    """Evaluates strategy classifier against ground truth labels."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results = {}
    
    def log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def load_labeled_data(
        self,
        labeled_dir: Path,
        speed_trends_path: Optional[Path] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Path]:
        """
        Load labeled log data and ground truth labels.
        
        Args:
            labeled_dir: Directory containing User*.log files and ground_truth_labels.json
            speed_trends_path: Optional path to speed/trends.csv (defaults to project speed folder)
        
        Returns:
            Tuple of (events, ground_truth_labels, trends_path)
        """
        labeled_dir = Path(labeled_dir)
        
        # Find log files
        log_files = list(labeled_dir.glob("User*.log"))
        if not log_files:
            raise FileNotFoundError(f"No User*.log files found in {labeled_dir}")
        
        self.log(f"📁 Found {len(log_files)} log files in {labeled_dir}")
        
        # Load ground truth labels
        labels_path = labeled_dir / "ground_truth_labels.json"
        if not labels_path.exists():
            # Try CSV format
            labels_path = labeled_dir / "ground_truth_labels.csv"
        
        if labels_path.exists():
            ground_truth = validate_ground_truth_labels(labels_path)
            self.log(f"📋 Loaded {len(ground_truth)} ground truth labels")
        else:
            self.log(f"⚠️  No ground truth labels found. Creating template...")
            from tests.ground_truth_format import create_template_labels_file
            template_path = labeled_dir / "ground_truth_labels.json"
            create_template_labels_file(
                output_path=template_path,
                log_path=log_files[0] if log_files else None,
            )
            raise FileNotFoundError(
                f"No ground truth labels found. Created template at {template_path}. "
                "Please fill it in and re-run."
            )
        
        # Parse events from log files
        events = parse_session(log_files, session_id="labeled_session")
        self.log(f"📊 Parsed {len(events)} events from logs")
        
        # Find speed trends
        if speed_trends_path is None:
            speed_trends_path = PROJECT_ROOT / "speed" / "trends.csv"
        
        if not speed_trends_path.exists():
            raise FileNotFoundError(
                f"Speed trends file not found: {speed_trends_path}. "
                "The classifier needs speed/trends.csv for segmentation."
            )
        
        return events, ground_truth, speed_trends_path
    
    def run_classifier(
        self,
        events: List[Dict[str, Any]],
        speed_trends_path: Path,
        session_id: str = "eval_session",
    ) -> pd.DataFrame:
        """
        Run the strategy classifier pipeline.
        
        Returns:
            DataFrame with predictions
        """
        self.log(f"🔄 Running classifier pipeline...")
        
        predictions_df, segments = run_full_pipeline_for_session(
            events=events,
            speed_csv_path=speed_trends_path,
            session_id=session_id,
        )
        
        # Add dominant predicted strategy column
        def get_dominant_strategy(row):
            best_strat = None
            best_prob = -1
            for strat in ALL_STRATEGIES:
                prob_col = f"{strat}_prob"
                if prob_col in row.index:
                    prob = row[prob_col]
                    if pd.notna(prob) and prob > best_prob:
                        best_prob = prob
                        best_strat = strat
            return best_strat if best_strat else "no_strategy"
        
        predictions_df["pred_strategy"] = predictions_df.apply(get_dominant_strategy, axis=1)
        
        self.log(f"✅ Generated {len(predictions_df)} prediction rows")
        
        return predictions_df
    
    def match_predictions_to_labels(
        self,
        predictions_df: pd.DataFrame,
        ground_truth: List[Dict[str, Any]],
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Match classifier predictions to ground truth labels.
        
        Returns:
            Tuple of (y_true, y_pred, match_details)
        """
        y_true = []
        y_pred = []
        match_details = []
        
        for label in ground_truth:
            user_id = label["user_id"]
            start = label["start_time"]
            end = label["end_time"]
            true_strategy = label["strategy"]
            
            # Find predictions overlapping this time window
            mask = (
                (predictions_df["user_id"] == user_id) &
                (predictions_df["seg_start"] < end) &
                (predictions_df["seg_end"] > start)
            )
            matching_preds = predictions_df[mask]
            
            if len(matching_preds) == 0:
                pred_strategy = "no_prediction"
                confidence = 0.0
                match_details.append({
                    "label": label,
                    "prediction": pred_strategy,
                    "confidence": confidence,
                    "num_matching_segments": 0,
                    "match": False,
                })
            else:
                # Get mode prediction, weighted by segment overlap duration
                pred_strategy = matching_preds["pred_strategy"].mode()
                pred_strategy = pred_strategy.iloc[0] if len(pred_strategy) > 0 else "no_strategy"
                
                # Calculate confidence as average probability of the predicted strategy
                prob_col = f"{pred_strategy}_prob"
                if prob_col in matching_preds.columns:
                    confidence = matching_preds[prob_col].mean()
                else:
                    confidence = 0.5
                
                match_details.append({
                    "label": label,
                    "prediction": pred_strategy,
                    "confidence": confidence,
                    "num_matching_segments": len(matching_preds),
                    "match": pred_strategy == true_strategy,
                })
            
            y_true.append(true_strategy)
            y_pred.append(pred_strategy)
        
        return y_true, y_pred, match_details
    
    def calculate_metrics(
        self,
        y_true: List[str],
        y_pred: List[str],
    ) -> Dict[str, Any]:
        """Calculate evaluation metrics."""
        
        unique_labels = sorted(set(y_true + y_pred))
        
        metrics = {
            "overall_accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        }
        
        # Per-strategy metrics
        metrics["per_strategy"] = {}
        for strategy in set(y_true):
            y_true_binary = [1 if y == strategy else 0 for y in y_true]
            y_pred_binary = [1 if y == strategy else 0 for y in y_pred]
            
            if sum(y_true_binary) > 0:
                metrics["per_strategy"][strategy] = {
                    "precision": precision_score(y_true_binary, y_pred_binary, zero_division=0),
                    "recall": recall_score(y_true_binary, y_pred_binary, zero_division=0),
                    "f1": f1_score(y_true_binary, y_pred_binary, zero_division=0),
                    "support": sum(y_true_binary),
                }
        
        # Confusion matrix
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=unique_labels).tolist()
        metrics["confusion_matrix_labels"] = unique_labels
        
        # Classification report
        metrics["classification_report"] = classification_report(
            y_true, y_pred,
            labels=unique_labels,
            zero_division=0,
        )
        
        # Raw data
        metrics["y_true"] = y_true
        metrics["y_pred"] = y_pred
        
        return metrics
    
    def print_report(
        self,
        metrics: Dict[str, Any],
        match_details: List[Dict],
        title: str = "Evaluation Report",
    ):
        """Print a formatted evaluation report."""
        
        print("\n" + "=" * 70)
        print(f"📊 {title}")
        print("=" * 70)
        
        print(f"\n🎯 Overall Accuracy:  {metrics['overall_accuracy']:.2%}")
        print(f"📈 Macro F1-Score:    {metrics['macro_f1']:.2%}")
        print(f"📈 Weighted F1-Score: {metrics['weighted_f1']:.2%}")
        print(f"📈 Macro Precision:   {metrics['macro_precision']:.2%}")
        print(f"📈 Macro Recall:      {metrics['macro_recall']:.2%}")
        
        print("\n" + "-" * 70)
        print("Per-Strategy Performance:")
        print("-" * 70)
        print(f"{'Strategy':<35} {'Prec':>8} {'Recall':>8} {'F1':>8} {'Support':>8}")
        print("-" * 70)
        
        for strategy, m in sorted(metrics["per_strategy"].items()):
            print(f"{strategy:<35} {m['precision']:>8.2%} {m['recall']:>8.2%} "
                  f"{m['f1']:>8.2%} {m['support']:>8}")
        
        print("\n" + "-" * 70)
        print("Match Details:")
        print("-" * 70)
        
        correct = sum(1 for d in match_details if d["match"])
        print(f"Correct: {correct}/{len(match_details)} ({correct/len(match_details)*100:.1f}%)\n")
        
        for i, detail in enumerate(match_details):
            label = detail["label"]
            status = "✅" if detail["match"] else "❌"
            print(f"{status} [{label['start_time']:.1f}s - {label['end_time']:.1f}s] "
                  f"User {label['user_id']}")
            print(f"   True: {label['strategy']}")
            print(f"   Pred: {detail['prediction']} (conf: {detail['confidence']:.2f})")
            if not detail["match"] and label.get("notes"):
                print(f"   Notes: {label['notes']}")
            print()
        
        print("\n" + "-" * 70)
        print("Confusion Matrix:")
        print("-" * 70)
        
        labels = metrics["confusion_matrix_labels"]
        cm = np.array(metrics["confusion_matrix"])
        
        # Print header (abbreviated labels)
        print(f"{'True\\Pred':<20}", end="")
        for label in labels:
            abbrev = label[:12] + ".." if len(label) > 14 else label
            print(f"{abbrev:<15}", end="")
        print()
        
        # Print rows
        for i, true_label in enumerate(labels):
            abbrev = true_label[:18] + ".." if len(true_label) > 20 else true_label
            print(f"{abbrev:<20}", end="")
            for j in range(len(labels)):
                print(f"{cm[i][j]:<15}", end="")
            print()
        
        print("\n" + "=" * 70)
    
    def evaluate(
        self,
        labeled_dir: Path = None,
        speed_trends_path: Path = None,
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        Run full evaluation pipeline.
        
        Args:
            labeled_dir: Directory with labeled logs and ground truth
            speed_trends_path: Path to speed trends CSV
            save_results: Whether to save results to JSON
        
        Returns:
            Evaluation results dictionary
        """
        if labeled_dir is None:
            labeled_dir = PROJECT_ROOT / "logs" / "Labelled"
        
        labeled_dir = Path(labeled_dir)
        
        # Load data
        events, ground_truth, trends_path = self.load_labeled_data(
            labeled_dir=labeled_dir,
            speed_trends_path=speed_trends_path,
        )
        
        # Run classifier
        predictions_df = self.run_classifier(
            events=events,
            speed_trends_path=trends_path,
        )
        
        # Match predictions to labels
        y_true, y_pred, match_details = self.match_predictions_to_labels(
            predictions_df=predictions_df,
            ground_truth=ground_truth,
        )
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred)
        
        # Print report
        self.print_report(metrics, match_details, title=f"Evaluation: {labeled_dir.name}")
        
        # Save results
        if save_results:
            results_path = labeled_dir / "evaluation_results.json"
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "labeled_dir": str(labeled_dir),
                "num_labels": len(ground_truth),
                "metrics": {
                    k: v for k, v in metrics.items()
                    if k not in ("confusion_matrix", "classification_report")
                },
                "confusion_matrix": metrics["confusion_matrix"],
                "confusion_matrix_labels": metrics["confusion_matrix_labels"],
                "match_details": [
                    {
                        "true": d["label"]["strategy"],
                        "pred": d["prediction"],
                        "start": d["label"]["start_time"],
                        "end": d["label"]["end_time"],
                        "user_id": d["label"]["user_id"],
                        "match": d["match"],
                    }
                    for d in match_details
                ],
            }
            
            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            
            self.log(f"\n💾 Results saved to {results_path}")
            
            # Also save predictions
            pred_path = labeled_dir / "classifier_predictions.csv"
            predictions_df.to_csv(pred_path, index=False)
            self.log(f"💾 Predictions saved to {pred_path}")
        
        self.results = metrics
        return metrics


def run_synthetic_test():
    """Run evaluation using synthetically generated test data."""
    print("\n" + "=" * 70)
    print("🧪 SYNTHETIC DATA TEST")
    print("=" * 70)
    
    from tests.test_classifier_accuracy import run_classifier_test
    
    return run_classifier_test(test_name="synthetic_eval")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Strategy Classifier Accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate against manually labeled data
  python evaluate_classifier.py --labeled-dir logs/Labelled
  
  # Run synthetic test only  
  python evaluate_classifier.py --synthetic
  
  # Run both manual and synthetic evaluation
  python evaluate_classifier.py --all
  
  # Create a ground truth labels template
  python evaluate_classifier.py --create-template logs/Labelled
        """
    )
    
    parser.add_argument(
        "--labeled-dir", type=Path,
        help="Directory containing labeled logs and ground_truth_labels.json"
    )
    parser.add_argument(
        "--speed-trends", type=Path,
        help="Path to speed/trends.csv (defaults to project speed folder)"
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help="Run synthetic data test"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run both manual and synthetic evaluation"
    )
    parser.add_argument(
        "--create-template", type=Path,
        help="Create a ground truth labels template in the specified directory"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Create template
    if args.create_template:
        from tests.ground_truth_format import create_template_labels_file
        template_dir = Path(args.create_template)
        log_files = list(template_dir.glob("User*.log"))
        create_template_labels_file(
            output_path=template_dir / "ground_truth_labels.json",
            log_path=log_files[0] if log_files else None,
        )
        return
    
    # Run evaluations
    results = []
    
    if args.all or args.labeled_dir:
        labeled_dir = args.labeled_dir or (PROJECT_ROOT / "logs" / "Labelled")
        try:
            evaluator = ClassifierEvaluator(verbose=not args.quiet)
            metrics = evaluator.evaluate(
                labeled_dir=labeled_dir,
                speed_trends_path=args.speed_trends,
            )
            results.append(("Manual Labels", metrics))
        except FileNotFoundError as e:
            print(f"\n⚠️  Manual evaluation skipped: {e}")
    
    if args.all or args.synthetic:
        try:
            metrics = run_synthetic_test()
            results.append(("Synthetic Test", metrics))
        except Exception as e:
            print(f"\n⚠️  Synthetic test failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print("📋 EVALUATION SUMMARY")
        print("=" * 70)
        for name, metrics in results:
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['overall_accuracy']:.2%}")
            print(f"  Macro F1:  {metrics['macro_f1']:.2%}")
            print(f"  Precision: {metrics['macro_precision']:.2%}")
            print(f"  Recall:    {metrics['macro_recall']:.2%}")
    else:
        print("\nNo evaluations run. Use --help for usage information.")


if __name__ == "__main__":
    main()
