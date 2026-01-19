# Strategy Classifier Testing Framework

This directory contains tools for testing the accuracy of the strategy classifier by comparing its predictions against ground truth labels.

## Overview

The testing framework provides:
1. **Synthetic data generation** - Create mock labeled interaction data with known strategy patterns
2. **Manual labeling support** - Format specification for manually labeling real interaction data
3. **Classifier evaluation** - Calculate accuracy, F1 score, precision, recall, and confusion matrix

## Quick Start

### 1. Run evaluation against manually labeled data

```bash
# From project root
python tests/evaluate_classifier.py --labeled-dir logs/Labelled
```

### 2. Run synthetic test

```bash
python tests/evaluate_classifier.py --synthetic
```

### 3. Run both

```bash
python tests/evaluate_classifier.py --all
```

## Files

| File | Description |
|------|-------------|
| `test_classifier_accuracy.py` | Synthetic data generator and evaluation runner |
| `ground_truth_format.py` | Ground truth label format documentation and utilities |
| `evaluate_classifier.py` | Main evaluation script for both manual and synthetic tests |

## Creating Ground Truth Labels

### Step 1: Create a template

```bash
python tests/ground_truth_format.py --create-template logs/Labelled/ground_truth_labels.json --log-file logs/Labelled/User0.log
```

### Step 2: Edit the labels file

Open `logs/Labelled/ground_truth_labels.json` and edit the labels:

```json
[
  {
    "user_id": 0,
    "strategy": "goal_directed_tuning",
    "start_time": 38.5,
    "end_time": 90.0,
    "notes": "User making small consistent adjustments toward target"
  },
  {
    "user_id": 0,
    "strategy": "random_trial_error",
    "start_time": 90.0,
    "end_time": 120.0,
    "notes": "Erratic parameter changes"
  }
]
```

### Step 3: Validate labels

```bash
python tests/ground_truth_format.py --validate logs/Labelled/ground_truth_labels.json
```

## Available Strategies

Run this to see all available strategies:

```bash
python tests/ground_truth_format.py --list-strategies
```

Currently supported strategies:

**Exploration:**
- `structured_exploration` - Systematic exploration of parameter space
- `random_trial_error` - Erratic parameter changes with large variance
- `systematic_parameter_sweep` - One parameter varied while others held constant

**Tuning:**
- `goal_directed_tuning` - Focused adjustments toward a specific goal
- `iterative_finetuning` - Repeated adjustments with occasional reversals
- `incremental_adjustment` - Small incremental changes to parameters

**Repetition:**
- `repetition_practice` - Repeating similar actions
- `trial_repetition_improvement` - Repeating with improvement
- `trial_repetition_no_improvement` - Repeating without improvement

**Error/Backtracking:**
- `backtracking_recovery` - Going back to fix errors
- `undo_correction` - Using undo to correct mistakes
- `loop_stuck_state` - Stuck in a repetitive loop

**Behavioural:**
- `help_seeking_pause` - Pausing to seek help
- `inactivity_wait` - Waiting/inactive period
- `playful_inefficient` - Playing around inefficiently

## Converting Timestamps

Log timestamps are in format `HH:MM:SS.SSSSSSS`. To convert to seconds:

| Timestamp | Seconds |
|-----------|---------|
| `00:01:30.5` | 90.5 |
| `00:05:00.0` | 300.0 |
| `00:07:15.7` | 435.7 |

## Output Files

After evaluation, you'll find:

- `evaluation_results.json` - Full evaluation metrics in JSON format
- `classifier_predictions.csv` - Raw predictions from the classifier

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Accuracy** | Percentage of correctly classified segments |
| **Precision** | Of all segments predicted as strategy X, how many were actually X? |
| **Recall** | Of all segments that were actually strategy X, how many did we predict as X? |
| **F1 Score** | Harmonic mean of precision and recall |
| **Macro F1** | Average F1 across all strategies (treats all strategies equally) |
| **Weighted F1** | Average F1 weighted by support (number of samples per strategy) |

## Improving Classifier Accuracy

Based on evaluation results, you can:

1. **Adjust rule thresholds** in `strategy_classifier/strategy_rules/*.py`
2. **Add/modify features** in `strategy_classifier/features.py`
3. **Create more labeled data** for training ML-based classifiers
4. **Review confusion matrix** to understand which strategies are confused

## Example Workflow

```bash
# 1. List available strategies
python tests/ground_truth_format.py --list-strategies

# 2. Create template for labeling
python tests/ground_truth_format.py --create-template logs/Labelled/ground_truth_labels.json

# 3. Manually edit ground_truth_labels.json with your observations

# 4. Validate your labels
python tests/ground_truth_format.py --validate logs/Labelled/ground_truth_labels.json

# 5. Run evaluation
python tests/evaluate_classifier.py --labeled-dir logs/Labelled

# 6. Review results in logs/Labelled/evaluation_results.json
```
