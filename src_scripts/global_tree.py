#!/usr/bin/env python3
# --------------------------------------------------------------
# train_global_tree.py
# Train & plot a single global decision tree for strategies
# --------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree

from strategy_classifier.constants import ALL_STRATEGIES
from strategy_classifier.features import DEFAULT_FEATURE_COLS


def load_data(csv_path: str | Path = "segment_strategy_predictions.csv") -> pd.DataFrame:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `python run_all.py` first "
            "to generate segment_strategy_predictions.csv."
        )
    df = pd.read_csv(csv_path)
    return df


def compute_global_label(df: pd.DataFrame) -> pd.DataFrame:
    labels = []
    max_probs = []

    for _, row in df.iterrows():
        # treat segments with no user actions as unlabeled
        if int(row.get("has_events", 1)) == 0 or int(row.get("num_actions", 0)) == 0:
            labels.append(None)
            max_probs.append(0.0)
            continue

        best_name = None
        best_prob = -1.0
        for strat in ALL_STRATEGIES:
            col = f"{strat}_prob"
            if col not in row:
                continue
            p = row[col]
            if p > best_prob:
                best_prob = p
                best_name = strat

        labels.append(best_name)
        max_probs.append(best_prob)

    df2 = df.copy()
    df2["global_strategy_label"] = labels
    df2["global_strategy_max_prob"] = max_probs
    return df2


def train_global_tree(
    df: pd.DataFrame,
    feature_cols=None,
    max_depth: int = 5,
) -> DecisionTreeClassifier:
    if feature_cols is None:
        feature_cols = DEFAULT_FEATURE_COLS

    if "global_strategy_label" not in df.columns:
        df = compute_global_label(df)

    # drop rows without label and rows with no events to avoid leakage
    df_train = df.dropna(subset=["global_strategy_label"]).copy()
    if "has_events" in df_train.columns:
        df_train = df_train[df_train["has_events"].astype(int) > 0].copy()

    if df_train.empty:
        raise RuntimeError("No labeled rows available to train the global tree.")

    X = df_train[feature_cols]
    y = df_train["global_strategy_label"]

    clf = DecisionTreeClassifier(max_depth=max_depth, criterion="entropy", random_state=42)
    clf.fit(X, y)

    # Simple training accuracy (just to see it works)
    acc = clf.score(X, y)
    print(f"[INFO] Global decision tree training accuracy (on full data): {acc:.3f}")
    print(f"[INFO] Number of training segments: {len(df_train)}")
    print(f"[INFO] Number of classes: {y.nunique()}")

    return clf, feature_cols, sorted(y.unique())


def plot_global_tree(
    clf: DecisionTreeClassifier,
    feature_names,
    class_names,
    out_path: str | Path = "global_strategy_tree.png",
):
    """
    Plot and save the global decision tree.
    """
    out_path = Path(out_path)

    plt.figure(figsize=(24, 16))  # big figure for a big tree
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.title("Global Strategy Decision Tree", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"[INFO] Saved global decision tree plot to: {out_path}")
    # If you want to see it interactively:
    # plt.show()
    plt.close()


def main():
    print("[INFO] Loading segment-level predictions...")
    df = load_data("segment_strategy_predictions.csv")

    print("[INFO] Computing global labels (argmax over rule-based strategies)...")
    df = compute_global_label(df)

    print("[INFO] Training global decision tree...")
    clf, feature_cols, classes = train_global_tree(
        df,
        feature_cols=DEFAULT_FEATURE_COLS,
        max_depth=5,   # adjust if you want a deeper or simpler tree
    )

    print("[INFO] Plotting global decision tree...")
    plot_global_tree(
        clf,
        feature_names=feature_cols,
        class_names=classes,
        out_path="global_strategy_tree.png",
    )

    # Optional: save back the labels if you want them in a CSV
    df.to_csv("segment_strategy_with_global_label.csv", index=False)
    print("✔ Saved: segment_strategy_with_global_label.csv")
    print("✔ Saved: global_strategy_tree.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
