import streamlit as st
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

from decisionTrees import (
    parse_session,
    build_player_features,
    train_strategy_tree,
    add_strategy_predictions,
    ensure_label_diversity,
)

# --------- Streamlit config ---------
st.set_page_config(
    page_title="4-Legged Robot – Strategy & Coordination Analysis",
    layout="wide",
)


# --------- Helpers ---------
def load_session():
    """
    Loads User0.log .. User3.log from the same folder as this app.
    """
    base = Path(__file__).parent
    log_paths = [base / f"User{i}.log" for i in range(4)]

    missing = [p for p in log_paths if not p.exists()]
    if missing:
        st.error(
            "Could not find the following log files in the app directory:\n"
            + "\n".join(str(m) for m in missing)
        )
        return None

    events = parse_session(log_paths, session_id="session_1")
    if not events:
        st.error("No events parsed from logs. Check file content/format.")
        return None

    return events


def make_tree_figure(clf, feature_names, class_names):
    fig, ax = plt.subplots(figsize=(16, 9))
    plot_tree(
        clf,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
    )
    plt.tight_layout()
    return fig


# --------- Main UI ---------
st.title("4-Legged Robot – Strategy & Coordination Analysis")

st.markdown(
    """
This dashboard:

- Parses `User0.log` – `User3.log`
- Uses **rule-based logic** to assign leadership (who owns the shared frequency knob)
- Extracts **strategy features** (HOTAT / VOTAT / Mixed)
- Analyzes **coordination** (straight vs diagonal vs pairwise)
- Trains a **decision tree** to classify strategies
- Shows a **confidence value** and a **strategy analysis list** per player
"""
)

st.sidebar.header("Analysis Controls")
if st.sidebar.button("Reload & Recompute"):
    st.experimental_rerun()

events = load_session()
if events is None:
    st.stop()

# Build per-player features
df_players = build_player_features(events)
if df_players.empty:
    st.error("No per-player data extracted.")
    st.stop()

st.subheader("Raw Per-Player Feature Table")
st.dataframe(df_players.set_index(["session_id", "user_id"]))

# Strategy tree features (must match decisionTrees.py)
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

# Ensure we have enough label variety
df_players = ensure_label_diversity(df_players, "strategy_full_label")

# Train strategy decision tree
clf, test_results = train_strategy_tree(
    df_players,
    feature_cols=strategy_feature_cols,
    label_col="strategy_full_label",
    max_depth=4,
)

# Add predictions & confidence
df_players_pred = add_strategy_predictions(
    df_players,
    clf,
    feature_cols=strategy_feature_cols,
    label_col="strategy_full_label",
)

st.subheader("Per-Player Strategy Predictions")
st.dataframe(
    df_players_pred[
        [
            "session_id",
            "user_id",
            "leader_role",
            "strategy_rule_label",
            "coord_style",
            "strategy_full_label",
            "strategy_tree_pred",
            "strategy_tree_confidence",
            "coord_partners",
            "strategy_analysis_string",
        ]
    ].set_index(["session_id", "user_id"])
)

# --------- Per-player cards ---------
st.subheader("Personal Behavior Analysis")

for _, row in df_players_pred.iterrows():
    with st.expander(f"Player {row['user_id']} – {row['leader_role'].capitalize()}"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Leadership (rule-based)**")
            st.write(f"Role: `{row['leader_role']}`")
            st.write(f"Frequency changes: `{row['freq_changes']}`")
            st.write(f"Frequency share: `{row['freq_share']:.2f}`")
            st.write(f"Initiations: `{row['initiations']}`")
            st.write(f"Reactions: `{row['reactions']}`")
            st.write(f"Lead fraction: `{row['lead_fraction']:.2f}`")

        with col2:
            st.markdown("**Parameter Strategy (HOTAT / VOTAT / Mixed)**")
            st.write(f"Total param changes: `{row['total_param_changes']}`")
            st.write(f"# parameters used: `{row['num_params_used']}`")
            st.write(f"Dominant param share: `{row['dominant_param_share']:.2f}`")
            st.write(
                "Single-param clusters (VOTAT-like metric): "
                f"`{row['single_param_cluster_ratio']:.2f}`"
            )
            st.write(f"Rule-based strategy: `{row['strategy_rule_label']}`")

        with col3:
            st.markdown("**Coordination & Tree Output**")
            st.write(f"Session coordination style: `{row['coord_style']}`")
            st.write(
                f"Straight coord score: `{row['straight_coord_score']:.2f}`"
            )
            st.write(
                f"Diagonal coord score: `{row['diagonal_coord_score']:.2f}`"
            )
            st.write(f"Pairwise partners: `{row['coord_partners']}`")
            st.write("---")
            st.write(f"Tree strategy: `{row['strategy_tree_pred']}`")
            st.write(
                f"Confidence: `{row['strategy_tree_confidence']:.2f}`"
            )

        st.markdown("**Strategy analysis**")
        analysis = row["strategy_analysis"]
        if isinstance(analysis, list):
            for item in analysis:
                st.write("•", item)
        else:
            st.write(row["strategy_analysis_string"])

# --------- Strategy tree visualization ---------
st.subheader("Decision Tree Structure")

if clf is None:
    st.info(
        "Not enough class variety to train a strategy decision tree yet.\n"
        "You’ll see a tree once there are at least 2 different strategy_full_label classes."
    )
else:
    class_names = list(sorted(df_players_pred["strategy_full_label"].unique()))
    fig_tree = make_tree_figure(
        clf,
        feature_names=strategy_feature_cols,
        class_names=class_names,
    )
    st.pyplot(fig_tree)

# --------- Download features ---------
st.subheader("Download Features as CSV")
csv = df_players_pred.to_csv(index=False)
st.download_button(
    label="Download per-player features (CSV)",
    data=csv,
    file_name="robot_strategy_features.csv",
    mime="text/csv",
)
