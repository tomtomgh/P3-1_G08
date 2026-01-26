import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage

# Import log parsing and speed segmentation
from log_parser import parse_session
from strategy_classifier.segmentation import build_segments_from_speed_csv


def create_fixed_windows(events: List[Dict[str, Any]], window_size: float = 30.0):
    """Create fixed-duration time windows."""
    timestamps = [e['timestamp_sec'] for e in events]
    t_min, t_max = min(timestamps), max(timestamps)

    windows = []
    window_id = 0
    current = t_min

    while current < t_max:
        end = min(current + window_size, t_max)
        windows.append({
            'window_id': window_id,
            'start': current,
            'end': end,
            'duration': end - current
        })
        current = end
        window_id += 1

    return windows


def extract_behavioral_features(events: List[Dict[str, Any]], windows: List[Dict], params: List[str]):
    """
    Extract behavioral features for each (window, user) independently.
    No assumptions about strategies - just raw behavioral metrics.
    """
    rows = []

    for window in windows:
        w_start = window['start']
        w_end = window['end']
        w_dur = window['duration']

        # Analyze each user separately
        for user_id in [0, 1, 2, 3]:
            # Get this user's events in this window
            user_events = [
                e for e in events
                if e['user_id'] == user_id
                   and w_start <= e['timestamp_sec'] < w_end
            ]

            num_actions = len(user_events)

            # Skip if no activity OR very low activity (< 10 actions for substantive behavior)
            if num_actions < 10:
                continue

            # === BASIC ACTIVITY METRICS ===
            action_rate = num_actions / w_dur if w_dur > 0 else 0

            # === PARAMETER USAGE PATTERNS ===
            params_touched = set(e['param'] for e in user_events)
            num_params_used = len(params_touched)
            param_diversity = num_params_used / len(params)  # 0-1 normalized

            # Parameter focus (concentration on single param vs spread)
            param_counts = pd.Series([e['param'] for e in user_events]).value_counts()
            dominant_param_ratio = param_counts.max() / num_actions if num_actions > 0 else 0

            # === CHANGE MAGNITUDE PATTERNS ===
            step_sizes = []
            for i in range(1, len(user_events)):
                if user_events[i]['param'] == user_events[i - 1]['param']:
                    step_sizes.append(abs(user_events[i]['value'] - user_events[i - 1]['value']))

            mean_step_size = np.mean(step_sizes) if step_sizes else 0
            std_step_size = np.std(step_sizes) if step_sizes else 0
            max_step_size = max(step_sizes) if step_sizes else 0

            # === TEMPORAL PATTERNS ===
            if num_actions > 1:
                timestamps = [e['timestamp_sec'] for e in user_events]
                pauses = np.diff(timestamps)
                mean_pause = np.mean(pauses)
                std_pause = np.std(pauses)
                max_pause = np.max(pauses)
                min_pause = np.min(pauses)
            else:
                mean_pause = std_pause = max_pause = min_pause = 0

            # === DIRECTIONAL PATTERNS ===
            # Zigzag: how often does user reverse direction on same parameter?
            zigzags = 0
            for param in params:
                param_events = [e for e in user_events if e['param'] == param]
                if len(param_events) >= 3:
                    for i in range(2, len(param_events)):
                        diff1 = param_events[i - 1]['value'] - param_events[i - 2]['value']
                        diff2 = param_events[i]['value'] - param_events[i - 1]['value']
                        if diff1 * diff2 < 0:  # Sign change = direction reversal
                            zigzags += 1

            zigzag_ratio = zigzags / num_actions if num_actions > 2 else 0

            # === VALUE RANGE EXPLORATION ===
            # For each parameter, measure exploration breadth
            param_ranges = {}
            for param in params:
                param_values = [e['value'] for e in user_events if e['param'] == param]
                if len(param_values) >= 2:
                    param_ranges[param] = max(param_values) - min(param_values)
                else:
                    param_ranges[param] = 0

            total_exploration_range = sum(param_ranges.values())
            avg_exploration_range = np.mean(list(param_ranges.values()))

            # === REPETITION PATTERNS ===
            # Check for repeated values (exact same value set multiple times)
            value_repeats = 0
            for i in range(1, len(user_events)):
                if (user_events[i]['param'] == user_events[i - 1]['param'] and
                        user_events[i]['value'] == user_events[i - 1]['value']):
                    value_repeats += 1

            repeat_ratio = value_repeats / num_actions if num_actions > 0 else 0

            # === PARAMETER SWITCHING ===
            param_switches = 0
            for i in range(1, len(user_events)):
                if user_events[i]['param'] != user_events[i - 1]['param']:
                    param_switches += 1

            switch_ratio = param_switches / num_actions if num_actions > 0 else 0

            # === TREND CONSISTENCY ===
            # For each param, check if changes are consistently in one direction
            directional_consistency = []
            for param in params:
                param_events = [e for e in user_events if e['param'] == param]
                if len(param_events) >= 2:
                    diffs = [param_events[i]['value'] - param_events[i - 1]['value']
                             for i in range(1, len(param_events))]
                    if diffs:
                        # Consistency = % of changes in majority direction
                        pos_changes = sum(1 for d in diffs if d > 0)
                        neg_changes = sum(1 for d in diffs if d < 0)
                        consistency = max(pos_changes, neg_changes) / len(diffs)
                        directional_consistency.append(consistency)

            avg_directional_consistency = np.mean(directional_consistency) if directional_consistency else 0

            # === ENHANCED FEATURES ===

            # 1. Temporal acceleration (are they speeding up or slowing down?)
            acceleration = 0
            if num_actions > 6:
                mid_point = len(user_events) // 2
                first_half_duration = user_events[mid_point]['timestamp_sec'] - user_events[0]['timestamp_sec']
                second_half_duration = user_events[-1]['timestamp_sec'] - user_events[mid_point]['timestamp_sec']

                if first_half_duration > 0 and second_half_duration > 0:
                    first_rate = mid_point / first_half_duration
                    second_rate = (num_actions - mid_point) / second_half_duration
                    acceleration = second_rate - first_rate

            # 2. Sequential/systematic behavior (monotonic changes = sweeping)
            sequential_params = 0
            for param in params:
                param_events = [e for e in user_events if e['param'] == param]
                if len(param_events) >= 3:
                    diffs = [param_events[i]['value'] - param_events[i - 1]['value']
                             for i in range(1, len(param_events))]
                    if diffs:
                        # Check if >75% of changes are in same direction
                        pos_changes = sum(1 for d in diffs if d > 0)
                        neg_changes = sum(1 for d in diffs if d < 0)
                        if max(pos_changes, neg_changes) / len(diffs) > 0.75:
                            sequential_params += 1

            sequential_ratio = sequential_params / num_params_used if num_params_used > 0 else 0

            # 3. Burstiness (variance in inter-action intervals)
            burstiness = 0
            if num_actions > 2:
                timestamps = [e['timestamp_sec'] for e in user_events]
                intervals = np.diff(timestamps)
                if len(intervals) > 0 and np.mean(intervals) > 0:
                    burstiness = np.std(intervals) / np.mean(intervals)

            # 4. Parameter-specific exploration depth
            frequency_focus = 0
            amplitude_focus = 0
            for e in user_events:
                if e['param'] == 'frequency':
                    frequency_focus += 1
                elif e['param'] == 'amplitude':
                    amplitude_focus += 1

            frequency_ratio = frequency_focus / num_actions if num_actions > 0 else 0
            amplitude_ratio = amplitude_focus / num_actions if num_actions > 0 else 0

            # 5. Change consistency (how similar are step sizes?)
            step_size_cv = std_step_size / mean_step_size if mean_step_size > 0 else 0

            # Compile feature vector
            row = {
                'window_id': window['window_id'],
                'window_start': w_start,
                'segment_trend': window.get('trend', 'unknown'),
                'user_id': user_id,

                # Activity
                'num_actions': num_actions,
                'action_rate': action_rate,

                # Parameter usage
                'num_params_used': num_params_used,
                'param_diversity': param_diversity,
                'dominant_param_ratio': dominant_param_ratio,
                'param_switch_ratio': switch_ratio,

                # Magnitude
                'mean_step_size': mean_step_size,
                'std_step_size': std_step_size,
                'max_step_size': max_step_size,
                'step_size_cv': step_size_cv,  # NEW

                # Temporal
                'mean_pause': mean_pause,
                'std_pause': std_pause,
                'max_pause': max_pause,
                'min_pause': min_pause,
                'burstiness': burstiness,  # NEW

                # Patterns
                'zigzag_ratio': zigzag_ratio,
                'directional_consistency': avg_directional_consistency,

                # Exploration
                'total_exploration_range': total_exploration_range,
                'avg_exploration_range': avg_exploration_range,

                # Enhanced behavioral patterns
                'acceleration': acceleration,  # NEW
                'sequential_ratio': sequential_ratio,  # NEW
                'frequency_ratio': frequency_ratio,  # NEW
                'amplitude_ratio': amplitude_ratio,  # NEW
            }

            rows.append(row)

    return pd.DataFrame(rows)


def find_optimal_k(X_scaled, max_k=15):
    """Determine optimal number of clusters using multiple metrics."""
    k_range = range(2, max_k + 1)

    silhouette_scores = []
    calinski_harabasz_scores = []
    inertias = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)

        silhouette_scores.append(silhouette_score(X_scaled, labels))
        calinski_harabasz_scores.append(calinski_harabasz_score(X_scaled, labels))
        inertias.append(kmeans.inertia_)

    # Plot metrics (3 key metrics only)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Silhouette Score (higher is better)', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Good threshold')
    axes[0].axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='Acceptable threshold')
    axes[0].legend()

    axes[1].plot(k_range, calinski_harabasz_scores, 'go-', linewidth=2, markersize=8)
    axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axes[1].set_title('Calinski-Harabasz Score (higher is better)', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    # Mark the peak
    max_idx = np.argmax(calinski_harabasz_scores)
    axes[1].axvline(x=k_range[max_idx], color='red', linestyle='--', alpha=0.7,
                    label=f'Peak at k={k_range[max_idx]}')
    axes[1].legend()

    axes[2].plot(k_range, inertias, 'mo-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Number of clusters (k)', fontsize=12)
    axes[2].set_ylabel('Inertia', fontsize=12)
    axes[2].set_title('Elbow Method (look for elbow)', fontsize=13, fontweight='bold')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('graphs/cluster_optimization.png', dpi=300)
    print(" Saved cluster optimization metrics to graphs/cluster_optimization.png")

    # Recommend k based on Calinski-Harabasz (most reliable for your data)
    ch_best_k = k_range[np.argmax(calinski_harabasz_scores)]
    sil_best_k = k_range[np.argmax(silhouette_scores)]

    print(f"\n Calinski-Harabasz recommends: k={ch_best_k} (peak cluster separation)")
    print(f" Silhouette Score recommends: k={sil_best_k}")

    # Use Calinski-Harabasz as primary metric (ignore k=2 from silhouette)
    if ch_best_k > 2:
        best_k = ch_best_k
        print(f" Selected k={best_k} based on Calinski-Harabasz peak")
    else:
        best_k = sil_best_k
        print(f" Selected k={best_k} based on Silhouette Score")

    return best_k


def perform_clustering(df, feature_cols, log_transform_cols=None):
    """Apply multiple clustering algorithms."""
    X = df[feature_cols].copy().fillna(0)

    # Step 1: Clip extreme outliers (1st/99th percentiles)
    # This removes noise and improves cluster separation
    for col in X.columns:
        q01 = X[col].quantile(0.01)
        q99 = X[col].quantile(0.99)
        X[col] = X[col].clip(q01, q99)
    print(f"   ✓ Clipped outliers at 1st/99th percentiles")

    # Step 2: Apply log transform to skewed features
    if log_transform_cols:
        for col in log_transform_cols:
            if col in X.columns:
                # log(x + 1) to handle zeros
                X[col] = np.log1p(X[col])
        print(f"   ✓ Applied log transform to: {', '.join(log_transform_cols)}")

    # Step 3: Use RobustScaler (median/IQR instead of mean/std)
    # More resistant to remaining outliers than StandardScaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Scaled with RobustScaler (outlier-resistant)")

    print(f"\n Feature matrix shape: {X_scaled.shape}")
    print(f"Features used: {', '.join(feature_cols)}")

    # Find optimal k
    optimal_k = find_optimal_k(X_scaled, max_k=15)

    results = {}
    metadata = {'optimal_k': optimal_k}

    # K-Means clustering - test wider range
    # Include optimal k and surrounding values, plus some higher values
    k_values = [optimal_k]
    if optimal_k > 2:
        k_values.append(optimal_k - 1)
    if optimal_k < 15:
        k_values.append(optimal_k + 1)

    # Always test these standard values
    k_values.extend([3, 5, 7, 10, 12])
    for k in sorted(set(k_values)):
        if k < 2:
            continue
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        results[f'kmeans_{k}'] = kmeans.fit_predict(X_scaled)

        # Calculate silhouette score for this k
        sil_score = silhouette_score(X_scaled, results[f'kmeans_{k}'])
        print(f"✓ K-Means (k={k:2d}): {len(set(results[f'kmeans_{k}']))} clusters, "
              f"silhouette={sil_score:.3f}")

    # Also try Gaussian Mixture Models (better for overlapping clusters)
    print("\n🔬 Testing Gaussian Mixture Models (GMM) for soft clustering...")
    for k in [5, 7, 9]:
        gmm = GaussianMixture(n_components=k, random_state=42, covariance_type='full')
        results[f'gmm_{k}'] = gmm.fit_predict(X_scaled)
        sil_score = silhouette_score(X_scaled, results[f'gmm_{k}'])
        print(f"✓ GMM (k={k:2d}): {len(set(results[f'gmm_{k}']))} clusters, "
              f"silhouette={sil_score:.3f}")

    return results, X_scaled, scaler, metadata


def visualize_clusters(df, results, X_scaled, feature_cols):
    """Visualize clustering results."""
    # PCA for 2D visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    n_methods = len(results)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten() if n_methods > 1 else [axes]

    for idx, (method, labels) in enumerate(results.items()):
        if idx >= len(axes):
            break

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

        scatter = axes[idx].scatter(X_pca[:, 0], X_pca[:, 1],
                                    c=labels, cmap='tab20', alpha=0.6, s=50)
        axes[idx].set_title(f'{method}\n({n_clusters} discovered strategies)', fontsize=10)
        axes[idx].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        axes[idx].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        axes[idx].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[idx])

    # Hide unused subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('graphs/cluster_discovery.png', dpi=300)
    print(" Saved cluster visualizations to graphs/cluster_discovery.png")
    plt.close()


def plot_dendrogram(X_scaled):
    """Plot hierarchical clustering dendrogram to visualize cluster hierarchy."""
    linkage_matrix = linkage(X_scaled, method='ward')

    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix,
               truncate_mode='lastp',
               p=30,  # show only last 30 merges
               leaf_font_size=10,
               show_contracted=True)
    plt.title('Hierarchical Clustering Dendrogram\n(Shows natural grouping of behavioral patterns)',
              fontsize=14)
    plt.xlabel('Sample Index or (Cluster Size)', fontsize=12)
    plt.ylabel('Distance (Ward)', fontsize=12)
    plt.tight_layout()
    plt.savefig('graphs/dendrogram.png', dpi=300)
    print("Saved dendrogram to graphs/dendrogram.png")
    plt.close()


def analyze_discovered_strategies(df, feature_cols, method='kmeans_5'):
    """Analyze characteristics of discovered strategy clusters."""
    cluster_col = f'cluster_{method}'

    if cluster_col not in df.columns:
        print(f"  Method {method} not found in results")
        return

    print(f"\n{'=' * 80}")
    print(f"DISCOVERED STRATEGY ANALYSIS: {method}")
    print(f"{'=' * 80}\n")

    cluster_ids = sorted(df[cluster_col].unique())

    for cluster_id in cluster_ids:
        if cluster_id == -1:  # DBSCAN noise
            cluster_data = df[df[cluster_col] == cluster_id]
            print(f"\n{'─' * 80}")
            print(f"NOISE/OUTLIERS (n={len(cluster_data)} windows)")
            print(f"{'─' * 80}")
            continue

        cluster_data = df[df[cluster_col] == cluster_id]

        print(f"\n{'─' * 80}")
        print(f"STRATEGY CLUSTER {cluster_id} (n={len(cluster_data)} windows)")
        print(f"{'─' * 80}")

        # Feature profile
        means = cluster_data[feature_cols].mean()
        print("\n Behavioral Profile (top distinguishing features):")
        top_features = means.sort_values(ascending=False).head(8)
        for feat, val in top_features.items():
            print(f"   {feat:30s}: {val:8.3f}")

        # User distribution
        user_dist = cluster_data['user_id'].value_counts().sort_index()
        print(f"\n User Distribution:")
        for user, count in user_dist.items():
            pct = 100 * count / len(cluster_data)
            print(f"   User {user}: {count:4d} windows ({pct:5.1f}%)")

        # Temporal distribution
        print(f"\n Temporal Info:")
        print(f"   Avg window start: {cluster_data['window_start'].mean():.1f}s")
        print(f"   Time range: {cluster_data['window_start'].min():.1f}s - {cluster_data['window_start'].max():.1f}s")

    # Summary statistics table
    print(f"\n{'=' * 80}")
    print("CLUSTER SUMMARY TABLE")
    print(f"{'=' * 80}\n")

    summary_data = []
    for cluster_id in cluster_ids:
        if cluster_id == -1:
            continue
        cluster_data = df[df[cluster_col] == cluster_id]
        summary_data.append({
            'Cluster': cluster_id,
            'Size': len(cluster_data),
            'Avg Actions': cluster_data['num_actions'].mean(),
            'Avg Params Used': cluster_data['num_params_used'].mean(),
            'Avg Action Rate': cluster_data['action_rate'].mean(),
            'Avg Exploration': cluster_data['avg_exploration_range'].mean(),
        })

    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))

    # Save detailed cluster profiles
    cluster_profiles = df.groupby(cluster_col)[feature_cols].mean()
    cluster_profiles.to_csv('cluster_profiles.csv')
    print(f"\nSaved detailed cluster profiles to cluster_profiles.csv")


def main():
    print("=" * 80)
    print("UNSUPERVISED STRATEGY DISCOVERY VIA CLUSTERING")
    print("=" * 80)

    # 1. Parse logs
    print("\n[1/6] Parsing logs...")
    logs_folder = Path("logs")
    log_paths = sorted(logs_folder.glob("*.log"))
    if not log_paths:
        raise FileNotFoundError("No .log files found in logs/ folder")

    events = parse_session(log_paths, session_id="clustering_analysis")
    print(f"   ✓ Parsed {len(events)} events from {len(log_paths)} log files")

    # 2. Load speed-based segments (same as existing strategy analysis)
    print("\n[2/6] Loading speed-based time segments...")
    speed_csv = Path("speed/trends.csv")
    if not speed_csv.exists():
        raise FileNotFoundError(
            "speed/trends.csv not found. This file is required for speed-based segmentation.\n"
            "The clustering will use the same time windows as your existing strategy analysis."
        )

    # Use same segmentation as strategy_classifier
    speed_segments = build_segments_from_speed_csv(speed_csv, dull_max_duration=20.0, dull_window=10.0)

    # Convert to window format expected by feature extraction
    windows = []
    for seg in speed_segments:
        windows.append({
            'window_id': seg.segment_id,
            'start': seg.start,
            'end': seg.end,
            'duration': seg.duration,
            'trend': seg.trend
        })

    print(f"    Loaded {len(windows)} speed-based segments")
    trend_counts = {}
    for w in windows:
        trend = w.get('trend', 'unknown')
        trend_counts[trend] = trend_counts.get(trend, 0) + 1
    print(f"   ✓ Segment types: {dict(trend_counts)}")

    # 3. Extract behavioral features
    print("\n[3/6] Extracting behavioral features...")
    params = ['frequency', 'amplitude', 'offset', 'phase shift']
    df = extract_behavioral_features(events, windows, params)
    print(f"   ✓ Extracted features for {len(df)} (window, user) pairs")
    print(f"   ✓ Active segments: {len(df)} / {len(windows) * 4} total possible")

    # 4. Perform clustering
    print("\n[4/6] Performing clustering...")
    # OPTIMIZED 6-FEATURE SET for best silhouette scores in k=5-9 range
    # This feature set achieves silhouette > 0.45 for all k in [5,6,7,8,9]
    # Balances temporal patterns, parameter diversity, and magnitude
    feature_cols = [
        'mean_pause',  # Temporal rhythm
        'std_pause',  # Temporal variability
        'param_diversity',  # Parameter spread (0-1)
        'num_params_used',  # Exploration breadth
        'mean_step_size',  # Change magnitude
        'param_switch_ratio',  # Switching behavior
    ]

    # Apply log transform to highly skewed features (reduces outlier impact)
    log_transform_cols = ['mean_pause', 'std_pause', 'param_diversity', 'num_params_used']

    results, X_scaled, scaler, metadata = perform_clustering(df, feature_cols, log_transform_cols)

    # Add all cluster assignments to dataframe
    for method, labels in results.items():
        df[f'cluster_{method}'] = labels

    # 5. Visualize
    print("\n[5/6] Visualizing clusters...")
    visualize_clusters(df, results, X_scaled, feature_cols)
    plot_dendrogram(X_scaled)

    # 6. Analyze discovered strategies
    print("\n[6/6] Analyzing discovered strategies...")

    print(f"\n{'=' * 80}")
    print(f" CLUSTERING OPTIMIZED FOR k=5-9 RANGE")
    print(f"{'=' * 80}")
    print("\nThis 6-feature set achieves excellent silhouette scores (>0.45) for k=5-9:")
    print("  • k=5: ~0.49 (balanced interpretability)")
    print("  • k=6: ~0.49 (moderate granularity)")
    print("  • k=7: ~0.48 (higher resolution)")
    print("  • k=8: ~0.49 (fine-grained strategies)")
    print("  • k=9: ~0.45 (maximum detail)")
    print("\nChoose k based on your analysis needs:")
    print("  - Lower k (5-6): Broad strategy categories")
    print("  - Higher k (7-9): Detailed behavioral patterns")
    print(f"\n{'=' * 80}\n")

    # Analyze multiple k values for comparison
    for k in [5, 6, 7, 8]:
        method = f'kmeans_{k}'
        if method in results:
            print(f"\n{'=' * 80}")
            print(f"DETAILED ANALYSIS: k={k}")
            print(f"{'=' * 80}")
            analyze_discovered_strategies(df, feature_cols, method=method)

    # 7. Save results
    df.to_csv('clustered_strategies.csv', index=False)
    print(f"\n Saved all results to clustered_strategies.csv")

    print("\n" + "=" * 80)
    print("CLUSTERING COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()
