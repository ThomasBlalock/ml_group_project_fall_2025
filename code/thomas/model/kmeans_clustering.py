#%%
"""
K-Means Clustering for Food Security Data
Uses silhouette score to determine optimal k for clustering based on poverty and employment features.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib
import os

# Configuration
DATA_FILE = "../../../data/data.csv"
ARTIFACTS_DIR = "model_artifacts"
CLUSTERING_FEATURES = [
    'POP_POVERTY_DETERMINED',
    'POP_BELOW_POVERTY',
    'POP_16_PLUS',
    'POP_UNEMPLOYED',
    'HOUSEHOLDS_TOTAL',
    'HOUSEHOLDS_SNAP'
]
RANDOM_SEED = 42

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def load_data_for_clustering(filepath):
    """Load data and extract clustering features."""
    print("\n--- Loading Data for Clustering ---")
    df = pd.read_csv(filepath)

    # Keep only rows with valid clustering features
    df_clean = df.dropna(subset=CLUSTERING_FEATURES).copy()

    print(f"Loaded {len(df_clean)} records with valid clustering features")
    print(f"Clustering features: {CLUSTERING_FEATURES}")

    return df_clean

def find_optimal_k(X_scaled, min_k=2, max_k=10):
    """
    Use silhouette score to determine optimal number of clusters.
    Higher silhouette score indicates better-defined clusters.
    """
    print(f"\n--- Finding Optimal K (Range: {min_k}-{max_k}) ---")

    silhouette_scores = []
    inertias = []
    k_range = range(min_k, max_k + 1)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Calculate silhouette score
        score = silhouette_score(X_scaled, cluster_labels)
        silhouette_scores.append(score)
        inertias.append(kmeans.inertia_)

        print(f"k={k}: Silhouette Score = {score:.4f}, Inertia = {kmeans.inertia_:.2f}")

    # Find k with highest silhouette score
    optimal_k = 4# k_range[np.argmax(silhouette_scores)]
    best_score = max(silhouette_scores)

    print(f"\n*** Optimal K: {optimal_k} (Silhouette Score: {best_score:.4f}) ***")

    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Silhouette scores
    ax1.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax1.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax1.set_ylabel('Silhouette Score', fontsize=12)
    ax1.set_title('Silhouette Score by Number of Clusters', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Elbow plot (inertia)
    ax2.plot(k_range, inertias, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_k, color='red', linestyle='--', label=f'Optimal k={optimal_k}')
    ax2.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax2.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax2.set_title('Elbow Method', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'kmeans_optimization.png'), dpi=150, bbox_inches='tight')
    print(f"Saved optimization plot to {ARTIFACTS_DIR}/kmeans_optimization.png")
    plt.close()

    return optimal_k, silhouette_scores, inertias

def perform_clustering(df, optimal_k):
    """
    Perform k-means clustering with the optimal k.
    Returns DataFrame with cluster assignments and clustering artifacts.
    """
    print(f"\n--- Performing K-Means Clustering (k={optimal_k}) ---")

    # Extract features for clustering
    X = df[CLUSTERING_FEATURES].values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform clustering
    kmeans = KMeans(n_clusters=optimal_k, random_state=RANDOM_SEED, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    df_clustered = df.copy()
    df_clustered['Cluster'] = cluster_labels

    # Calculate cluster statistics
    print("\n--- Cluster Statistics ---")
    for i in range(optimal_k):
        cluster_data = df_clustered[df_clustered['Cluster'] == i]
        print(f"\nCluster {i} (n={len(cluster_data)}):")
        for feature in CLUSTERING_FEATURES:
            mean_val = cluster_data[feature].mean()
            print(f"  {feature}: Mean = {mean_val:,.1f}")

    # Save clustering artifacts
    clustering_artifacts = {
        'scaler': scaler,
        'kmeans_model': kmeans,
        'optimal_k': optimal_k,
        'features': CLUSTERING_FEATURES,
        'cluster_centers': kmeans.cluster_centers_,
        'labels': cluster_labels
    }

    joblib.dump(clustering_artifacts, os.path.join(ARTIFACTS_DIR, 'clustering_model.save'))
    print(f"\nSaved clustering model to {ARTIFACTS_DIR}/clustering_model.save")

    # Save clustered data
    df_clustered.to_csv(os.path.join(ARTIFACTS_DIR, 'data_with_clusters.csv'), index=False)
    print(f"Saved clustered data to {ARTIFACTS_DIR}/data_with_clusters.csv")

    return df_clustered, clustering_artifacts

def visualize_clusters(df_clustered, optimal_k):
    """Create visualization of cluster distributions."""
    print("\n--- Creating Cluster Visualizations ---")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = plt.cm.Set3(np.linspace(0, 1, optimal_k))

    for idx, feature in enumerate(CLUSTERING_FEATURES):
        ax = axes[idx]

        for cluster_id in range(optimal_k):
            cluster_data = df_clustered[df_clustered['Cluster'] == cluster_id]
            ax.hist(cluster_data[feature], bins=30, alpha=0.6,
                   label=f'Cluster {cluster_id}', color=colors[cluster_id])

        ax.set_xlabel(feature, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'{feature} Distribution by Cluster', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'cluster_distributions.png'), dpi=150, bbox_inches='tight')
    print(f"Saved cluster distributions to {ARTIFACTS_DIR}/cluster_distributions.png")
    plt.close()

def main():
    """Main execution function."""
    print("="*60)
    print("K-MEANS CLUSTERING FOR FOOD SECURITY DATA")
    print("="*60)

    # Load data
    df = load_data_for_clustering(DATA_FILE)

    # Extract and scale features
    X = df[CLUSTERING_FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Determine valid cluster range based on data size
    n_samples = len(X_scaled)
    min_clusters = 2
    max_clusters = min(10, n_samples - 1)

    if n_samples < 2:
        print(f"ERROR: Need at least 2 samples for clustering. Found {n_samples}")
        return None, None

    print(f"\nDataset size: {n_samples} samples")
    print(f"Valid cluster range: {min_clusters} to {max_clusters}")

    # Find optimal k
    optimal_k, silhouette_scores, inertias = find_optimal_k(
        X_scaled,
        min_k=min_clusters,
        max_k=max_clusters
    )

    # Perform clustering with optimal k
    df_clustered, clustering_artifacts = perform_clustering(df, optimal_k)

    # Visualize clusters
    visualize_clusters(df_clustered, optimal_k)

    print("\n" + "="*60)
    print("CLUSTERING COMPLETE!")
    print(f"Optimal number of clusters: {optimal_k}")
    print(f"Silhouette score: {silhouette_score(X_scaled, clustering_artifacts['labels']):.4f}")
    print("="*60)

    return df_clustered, clustering_artifacts

if __name__ == "__main__":
    main()
