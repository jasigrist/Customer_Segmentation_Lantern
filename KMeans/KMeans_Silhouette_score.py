# =============================================================================
# K-MEANS VALIDATION VIA SILHOUETTE SCORE ANALYSIS
# =============================================================================
# Purpose: Evaluate K-Means clustering quality across k=2-10 clusters
# Input: Normalized load profiles for one or two building types
# Output: Silhouette scores and comparison plot
# Metrics: Silhouette score (range: -1 to 1, higher is better)
# =============================================================================

import polars as pl
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans


# %%

def validation_kMeans(main_path, train_data1, train_data2, dataset, building_type):
    """
    Validate K-Means clustering performance across different numbers of clusters.
    
    Computes Silhouette scores for k=2-10 and generates comparison plots for
    one or two datasets (e.g., flats vs. houses).
    
    Parameters
    ----------
    main_path : str
        Root directory path for saving outputs.
    train_data1 : array-like, shape (n_samples, n_features)
        Primary dataset (normalized load profiles).
    train_data2 : array-like or None
        Secondary dataset for comparison. If None, only train_data1 is analyzed.
    dataset : str
        Dataset identifier (used in file naming).
    building_type : str
        Building type label (e.g., 'flats', 'houses') for plot titles and filenames.
    
    Returns
    -------
    df_scores1 : polars.DataFrame
        Silhouette scores for primary dataset with columns: 'no_of_clusters', 'silhouette_score'.
    df_scores2 : polars.DataFrame or None
        Silhouette scores for secondary dataset, or None if not provided.
    """
    seed_random = 42
    k_values_to_try = np.arange(2, 11)

    def compute_scores(train_data):
        """Compute Silhouette scores for varying cluster counts."""
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1]))
        df_scores = []
        for n_clusters in k_values_to_try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed_random)
            labels_clusters = kmeans.fit_predict(train_data)
            
            sh = silhouette_score(train_data, labels_clusters)
            df_scores.append({"no_of_clusters": n_clusters, "silhouette_score": sh})
        df_scores = pl.DataFrame(df_scores)
        return df_scores
    
    # Compute scores for primary dataset
    df_scores1 = compute_scores(train_data1)

    # Check if secondary dataset is provided and distinct from primary
    two_datasets = train_data2 is not None and train_data2 is not train_data1
    if two_datasets:
        df_scores2 = compute_scores(train_data2)
    else:
        df_scores2 = None

    # Create comparison plot
    plt.figure(figsize=(8, 5))

    if two_datasets:
        # Plot both datasets for comparison
        plt.plot(df_scores1['no_of_clusters'], df_scores1['silhouette_score'], marker='o', color='#1f77b4', label='Flats')
        plt.plot(df_scores2['no_of_clusters'], df_scores2['silhouette_score'], marker='s', color='#ff7f0e', label='Houses')
    else:
        # Plot single dataset
        plt.plot(df_scores1['no_of_clusters'], df_scores1['silhouette_score'], marker='o', color='#1f77b4', label=building_type.capitalize())

    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.ylim(0.0, 1)
    plt.legend()
    plt.grid(False)
    plt.savefig(f"{main_path}/Results/{dataset}/Plots/Cluster/KMeans_Silhouette_{building_type}.png", bbox_inches='tight')
    plt.show()
    
    return df_scores1, df_scores2

# %%



