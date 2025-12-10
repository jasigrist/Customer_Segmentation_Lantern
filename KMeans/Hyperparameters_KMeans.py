# =============================================================================
# K-MEANS HYPERPARAMETER OPTIMIZATION FOR LOAD PROFILE CLUSTERING
# =============================================================================
# Purpose: Grid search over k-means hyperparameters using silhouette + Davies-Bouldin scores
# Input: Normalized daily load profiles (n_households × 96 timepoints)
# Output: Optimal {n_clusters, n_init, max_iter} minimizing Davies-Bouldin score
# Metrics: Silhouette (cluster cohesion/separation), DB (cluster similarity)
# Search space: k=3-10, n_init=1-5, max_iter=100-500 → 40 combinations
# =============================================================================

import numpy as np
import itertools
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd

logging.getLogger('matplotlib.font_manager').disabled = True

# =============================================================================
# HYPERPARAMETER GRID SEARCH: Exhaustive evaluation of k-means configurations
# =============================================================================
# Tests all combinations of:
# - n_clusters: 3-10 (avoids k=1,2; typical for customer segmentation)
# - init: 'k-means++' (commented: 'random' for comparison)
# - n_init: 1-5 (multiple random initializations per config)
# - max_iter: 100,200,300,400,500 (convergence control)
# Skips configs producing single cluster (invalid silhouette/DB scores)
# Ranks by Davies-Bouldin score (lower=better cluster separation)

def optimize_kmeans(train_data):
    """
    Grid search k-means hyperparameters on normalized load profiles.
    
    Parameters
    ----------
    train_data : np.ndarray
        Shape (n_households, 96) or (n_households, 96, 1); normalized daily profiles
        
    Returns
    -------
    dict
        Best hyperparameters: {'n_clusters', 'init', 'n_init', 'max_iter', 
                              'silhouette_score', 'db_score'}
    """

    X = train_data.squeeze()  # adjust shape if needed

    # Define hyperparameter grid (40 combinations total)
    param_grid = {
        'n_clusters': np.arange(3, 11, 1), 
        'init': ['k-means++'],
        'n_init': np.arange(1, 6, 1), 
        'max_iter': np.arange(100, 600, 100), 
    }

    results = []

    # Exhaustive grid search: 8×1×5×5 = 40 configurations
    for n_clusters, init, n_init, max_iter in itertools.product(
        param_grid['n_clusters'],
        param_grid['init'],
        param_grid['n_init'],
        param_grid['max_iter']
    ):
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            random_state=42
        )

        # Fit and predict cluster labels
        labels = kmeans.fit_predict(X)

        if len(set(labels)) > 1:
            sil_score = silhouette_score(X, labels)
            db_score = davies_bouldin_score(X, labels)
            results.append({
                'n_clusters': n_clusters,
                'init': init,
                'n_init': n_init,
                'max_iter': max_iter,
                'silhouette_score': sil_score,
                'db_score': db_score
            })

    # Rank configurations by Davies-Bouldin (ascending=better)
    results_df = pd.DataFrame(results).sort_values(by='db_score', ascending=True)

    # Uncomment to inspect full ranking:    
    #print(results_df)

    # Output best combination (lowest db_score)
    best_result = results_df.iloc[0]
    print("\nBest combination of hyperparameters:")
    print(best_result)
    return best_result



