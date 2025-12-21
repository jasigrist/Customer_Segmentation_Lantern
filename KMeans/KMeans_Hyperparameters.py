# =============================================================================
# K-MEANS HYPERPARAMETER OPTIMIZATION FOR LOAD PROFILE CLUSTERING
# =============================================================================
# Purpose: Grid search over k-means hyperparameters using silhouette scores
# Input: Normalized daily load profiles (n_households × 96 timepoints)
# Output: Optimal {n_clusters, n_init, max_iter} maximizing Silhouette score
# Metrics: Silhouette (cluster cohesion/separation), DB (cluster similarity)
# Search space: k=3-10, n_init=1-5, max_iter=100-500
# =============================================================================

import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd



def optimize_kmeans(train_data):
    """
    Optimize K-Means hyperparameters via exhaustive grid search.
    
    Parameters
    ----------
    train_data : array-like, shape (n_samples, n_features)
        Normalized load profiles (hourly consumption patterns).
    
    Returns
    -------
    best_result : pandas.Series
        Dictionary-like object containing optimal hyperparameters and their 
        corresponding Silhouette and Davies-Bouldin scores.
    """

    X = train_data.squeeze()  # Shape: (n_samples, n_features)

    # Define hyperparameter grid to search
    param_grid = {
        'n_clusters': np.arange(3, 11, 1), 
        'init': ['k-means++'],
        'n_init': np.arange(1, 6, 1), 
        'max_iter': np.arange(100, 600, 100), 
    }

    results = []

    # Exhaustive grid search
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

        # Fit model and obtain cluster assignments
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

    # Rank configurations by Silhouette score
    results_df = pd.DataFrame(results).sort_values(by='silhouette_score', ascending=False)

    # Uncomment to inspect full ranking:    
    #print(results_df)

    # Output best combination (highest silhouette_score)
    best_result = results_df.iloc[0]
    print("\nBest combination of hyperparameters:")
    print(best_result)
    return best_result



