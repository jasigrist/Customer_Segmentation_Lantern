# %%

import numpy as np
import itertools

import logging

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
import pandas as pd


logging.getLogger('matplotlib.font_manager').disabled = True

# %%
def optimize_kmeans(train_data):


    X = train_data.squeeze()  # adjust shape if needed

    param_grid = {
        'n_clusters': np.arange(2, 11, 1), 
        #'init': ['k-means++', 'random'],
        'init': ['k-means++'],
        'n_init': np.arange(1, 6, 1), 
        'max_iter': np.arange(100, 600, 100), 
    }

    results = []
    import itertools

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

    results_df = pd.DataFrame(results).sort_values(by='db_score', ascending=True)
    #print(results_df)

    # Output best combination (highest silhouette score)
    best_result = results_df.iloc[0]
    print("\nBest combination of hyperparameters:")
    print(best_result)
    return best_result

# %%



