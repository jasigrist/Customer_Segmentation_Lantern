# %% [markdown]

# %%
import polars as pl
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans


# %%
def validation_kMeans(train_data1, train_data2):
    seed_random = 42
    k_values_to_try = np.arange(2, 11)
    
    def compute_scores(train_data):
        train_data = train_data.reshape((train_data.shape[0], train_data.shape[1]))
        df_scores = []
        for n_clusters in k_values_to_try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed_random)
            labels_clusters = kmeans.fit_predict(train_data)
            
            sh = silhouette_score(train_data, labels_clusters)
            df_scores.append({"no_of_clusters": n_clusters, "silhouette_score": sh})
        df_scores = pl.DataFrame(df_scores)
        return df_scores
    
    df_scores1 = compute_scores(train_data1)
    df_scores2 = compute_scores(train_data2)

    plt.figure(figsize=(8, 5))
    plt.plot(df_scores1['no_of_clusters'], df_scores1['silhouette_score'], marker='o', color='#1f77b4', label='Flats')
    plt.plot(df_scores2['no_of_clusters'], df_scores2['silhouette_score'], marker='s', color='#ff7f0e', label='Houses')
    plt.title("Silhouette Score vs Number of Clusters")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.ylim(0.0, 1)
    plt.legend()
    plt.grid(False)
    plt.savefig("/Users/jansigrist/Documents/SP/Customer_Segmentation_Lantern/Results/Plots/Cluster/KMeans_Silhouette.png", bbox_inches='tight')
    plt.show()
    
    print(f"Minimum Silhouette Score Dataset 1: {max(df_scores1['silhouette_score'])}")
    print(f"Minimum Silhouette Score Dataset 2: {max(df_scores2['silhouette_score'])}")
    
    return df_scores1, df_scores2

# %%



